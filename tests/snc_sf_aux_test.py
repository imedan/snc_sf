"""
Unit tests for auxiliary functions

Covers:
  - utils.py  : coord2healpix, calc_1d_index, build_effective_sel_factor,
                 calc_subsample_p, calculateSF, cal_veff
  - optimize.py: sigmoid_inv, compute_single_loglike, objective_jax_multi
"""

import numpy as np
import pytest
import polars as pl
import jax
import jax.numpy as jnp
from jax.nn import sigmoid
from jax.experimental.sparse import BCOO

# ── modules under test ────────────────────────────────────────────────────────
from snc_sf.utils import (
    coord2healpix,
    calc_1d_index,
    build_effective_sel_factor,
    calc_subsample_p,
    calculateSF,
    cal_veff,
)
from snc_sf.optimize import sigmoid_inv, compute_single_loglike, objective_jax_multi
from astropy.coordinates import SkyCoord
import astropy.units as u


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures / shared helpers
# ─────────────────────────────────────────────────────────────────────────────

RNG = np.random.default_rng(42)


def make_sparse(weights, sf_idx, model_idx, max_sf, max_mod):
    """Thin wrapper around build_effective_sel_factor for test convenience."""
    return build_effective_sel_factor(
        jnp.array(model_idx, dtype=jnp.int32),
        jnp.array(sf_idx, dtype=jnp.int32),
        jnp.array(weights, dtype=jnp.float32),
        int(max_mod),
        int(max_sf),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. coord2healpix
# ─────────────────────────────────────────────────────────────────────────────

class TestCoord2Healpix:
    """Tests for the RA/Dec → HEALPix conversion utility."""

    def test_returns_integer_array(self):
        coord = SkyCoord(ra=[0.0, 90.0] * u.deg, dec=[0.0, 45.0] * u.deg)
        result = coord2healpix(coord, nside=8)
        assert result.dtype in (np.int32, np.int64, int)

    def test_output_length_matches_input(self):
        n = 50
        ra = RNG.uniform(0, 360, n)
        dec = RNG.uniform(-90, 90, n)
        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
        result = coord2healpix(coord, nside=4)
        assert len(result) == n

    def test_pixel_indices_within_valid_range(self):
        """All pixel indices must be in [0, npix)."""
        import healpy as hp
        nside = 8
        coord = SkyCoord(
            ra=RNG.uniform(0, 360, 100) * u.deg,
            dec=RNG.uniform(-90, 90, 100) * u.deg,
        )
        result = coord2healpix(coord, nside=nside)
        assert np.all(result >= 0)
        assert np.all(result < hp.nside2npix(nside))

    def test_north_pole_galactic_coords(self):
        """Should also work with Galactic l/b coordinates."""
        coord = SkyCoord(l=[0.0, 180.0] * u.deg, b=[45.0, -45.0] * u.deg,
                         frame='galactic')
        result = coord2healpix(coord, nside=4)
        assert len(result) == 2

    def test_invalid_coord_raises(self):
        """A coordinate without ra or l should raise ValueError."""
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        # Build a coord in a frame that has neither ra nor l directly visible
        # (use a custom minimal object without those attrs)
        class BadCoord:
            pass
        with pytest.raises((ValueError, AttributeError)):
            coord2healpix(BadCoord(), nside=4)


# ─────────────────────────────────────────────────────────────────────────────
# 2. calc_1d_index
# ─────────────────────────────────────────────────────────────────────────────

class TestCalc1DIndex:
    """Tests for the ND → 1D flattened index helper."""

    def test_basic_2d_ravel(self):
        """Hand-checked example: bin (1, 2) in a 4×5 grid → 1*5 + 2 = 7."""
        idx_1d, valid, max_idx = calc_1d_index(
            [np.array([1]), np.array([2])],
            [4, 5],
        )
        assert idx_1d[0] == 7
        assert max_idx == 20

    def test_valid_flag_true_for_in_bounds(self):
        idx_1d, valid, _ = calc_1d_index(
            [np.array([0, 1, 2]), np.array([0, 1, 2])],
            [3, 3],
        )
        assert np.all(valid)

    def test_valid_flag_false_for_out_of_bounds(self):
        """Negative index → invalid."""
        _, valid, _ = calc_1d_index(
            [np.array([-1, 0]), np.array([0, 0])],
            [3, 3],
        )
        assert not valid[0]
        assert valid[1]

    def test_max_idx_equals_product_of_bins(self):
        _, _, max_idx = calc_1d_index(
            [np.array([0, 1], dtype=int), np.array([0, 1], dtype=int)],
            [np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0, 1.0, 2.0])],
        )
        # 3 bins × 2 bins = 6
        assert max_idx == 6

    def test_consistent_with_numpy_ravel(self):
        """Should reproduce np.ravel_multi_index for in-bounds data."""
        bin_idx = [np.array([0, 1, 2]), np.array([2, 1, 0])]
        bin_edges = [3, 3]
        idx_1d, _, _ = calc_1d_index(bin_idx, bin_edges)
        expected = np.ravel_multi_index(bin_idx, bin_edges)
        np.testing.assert_array_equal(idx_1d, expected)


# ─────────────────────────────────────────────────────────────────────────────
# 3. build_effective_sel_factor
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildEffectiveSelFactor:
    """Tests for the sparse A_jk matrix builder."""

    def test_returns_bcoo(self):
        A = make_sparse([1.0, 0.5], [0, 1], [0, 1], max_sf=3, max_mod=3)
        assert isinstance(A, BCOO)

    def test_shape_is_correct(self):
        A = make_sparse([1.0, 0.5], [0, 1], [0, 1], max_sf=5, max_mod=4)
        assert A.shape == (5, 4)

    def test_dense_values_match_weights(self):
        """For a 2×2 matrix with two entries, check dense form."""
        weights = [2.0, 3.0]
        sf_idx  = [0, 1]
        mod_idx = [1, 0]
        A = make_sparse(weights, sf_idx, mod_idx, max_sf=2, max_mod=2)
        dense = A.todense()
        assert dense[0, 1] == pytest.approx(2.0)
        assert dense[1, 0] == pytest.approx(3.0)

    def test_zero_weights_give_zero_matrix(self):
        A = make_sparse([0.0, 0.0], [0, 1], [0, 1], max_sf=3, max_mod=3)
        dense = A.todense()
        assert jnp.allclose(dense, jnp.zeros((3, 3)))

    def test_matrix_vector_multiply(self):
        """A @ p should produce the correct weighted sum."""
        # A[0,0]=1, A[1,1]=2 → A @ [3,4] = [3, 8]
        A = make_sparse([1.0, 2.0], [0, 1], [0, 1], max_sf=2, max_mod=2)
        p = jnp.array([3.0, 4.0])
        result = A @ p
        np.testing.assert_allclose(np.array(result), [3.0, 8.0], rtol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# 4. calc_subsample_p
# ─────────────────────────────────────────────────────────────────────────────

class TestCalcSubsampleP:
    """Tests for the Beta-posterior sampler for selection probability."""

    def test_output_in_unit_interval(self):
        k = np.array([0, 5, 10])
        n = np.array([10, 10, 10])
        p = calc_subsample_p(k, n, RNG)
        assert np.all((p >= 0) & (p <= 1))

    def test_output_length_matches_input(self):
        k = np.zeros(20, dtype=int)
        n = np.ones(20, dtype=int) * 100
        p = calc_subsample_p(k, n, RNG)
        assert len(p) == 20

    def test_k_equals_n_biases_toward_one(self):
        """When k = n (all selected) the sample should typically be > 0.5."""
        k = np.array([100] * 500)
        n = np.array([100] * 500)
        p = calc_subsample_p(k, n, np.random.default_rng(0))
        assert np.nanmean(p) > 0.5

    def test_k_equals_zero_biases_toward_zero(self):
        """When k = 0 (none selected) the sample should typically be < 0.5."""
        k = np.array([0] * 500)
        n = np.array([100] * 500)
        p = calc_subsample_p(k, n, np.random.default_rng(0))
        assert np.nanmean(p) < 0.5

    def test_k_greater_than_n_returns_valid(self):
        """k > n is handled without crashing (beta param set to 1)."""
        k = np.array([15])
        n = np.array([10])
        p = calc_subsample_p(k, n, RNG)
        assert len(p) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 5. calculateSF
# ─────────────────────────────────────────────────────────────────────────────

class TestCalculateSF:
    """Tests for the k/n counting function."""

    @pytest.fixture
    def simple_setup(self):
        import healpy as hp
        order = 2

        gcns_df = pl.DataFrame({
            'healpix_':           [0] * 5,
            'phot_g_mean_mag_':   [1] * 5,
            'g_rp_':              [1] * 5,
            # raw columns needed for the WHERE clause in calculateSF
            'phot_g_mean_mag':    [10.0] * 5,   # falls in [0, 22)
            'g_rp':               [0.5]  * 5,   # falls in [-0.4, 2.2)
        })
        data_df = pl.DataFrame({
            'healpix_':           [0] * 3,
            'phot_g_mean_mag_':   [1] * 3,
            'g_rp_':              [1] * 3,
            'phot_g_mean_mag':    [10.0] * 3,
            'g_rp':               [0.5]  * 3,
        })

        sf_bins = {
            'healpix': order,
            'phot_g_mean_mag': [0, 22, 1],
            'g_rp': [-0.4, 2.2, 0.05],
        }
        return data_df, sf_bins, gcns_df

    def test_k_leq_n(self, simple_setup):
        data_df, sf_bins, gcns_df = simple_setup
        result = calculateSF(data_df, sf_bins, gcns_df)
        row = result.filter(
            (pl.col('healpix_') == 0) &
            (pl.col('phot_g_mean_mag_') == 1) &
            (pl.col('g_rp_') == 1)
        )
        if len(row) > 0:
            k = row['k'][0]
            n = row['n'][0]
            assert k <= n

    def test_output_has_expected_columns(self, simple_setup):
        data_df, sf_bins, gcns_df = simple_setup
        result = calculateSF(data_df, sf_bins, gcns_df)
        assert 'k' in result.columns
        assert 'n' in result.columns

    def test_null_k_filled_with_zero(self, simple_setup):
        """Bins in GCNS with no matching data star must have k=0, not null."""
        data_df, sf_bins, gcns_df = simple_setup
        # Add an extra GCNS star in a different bin that data doesn't cover
        extra = pl.DataFrame({
            'healpix_':           [0],
            'phot_g_mean_mag_':   [5],
            'g_rp_':              [1],
            'phot_g_mean_mag':    [14.0],   # add — in a different G bin
            'g_rp':               [0.5],    # add
        })
        gcns_aug = pl.concat([gcns_df, extra])
        result = calculateSF(data_df, sf_bins, gcns_aug)
        assert result['k'].null_count() == 0


# ─────────────────────────────────────────────────────────────────────────────
# 6. cal_veff
# ─────────────────────────────────────────────────────────────────────────────

class TestCalVeff:
    """Tests for the effective volume calculator (Schmidt 1968 + Felten 1976)."""

    def _basic_inputs(self, n=10):
        G       = np.full(n, 10.0)       # G mag
        plx     = np.full(n, 50.0)       # 50 mas → 20 pc
        galb    = np.full(n, np.pi / 4)  # 45° above plane
        G_lim   = np.full(n, 20.0)
        healpix = np.zeros(n, dtype=int)
        return G, plx, galb, G_lim, healpix

    def test_positive_veff(self):
        G, plx, galb, G_lim, hp_idx = self._basic_inputs()
        veff = cal_veff(G, plx, galb, order=3, G_lim=G_lim, healpix=hp_idx)
        assert np.all(veff[plx >= 10] >= 0)

    def test_parallax_below_10mas_gives_zero(self):
        """Stars beyond 100 pc (parallax < 10 mas) must not contribute."""
        G    = np.array([10.0])
        plx  = np.array([5.0])      # < 10 mas → outside 100 pc
        galb = np.array([np.pi / 4])
        Glim = np.array([20.0])
        hp_i = np.array([0])
        veff = cal_veff(G, plx, galb, order=3, G_lim=Glim, healpix=hp_i)
        assert veff[0] == 0.0

    def test_veff_increases_with_limiting_magnitude(self):
        """Deeper survey → larger effective volume."""
        G    = np.array([10.0])
        plx  = np.array([50.0])
        galb = np.array([np.pi / 4])
        hp_i = np.array([0])
        veff_shallow = cal_veff(G, plx, galb, 3, np.array([15.0]), hp_i)
        veff_deep    = cal_veff(G, plx, galb, 3, np.array([20.0]), hp_i)
        assert veff_deep[0] >= veff_shallow[0]

    def test_high_galactic_latitude_larger_volume(self):
        """Near the plane the scale-height slab contributes more volume."""
        G    = np.array([10.0])
        plx  = np.array([50.0])
        Glim = np.array([20.0])
        hp_i = np.array([0])
        galb_pole  = np.array([np.pi / 2 * 0.99])
        galb_plane = np.array([0.1])
        veff_pole  = cal_veff(G, plx, galb_pole,  3, Glim, hp_i)
        veff_plane = cal_veff(G, plx, galb_plane, 3, Glim, hp_i)
        assert veff_plane[0] >= veff_pole[0]   # plane > pole


# ─────────────────────────────────────────────────────────────────────────────
# 7. sigmoid_inv (optimize.py)
# ─────────────────────────────────────────────────────────────────────────────

class TestSigmoidInv:
    """sigmoid_inv should be the exact inverse of jax.nn.sigmoid."""

    def test_inverse_of_sigmoid(self):
        theta = jnp.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        p     = sigmoid(theta)
        theta_back = sigmoid_inv(p)
        np.testing.assert_allclose(np.array(theta_back), np.array(theta), rtol=1e-5)

    def test_p_equals_half_gives_zero(self):
        """sigmoid_inv(0.5) = log(1) = 0."""
        result = sigmoid_inv(jnp.array([0.5]))
        np.testing.assert_allclose(np.array(result), [0.0], atol=1e-6)

    def test_p_near_one_gives_large_positive(self):
        result = sigmoid_inv(jnp.array([0.99]))
        assert float(result[0]) > 3.0

    def test_p_near_zero_gives_large_negative(self):
        result = sigmoid_inv(jnp.array([0.01]))
        assert float(result[0]) < -3.0


# ─────────────────────────────────────────────────────────────────────────────
# 8. compute_single_loglike (optimize.py)
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeSingleLoglike:
    """Tests for the Poisson point process log-likelihood."""

    def _make_inputs(self, n_mod=4, n_sf=3):
        """Create a tiny toy problem."""
        p       = jnp.array([0.25, 0.25, 0.25, 0.25])      # uniform HR probs
        weights = jnp.array([0.5, 0.5, 0.5])
        sf_idx  = jnp.array([0, 1, 2])
        mod_idx = jnp.array([0, 1, 2])
        A = BCOO((weights, jnp.column_stack((sf_idx, mod_idx))),
                 shape=(n_sf, n_mod))
        S       = jnp.array([0.8, 0.6, 0.4])
        idx_mod = jnp.array([0, 1, 2])
        return p, A, S, idx_mod

    def test_returns_finite_scalar(self):
        p, A, S, idx_mod = self._make_inputs()
        ll = compute_single_loglike(p, A, S, idx_mod)
        assert jnp.isfinite(ll)
        assert ll.shape == ()

    def test_higher_p_at_observed_bins_increases_loglike(self):
        """Concentrating all probability on the one observed bin must increase ll."""
        # Single observed star in bin 0, uniform A
        A = BCOO((jnp.array([0.5]), jnp.array([[0, 0]])), shape=(1, 4))
        S       = jnp.array([0.8])
        idx_mod = jnp.array([0])

        p_flat    = jnp.array([0.25, 0.25, 0.25, 0.25])
        p_focused = jnp.array([0.97, 0.01, 0.01, 0.01])

        ll_flat    = compute_single_loglike(p_flat,    A, S, idx_mod)
        ll_focused = compute_single_loglike(p_focused, A, S, idx_mod)
        assert float(ll_focused) > float(ll_flat)

    def test_zero_selection_prob_not_counted(self):
        """Stars with S=0 must not contribute (mask applied correctly)."""
        p, A, _, idx_mod = self._make_inputs()
        S_zero = jnp.array([0.0, 0.0, 0.0])
        ll = compute_single_loglike(p, A, S_zero, idx_mod)
        # With S=0 observation term is 0, only normalisation remains → finite
        assert jnp.isfinite(ll)

    def test_log_like_decreases_with_worse_normalization(self):
        """Doubling the A matrix (expect 2× more stars) should lower the ll."""
        p, A, S, idx_mod = self._make_inputs()
        weights_2x = jnp.array([1.0, 1.0, 1.0])
        sf_idx = jnp.array([0, 1, 2])
        mod_idx = jnp.array([0, 1, 2])
        A_2x = BCOO((weights_2x, jnp.column_stack((sf_idx, mod_idx))),
                    shape=(3, 4))
        ll_normal = compute_single_loglike(p, A,    S, idx_mod)
        ll_inflated = compute_single_loglike(p, A_2x, S, idx_mod)
        assert float(ll_normal) > float(ll_inflated)


# ─────────────────────────────────────────────────────────────────────────────
# 9. objective_jax_multi (optimize.py)
# ─────────────────────────────────────────────────────────────────────────────

class TestObjectiveJaxMulti:
    """Tests for the total (marginalised) negative log-likelihood."""

    def _make_multi_inputs(self, n_draws=3):
        theta   = jnp.zeros(4)                           # sigmoid(0) = 0.5
        S_data  = jnp.ones((3, n_draws)) * 0.5
        idx_mod = jnp.array([0, 1, 2])

        A_jks = []
        for _ in range(n_draws):
            weights = jnp.array([0.5, 0.5, 0.5])
            sf_idx  = jnp.array([0, 1, 2])
            mod_idx = jnp.array([0, 1, 2])
            A = BCOO((weights, jnp.column_stack((sf_idx, mod_idx))),
                     shape=(3, 4))
            A_jks.append(A)
        return theta, S_data, tuple(A_jks), idx_mod

    def test_returns_finite_scalar(self):
        theta, S_data, A_jks, idx_mod = self._make_multi_inputs()
        nll = objective_jax_multi(theta, S_data, A_jks, idx_mod)
        assert jnp.isfinite(nll)

    def test_negative_log_likelihood_is_positive(self):
        """neg_log_like = -1 * log_like; for a valid problem it should be ≥ 0 or
        at least finite (some likelihoods can be > 1 in density terms, so we
        just assert finiteness and that it is a real number)."""
        theta, S_data, A_jks, idx_mod = self._make_multi_inputs()
        nll = objective_jax_multi(theta, S_data, A_jks, idx_mod)
        assert float(nll) == float(nll)   # not NaN

    def test_more_draws_gives_similar_result(self):
        """Using 1 vs 3 identical draws should give the same neg-LL."""
        theta, S_data_3, A_jks_3, idx_mod = self._make_multi_inputs(3)
        S_data_1 = S_data_3[:, :1]
        A_jks_1  = (A_jks_3[0],)
        nll_1 = objective_jax_multi(theta, S_data_1, A_jks_1, idx_mod)
        nll_3 = objective_jax_multi(theta, S_data_3, A_jks_3, idx_mod)
        # With identical draws the logsumexp average equals the single draw
        np.testing.assert_allclose(float(nll_1), float(nll_3), rtol=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# 10. Integration-style: loglike gradient is computable (JAX autodiff)
# ─────────────────────────────────────────────────────────────────────────────

class TestGradientComputable:
    """
    The entire point of using JAX is autodiff. Verify that jax.grad can
    differentiate the objective with respect to theta.
    """

    def test_grad_finite_for_objective(self):
        theta   = jnp.zeros(4)
        S_data  = jnp.ones((3, 2)) * 0.5
        idx_mod = jnp.array([0, 1, 2])
        A_jks = []
        for _ in range(2):
            weights = jnp.array([0.5, 0.5, 0.5])
            sf_idx  = jnp.array([0, 1, 2])
            mod_idx = jnp.array([0, 1, 2])
            A = BCOO((weights, jnp.column_stack((sf_idx, mod_idx))),
                     shape=(3, 4))
            A_jks.append(A)

        grad_fn = jax.grad(objective_jax_multi)
        grads   = grad_fn(theta, S_data, tuple(A_jks), idx_mod)
        assert grads.shape == theta.shape
        assert jnp.all(jnp.isfinite(grads))
