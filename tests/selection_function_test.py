"""
Unit tests for SNCSelectionFunction (snc_sf.selection_function.py).

Below creates fake datasets and bypasses file I/O that would typically be done
"""

import numpy as np
import pytest
import polars as pl
import healpy as hp
from unittest.mock import patch, MagicMock
from io import BytesIO
import tempfile, os

import jax.numpy as jnp
from jax.experimental.sparse import BCOO

from snc_sf.selection_function import SNCSelectionFunction


# =======================================
# BUILD FAKE DATA
# =======================================

N_GCNS = 40   # fake GCNS stars
N_DATA = 15   # fake observed SNC stars
N_DIST = 99   # distance posterior samples

RNG = np.random.default_rng(42)

SF_BINS = {
    'healpix': 2,
    'phot_g_mean_mag': [0, 22, 1],
    'g_rp': [-0.4, 2.2, 0.05],
}
MG_BINS = [0, 20, 0.5]


def _make_fake_gcns(n=N_GCNS):
    """Return a polars DataFrame mimicking the GCNS CSV."""
    source_ids = np.arange(1000, 1000 + n, dtype=np.int64)
    ra  = RNG.uniform(0,   360, n).astype(np.float32)
    dec = RNG.uniform(-60,  60, n).astype(np.float32)
    plx = RNG.uniform(15,   90, n).astype(np.float32)
    G   = RNG.uniform(8,   17, n).astype(np.float32)
    Grp = RNG.uniform(0.3,  1.8, n).astype(np.float32)
    bprp = RNG.uniform(0.3,  2.5, n).astype(np.float32)

    data = {
        'source_id':                  source_ids,
        'ra':                         ra,
        'ra_error':                   np.full(n, 0.01, np.float32),
        'dec':                        dec,
        'dec_error':                  np.full(n, 0.01, np.float32),
        'parallax':                   plx,
        'parallax_error':             np.full(n, 0.1, np.float32),
        'phot_g_mean_mag':            G,
        'phot_rp_mean_mag':           G - Grp,
        'phot_bp_mean_mag':           bprp + (G - Grp),
        'phot_g_mean_flux_over_error': np.full(n, 500.0, np.float32),
        'ruwe':                       np.full(n, 1.0, np.float32),
        'ipd_frac_multi_peak':        np.zeros(n, np.int32),
    }
    # 99 distance posterior columns (1/parallax ± small noise, in kpc)
    for i in range(1, N_DIST + 1):
        data[f'Dist{i}'] = (1.0 / (plx + RNG.normal(0, 0.5, n))).astype(np.float32)

    # Add GaiaEDR3 column so the distpdf join works
    data['GaiaEDR3'] = source_ids.copy()
    return pl.DataFrame(data)


def _make_fake_data_csv(gcns_df, n_obs=N_DATA):
    """Write a minimal observed-data CSV using the first n_obs GCNS source IDs."""
    source_ids = gcns_df['source_id'][:n_obs].to_numpy()
    df = pl.DataFrame({'source_id': source_ids})
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv',
                                     delete=False) as f:
        df.write_csv(f.name)
        return f.name


def _make_fake_maglim():
    """Return a minimal astropy-FITS-like mock for the maglim HEALPix file."""
    n_pix = hp.order2npix(5)
    maglim_arr = np.full(n_pix, 20.0, dtype=np.float32)

    mock_hdu = MagicMock()
    mock_hdu[1].data = np.array(
        list(zip(maglim_arr)),
        dtype=[('mag80', np.float32)]
    )
    return mock_hdu


# =======================================
# Create a patched SNCSelectionFunction instance
# =======================================

@pytest.fixture(scope='module')
def sf():
    """
    Build an SNCSelectionFunction with:
      - all file checks bypassed
      - GCNS CSV and distpdf CSV replaced with in-memory fake data
      - FITS maglim replaced with a mock
      - gaiaunlimited DR3SelectionFunctionTCG replaced with a stub
        that returns completeness = 1.0 for all stars
      - calc_SF=True, mean=True
    """
    fake_gcns   = _make_fake_gcns()
    # Split gcns into main table and distpdf (join key is GaiaEDR3 / source_id)
    dist_cols   = [f'Dist{i}' for i in range(1, N_DIST + 1)]
    gcns_main   = fake_gcns.drop(['GaiaEDR3'] + dist_cols)
    gcns_dist   = fake_gcns.select(['GaiaEDR3'] + dist_cols)

    data_file   = _make_fake_data_csv(fake_gcns)
    fake_maglim = _make_fake_maglim()

    # Stub for DR3SelectionFunctionTCG — always returns completeness = 1
    mock_dr3 = MagicMock()
    mock_dr3.return_value.query.return_value = np.ones(max(N_GCNS, N_DATA),
                                                        dtype=np.float32)

    def fake_read_csv(path_or_name):
        """Return the correct fake DataFrame depending on which file is asked for."""
        if 'distpdf' in str(path_or_name) or 'GNSC_dist' in str(path_or_name):
            return gcns_dist
        return gcns_main   # GCNS-result.csv

    fake_open_binary_gcns  = MagicMock(); fake_open_binary_gcns.name = 'GCNS-result.csv'
    fake_open_binary_dist  = MagicMock(); fake_open_binary_dist.name = 'GNSC_distpdf.csv'
    fake_open_binary_fits  = MagicMock(); fake_open_binary_fits.name = 'GCNS_healpix_maglim.fit'

    def fake_open_binary(pkg, fname):
        if 'distpdf' in fname:
            return fake_open_binary_dist
        if 'maglim' in fname or fname.endswith('.fit'):
            return fake_open_binary_fits
        return fake_open_binary_gcns

    with patch('snc_sf.selection_function.os.path.isfile', return_value=True), \
         patch('snc_sf.selection_function.open_binary', side_effect=fake_open_binary), \
         patch('snc_sf.selection_function.pl.read_csv', side_effect=fake_read_csv), \
         patch('snc_sf.selection_function.fits.open', return_value=fake_maglim), \
         patch('snc_sf.selection_function.DR3SelectionFunctionTCG', mock_dr3):

        obj = SNCSelectionFunction(
            data_file=data_file,
            sf_bins=SF_BINS,
            MG_bins=MG_BINS,
            mean=True,
            calc_SF=True,
        )

    os.unlink(data_file)
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# 1.  __init__ – attribute existence and basic shapes
# ─────────────────────────────────────────────────────────────────────────────

class TestInit:
    """After construction, all documented public attributes must be present
    and have the right types / shapes."""

    def test_data_is_polars_dataframe(self, sf):
        assert isinstance(sf.data, pl.DataFrame)

    def test_gcns_is_polars_dataframe(self, sf):
        assert isinstance(sf.gcns, pl.DataFrame)

    def test_data_length_leq_gcns_length(self, sf):
        # data is filtered to only GCNS source_ids
        assert len(sf.data) <= len(sf.gcns)

    def test_data_contains_required_columns(self, sf):
        required = {'source_id', 'ra', 'dec', 'parallax',
                    'phot_g_mean_mag', 'g_rp', 'healpix_', 'MG', 'MG_',
                    'phot_g_mean_mag_', 'g_rp_', 'k', 'n'}
        assert required.issubset(set(sf.data.columns))

    def test_gcns_contains_required_columns(self, sf):
        required = {'source_id', 'ra', 'dec', 'parallax',
                    'phot_g_mean_mag', 'g_rp', 'healpix_', 'MG', 'MG_',
                    'phot_g_mean_mag_', 'g_rp_', 'k', 'n'}
        assert required.issubset(set(sf.gcns.columns))

    def test_sf_bins_stored(self, sf):
        assert sf.sf_bins == SF_BINS

    def test_mg_bins_stored(self, sf):
        assert sf.MG_bins == MG_BINS

    def test_coord_is_skycoord(self, sf):
        from astropy.coordinates import SkyCoord
        assert isinstance(sf.coord, SkyCoord)

    def test_coord_gcns_is_skycoord(self, sf):
        from astropy.coordinates import SkyCoord
        assert isinstance(sf.coord_gcns, SkyCoord)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  __init__ – bin index columns are correct
# ─────────────────────────────────────────────────────────────────────────────

class TestBinIndexColumns:
    """The digitised index columns (healpix_, phot_g_mean_mag_, g_rp_, MG_)
    must be within the valid bin range for every row."""

    def test_healpix_index_in_range(self, sf):
        nside  = 2 ** SF_BINS['healpix']
        n_pix  = hp.nside2npix(nside)
        assert sf.data['healpix_'].min() >= 0
        assert sf.data['healpix_'].max() < n_pix

    def test_gmag_index_in_range(self, sf):
        max_ind  = len(np.arange(*SF_BINS['phot_g_mean_mag']))
        assert sf.data['phot_g_mean_mag_'].min() >= 0
        assert sf.data['phot_g_mean_mag_'].max() < max_ind

    def test_grp_index_in_range(self, sf):
        max_ind  = len(np.arange(*SF_BINS['g_rp']))
        assert sf.data['g_rp_'].min() >= 0
        assert sf.data['g_rp_'].max() < max_ind

    def test_mg_index_in_range(self, sf):
        max_ind  = len(np.arange(*MG_BINS))
        assert sf.data['MG_'].min() >= 0
        assert sf.data['MG_'].max() < max_ind

    def test_gmag_bin_index_dtype_is_integer(self, sf):
        assert sf.data['phot_g_mean_mag_'].dtype in (
            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        )

    def test_grp_bin_index_dtype_is_integer(self, sf):
        assert sf.data['g_rp_'].dtype in (
            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        )

    def test_mg_bin_index_dtype_is_integer(self, sf):
        assert sf.data['MG_'].dtype in (
            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3.  __init__ – pre-filtering works
# ─────────────────────────────────────────────────────────────────────────────

class TestPreFiltering:
    """pre_filt should reduce the GCNS size and propagate to all dependent steps."""

    def test_pre_filt_reduces_gcns_size(self):
        fake_gcns   = _make_fake_gcns()
        dist_cols   = [f'Dist{i}' for i in range(1, N_DIST + 1)]
        gcns_main   = fake_gcns.drop(['GaiaEDR3'] + dist_cols)
        gcns_dist   = fake_gcns.select(['GaiaEDR3'] + dist_cols)
        data_file   = _make_fake_data_csv(fake_gcns)
        fake_maglim = _make_fake_maglim()
        mock_dr3    = MagicMock()
        mock_dr3.return_value.query.return_value = np.ones(N_GCNS, np.float32)

        def fake_read_csv(path_or_name):
            if 'distpdf' in str(path_or_name) or 'GNSC_dist' in str(path_or_name):
                return gcns_dist
            return gcns_main

        fake_ob_gcns = MagicMock(); fake_ob_gcns.name = 'GCNS-result.csv'
        fake_ob_dist = MagicMock(); fake_ob_dist.name = 'GNSC_distpdf.csv'
        fake_ob_fits = MagicMock(); fake_ob_fits.name = 'GCNS_healpix_maglim.fit'

        def fake_open_binary(pkg, fname):
            if 'distpdf' in fname: return fake_ob_dist
            if 'maglim' in fname:  return fake_ob_fits
            return fake_ob_gcns

        # Keep only the first half of GCNS stars
        keep_ids = fake_gcns['source_id'][:N_GCNS // 2].to_list()
        pre_filt = pl.col('source_id').is_in(keep_ids)

        with patch('snc_sf.selection_function.os.path.isfile', return_value=True), \
             patch('snc_sf.selection_function.open_binary', side_effect=fake_open_binary), \
             patch('snc_sf.selection_function.pl.read_csv', side_effect=fake_read_csv), \
             patch('snc_sf.selection_function.fits.open', return_value=fake_maglim), \
             patch('snc_sf.selection_function.DR3SelectionFunctionTCG', mock_dr3):

            sf_filtered = SNCSelectionFunction(
                data_file=data_file,
                sf_bins=SF_BINS,
                MG_bins=MG_BINS,
                mean=True,
                calc_SF=False,
                pre_filt=pre_filt,
            )

        os.unlink(data_file)
        assert len(sf_filtered.gcns) <= N_GCNS // 2 + 1  # ≤ because join may drop rows


# ─────────────────────────────────────────────────────────────────────────────
# 4.  calculate_selection_func / sample_posterior
# ─────────────────────────────────────────────────────────────────────────────

class TestSelectionFuncColumns:
    """After calculate_selection_func the key columns k, n, pselect_samps,
    completeness and Veff_samps must be present in both data and gcns."""

    def test_subsamp_has_k_and_n(self, sf):
        assert 'k' in sf.subsamp.columns
        assert 'n' in sf.subsamp.columns

    def test_k_leq_n_everywhere(self, sf):
        assert (sf.subsamp['k'] <= sf.subsamp['n']).all()

    def test_k_no_nulls(self, sf):
        assert sf.subsamp['k'].null_count() == 0

    def test_n_positive(self, sf):
        assert (sf.subsamp['n'] >= 0).all()

    def test_data_has_pselect_samps(self, sf):
        # pselect_samps is joined onto data via subsamp
        assert 'pselect_samps' in sf.data.columns

    def test_gcns_has_pselect_samps(self, sf):
        assert 'pselect_samps' in sf.gcns.columns

    def test_data_has_completeness(self, sf):
        assert 'completeness' in sf.data.columns

    def test_gcns_has_completeness(self, sf):
        assert 'completeness' in sf.gcns.columns

    def test_gcns_has_veff_samps(self, sf):
        assert 'Veff_samps' in sf.gcns.columns

    def test_data_has_veff_samps(self, sf):
        assert 'Veff_samps' in sf.data.columns

    def test_pselect_samps_in_unit_interval(self, sf):
        """All sampled selection probabilities must be in [0, 1]."""
        vals = sf.gcns['pselect_samps'].to_numpy()
        assert np.all((vals >= 0) & (vals <= 1))

    def test_veff_non_negative(self, sf):
        vals = sf.gcns['Veff_samps'].to_numpy()
        assert np.all(vals >= 0)

    def test_bins_with_k_zero_have_zero_pselect(self, sf):
        """
        Bins where k=0 (no SDSS-V detection) must have pselect=0 on GCNS,
        because the code explicitly zeros those out (the paper enforces strict 0
        probability for unvisited fields).
        """
        zero_hp = sf.gcns.filter(pl.col('k') == 0)
        if len(zero_hp) > 0:
            p_vals = zero_hp['pselect_samps'].to_numpy()
            assert np.all(p_vals == 0)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  evalutate_Ajk
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluateAjk:
    """The A_jk sparse matrices must have correct shapes and contain
    non-negative entries."""

    @pytest.fixture(autouse=True)
    def run_ajk(self, sf):
        sf.evalutate_Ajk(weight_volume=False)

    def test_a_jks_is_list(self, sf):
        assert isinstance(sf.A_jks, list)

    def test_a_jks_length_equals_nsamps(self, sf):
        assert len(sf.A_jks) == sf.nsamps

    def test_each_element_is_bcoo(self, sf):
        for A in sf.A_jks:
            assert isinstance(A, BCOO)

    def test_a_jk_shape_rows_is_max_sf_idx(self, sf):
        """Number of rows = max possible selection-function bin index."""
        for A in sf.A_jks:
            assert A.shape[0] == sf.max_idx_sel_gcns

    def test_a_jk_shape_cols_is_max_mod_idx(self, sf):
        """Number of columns = max possible HR-diagram bin index."""
        for A in sf.A_jks:
            assert A.shape[1] == sf.max_idx_mod_gcns

    def test_a_jk_values_non_negative(self, sf):
        for A in sf.A_jks:
            dense = np.array(A.todense())
            assert np.all(dense >= 0)

    def test_weight_volume_flag_stored(self, sf):
        assert sf.weight_volume is False

    def test_gcns_valid_shape(self, sf):
        assert sf.gcns_valid.shape == (len(sf.gcns),)

    def test_gcns_valid_is_boolean(self, sf):
        assert sf.gcns_valid.dtype == bool

    def test_ajk_with_volume_weight_differs(self, sf):
        """Volume-weighted A_jk values should differ from unweighted ones."""
        sf.evalutate_Ajk(weight_volume=False)
        dense_no_vol = np.array(sf.A_jks[0].todense())

        sf.evalutate_Ajk(weight_volume=True)
        dense_vol = np.array(sf.A_jks[0].todense())

        # Restore for other tests
        sf.evalutate_Ajk(weight_volume=False)

        # Not identical (unless all Veff happen to equal 1, which is unlikely)
        assert not np.allclose(dense_no_vol, dense_vol)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  estimate_beta_prior
# ─────────────────────────────────────────────────────────────────────────────

class TestEstimateBetaPrior:
    """Test through the real instance rather than a stub."""

    def test_returns_integer(self, sf):
        filter_data = sf.data
        result = sf.estimate_beta_prior(filter_data)
        assert isinstance(result, int)

    def test_beta_at_least_two(self, sf):
        result = sf.estimate_beta_prior(sf.data)
        assert result >= 2

    def test_rare_subpop_gives_large_beta(self, sf):
        """Filter to just 1 star → fraction ≈ 1/N_DATA → large beta."""
        tiny = sf.data.head(1)
        result = sf.estimate_beta_prior(tiny)
        assert result > 5

    def test_all_data_selected_floors_to_two(self, sf):
        """When every star is in the subpopulation fo = 1 → beta = 2."""
        result = sf.estimate_beta_prior(sf.data)
        # fo = len(sf.data)/len(sf.data) = 1.0 → (1-1)/1 = 0 → floored to 2
        assert result == 2
