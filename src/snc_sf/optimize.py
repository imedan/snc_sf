import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax.nn import sigmoid
from jax.experimental.sparse import BCOO
try:
    from jaxlib._jax import ArrayImpl
except ModuleNotFoundError:
    from jaxlib.xla_extension import ArrayImpl


@jax.jit
def compute_single_loglike(p: ArrayImpl, A_jk_bcoo: BCOO,
                           S: ArrayImpl, idx_mod: ArrayImpl) -> float:
    """
    Compute the log likelihood for a single posterior draw
    of the selection function

    Parameters
    ----------
    p: jaxlib._jax.ArrayImpl
        Probability of a subpopulation of GCNS stars being in bins
        across the HR diagram. This is a 1D array raveled
        from a 2D array of G - RP vs M_G
    
    A_jk_bcoo: jax.experimental.sparse.BCOO
        Sparse matrix of the sum of the selection function values
        for the intersection of each selection function bin with
        each HR diagram bin. Should be of shape
        (1D ravel of selection bins, 1D ravel of HR diagram bins)
    
    S: jaxlib._jax.ArrayImpl
        Selection function values for the obserbed data in a subpopulation.

    idx_mod: jaxlib._jax.ArrayImpl
        The HR diagram 1D raveled indexes on the HR diagram of the observed
        data in the subpopulation
    
    Returns
    --------
    log_like: float
        The poisson point process log likelihood
    """
    Z = A_jk_bcoo @ p
    mask = (S > 0)
    p_obs = p[idx_mod] * S
    safe_p_obs = jnp.where(mask, jnp.maximum(p_obs, 1e-300), 1.0)
    log_like_obs = jnp.sum(jnp.log(safe_p_obs))
    log_like_norm = jnp.sum(Z)
    log_like = log_like_obs - log_like_norm
    return log_like


def objective_jax_multi(theta: ArrayImpl, S_data: ArrayImpl, A_jks_bcoo: tuple,
                        idx_mod_data: ArrayImpl) -> float:
    """
    Compute the total log likelihood

    Parameters
    ----------
    theta: jaxlib._jax.ArrayImpl
        The sigmoid of theta gives the probability of a subpopulation
        of GCNS stars being in bins
        across the HR diagram. This is a 1D array raveled
        from a 2D array of G - RP vs M_G.

    S_data: jaxlib._jax.ArrayImpl
        Selection function values for the obserbed data in a subpopulation.
    
    A_jks_bcoo: tuple
        Sparse matrix of the sum of the selection function values
        for the intersection of each selection function bin with
        each HR diagram bin. Should be of shape
        (1D ravel of selection bins, 1D ravel of HR diagram bins). This should
        be a tuple where each index is for a different posterior draw

    idx_mod_data: jaxlib._jax.ArrayImpl
        The HR diagram 1D raveled indexes on the HR diagram of the observed
        data in the subpopulation
    
    Returns
    --------
    neg_log_like: float
        The total poisson point process negative log likelihood
    """
    p = sigmoid(theta)
    p = jnp.nan_to_num(p, nan=0, posinf=1, neginf=0)

    log_like_samples = []
    for i in range(len(A_jks_bcoo)):
        log_like_samples.append(compute_single_loglike(p, A_jks_bcoo[i],
                                                       S_data[:, i], idx_mod_data))
    log_like_samples = jnp.stack(log_like_samples)

    log_like_samples = jnp.where(jnp.isfinite(log_like_samples), log_like_samples, -1e10)

    neg_log_like = -1.0 * (logsumexp(log_like_samples) - jnp.log(len(A_jks_bcoo)))
    return neg_log_like


@jax.jit
def sigmoid_inv(y: ArrayImpl):
    """
    Inverse of the sigmoid
    """
    return jnp.log(y / (1 - y))
