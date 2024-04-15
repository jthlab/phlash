import jax
import jax.numpy as jnp
from jax import vmap
from jax.scipy.special import xlogy
from jaxtyping import Array, Float, Float64, Int8, Int64

import phlash.hmm
from phlash.params import MCMCParams, PSMCParams


def log_prior(mcp: MCMCParams) -> float:
    ret = sum(
        jax.scipy.stats.norm.logpdf(a, loc=mu, scale=sigma).sum()
        for (a, mu, sigma) in [
            (jnp.log(mcp.rho_over_theta), 0.0, 1.0),
        ]
    )
    ret -= mcp.alpha * jnp.sum(jnp.diff(mcp.log_c) ** 2)
    x, _ = jax.flatten_util.ravel_pytree(mcp)
    ret -= mcp.beta * x.dot(x)
    return ret


def log_density(
    mcp: MCMCParams,
    c: Float64[Array, "3"],
    inds: Int64[Array, "batch"],
    warmup: Int8[Array, "c ell"],
    kern: "phlash.gpu.PSMCKernel",
    afs: Int64[Array, "n"],
    afs_transform: Float[Array, "m n"] = None,
) -> float:
    r"""
    Computes the log density of a statistical model by combining the contributions from
    the prior, the hidden Markov model (HMM), and the allele frequency spectrum (AFS)
    model, weighted by given coefficients.

    Args:
        mcp: The Markov Chain Monte Carlo parameters used to specify the model.
        c: Weights for each component of the density - prior, HMM model, and AFS model.
        inds: Mini-batch indices for selecting subsets of the data.
        data: Data matrix used in the model computation.
        kern: An instantiated PSMC Kernel used in the computation.
        afs: The allele frequency spectrum data.
        use_folded_afs: Whether to fold the afs, if ancestral allele is not known.

    Returns:
        The log density, or negative infinity where the result is not finite.
    """
    dm = mcp.to_dm()
    pp = PSMCParams.from_dm(dm)
    pis = vmap(lambda pp, d: phlash.hmm.psmc_ll(pp, d)[0], (None, 0))(
        pp, warmup
    )  # (I, M)
    pps = vmap(lambda pi: pp._replace(pi=pi))(pis)
    l1 = log_prior(mcp)
    l2 = vmap(kern.loglik, (0, 0))(pps, inds).sum()
    if afs is not None:
        n = len(afs) + 1
        if afs_transform is None:
            T = jnp.eye(n - 1)
        else:
            T = afs_transform
        assert T.ndim == 2
        assert T.shape[1] == n - 1
        etbl = dm.eta.etbl(n)
        esfs = etbl / etbl.sum()
        l3 = xlogy(T @ afs, T @ esfs).sum()
    else:
        l3 = 0.0
    ll = jnp.array([l1, l2, l3])
    ret = jnp.dot(c, ll)
    return jnp.where(jnp.isfinite(ret), ret, -jnp.inf)
