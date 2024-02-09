import jax
import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array, Float64, Int8, Int64

import phlash.hmm
from phlash.params import MCMCParams, PSMCParams
from phlash.util import fold_afs


def log_prior(mcp: MCMCParams) -> float:
    dm = mcp.to_dm()
    ret = sum(
        jax.scipy.stats.norm.logpdf(a, loc=mu, scale=sigma).sum()
        for (a, mu, sigma) in [
            # (jnp.log(dm.eta.c), 0.0, 3.0),
            (mcp.log_rho, jnp.log(dm.theta), 1.0),
        ]
    )
    ret -= mcp.alpha * jnp.sum(jnp.diff(mcp.c_tr) ** 2)
    return ret


def log_density(
    mcp: MCMCParams,
    c: Float64[Array, "3"],
    inds: Int64[Array, "batch"],
    warmup: Int8[Array, "c ell"],
    kern: "phlash.gpu.PSMCKernel",
    afs: Int64[Array, "n"],
    use_folded_afs: bool,
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
        etbl = dm.eta.etbl(n)
        esfs = etbl / etbl.sum()
        f_afs, f_esfs = map(fold_afs, (afs, esfs))
        l3 = jnp.where(
            use_folded_afs,
            jax.scipy.special.xlogy(f_afs, f_esfs).sum(),
            jax.scipy.special.xlogy(afs, esfs).sum(),
        )

    else:
        l3 = 0.0
    ll = jnp.array([l1, l2, l3])
    ret = jnp.dot(c, ll)
    return jnp.where(jnp.isfinite(ret), ret, -jnp.inf)
