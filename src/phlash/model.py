import jax
import jax.numpy as jnp
from jax import vmap
from jax.scipy.special import xlogy
from jaxtyping import Array, Float, Float64, Int8, Int64

import phlash.hmm
from phlash.ld.expected import expected_ld
from phlash.params import MCMCParams


def log_prior(mcp: MCMCParams) -> float:
    ret = sum(
        jax.scipy.stats.norm.logpdf(a, loc=mu, scale=sigma).sum()
        for (a, mu, sigma) in [
            (mcp.log_rho_over_theta, 0.0, 1.0),
        ]
    )
    ret -= mcp.alpha * jnp.sum(jnp.diff(mcp.log_c) ** 2)
    x, _ = jax.flatten_util.ravel_pytree(mcp)
    ret -= mcp.beta * x.dot(x)
    return ret


def log_density(
    mcp: MCMCParams,
    c: Float64[Array, "3"],
    inds: Int64[Array, "batch"],  # noqa: F821
    warmup: Int8[Array, "c ell"],
    kern: "phlash.gpu.PSMCKernel",
    afs: Int64[Array, "n"] = None,  # noqa: F821
    ld: dict[tuple[float, float], dict] = None,
    afs_transform: dict[int, Float[Array, "m n"]] = None,  # noqa: F722
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
    pp = mcp.to_pp()
    if warmup is None:
        pis = vmap(lambda _: pp.pi)(inds)  # (I, M)
    else:
        pis = vmap(lambda pp, d: phlash.hmm.psmc_ll(pp, d)[0], (None, 0))(
            pp, warmup
        )  # (I, M)
    pis = pis.clip(0, 1)
    pps = vmap(lambda pi: pp._replace(pi=pi))(pis)
    l1 = log_prior(mcp)
    l2 = vmap(kern.loglik, (0, 0))(pps, inds).sum()

    # afs contribution, if present
    l3 = 0.0
    if afs is not None:
        for n in afs:
            T = afs_transform.get(n, jnp.eye(n - 1))
            assert T.ndim == 2
            assert T.shape[1] == n - 1
            etbl = dm.eta.etbl(n)
            esfs = etbl / etbl.sum()
            l3 += xlogy(T @ afs[n], T @ esfs).sum()

    # ld contribution, if present
    l4 = 0.0
    if ld:
        eta = dm.eta
        e0 = jnp.eye(len(eta.c))[0]
        cg = e0 * eta.c + (1 - e0) * jax.lax.stop_gradient(eta.c)
        eta_ld = eta._replace(c=cg)

        @vmap
        def f(ab, d):
            def f(r):
                # dmr = dm.rescale(dm.theta / 4 / mcp.N0)
                return expected_ld(eta_ld, r * 2 * mcp.N0, dm.theta).norm()

            y = vmap(f)(ab)
            expected = vmap(jnp.mean)(y)
            # expected = f((b - a) / 2)
            # jax.debug.print("obs:{} exp:{}", d["mu"], expected)
            x = d["mu"] - expected
            return -x @ jnp.linalg.solve(d["Sigma"], x)

        ab = jnp.array(list(ld.keys()))
        d = jax.tree.map(lambda *a: jnp.array(a), *ld.values())
        l4s = f(ab, d)
        # jax.debug.print("l4s:{}", l4s)
        l4 += l4s.sum()

    ll = jnp.array([l1, l2, l3, l4])
    ret = jnp.dot(c, ll)
    return jnp.where(jnp.isfinite(ret), ret, -jnp.inf)
