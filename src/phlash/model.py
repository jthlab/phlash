from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
from jax import vmap
from jax.scipy.special import xlogy
from jaxtyping import Array, Float, Float64, Int8, Int64

import phlash.hmm
from phlash.ld.expected import expected_ld
from phlash.params import MCMCParams
from phlash.util import softplus_inv


@jax.tree_util.register_dataclass
@dataclass
class PhlashMCMCParams(MCMCParams):
    c_tr: jax.Array

    @property
    def c(self):
        return jax.nn.softplus(self.c_tr)

    @property
    def log_c(self):
        return jnp.log(self.c)

    def to_dm(self) -> phlash.size_history.DemographicModel:
        c = jnp.array(self.pattern.expand(self.c))
        eta = phlash.size_history.SizeHistory(t=self.times, c=c)
        assert eta.t.shape == eta.c.shape
        return phlash.size_history.DemographicModel(
            eta=eta, theta=self.theta, rho=self.rho
        )

    @classmethod
    def from_linear(
        cls,
        c: jax.Array,
        pattern_str: str,
        t1: float,
        tM: float,
        theta: float,
        rho: float,
        alpha: float = 0.0,
        beta: float = 0.0,
        window_size: int = 100,
        N0: float = None,
    ):
        mcp = MCMCParams.from_linear(
            pattern_str=pattern_str,
            t1=t1,
            tM=tM,
            theta=theta,
            rho=rho,
            alpha=alpha,
            beta=beta,
            window_size=window_size,
            N0=N0,
        )
        return cls(c_tr=softplus_inv(c), **asdict(mcp))


def log_prior(mcp: MCMCParams) -> float:
    ret = 0.0
    ret += sum(
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
    weights: Float64[Array, "4"],
    inds: tuple[int, int],  # Int64[Array, "batch"],  # noqa: F821
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
    pi = pp.pi
    if warmup is not None:
        pi = phlash.hmm.psmc_ll(pp, warmup)[0].clip(0, 1)
    pp = pp._replace(pi=pi)
    l1 = log_prior(mcp)
    l2 = kern.loglik(pp, inds)

    # afs contribution, if present
    l3 = 0.0
    if afs is not None:
        for n in afs:
            assert len(n) == 1
            T = afs_transform.get(n, lambda a: a)
            etbl = dm.eta.etbl(n[0])
            esfs = etbl / etbl.sum()
            l3 += xlogy(T(afs[n][1:-1]), T(esfs)).sum()

    # ld contribution, if present
    l4 = 0.0
    if ld:

        @vmap
        def f(ab, d):
            @vmap
            def f(r):
                # dmr = dm.rescale(dm.theta / 4 / mcp.N0)
                return expected_ld(dm.eta, r * 4 * mcp.N0, dm.theta).norm()

            # a, b = ab
            # x = jnp.geomspace(a, b, 8)
            # y = f(x)
            expected = f(ab).mean(0)
            # expected = vmap(jnp.trapezoid, (1, None))(y, x) / (b - a)
            # jax.debug.print("obs:{} exp:{}", d["mu"], expected)
            u = d["mu"] - expected
            return -u @ jnp.linalg.solve(d["Sigma"], u)

        ab = jnp.array(list(ld.keys()))
        d = jax.tree.map(lambda *a: jnp.array(a), *ld.values())
        l4s = f(ab, d)
        # jax.debug.print("l4s:{}", l4s)
        l4 += l4s.sum()

    ll = jnp.array([l1, l2, l3, l4])
    ret = jnp.dot(weights, ll)
    return jnp.where(jnp.isfinite(ret), ret, -jnp.inf)
