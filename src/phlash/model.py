from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
from jax import vmap
from jax.scipy.special import xlogy
from jaxtyping import Array, Float, Float64, Int8, Int64

import phlash.hmm
from phlash.ld.expected import expected_ld
from phlash.params import MCMCParams


@jax.tree_util.register_dataclass
@dataclass
class PhlashMCMCParams(MCMCParams):
    c_tr: jax.Array

    @property
    def c(self):
        # return jax.nn.softplus(self.c_tr)
        return jnp.exp(self.c_tr)

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
        window_size: int = 100,
        N0: float = None,
    ):
        mcp = MCMCParams.from_linear(
            pattern_str=pattern_str,
            t1=t1,
            tM=tM,
            theta=theta,
            rho=rho,
            window_size=window_size,
            N0=N0,
        )
        return cls(c_tr=jnp.log(c), **asdict(mcp))


def log_prior(mcp: MCMCParams, alpha: float, beta: float) -> float:
    ret = 0.0
    ret += sum(
        jax.scipy.stats.norm.logpdf(a, loc=mu, scale=sigma).sum()
        for (a, mu, sigma) in [
            (mcp.log_rho_over_theta, 0.0, 1.0),
        ]
    )
    ret -= alpha * jnp.sum(jnp.diff(mcp.c_tr) ** 2)
    x, _ = jax.flatten_util.ravel_pytree(mcp)
    ret -= beta * x.dot(x)
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
    alpha: float = 1.0,
    beta: float = 1.0,
    _components: bool = False,
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
    l1 = log_prior(mcp, alpha, beta)

    # sequence contribution, if present
    l2 = 0.0
    if inds is not None:
        pp = mcp.to_pp()
        pi = pp.pi
        if warmup is not None:
            pi = phlash.hmm.psmc_ll(pp, warmup)[0].clip(0, 1)
        pp = pp._replace(pi=pi)
        l2 = kern.loglik(pp, inds)

    # afs contribution, if present
    l3 = 0.0
    if afs is not None:
        l3 = _loglik_afs(dm, afs, afs_transform)

    # ld contribution, if present
    l4 = 0.0
    if ld is not None:
        l4 = _loglik_ld(dm, mcp.N0, ld)

    lls = jnp.array([l1, l2, l3, l4])

    if _components:
        return lls

    ret = jnp.dot(weights, lls)
    return jnp.where(jnp.isfinite(ret), ret, -jnp.inf)


def _loglik_afs(dm, afs, afs_transform):
    ll = 0.0
    for n in afs:
        assert len(n) == 1
        T = afs_transform.get(n, lambda a: a)
        etbl = dm.eta.etbl(n[0])
        esfs = etbl / etbl.sum()
        ll += xlogy(T(afs[n]), T(esfs)).sum()
    return ll


def _loglik_ld(dm, N0, ld):
    @vmap
    def f(ab, d):
        @vmap
        def f(r):
            dmr = dm.rescale(dm.theta / 4 / N0)
            e = expected_ld(dmr.eta, 2 * r, 2 * dmr.theta)
            return jnp.array([e["D2/pi2"], e["Dz/pi2"]])

        # a, b = ab
        # x = jnp.geomspace(a, b, 8)
        # y = f(x)
        # expected = vmap(jnp.trapezoid, (1, None))(y, x) / (b - a)
        expected = f(ab).mean(0)
        # jax.debug.print("obs:{} exp:{}", d["mu"], expected)
        u = d["mu"] - expected
        S = 1e-6 * jnp.eye(len(expected)) + d["Sigma"]
        return -u @ jnp.linalg.solve(S, u)

    ab = jnp.array(list(ld.keys()))
    d = jax.tree.map(lambda *a: jnp.array(a), *ld.values())
    lls = f(ab, d)
    # jax.debug.print("l4s:{}", l4s)
    return lls.sum()
