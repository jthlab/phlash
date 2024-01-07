"Different parameterizations needed for MCMC and HMM"
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from jaxtyping import Array, Float

import eastbay.size_history
import eastbay.transition
from eastbay.pattern import Pattern


def softplus_inv(y):
    # y > 0
    return y + jnp.log1p(-jnp.exp(-y))


class PSMCParams(NamedTuple):
    b: Float[Array, "M"]
    d: Float[Array, "M"]
    u: Float[Array, "M"]
    v: Float[Array, "M"]
    emis0: Float[Array, "M"]
    emis1: Float[Array, "M"]
    pi: Float[Array, "M"]

    @property
    def M(self) -> int:
        "The number of discretization intervals"
        M = self.d.shape[-1]
        assert all(a.shape[-1] == M for a in self)
        return M

    @classmethod
    def from_dm(cls, dm: eastbay.size_history.DemographicModel) -> "PSMCParams":
        "Initialize parameterns from a demographic model"
        assert dm.M == 16, "require M=16"
        u = dm.theta * dm.eta.ect()
        emis0 = jnp.exp(-u)
        emis1 = -jnp.expm1(-u)
        pi = dm.eta.pi
        A = eastbay.transition.transition_matrix(dm)
        emis0, emis1, pi, A = jax.tree_map(
            lambda a: a.clip(1e-20, 1.0 - 1e-20), (emis0, emis1, pi, A)
        )
        b, d, u = (jnp.diag(A, i) for i in [-1, 0, 1])
        v = A[0, 1:] / A[0, 1]
        ut = u / v
        return cls(
            b=jnp.append(b, 0.0),
            d=d,
            u=jnp.append(ut, 0.0),
            v=jnp.insert(v, 0, 0.0),
            emis0=emis0,
            emis1=emis1,
            pi=pi,
        )


@jdc.pytree_dataclass
class MCMCParams:
    pattern: jdc.Static[str]
    t_tr: jax.Array
    c_tr: jax.Array
    log_rho: float
    theta: jdc.Static[float]
    log_alpha: jdc.Static[float]

    @classmethod
    def from_linear(
        cls,
        pattern: str,
        t1: float,
        tM: float,
        c: jax.Array,
        theta: float,
        rho: float,
        alpha: float,
    ) -> "MCMCParams":
        dtM = tM - t1
        assert len(Pattern(pattern)) == len(c)  # one c per epoch
        return cls(
            pattern=pattern,
            c_tr=softplus_inv(c),
            t_tr=jnp.log(jnp.array([t1, dtM])),
            theta=theta,
            log_rho=jnp.log(rho),
            log_alpha=jnp.log(alpha),
        )

    def to_dm(self) -> eastbay.size_history.DemographicModel:
        pat = Pattern(self.pattern)
        assert len(pat) == len(self.c)
        t1, dtM = self.t
        tM = t1 + dtM
        t = jnp.insert(jnp.geomspace(t1, tM, pat.M - 1), 0, 0.0)
        c = jnp.array(pat.expand(self.c))
        eta = eastbay.size_history.SizeHistory(t=t, c=c)
        assert eta.t.shape == eta.c.shape
        return eastbay.size_history.DemographicModel(
            eta=eta, theta=self.theta, rho=self.rho
        )

    @property
    def M(self):
        return Pattern(self.pattern).M

    @property
    def rho(self):
        return jnp.exp(self.log_rho)

    @property
    def alpha(self):
        return jnp.exp(self.log_alpha)

    @property
    def t(self):
        return jnp.exp(self.t_tr)

    @property
    def c(self):
        return jax.nn.softplus(self.c_tr)
