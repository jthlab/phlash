"Different parameterizations needed for MCMC and HMM"

from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

import phlash.size_history
import phlash.transition
from phlash.util import Pattern


class PSMCParams(NamedTuple):
    b: Float[Array, "M"]
    d: Float[Array, "M"]
    u: Float[Array, "M"]
    v: Float[Array, "M"]
    lam: Float[Array, "M"]
    pi: Float[Array, "M"]

    @property
    def M(self) -> int:
        "The number of discretization intervals"
        M = self.d.shape[-1]
        assert all(a.shape[-1] == M for a in self)
        return M

    @classmethod
    def from_dm(cls, dm: phlash.size_history.DemographicModel) -> "PSMCParams":
        "Initialize parameters from a demographic model"
        assert dm.M == 16, "require M=16"
        lam = dm.theta * dm.eta.ect()
        pi = dm.eta.pi
        A = phlash.transition.transition_matrix(dm)
        pi, A = jax.tree.map(lambda a: a.clip(1e-20, 1.0 - 1e-20), (pi, A))
        b, d, u = (jnp.diag(A, i) for i in [-1, 0, 1])
        v = A[0, 1:] / A[0, 1]
        ut = u / v
        return cls(
            b=jnp.append(b, 0.0),
            d=d,
            u=jnp.append(ut, 0.0),
            v=jnp.insert(v, 0, 0.0),
            lam=lam,
            pi=pi,
        )


def static_field(**kw):
    return field(metadata=dict(static=True), **kw)


@dataclass(kw_only=True)
class MCMCParams:
    pattern_str: str = static_field()
    t_tr: jax.Array
    log_rho_over_theta: float
    theta: float = static_field()
    window_size: int = static_field()
    N0: float = static_field(default=None)

    def to_pp(self) -> PSMCParams:
        dm = self.to_dm()
        dm = dm._replace(rho=self.window_size * dm.rho)
        return PSMCParams.from_dm(dm)

    # @classmethod
    # def default(cls, theta):
    #     return cls.from_linear(
    #         pattern = "14*1+1*2",
    #         t1=1e-4,
    #         tM=15.0,
    #         log_rho_over_theta=0.,
    #         theta=theta,
    #         alpha=0.0,
    #         beta=0.0,
    #         window_size=100,
    #         N0=1e4,
    #     )

    @property
    def pattern(self) -> Pattern:
        return Pattern(self.pattern_str)

    @classmethod
    def from_linear(
        cls,
        pattern_str: str,
        t1: float,
        tM: float,
        theta: float,
        rho: float,
        window_size: int = 100,
        N0: float = None,
    ) -> "MCMCParams":
        dtM = tM - t1
        t_tr = jnp.array([jnp.log(t1), jnp.log(dtM)])
        return cls(
            pattern_str=pattern_str,
            t_tr=t_tr,
            log_rho_over_theta=jnp.log(rho / theta),
            theta=theta,
            window_size=window_size,
            N0=N0,
        )

    @property
    def times(self):
        t1, tM = self.t
        pi = jnp.ones(self.M - 2) / (self.M - 2)
        Pi = jnp.insert(jnp.cumsum(pi), 0, 0.0)
        t = t1 * (tM / t1) ** Pi
        return jnp.insert(t, 0, 0.0)

    @property
    def t1(self):
        return self.t[0]

    @property
    def tM(self):
        return self.t[1]

    @property
    def M(self):
        return self.pattern.M

    @property
    def rho_over_theta(self):
        return jnp.exp(self.log_rho_over_theta)

    @property
    def rho(self):
        return self.rho_over_theta * self.theta

    @property
    def t(self):
        t = jnp.exp(self.t_tr)
        t1 = t[..., 0]
        dtM = t[..., 1]
        tM = t1 + dtM
        return t1, tM
