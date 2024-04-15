"Different parameterizations needed for MCMC and HMM"
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax_dataclasses as jdc
from jaxtyping import Array, Float

import phlash.size_history
import phlash.transition
from phlash.util import Pattern, softplus_inv


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
    def from_dm(cls, dm: phlash.size_history.DemographicModel) -> "PSMCParams":
        "Initialize parameters from a demographic model"
        assert dm.M == 16, "require M=16"
        u = 2 * dm.theta * dm.eta.ect()
        emis0 = jnp.exp(-u)
        emis1 = -jnp.expm1(-u)
        pi = dm.eta.pi
        A = phlash.transition.transition_matrix(dm)
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
    rho_over_theta_tr: float
    theta: jdc.Static[float]
    alpha: jdc.Static[float]
    beta: jdc.Static[float]

    @classmethod
    def from_linear(
        cls,
        pattern: str,
        t1: float,
        tM: float,
        c: jax.Array,
        theta: float,
        rho: float,
        alpha: float = 0.0,
        beta: float = 0.0,
    ) -> "MCMCParams":
        dtM = tM - t1
        t_tr = jnp.array([jnp.log(t1), jnp.log(dtM)])
        assert len(Pattern(pattern)) == len(c)  # one c per epoch
        rho_over_theta_tr = jsp.special.logit((rho / theta - 0.1) / 9.9)
        return cls(
            pattern=pattern,
            c_tr=softplus_inv(c),
            t_tr=t_tr,
            rho_over_theta_tr=rho_over_theta_tr,
            theta=theta,
            alpha=alpha,
            beta=beta,
        )

    def to_dm(self) -> phlash.size_history.DemographicModel:
        pat = Pattern(self.pattern)
        assert len(pat) == len(self.c)
        t1, tM = self.t
        t = jnp.insert(jnp.geomspace(t1, tM, pat.M - 1), 0, 0.0)
        c = jnp.array(pat.expand(self.c))
        eta = phlash.size_history.SizeHistory(t=t, c=c)
        assert eta.t.shape == eta.c.shape
        return phlash.size_history.DemographicModel(
            eta=eta, theta=self.theta, rho=self.rho
        )

    @property
    def M(self):
        return Pattern(self.pattern).M

    @property
    def rho_over_theta(self):
        # this transformation ensures that rho/theta is in [.1, 10]
        return 0.1 + 9.9 * jsp.special.expit(self.rho_over_theta_tr)

    @property
    def rho(self):
        return self.rho_over_theta * self.theta

    @property
    def t(self):
        t1, dtM = jnp.exp(self.t_tr)
        tM = t1 + dtM
        return t1, tM

    @property
    def c(self):
        return jax.nn.softplus(self.c_tr)

    @property
    def log_c(self):
        return jnp.log(self.c)
