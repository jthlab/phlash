"Different parameterizations needed for MCMC and HMM"

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax_dataclasses as jdc
from jaxtyping import Array, Float
from phlashlib.params import PSMCParams as _PSMCParams

from phlash._phlashlib import as_piecewise_constant, rho_to_phlashlib, theta_to_phlashlib
import phlash.size_history
from phlash.util import Pattern, softplus_inv


class PSMCParams(_PSMCParams):

    @classmethod
    def from_dm(cls, dm: phlash.size_history.DemographicModel) -> "PSMCParams":
        "Initialize parameters from a demographic model"
        assert dm.M == 16, "require M=16"
        return cls.from_piecewise_const(
            eta=as_piecewise_constant(dm.eta),
            theta=theta_to_phlashlib(dm.theta),
            rho=rho_to_phlashlib(dm.rho),
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
