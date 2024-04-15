import operator
from functools import singledispatch, singledispatchmethod

import jax
import jax.numpy as jnp
from jax import lax, vmap
from jaxtyping import Array, Int8
from loguru import logger

from phlash.params import PSMCParams
from phlash.size_history import DemographicModel


class PureJaxPSMCKernel:
    "Pure Jax implementation of the PSMC kernel, used as a fallback if no GPU."

    def __init__(self, M, data, double_precision=False, num_gpus: int = None):
        if num_gpus is not None:
            logger.warning("num_gpus is ignored in pure Jax kernel")
        self.data = jnp.array(data)
        self.double_precision = double_precision
        self.M = M

    @property
    def float_type(self):
        if self.double_precision:
            return jnp.float64
        return jnp.float32

    @singledispatchmethod
    def loglik(self, pp: PSMCParams, index: int):
        return psmc_ll(pp, self.data[index])[1]

    # convenience overload mostly to help test code
    @loglik.register
    def _(self, dm: DemographicModel, index):
        return self.loglik(PSMCParams.from_dm(dm), index)

    def __call__(
        self, pp: PSMCParams, index: int, grad: bool
    ) -> tuple[float, PSMCParams]:
        index = jnp.array(index)
        assert index.ndim in (0, 1)
        f = self.loglik
        if grad:
            f = jax.value_and_grad(f)
        if index.ndim == 1:
            f = vmap(f, in_axes=(None, 0))
        return f(pp, index)


def matvec_smc(v, pp: PSMCParams):
    # v @ A where A is the SMC' transition matrix.
    vr = lax.associative_scan(operator.add, jnp.append(v, 0.0)[1:], reverse=True)
    lower = vr * pp.b

    def f(s, tup):
        ppi, vi = tup
        t = s * ppi.v
        s += ppi.u * vi
        return s, t

    _, upper = lax.scan(f, 0.0, (pp, v))

    return lower + pp.d * v + upper


@singledispatch
def psmc_ll(pp: PSMCParams, data: Int8[Array, "L"]) -> tuple[jax.Array, float]:
    # for missing data, set emis[-1] = 1.
    emis = jnp.array([pp.emis0, pp.emis1, jnp.ones_like(pp.emis0)])

    @jax.remat
    def fwd(tup, ob):
        alpha_hat, ll = tup
        alpha_hat = matvec_smc(alpha_hat, pp)
        alpha_hat *= emis[ob]
        c = alpha_hat.sum()
        return (alpha_hat / c, ll + jnp.log(c)), None

    init = (pp.pi, 0.0)
    return lax.scan(fwd, init, data)[0]


@psmc_ll.register
def _(dm: DemographicModel, data: jax.Array) -> float:
    return psmc_ll(PSMCParams.from_dm(dm), data)
