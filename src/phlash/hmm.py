from functools import singledispatch, singledispatchmethod

import jax
import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array, Int8
from loguru import logger
from phlashlib.hmm import _matvec_smc
from phlashlib.hmm import forward as _forward

from phlash.params import PSMCParams
from phlash.size_history import DemographicModel


def matvec_smc(v, pp):
    return _matvec_smc(jnp.asarray(v), pp)


class PureJaxPSMCKernel:
    "Pure JAX implementation of the PSMC kernel, used as a fallback if no GPU."

    def __init__(self, M, data, double_precision=False, num_gpus: int = None):
        if num_gpus is not None:
            logger.warning("num_gpus is ignored in pure JAX kernel")
        self.data = jnp.asarray(data, dtype=jnp.int8)
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

    @loglik.register
    def _(self, dm: DemographicModel, index):
        return self.loglik(PSMCParams.from_dm(dm), index)

    def __call__(
        self, pp: PSMCParams, index: int, grad: bool
    ) -> tuple[float, PSMCParams]:
        index = jnp.asarray(index)
        assert index.ndim in (0, 1)
        f = self.loglik
        if grad:
            f = jax.value_and_grad(f)
        if index.ndim == 1:
            f = vmap(f, in_axes=(None, 0))
        return f(pp, index)


@singledispatch
def psmc_ll(pp: PSMCParams, data: Int8[Array, "L"]) -> tuple[jax.Array, float]:
    return _forward(pp, data)


@psmc_ll.register
def _(dm: DemographicModel, data: jax.Array) -> float:
    return psmc_ll(PSMCParams.from_dm(dm), data)
