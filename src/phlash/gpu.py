from functools import singledispatchmethod

import jax
import jax.numpy as jnp
from loguru import logger
from phlashlib.gpu import _gpu_ll

from phlash._phlashlib import as_phlashlib_params, safe_log_params
from phlash.params import PSMCParams
from phlash.size_history import DemographicModel


class PSMCKernel:
    """Compatibility wrapper around the phlashlib GPU likelihood kernel."""

    def __init__(self, M, data, double_precision=False, num_gpus: int = None):
        if num_gpus is not None:
            logger.warning("num_gpus is ignored by the phlashlib-backed GPU kernel")
        self.data = jnp.asarray(data, dtype=jnp.int8)
        self.double_precision = double_precision
        self.M = M

    @property
    def float_type(self):
        if self.double_precision:
            return jnp.float64
        return jnp.float32

    def _kernel_params(self, pp: PSMCParams) -> PSMCParams:
        return jax.tree.map(lambda a: jnp.asarray(a, dtype=self.float_type), pp)

    @singledispatchmethod
    def loglik(self, pp: PSMCParams, index: int):
        pp = as_phlashlib_params(self._kernel_params(pp))
        return _gpu_ll(safe_log_params(pp), self.data[index])

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
            first_leaf = jax.tree.leaves(pp)[0]
            in_axes = (0, 0) if first_leaf.ndim > 1 else (None, 0)
            f = jax.vmap(f, in_axes=in_axes)
        return f(pp, index)
