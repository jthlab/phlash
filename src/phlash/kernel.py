from jax.typing import ArrayLike
from loguru import logger

from phlash.hmm import PureJaxPSMCKernel


def get_kernel(M: int, data: ArrayLike, double_precision: bool):
    try:
        # defer loading the gpu module until necessary, to keep from having to init
        # CUDA on overall package load.
        from phlash.gpu import PSMCKernel

        return PSMCKernel(M=M, data=data, double_precision=double_precision)
    except (ImportError, RuntimeError) as e:
        logger.warning(
            "Error when loading GPU code, falling back on pure JAX implmentation. "
            "This will be **much slower**. Error was: {}",
            str(e),
        )
        return PureJaxPSMCKernel(
            M=M,
            data=data,
            double_precision=double_precision,
        )
