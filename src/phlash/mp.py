import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


def _set_jax_to_cpu():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import jax

    jax.config.update("jax_platforms", "cpu")


class JaxCpuProcessPoolExecutor(ProcessPoolExecutor):
    "class which prevents worker threads from allocating GPU memory when loading Jax"

    def __init__(self, *args, **kwargs):
        # Set the initializer to configure JAX to use CPU in worker processes
        spawn = multiprocessing.get_context("spawn")
        super().__init__(*args, initializer=_set_jax_to_cpu, mp_context=spawn, **kwargs)


if os.environ.get("PHLASH_DISABLE_MP"):
    JaxCpuProcessPoolExecutor = ThreadPoolExecutor
