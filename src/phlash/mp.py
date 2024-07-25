import multiprocessing
import os
from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor
from threading import Lock


def _set_jax_to_cpu():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    import jax

    jax.config.update("jax_platforms", "cpu")


class JaxCpuProcessPoolExecutor(ProcessPoolExecutor):
    "class which prevents worker threads from allocating GPU memory when loading Jax"

    def __init__(self, *args, **kwargs):
        # Set the initializer to configure JAX to use CPU in worker processes
        spawn = multiprocessing.get_context("spawn")
        super().__init__(*args, initializer=_set_jax_to_cpu, mp_context=spawn, **kwargs)


class DummyExecutor(Executor):
    def __init__(self, *args, **kwargs):
        _set_jax_to_cpu()
        self._shutdown = False
        self._shutdownLock = Lock()

    def submit(self, fn, *args, **kwargs):
        with self._shutdownLock:
            if self._shutdown:
                raise RuntimeError("cannot schedule new futures after shutdown")

            f = Future()
            try:
                result = fn(*args, **kwargs)
            except BaseException as e:
                f.set_exception(e)
            else:
                f.set_result(result)

            return f

    def shutdown(self, wait=True):
        with self._shutdownLock:
            self._shutdown = True


if os.environ.get("PHLASH_DISABLE_MP"):
    JaxCpuProcessPoolExecutor = DummyExecutor
