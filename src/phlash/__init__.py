"Bayesian inference of ancestral size history."

import os

# this needs to occur before jax loads
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import sys

import jax
from loguru import logger

from phlash.data import contig
from phlash.mcmc import fit

jax.config.update("jax_enable_x64", True)
if jax.local_devices()[0].platform != "gpu":
    logger.warning(
        "Detected that Jax is not running on GPU; you appear to have "
        "CPU-mode Jax installed. Performance may be improved by installing "
        "Jax-GPU instead. For installation instructions see:\n\n\t{}\n",
        "https://github.com/google/jax?tab=readme-ov-file#installation",
    )

__all__ = ["fit", "contig"]

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError
