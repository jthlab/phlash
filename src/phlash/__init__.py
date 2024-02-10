"Bayesian inference of ancestral size history."

import os
import warnings

# ignore some annoying warnings that show up in the packages we rely on
warnings.filterwarnings(action="ignore", module="stdpopsim", category=FutureWarning)

# this needs to occur before jax loads
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys

import jax

jax.config.update("jax_enable_x64", True)

from phlash.data import contig
from phlash.mcmc import fit

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
