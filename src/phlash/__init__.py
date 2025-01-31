"Bayesian inference of ancestral size history."

# ruff: noqa: E402

import os
import warnings

import platformdirs

# ignore some annoying warnings that show up in the packages we rely on
for w in (FutureWarning, UserWarning):
    warnings.filterwarnings(action="ignore", module="stdpopsim", category=w)

# this needs to occur before jax loads
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", platformdirs.user_cache_dir("phlash"))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)


import sys

from phlash.data import Contig

# from phlash.fit.base import fit
from phlash.psmc import psmc
from phlash.size_history import DemographicModel, SizeHistory
from phlash.util import plot_posterior

__all__ = ["Contig", "psmc", "DemographicModel", "SizeHistory", "plot_posterior"]

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
