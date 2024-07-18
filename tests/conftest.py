from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from pytest import fixture

from phlash.kernel import get_kernel
from phlash.params import PSMCParams
from phlash.size_history import DemographicModel, SizeHistory

jax.config.update("jax_enable_x64", True)


@fixture
def test_assets():
    return Path(__file__).parent / "fixtures"


@fixture
def psmcfa_file(test_assets):
    return test_assets / "sample.psmcfa"


@fixture(params=[0, 1, 2])
def rng(request):
    return np.random.default_rng(request.param)


@fixture
def data(rng):
    ret = np.sum(rng.uniform(size=(10, 11, 100)) < 0.05, 2)
    return np.stack([np.full_like(ret, 100), ret], 2).astype(np.int8)


@fixture
def dm():
    return DemographicModel.default(pattern="16*1", theta=1e-2, rho=1e-2)


@fixture
def pp(dm) -> PSMCParams:
    return PSMCParams.from_dm(dm)


@fixture
def kern(data):
    return get_kernel(M=16, data=data, double_precision=True)


@fixture
def random_eta(rng):
    def f():
        log_dt, log_c = rng.normal(size=(2, 10))
        t = np.exp(log_dt).cumsum()
        t[0] = 0.0
        return SizeHistory(t=jnp.array(t), c=jnp.exp(log_c))

    return f


@fixture
def eta(random_eta):
    return random_eta()
