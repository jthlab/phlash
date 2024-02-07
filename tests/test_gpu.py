import jax
import jax.test_util
import numpy as np
from pytest import fixture

from phlash.params import PSMCParams

jax.config.update("jax_enable_x64", True)


@fixture
def data(rng):
    U = rng.uniform(size=(128, 1_000))
    data = (1 * (U < 0.01) + 2 * (U < 0.1) + 3 * (U < 0.2)).astype(np.int8)
    # randomly insert some missing data
    inds = rng.integers(0, data.size, size=int(0.01 * data.size))
    data.flat[inds] = -1
    return data.clip(-1, 1)


def rel_err(a, b):
    return np.abs(a - b) / np.abs(a)


def test_check_grads(dm, data, kern):
    jax.test_util.check_grads(
        lambda d: kern.loglik(d, 0), (dm,), order=1, modes=["rev"], rtol=1e-2
    )


def test_eq_grad_nograd(pp: PSMCParams, data, kern):
    "test that the likelihood is the same using either method"
    inds = np.arange(len(data))
    ll1, _ = kern(pp, inds, grad=True)
    ll2 = kern(pp, inds, grad=False)
    np.testing.assert_allclose(ll1, ll2)


# def test_skip_missing(pp: PSMCParams, rng: np.random.Generator, kern_cls):
#     U = rng.uniform(size=(100, 1000))
#     data = (U < 0.01).astype(np.int8)
#     # randomly insert some missing data
#     data[50, -500:] = -1
#     kern = kern_cls(M=pp.M, data=data, double_precision=True)
#     assert np.all(kern._L_max[:50] == 1000)
#     assert np.all(kern._L_max[51:] == 1000)
#     assert kern._L_max[50] == 500
