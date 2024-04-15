import jax
import jax.test_util
import numpy as np
from pytest import fixture

from phlash.hmm import psmc_ll
from phlash.kernel import get_kernel
from phlash.params import PSMCParams

jax.config.update("jax_enable_x64", True)


@fixture
def missing_data(data, rng):
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


def test_pyll_vs_cuda(dm, data, kern):
    ll1 = kern.loglik(dm, 0)
    ll2 = psmc_ll(dm, data[0])[1]
    np.testing.assert_allclose(ll1, ll2, rtol=1e-4)


def test_pyll_vs_cuda_missing(dm, missing_data):
    kern = get_kernel(M=16, data=missing_data, double_precision=True)
    ll1 = kern.loglik(dm, 0)
    ll2 = psmc_ll(dm, missing_data[0])[1]
    np.testing.assert_allclose(ll1, ll2, rtol=1e-4)


def test_pyll_vg_vs_cuda(dm, data, kern):
    ll1, dll1 = jax.value_and_grad(kern.loglik)(dm, 0)
    ll2, dll2 = jax.value_and_grad(lambda dm: psmc_ll(dm, data[0])[1])(dm)
    np.testing.assert_allclose(ll1, ll2, atol=1e-8, rtol=1e-5)
    for x, y in zip(dll1, dll2):
        np.testing.assert_allclose(x, y, atol=1e-8, rtol=1e-5)
