from functools import partial

import jax
import jax.test_util
import numpy as np
from pytest import fixture

from eastbay.gpu import PSMCKernel
from eastbay.params import PSMCParams


@fixture(params=[None, 1])
def num_gpus(request):
    return request.param


@fixture
def kern_cls(num_gpus):
    return partial(PSMCKernel, num_gpus=num_gpus)


@fixture
def data(rng):
    U = rng.uniform(size=(1_000, 1_000))
    data = (1 * (U < 0.01) + 2 * (U < 0.1) + 3 * (U < 0.2)).astype(np.int8)
    # randomly insert some missing data
    inds = rng.integers(0, data.size, size=int(0.01 * data.size))
    data.flat[inds] = -1
    return data.clip(-1, 1)


def rel_err(a, b):
    return np.abs(a - b) / np.abs(a)


def test_check_grads(dm, data, kern_cls):
    kern = kern_cls(M=dm.M, data=data, double_precision=True)
    jax.test_util.check_grads(
        lambda d: kern.loglik(d, 0), (dm,), order=1, modes=["rev"], rtol=1e-4
    )


def test_eq_grad_nograd(pp: PSMCParams, data, kern_cls):
    "test that the likelihood is the same using either method"
    kern = kern_cls(M=pp.M, data=data, double_precision=True)
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


# def test_matches_purepy(dm, data):
#     data = np.array([data[:, :10] * 10000, dtype=np.int8)
#     kern = PSMCKernel(dm.M, data)
#     tup = PSMCParams.from_dm(dm)
#     logtup = jax.tree_map(jnp.log, tup)
#     ll1, dll1 = grad_ll(tup, data[0], 0, True)
#     # ll2, dll2 = jax.value_and_grad(_psmc_ll)(logtup, 0, kern)
#     ll3, dll3 = kern(tup, 0)
#     breakpoint()
#     np.testing.assert_allclose(ll1, ll2)
#     # jax.tree_map(np.testing.assert_allclose, dll1, tuple(dll2))


def test_speed(benchmark, pp, rng, kern_cls):
    data = (rng.uniform(size=(1000, 1_000)) < 0.01).astype(np.int8)
    kern = kern_cls(M=16, data=data)
    f = jax.vmap(jax.grad(kern.loglik), (None, 0))
    inds = np.arange(1000)
    benchmark(f, pp, inds)


def test_nsight(pp, kern_cls, data):
    kern = kern_cls(M=16, data=data)
    f = jax.vmap(jax.grad(kern.loglik), (None, 0))
    inds = np.arange(len(data))
    res = f(pp, inds)
    print(res)
