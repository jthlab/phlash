import jax
import jax.test_util
import numpy as np

from phlash.hmm import psmc_ll
from phlash.model import _fold_afs


def test_pyll_vs_cuda(dm, data, kern):
    ll1 = kern.loglik(dm, 0)
    ll2 = psmc_ll(dm, data[0])[1]
    np.testing.assert_allclose(ll1, ll2, rtol=1e-5)


def test_pyll_grad_vs_cuda(dm, data, kern):
    ll1, dll1 = jax.value_and_grad(kern.loglik)(dm, 0)
    ll2, dll2 = jax.value_and_grad(lambda dm: psmc_ll(dm, data[0])[1])(dm)
    np.testing.assert_allclose(ll1, ll2, atol=1e-8, rtol=1e-5)
    for x, y in zip(dll1, dll2):
        np.testing.assert_allclose(x, y, atol=1e-3, rtol=1e-3)


def test_fold():
    for x, y in [
        ([], []),
        ([1], [1]),
        ([1, 2], [3]),
        (np.arange(5), [4, 4, 2]),
        (np.arange(6), [5, 5, 5]),
    ]:
        np.testing.assert_allclose(_fold_afs(x), y)
