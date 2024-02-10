import jax
import numpy as np
import scipy
from pytest import fixture
from scipy.integrate import quad

from phlash.size_history import SizeHistory, _expm1inv, _W_matrix


def test_pi():
    eta = SizeHistory(t=np.array([0.0, 1.0, 2.0, 3.0]), c=np.ones(4))
    S = eta.surv()
    np.testing.assert_allclose(S[0], np.exp(-1))
    q = scipy.stats.expon.ppf([0.1, 0.2, 0.3])
    eta = SizeHistory(t=np.concatenate([[0.0], q]), c=np.ones(4))
    S = eta.surv()
    np.testing.assert_allclose(S, [0.9, 0.8, 0.7, 0.0])
    q = scipy.stats.expon.ppf([0.25, 0.5, 0.75])
    eta = SizeHistory(t=np.concatenate([[0.0], q]), c=np.ones(4))
    np.testing.assert_allclose(eta.pi, 0.25)


@fixture
def eta(rng):
    log_dt, log_c = rng.normal(size=(2, 10))
    t = np.exp(log_dt).cumsum()
    t[0] = 0.0
    return SizeHistory(t=t, c=np.exp(log_c))


def test_R(eta, rng):
    t = rng.uniform(eta.t[0], eta.t[-1], size=10)
    R = jax.jit(eta.R.__call__)
    for tt in t:
        Ri, err = quad(eta, 0.0, tt, points=eta.t)
        np.testing.assert_allclose(R(tt), Ri, rtol=1e-5)


def test_etjj(eta):
    etjj = eta.etjj(10)
    assert np.all(etjj[1:] < etjj[:-1])


def test_mean1():
    eta = SizeHistory(t=np.array([0.0, np.inf]), c=np.ones(1))
    n = 20
    etjj = eta.etjj(n)
    k = np.arange(2, n + 1)
    np.testing.assert_allclose(etjj, 2 / k / (k - 1))


def test_W():
    n = 10
    W = _W_matrix(n)
    eta = SizeHistory(t=np.array([0.0, np.inf]), c=np.ones(1))
    etjj = eta.etjj(n)
    v = W @ etjj
    np.testing.assert_allclose(v, 2 / np.arange(1, n))


def test_tv():
    eta1 = SizeHistory(t=np.array([0.0]), c=np.ones(1))
    np.testing.assert_allclose(eta1.tv(eta1), 0.0)
    eta2 = eta1._replace(c=2 * eta1.c)
    tv1 = eta1.tv(eta2, 10)
    assert 0 <= tv1 <= 1


def test_l2():
    eta1 = SizeHistory(t=np.array([0.0]), c=np.ones(1))
    np.testing.assert_allclose(eta1.l2(eta1, 10.0), 0.0)
    eta2 = eta1._replace(c=2 * eta1.c)
    assert eta1.l2(eta2, 1.0) > 0


def test_expm1inv(rng):
    y = rng.normal(size=100) * 10
    np.testing.assert_allclose(1.0 / np.expm1(y), _expm1inv(y))
