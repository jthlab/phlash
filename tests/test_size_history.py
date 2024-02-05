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
    _W_matrix(10)


def test_expm1inv(rng):
    y = rng.normal(size=100) * 10
    np.testing.assert_allclose(1.0 / np.expm1(y), _expm1inv(y))
