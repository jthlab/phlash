import os.path
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from pytest import fixture
from scipy.integrate import quad

from phlash.size_history import SizeHistory, _expm1inv, _tv_helper, _W_matrix


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
    eta = SizeHistory(t=np.array([0.0]), c=np.ones(1))
    n = 20
    etjj = eta.etjj(n)
    k = np.arange(2, n + 1)
    np.testing.assert_allclose(etjj, 2 / k / (k - 1))


def test_W():
    n = 10
    W = _W_matrix(n)
    eta = SizeHistory(t=np.array([0.0]), c=np.ones(1))
    etjj = eta.etjj(n)
    v = W @ etjj
    np.testing.assert_allclose(v, 2 / np.arange(1, n))


def test_tv(eta):
    eta1 = eta
    np.testing.assert_allclose(eta1.tv(eta1), 0.0)
    eta2 = eta1._replace(c=2 * eta1.c)
    tv1 = eta1.tv(eta2, 10)
    assert 0 <= tv1 <= 1


def test_density(eta, rng):
    c = rng.normal() ** 2
    f = jax.jit(eta.density(c))
    I, err = quad(f, 0.0, np.inf)
    np.testing.assert_allclose(I, 1.0, atol=err)


def test_tv_quad(random_eta, rng):
    eta1 = random_eta()
    eta2 = random_eta()
    n = rng.integers(2, 20)
    c = 2 * n * (2 * n - 1) / 2
    f1 = eta1.density(c)
    f2 = eta2.density(c)
    g = jax.jit(lambda t: 0.5 * abs(f1(t) - f2(t)))
    t = sorted(set(eta1.t.tolist()) | set(eta2.t.tolist()))
    tv1a, err = quad(g, 0.0, t[-1], points=t[1:-1])
    tv1b, err = quad(g, t[-1], np.inf)
    tv1 = tv1a + tv1b
    tv2 = eta1.tv(eta2, n)
    np.testing.assert_allclose(tv1, tv2)


def test_l2():
    eta1 = SizeHistory(t=jnp.array([0.0]), c=jnp.ones(1))
    np.testing.assert_allclose(eta1.l2(eta1, 10.0), 0.0)
    eta2 = eta1._replace(c=2 * eta1.c)
    assert eta1.l2(eta2, 1.0) > 0


def test_l2_quad(random_eta, rng):
    eta1 = random_eta()
    eta2 = random_eta()
    T = 5.0 + abs(rng.normal())
    g = jax.jit(lambda t: (eta1(t, Ne=True) - eta2(t, Ne=True)) ** 2)
    t = np.array(sorted(set(eta1.t.tolist()) | set(eta2.t.tolist())))
    t = t[t < T]
    l1a, err = quad(g, 0.0, t[-1], points=t[1:-1])
    l1b, err = quad(g, t[-1], T)
    l1 = jnp.sqrt(l1a + l1b)
    l2 = eta1.l2(eta2, T)
    np.testing.assert_allclose(l1, l2)


def test_expm1inv(rng):
    y = rng.normal(size=100) * 10
    np.testing.assert_allclose(1.0 / np.expm1(y), _expm1inv(y))


def test_tv_helper(rng):
    ab1, ab2 = (a1, b1), (a2, b2) = rng.normal(size=(2, 2)) ** 2
    T = rng.normal() ** 2
    y1 = _tv_helper(ab1, ab2, T)
    y2, err = quad(
        jax.jit(
            lambda t: abs(a1 * jnp.exp(-(a1 * t + b1)) - a2 * jnp.exp(-(a2 * t + b2)))
        ),
        0.0,
        T,
    )
    np.testing.assert_allclose(y1, y2)


def test_tv_bug():
    fn = os.path.join(os.path.dirname(__file__), "fixtures", "tv_bug.pkl")
    eta1, eta2 = pickle.load(open(fn, "rb"))
    eta1, eta2 = (
        SizeHistory(t=jnp.array(eta.t), c=jnp.array(eta.c)) for eta in [eta1, eta2]
    )
    tv1 = eta1.tv(eta2)
    tv2 = eta2.tv(eta1)
    np.testing.assert_allclose(tv1, tv2)
    assert tv1 >= 0.0
    assert tv1 <= 1.0
    f1 = eta1.density()
    f2 = eta2.density()
    t = sorted({float(tt) for eta in [eta1, eta2] for tt in eta.t})
    f = jax.jit(lambda t: 0.5 * abs(f1(t) - f2(t)))
    I1, err1 = quad(f, 0.0, t[-1], points=t[1:-1], limit=2 * len(t))
    # quad really sucks at this second integral.
    t_max = 2 * t[-1]
    while f(t_max) > 1e-20:
        t_max *= 2
    I2, err2 = quad(f, t[-1], t_max)
    np.testing.assert_allclose(tv1, I1 + I2)
