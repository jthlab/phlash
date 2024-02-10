import jax.numpy as jnp
import numpy as np
import scipy
from pytest import fixture

from phlash.jax_ppoly import JaxPPoly


@fixture
def p(rng):
    x = np.r_[0.0, np.cumsum(rng.uniform(size=10)), np.inf]
    c = rng.uniform(size=(5, 11))
    return JaxPPoly(x=jnp.array(x), c=jnp.array(c))


def _to_spoly(p):
    return scipy.interpolate.PPoly(x=np.array(p.x), c=np.array(p.c))


@fixture
def q(p):
    return _to_spoly(p)


@fixture
def pconst():
    x = np.r_[0.0, np.cumsum(np.random.rand(10)), np.inf]
    c = np.random.rand(1, 11)
    return JaxPPoly(x=jnp.array(x), c=jnp.array(c))


@fixture
def qconst(pconst):
    return _to_spoly(pconst)


def test_eval(p, q):
    for t in np.random.rand(10):
        np.testing.assert_allclose(p(t), q(t))


def test_anti(p, q, rng):
    R1 = p.antiderivative()
    R2 = q.antiderivative()
    np.testing.assert_allclose(R1.c, R2.c)
    np.testing.assert_allclose(R1.x, R2.x)
    for t in rng.uniform(size=10):
        np.testing.assert_allclose(R1(t), R2(t))


def test_anti1(rng):
    p = JaxPPoly(x=jnp.zeros([1]), c=jnp.ones([1, 1]))
    R1 = p.antiderivative()
    for t in rng.uniform(size=10):
        np.testing.assert_allclose(t, R1(t))


def test_exp_integral(pconst, qconst):
    from scipy.integrate import quad

    for c in 0.01, 1, 10:
        c *= np.random.rand()
        i1 = pconst.scale(c).exp_integral()
        R = qconst.antiderivative()
        i2, err2 = quad(lambda x: np.exp(-c * R(x)), 0.0, R.x[-2], points=R.x[:-1])
        i3, err3 = quad(lambda x: np.exp(-c * R(x)), R.x[-2], np.inf)
        np.testing.assert_allclose(i1, i2 + i3, atol=err2 + err3)
