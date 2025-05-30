import jax
import jax.numpy as jnp
import numpy as np
import scipy
from pytest import fixture

from phlash.ppoly import PPoly


@fixture
def p(rng):
    x = np.r_[0.0, np.cumsum(rng.uniform(size=10)), np.inf]
    c = rng.uniform(size=(5, 11))
    return PPoly(x=jnp.array(x), c=jnp.array(c))


def _to_spoly(p):
    return scipy.interpolate.PPoly(x=np.array(p.x), c=np.array(p.c))


@fixture
def q(p):
    return _to_spoly(p)


@fixture
def pconst(rng):
    x = np.r_[0.0, np.cumsum(rng.uniform(size=10)), np.inf]
    c = rng.uniform(size=(1, 11))
    return PPoly(x=jnp.array(x), c=jnp.array(c))


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
    p = PPoly(x=jnp.zeros([1]), c=jnp.ones([1, 1]))
    R1 = p.antiderivative()
    for t in rng.uniform(size=10):
        np.testing.assert_allclose(t, R1(t))


def test_exp_integral(pconst, qconst, rng):
    from scipy.integrate import quad

    for t in np.r_[0.0, rng.uniform(size=3), np.inf]:
        const = np.random.normal()
        y1 = pconst.exp_integral(t=t, const=const)
        R = qconst.antiderivative()
        if np.isinf(t):
            y2a, erra = quad(
                lambda x: np.exp(-R(x) + const), 0.0, R.x[-2], points=R.x[:-2]
            )
            y2b, errb = quad(lambda x: np.exp(-R(x) + const), R.x[-2], np.inf)
            y2 = y2a + y2b
            err = erra + errb
        else:
            i = max(0, np.searchsorted(R.x, t) - 1)
            y2, err = quad(lambda x: np.exp(-R(x) + const), 0.0, t, points=R.x[:i])
        np.testing.assert_allclose(y1, y2, atol=2 * err)


def test_exp_integral_jittable(pconst):
    # check that the function is jittable
    f = jax.jit(pconst.exp_integral)
    f()


def test_scale_integrate(pconst):
    p2 = pconst.scale(2.0)
    R1 = pconst.antiderivative()
    R2 = p2.antiderivative()
    for t in np.random.rand(10):
        np.testing.assert_allclose(R1(t), R2(t) / 2.0)
