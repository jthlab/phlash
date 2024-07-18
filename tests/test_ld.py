import itertools as it

import jax
import jax.numpy as jnp
import mpmath
import numpy as np
from jax import jit
from quadax import quadgk
from scipy.integrate import dblquad, quad

from phlash.ld import S_integral, _g, expected_R2
from phlash.size_history import SizeHistory


def test_g(eta):
    #       g(t) = 2t(1-p_h) = 1 + exp(-2 Gamma(t)) * \int_0^t exp(2 Gamma(v)) dv
    #                 = 1 + \int_0^t exp(-2 [Gamma(t) - Gamma(v)]) dv
    # p_h = 1/2 - exp(-2 Gamma(t)) / 2t * \int_0^t exp(2 Gamma(v)) dv
    t = 1.0
    # closed-form approach
    y1 = _g(t, eta)
    Gamma = eta.R

    @jax.jit
    def f(v):
        return jnp.exp(2 * Gamma(v))

    q, _ = quad(f, 0.0, t, points=eta.t[:-1])
    p_h = 1 / 2 - jnp.exp(-2 * Gamma(t)) / 2 / t * q
    y2 = 2 * t * (1 - p_h)
    np.testing.assert_allclose(y1, y2, rtol=1e-5)


def test_S_integral(rng, eta):
    v = rng.uniform(0.0, 2.0)
    y1 = S_integral(v, eta)
    f = eta.density()

    @jax.jit
    def I(u, t):
        g = _g(t, eta)
        return f(t) * jnp.exp(-g * u)

    y2 = 0.0
    for t0, t1 in it.pairwise(np.append(eta.t, np.inf)):
        y2 += dblquad(I, t0, t1, lambda t: 0, lambda t: v)[0]
    np.testing.assert_allclose(y1, y2, rtol=1e-5)


def test_S_integral_Ne_1e4():
    eta = SizeHistory(t=np.zeros(1), c=np.ones(1))
    y1 = S_integral(v, eta)
    f = eta.density()

    @jax.jit
    def I(u, t):
        g = _g(t, eta)
        return f(t) * jnp.exp(-g * u)

    y2 = 0.0
    for t0, t1 in it.pairwise(np.append(eta.t, np.inf)):
        y2 += quadgk(I, t0, t1, v)[0]
    np.testing.assert_allclose(y1, y2, rtol=1e-5)


def test_S_integral_grad(rng, eta):
    v1, v2 = rng.uniform(0.0, 2.0, size=(2,))
    v2 += v1

    @jax.jit
    @jax.value_and_grad
    def f(c):
        eta_c = eta._replace(c=c)
        return jnp.diff(jax.vmap(S_integral, (0, None))(jnp.array([v1, v2]), eta_c))[0]

    f, df = f(eta.c)


def test_g_vs_constant(rng):
    gamma = rng.uniform()
    eta = SizeHistory(t=jnp.zeros(1), c=jnp.array([gamma]))
    t = rng.uniform()
    y1 = _g(t, eta)
    p_h = 1 / 2 + jnp.expm1(-2 * gamma * t) / 4 / gamma / t
    y2 = 2 * t * (1 - p_h)
    np.testing.assert_allclose(y1, y2)


def test_S_integral_vs_constant(rng):
    gamma = rng.uniform()
    eta = SizeHistory(t=jnp.zeros(1), c=jnp.array([gamma]))
    v = rng.uniform(0.0, 2.0)
    S1 = S_integral(v, eta)

    def integrand(u):
        def Power(x, y):
            return x**y

        def Gamma(x):
            return mpmath.gamma(x)

        def Re(x):
            return x.real

        def Im(x):
            return x.imag

        def GammaInc(x, y):
            return mpmath.gammainc(x, y)

        def Exp(x):
            return mpmath.exp(x)

        Pi = mpmath.pi
        # copied from mathematica/CForm
        ret = Im(
            (
                Power(2, (-1 + u / gamma) / 2.0)
                * (
                    Gamma((u + gamma) / (2.0 * gamma))
                    - GammaInc((u + gamma) / (2.0 * gamma), -0.5 * u / gamma)
                )
            )
            / (
                Exp((u + 1j * Pi * u) / (2.0 * gamma))
                * Power(u / gamma, (u + gamma) / (2.0 * gamma))
            )
        )
        return float(ret)

    S2, err = quad(integrand, 0.0, v)
    np.testing.assert_allclose(S1, S2, atol=2 * err)


def test_expected_R2(eta):
    ld_buckets = jnp.arange(0.5, 10.5, 0.5)
    res = expected_R2(eta, ld_buckets)
