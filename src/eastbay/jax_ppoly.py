from collections.abc import Callable
from numbers import Number
from typing import Any, NamedTuple

import jax.numpy as jnp
from jax import vmap


class RateFunction(NamedTuple):
    f: Callable[[float, Any, Any], float]

    def __call__(self, x, diff_params, nondiff_params):
        return self.f(
            x,
        )


class JaxPPoly(NamedTuple):
    """Piecewise polynomial, similar to scipy.interpolate.PPoly.

    p(t) = sum_i c[i,j] (t - x[j]) ** i, x[j] <= t < x[j + 1]
    """

    x: jnp.ndarray
    c: jnp.ndarray

    # allow scaling by a constant
    def __mul__(self, other):
        if isinstance(other, Number):
            return self._replace(c=jnp.array(self.c) * other)
        return NotImplemented

    def __call__(self, t):
        "Evaluate p(t)"
        # avoid negative indices
        i = jnp.maximum(0, jnp.searchsorted(self.x, t, side="right") - 1)
        ci = self.c[:, i]
        ti = t - self.x[i]
        return jnp.polyval(ci, ti)

    def antiderivative(self):
        c1 = vmap(jnp.polyint, in_axes=1, out_axes=1)(self.c)
        c0 = jnp.polyval(c1[:, :-1], jnp.diff(self.x)[:-1])
        z = jnp.zeros_like(c0[:1])
        d = jnp.cumsum(jnp.concatenate([z, c0]))
        e = jnp.concatenate([c1[:-1], d[None]])
        return JaxPPoly(x=self.x, c=e)

    def derivative(self):
        c1 = vmap(jnp.polyder, in_axes=1, out_axes=1)(self.c)
        c0 = jnp.polyval(c1[:, :-1], jnp.diff(self.x)[:-1])
        z = jnp.zeros_like(c0[:1])
        d = jnp.cumsum(jnp.concatenate([z, c0]))
        e = jnp.concatenate([c1[:-1], d[None]])
        return JaxPPoly(x=self.x, c=e)

    def exp_integral(self) -> float:
        r"""Compute the integral $\int_0^inf exp[-R(t)] dt$ for

            R(t) = \int_0^s self(s) ds.

        Returns:
            The value of the integral.

        Notes:
            Only works for piecewise constant rate functions.
        """
        # ET = \int_0^inf exp(-R(t))
        #    = \sum_{i=0}^{T - 1} \int_{t_i}^{t_{i+1}} exp(-R(t))
        #    = \sum_{i=0}^{T - 1} \int_{t_i}^{t_{i+1}} exp(-I_i) exp[-a_i (t-t_i)] / a
        #    = \sum_{i=0}^{T - 1} exp(-I_i) (1 - exp(-a_i * dt_i) / a_i
        #
        assert self.c.ndim == 2
        assert self.c.shape[0] == 1
        a = self.c[0]
        dt = jnp.diff(self.x)[:-1]
        # to prevent nan's from infecting the gradients, we handle the last epoch
        # separately.
        integrals = a[:-1] * dt
        z = jnp.zeros_like(a[:1])
        I = jnp.concatenate([z, jnp.cumsum(integrals)])  # noqa: E741
        exp_integrals = jnp.concatenate(
            [
                jnp.exp(-I[:-1]) * -jnp.expm1(-a[:-1] * dt) / a[:-1],
                jnp.exp(-I[-1:]) / a[-1:],
            ]
        )
        return exp_integrals.sum()

    # def inverse(self):
    #     """Return the inverse of this function. Only valid for continuous,
    #     strictly monotone, piecewise linear functions; this assumption is not
    #     checked.
    #     """
    #     assert self.c.shape[0] == 2, self.c.shape
    #     breaks = self.c[1]  # [0, p(t1), p(t2), ...]
    #     return TfPoly(
    #         x=tf.concat([breaks, [np.inf]], axis=0),
    #         c=tf.concat([1. / self.c[:1], [self.x[:-1]]], axis=0),
    #     )
