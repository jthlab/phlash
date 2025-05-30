from typing import NamedTuple

import jax.numpy as jnp
from jax import vmap


class JaxPPoly(NamedTuple):
    """Piecewise polynomial, similar to scipy.interpolate.PPoly.

    p(t) = sum_i c[i,j] (t - x[j]) ** i, x[j] <= t < x[j + 1]
    """

    x: jnp.ndarray
    c: jnp.ndarray

    # allow scaling by a constant
    def scale(self, x):
        return self._replace(c=jnp.array(self.c) * x)

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
        z = jnp.zeros([1])
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

    def exp_integral(self, t: float = jnp.inf, const: float = 0.0) -> float:
        r"""Compute the integral $\int_0^t exp[-R(u) + const] du$ for

            R(t) = \int_0^s self(s) ds.

        Args:
            t: The upper limit of the integral.
            const: A constant to add to the exponent.

        Returns:
            The value of the integral.

        Notes:
            Only works for piecewise constant rate functions.
        """
        # ET = \int_0^t exp(-R(u)) du
        #    = \sum_{i=0}^{T - 1} \int_{t_i}^{t_{i+1}} exp(-R(t))

        #    = \sum_{i=0}^{T - 1} \int_{t_i}^{t_{i+1}} exp(-I_i) exp[-a_i (t-t_i)] / a,
        #      if a != 0
        #    = \sum_{i=0}^{T - 1} exp(-I_i) (1 - exp(-a_i * dt_i) / a_i

        #    = \sum_{i=0}^{T - 1} dt_i exp(-I_i), if a = 0
        tinf = jnp.isinf(t)
        t_safe = jnp.where(tinf, 1.0, t)
        assert self.c.ndim == 2
        assert self.c.shape[0] == 1  # piecewise constant
        a = self.c[0]
        dt = jnp.diff(self.x)[:-1]
        # to prevent nan's from infecting the gradients, we handle the last epoch
        # separately.
        integrals = a[:-1] * dt
        z = jnp.zeros_like(a[:1])
        I = jnp.concatenate([z, jnp.cumsum(integrals)])  # noqa: E741
        a0 = jnp.isclose(a, 0.0)
        asafe = jnp.where(a0, 1.0, a)
        exp_integrals = jnp.concatenate(
            [
                jnp.exp(-I[:-1] + const)
                * jnp.where(a0[:-1], dt, -jnp.expm1(-asafe[:-1] * dt) / asafe[:-1]),
                # if a[-1] = 0 then the integral diverges so don't worry about it.
                jnp.exp(-I[-1:] + const) / a[-1:],
            ]
        )
        i = jnp.maximum(jnp.searchsorted(self.x, t_safe, side="right") - 1, 0)
        # c = jnp.exp(-I[i] + const) * -jnp.expm1(-a[i] * (t - self.x[i])) / a[i]
        c = jnp.exp(-I[i] + const) * jnp.where(
            a0[i],
            t_safe - self.x[i],
            -jnp.expm1(-asafe[i] * (t_safe - self.x[i])) / asafe[i],
        )
        mask = jnp.arange(len(self.x) - 1) < i
        return jnp.where(tinf, exp_integrals.sum(), (exp_integrals * mask).sum() + c)
