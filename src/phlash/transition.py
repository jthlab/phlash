from collections.abc import Callable

import jax
import jax.numpy as jnp
from phlashlib.transition import _expQ
from phlashlib.transition import transition_matrix as _transition_matrix

from phlash._phlashlib import as_piecewise_constant, rho_to_phlashlib
from phlash.size_history import DemographicModel


def transition_matrix(dm: DemographicModel, n: int = 2) -> jax.Array:
    return _transition_matrix(
        as_piecewise_constant(dm.eta),
        rho=rho_to_phlashlib(dm.rho),
        n=n,
    )


def q_s(*, eta, s, r) -> tuple[Callable[[float], float], float]:
    """Probability of transitioning from TMRCA=s at left locus to to TMRCA=s at
    right locus when recombination distance is r.

    Args:
        eta: SizeHistory
        s: float
        r: float

    Returns:
        q_ts: absolutely continuous part of the density
        p_s: weight of atom at t=s
    """

    def P(x):
        t_aug = jnp.sort(jnp.append(eta.t, x))
        i = jnp.searchsorted(t_aug, s, side="right")
        c_aug = eta(t_aug)
        dt = jnp.diff(t_aug)
        P = jax.vmap(_expQ, (0, 0, None))(dt * r, dt * c_aug[:-1], 2)
        Pcum = jax.lax.associative_scan(jnp.matmul, P)
        return Pcum[i - 1]

    R = eta.R
    P_s = P(s)

    def q_ts(t):
        return eta(t) * jnp.where(
            t < s,
            P(t)[0, 1],
            P_s[0, 1] * jnp.exp(-(R(t) - R(s))),
        )

    return q_ts, P_s[0, 0]
