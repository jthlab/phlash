import jax
import jax.numpy as jnp

from phlash.size_history import DemographicModel


def _expQ(r, c, n):
    u = jnp.sqrt((c * n) ** 2 - 2 * c * (n - 2) * r + r**2) / 2
    v = (r + c * n) / 2
    w = (r - c * n) / 2

    # t1 = exp(-v) * cosh(u) = .5 * (exp(u-v) + exp(-u-v))
    t1 = (jnp.exp(u - v) + jnp.exp(-(u + v))) / 2.0
    # t2 = exp(-v) * sinh(u) / u
    u_small = u < 1e-6
    u_safe = jnp.where(u_small, 1.0, u)
    t2 = jnp.where(
        u_small,
        jnp.exp(-v) * (1 + u_safe**2 / 6.0),
        (jnp.exp(u - v) - jnp.exp(-(u + v))) / 2.0 / u_safe,
    )
    P_11 = t1 - w * t2
    P_12 = r * t2
    P_21 = c * t2
    P_22 = t1 + w * t2
    return jnp.array(
        [
            [P_11, P_12, 1.0 - P_11 - P_12],
            [P_21, P_22, 1.0 - P_21 - P_22],
            [0.0, 0.0, 1.0],
        ]
    )


def transition_matrix(dm: DemographicModel, n: int = 2) -> jax.Array:
    c_adj = dm.eta.c * (n - 1)
    t = dm.eta.t
    dt = jnp.diff(t)
    ect = dm.eta.ect()

    t_aug = jnp.stack([dm.eta.t, dm.eta.ect()], 1).flatten()
    dt_aug = jnp.diff(t_aug)
    dt0 = jnp.isclose(dt_aug, 0.0)
    dt_safe = jnp.where(dt0, 1.0, dt_aug)
    cr = jnp.repeat(dm.eta.c, 2, axis=0)[:-1]
    P = jax.vmap(_expQ, (0, 0, None))(2 * dt_safe * dm.rho, dt_aug * cr, n)
    P = jnp.where(dt0[:, None, None], jnp.eye(3)[None], P)
    Pinf = jnp.array([[0.0, 0.0, 1.0]] * 3)
    P = jnp.concatenate([jnp.eye(3)[None], P, Pinf[None]], 0)
    Pcum = jax.lax.associative_scan(jnp.matmul, P)

    P_t = Pcum[::2]
    P_ect = Pcum[1::2]

    i, j, ell = jnp.ogrid[: dm.M, : dm.M, : dm.M]
    L = (jnp.diff(P_t[:, 0, 2])[j] * (i > j))[..., 0]
    # diagonal: not floating/no recomb
    d = P_ect[:, 0, 0]
    # diagonal: floating, but coalesces back before end of interval.
    d += P_ect[:, 0, 1] * (
        jnp.append(-jnp.expm1(-(t[1:] - ect[:-1]) * c_adj[:-1]), 1.0)
    )
    # recombines and coalesces in the interval
    d += P_ect[:, 0, 2] - P_t[:-1, 0, 2]
    D = jnp.diag(d)
    # upper triangle
    p = {}
    p["R<=i,C>i|T=i"] = P_ect[:, 0, 1] * (
        jnp.append(jnp.exp(-(t[1:] - ect[:-1]) * c_adj[:-1]), 0.0)
    )  # entries above first diagnoal
    p["C>i|C>i-1,T<i"] = jnp.append(jnp.exp(-dt * c_adj[:-1]), 0.0)
    p["C=i|C>i-1,T<i"] = jnp.append(-jnp.expm1(-dt * c_adj[:-1]), 1.0)
    p = jax.tree_map(lambda a: a.clip(1e-8, 1.0 - 1e-8), p)
    U = (
        p["R<=i,C>i|T=i"][i]
        * jnp.prod(
            p["C>i|C>i-1,T<i"][ell] ** ((i < ell) & (ell < j)), axis=2, keepdims=True
        )
        * p["C=i|C>i-1,T<i"][j]
        * (j > i)
    )[..., 0]
    M = L + D + U
    return M
