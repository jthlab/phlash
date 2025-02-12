"Calculation of expected LD curves"

import jax
import jax.numpy as jnp
from diffrax import Kvaerno3, ODETerm, PIDController, diffeqsolve

from phlash.ld.data import LdStats
from phlash.size_history import SizeHistory


def drift(c):
    D = jnp.array([[-3, 1, 1], [4, -5, 0], [0, 1, -2]])
    return D * c


def recomb(rho):
    R = jnp.array([[-2, 0, 0], [0, -1, 0], [0, 0, 0]])
    return rho / 2 * R


def mutation(theta):
    U = jnp.array([0, 0, 1.0])
    return theta / 2 * U


def stationary_ld(c, r, theta):
    D = drift(c)
    R = recomb(r)
    U = mutation(theta)
    return jnp.linalg.solve(D + R, -U * theta)


def f(t, y, args):
    eta = args["eta"]
    t = eta.t[-1] - t
    c = eta(t)
    mats = args["mats"]
    D = mats["D"] * c
    R = mats["R"]
    U = mats["U"]
    dld = (D + R) @ y["ld"] + U * y["h"]
    dh = -y["h"] * c + args["theta"]
    D2, Dz, pi2 = y["ld"]
    ret = dict(ld=dld, h=dh)
    return ret


def expected_ld_expm(eta: SizeHistory, r: float, theta: float) -> LdStats:
    # integrate the system forwards in time starting from the last epoch
    R = recomb(r)
    U = mutation(theta)
    D = drift(1.0)
    b = theta * jnp.array([0.0, 0.0, 0.0, 1.0])

    def f(y0, tup):
        ci, ti, ti1 = tup
        # form 4x4 rate matrix
        Q11 = ci * D + R
        Q12 = U[:, None]
        Q21 = jnp.zeros([1, 3])
        Q22 = -ci
        Q = jnp.block([[Q11, Q12], [Q21, Q22]])
        dt = ti1 - ti
        # alternatively, compute by eigenvalue decomposition?
        # if we assume that c is bounded by exp(10) <= 1e5 and r, theta are bounded by
        # 1, then the l1 norm of Q (largest absolute column sum) is bounded by
        # 6 * c <= 6e5, so log2(norm(Q)) <= 20
        e_tQ = jax.scipy.linalg.expm(Q * dt, max_squarings=20)
        y = e_tQ @ y0
        y += jnp.linalg.solve(Q, e_tQ @ b - b)
        return y, None

    y0 = jnp.append(stationary_ld(eta.c[-1], r, theta), theta)
    ld, _ = jax.lax.scan(f, y0, (eta.c[:-1], eta.t[:-1], eta.t[1:]), reverse=True)
    return {
        "D2/pi2": ld[0] / ld[2],
        "Dz/pi2": ld[1] / ld[2],
    }


def expected_ld_ode(eta: SizeHistory, r: float, theta: float) -> LdStats:
    ld0 = stationary_ld(eta.c[-1], r, theta)
    norm = {
        "D2/pi2": ld0[0] / ld0[2],
        "Dz/pi2": ld0[1] / ld0[2],
    }
    y0 = {"ld": ld0, "h": theta, "norm": norm}
    del y0["norm"]

    mats = {}
    mats["R"] = recomb(r)
    mats["U"] = mutation(theta)
    mats["D"] = drift(1.0)

    args = dict(mats=mats, eta=eta, theta=theta)

    term = ODETerm(f)
    solver = Kvaerno3()
    stepsize_controller = PIDController(
        rtol=1e-7, atol=1e-7, jump_ts=(eta.t[-1] - eta.t)[::-1]
    )
    if len(eta.t) == 1:
        dt0 = 1e-4
    else:
        dt0 = eta.t[1] / 10.0
    sol = diffeqsolve(
        term,
        solver,
        t0=0,
        t1=eta.t[-1],
        dt0=dt0,
        y0=y0,
        stepsize_controller=stepsize_controller,
        args=args,
    )
    ld = sol.ys["ld"][0]
    return {
        "D2/pi2": ld[0] / ld[2],
        "Dz/pi2": ld[1] / ld[2],
    }
    # norm = sol.ys["norm"]
    # return jax.tree.map(lambda a: a[0], norm)


expected_ld = expected_ld_expm
