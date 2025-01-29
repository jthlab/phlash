"Calculation of expected LD curves"

import jax
import jax.numpy as jnp
from diffrax import Kvaerno3, ODETerm, PIDController, diffeqsolve

from phlash.ld.data import LdStats
from phlash.size_history import SizeHistory

# this matters a lot for gradient accuracy
jax.config.update("jax_enable_x64", True)


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
    dnorm = {
        "D2/pi2": (pi2 * dld[0] - D2 * dld[2]) / pi2**2,
        "Dz/pi2": (pi2 * dld[1] - Dz * dld[2]) / pi2**2,
    }
    ret = dict(ld=dld, h=dh, norm=dnorm)
    # jax.debug.print('t:{} c:{} y:{} ret:{}', t, c, y, ret)
    return ret


def expected_ld(eta: SizeHistory, r: float, theta: float) -> LdStats:
    "expected LD statistics at recombination distance r"
    ld0 = stationary_ld(eta.c[-1], r, theta)
    norm = {
        "D2/pi2": ld0[0] / ld0[2],
        "Dz/pi2": ld0[1] / ld0[2],
    }
    y0 = {"ld": ld0, "h": theta, "norm": norm}

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
    sol = diffeqsolve(
        term,
        solver,
        t0=0,
        t1=eta.t[-1],
        dt0=1e-3,
        y0=y0,
        stepsize_controller=stepsize_controller,
        args=args,
    )
    e = sol.ys["norm"]
    return e  # LdStats(**d)
