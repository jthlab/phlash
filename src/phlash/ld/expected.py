"Calculation of expected LD curves"

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
    ld = (D + R) @ y["ld"] + U * y["h"]
    h = -y["h"] * c + args["theta"]
    ret = dict(ld=ld, h=h)
    # jax.debug.print('t:{} c:{} y:{} ret:{}', t, c, y, ret)
    return ret


def expected_ld(eta: SizeHistory, r: float, theta: float) -> LdStats:
    "expected LD statistics at recombination distance r"
    ld0 = stationary_ld(eta.c[-1], r, theta)
    y0 = {"ld": ld0, "h": theta}

    mats = {}
    mats["R"] = recomb(r)
    mats["U"] = mutation(theta)
    mats["D"] = drift(1.0)

    args = dict(mats=mats, eta=eta, theta=theta)

    term = ODETerm(f)
    solver = Kvaerno3()
    stepsize_controller = PIDController(
        rtol=1e-5, atol=1e-5, jump_ts=(eta.t[-1] - eta.t)[::-1]
    )
    sol = diffeqsolve(
        term,
        solver,
        t0=0,
        t1=eta.t[-1],
        dt0=None,
        y0=y0,
        stepsize_controller=stepsize_controller,
        args=args,
    )
    eld = sol.ys["ld"][0]
    d = dict(zip(["D2", "Dz", "pi2"], eld))
    return LdStats(**d)
