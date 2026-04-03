import jax
import jax.numpy as jnp
from phlashlib.iicr import PiecewiseConstant
from phlashlib.params import PSMCParams as PhlashlibPSMCParams


def as_piecewise_constant(eta) -> PiecewiseConstant:
    return PiecewiseConstant(
        t=jnp.asarray(eta.t),
        c=jnp.asarray(eta.c),
    )


def theta_to_phlashlib(theta):
    return jnp.asarray(theta) / 2.0


def rho_to_phlashlib(rho):
    return 2.0 * jnp.asarray(rho)


def safe_log_params(pp):
    return jax.tree.map(lambda a: jnp.log(jnp.clip(jnp.asarray(a), 1e-20)), pp)


def as_phlashlib_params(pp) -> PhlashlibPSMCParams:
    return PhlashlibPSMCParams(*[jnp.asarray(a) for a in pp])
