import operator
from functools import singledispatch

import jax
import jax.numpy as jnp
from jax import lax

from eastbay.params import PSMCParams
from eastbay.size_history import DemographicModel


def _assert_leaf_shape(pytree, sh):
    def f(a):
        assert a.shape == sh

    jax.tree_map(f, pytree)


def matvec_smc(v, pp: PSMCParams):
    # v @ A where A is the SMC' transition matrix.
    vr = lax.associative_scan(operator.add, jnp.append(v, 0.0)[1:], reverse=True)
    lower = vr * pp.b

    def f(s, tup):
        ppi, vi = tup
        t = s * ppi.v
        s += ppi.u * vi
        return s, t

    _, upper = lax.scan(f, 0.0, (pp, v))

    return lower + pp.d * v + upper


@singledispatch
def psmc_ll(pp: PSMCParams, data: jax.Array) -> float:
    emis = jnp.array([pp.emis0, pp.emis1, jnp.ones_like(pp.emis0)])

    @jax.remat
    def fwd(tup, ob):
        alpha_hat, ll = tup
        alpha_hat = matvec_smc(alpha_hat, pp)
        alpha_hat *= emis[ob]
        c = alpha_hat.sum()
        return (alpha_hat / c, ll + jnp.log(c)), None

    init = (pp.pi, 0.0)
    return lax.scan(fwd, init, data)[0]


@psmc_ll.register
def _(dm: DemographicModel, data: jax.Array) -> float:
    return psmc_ll(PSMCParams.from_dm(dm), data)
