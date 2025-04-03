"Likelihood of an ARG"

import jax
import jax.numpy as jnp
from jax import vmap
from jax.scipy.special import xlogy

import phlash.afs
import phlash.transition
from phlash.size_history import DemographicModel


def log_density(dm: DemographicModel, data, **kwargs):
    """Compute the log density of the data given the demographic model.

    Args:
        dm (DemographicModel): The demographic model.
        data [L, 2]: Array of (tmrca, span) pairs giving the TMRCA and span of each
            segment.
        **kwargs: Additional arguments.

    Notes:
        - Successive spans that have the same TMRCA should be merged into one span:
          <tmrca, span> + <tmrca, span> = <tmrca, span + span>.
    """
    r = dm.rho
    eta = dm.eta
    # always some probability of coalescing in any interval, avoid nan
    eta = eta._replace(c=jnp.maximum(eta.c, 1e-20))
    R = eta.R
    times = data[..., 0].reshape(-1)
    times = jnp.sort(times)
    cs = jax.vmap(eta)(times)
    dt = jnp.diff(times)
    tis = jnp.searchsorted(times, data[..., 0], side="right")  # [L, 2]

    def eQ(dt, c):
        dt_safe = jnp.where(dt > 0.0, dt, 1.0)
        ret = phlash.transition.expQ(dt_safe * r, dt_safe * c, 2)
        return jnp.where(dt > 0.0, ret, jnp.eye(3))

    P = jax.vmap(eQ)(dt, cs[:-1])

    def f(accum, Pi):
        A = accum @ Pi
        return A, A[0, :2]

    _, Pcum = jax.lax.scan(f, jnp.eye(3), P)
    log_Pcum = jnp.log(Pcum)
    # Pcum = jax.lax.associative_scan(jax.remat(jnp.matmul), P)[:, 0, :2]

    @vmap
    def g(i, j, span):
        # i, j = jnp.searchsorted(times, jnp.array([s, t]), side="right")
        s, t = times[i], times[j]
        log_eta_t = jnp.log(eta(t))
        log_p_span = span * log_Pcum[i - 1, 0]
        log_p_trans = jnp.select(
            [jnp.isclose(s, t), t > s, t < s],
            [
                log_Pcum[i - 1, 0],
                log_eta_t + log_Pcum[i - 1, 1] - (R(t) - R(s)),
                log_eta_t + log_Pcum[j - 1, 1],
            ],
        )
        return jnp.where(span > 0, log_p_trans + log_p_span, 0.0)

    @vmap
    def f(ti, span):
        return g(ti[:-1], ti[1:], span[:-1]).sum()

    spans = data[..., 1]
    ll_seq = f(tis, spans).sum()

    # afs
    ll_afs = 0.0
    afs = kwargs.get("afs", {})
    for n in afs:
        T = phlash.afs.fold_transform(n)
        e = eta.etbl(n)
        e /= e.sum()
        ll_afs += xlogy(T @ afs[n][1:-1], T @ e).sum()

    return ll_seq + ll_afs
