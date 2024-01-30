import jax
import jax.numpy as jnp
import jax.test_util
import numpy as np
import pytest
from csmp.transition import transition_matrix, transition_probs

from eastbay.model import _fold_afs


def test_par_ll(dm, data):
    pll = PSMCParLoglik(data)
    inds = jnp.arange(len(data) // 2)
    B = 3  # batch size
    dms = jax.vmap(lambda _: dm)(jnp.arange(B))
    ll = pll.loglik(dm, inds=inds)
    assert ll.shape == ()
    lls = jax.vmap(pll.loglik, (0, None))(dms, inds)
    assert lls.shape == (B,)
    np.testing.assert_allclose(ll, lls)


def test_par_ll_grad(dm, data):
    inds = jnp.arange(len(data))
    pll = PSMCParLoglik(data, double_precision=True)
    jax.test_util.check_grads(
        lambda d: pll.loglik(d, inds), (dm,), order=1, modes=["rev"]
    )


def test_pyll_vs_cuda(dm, data, kern):
    ll1 = kern.loglik(dm, 0)
    ll2 = psmc_ll(dm, data[0])[1]
    np.testing.assert_allclose(ll1, ll2, rtol=1e-5)


def test_pyll_vg_vs_cuda(dm, data, kern):
    ll1, dll1 = jax.value_and_grad(kern.loglik)(dm, 0)
    ll2, dll2 = jax.value_and_grad(lambda dm: psmc_ll(dm, data[0])[1])(dm)
    np.testing.assert_allclose(ll1, ll2, atol=1e-8, rtol=1e-5)
    for x, y in zip(dll1, dll2):
        np.testing.assert_allclose(x, y, atol=1e-8, rtol=1e-5)


def test_matvec(rng):
    dm = DemographicModel.default(pattern="16*1", theta=1e-2, rho=1e-2)
    p = transition_probs(dm.eta, dm.rho)
    A = transition_matrix(p)
    v = rng.uniform(size=16)
    v /= v.sum()
    v1 = v @ A
    pp = PSMCParams.from_dm(dm)
    v2 = matvec_smc(v, pp)
    np.testing.assert_allclose(v1, v2)


@pytest.mark.parametrize("double", [True, False])
def test_speed(rng, dm, benchmark, double):
    data = (rng.uniform(size=(300, 100_000)) < 0.05).astype(np.int8)
    pll = PSMCParLoglik(data, double_precision=double)
    inds = np.arange(300)
    f = jax.value_and_grad(pll.loglik)
    jf = jax.jit(f)
    ll, _ = jf(dm, inds)
    ll.block_until_ready()
    benchmark(jf, dm, inds)


def test_fold():
    for x, y in [
        ([], []),
        ([1], [1]),
        ([1, 2], [3]),
        (np.arange(5), [4, 4, 2]),
        (np.arange(6), [5, 5, 5]),
    ]:
        np.testing.assert_allclose(_fold_afs(x), y)
