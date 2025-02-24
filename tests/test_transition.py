import itertools as it

import jax
import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm

from phlash.transition import _expQ, q_s, transition_matrix


def _Q(r, c, n):
    return np.array(
        [
            [-r, r, 0.0],  # non-recombining/invisible recombined
            [1.0 * c, -(n * c), (n - 1) * c],  # floating
            [0.0, 0.0, -0.0],  # visibly recombined
        ]
    )


def test_expq(rng):
    for sigma in 1e-2, 1, 10, 100:
        r, c = sigma**2 * rng.chisquare(1, (2,))
        for n in [2, 10, 20, 50, 100]:
            Q = np.array(_Q(r, c, n))
            P1 = expm(Q)
            P2 = _expQ(r, c, n)
            np.testing.assert_allclose(P1, P2, rtol=1e-4)


def test_transition(dm):
    for n in 2, 5, 10, 50:
        M = transition_matrix(dm, n)
        assert np.all(M >= 0.0)
        np.testing.assert_allclose(M.sum(1), 1.0)


def test_qts_quad(random_eta, rng):
    eta = random_eta()
    s, t = rng.uniform(0.0, 2 * eta.t[-1], size=2)
    q, p_t_eq_s = q_s(eta=eta, s=s, r=1e-6)
    q = jax.jit(q)

    times = sorted([0.0, t, s, eta.t[-1]])
    intg = sum(quad(q, a, b, points=eta.t[1:-1])[0] for a, b in it.pairwise(times))
    intg += quad(q, times[-1], np.inf)[0]
    np.testing.assert_allclose(intg + p_t_eq_s, 1.0, atol=1e-5)
