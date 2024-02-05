import numpy as np
from scipy.linalg import expm

from phlash.transition import _expQ, transition_matrix


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
