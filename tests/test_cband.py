import numpy as np

from phlash.cband import _find_confidence_bands


def test_random_sampled_functions(rng):
    """
    Tests the find_confidence_bands function with randomly sampled functions.
    Checks if the solution satisfies basic confidence band criteria.
    Uses a fixture for the random number generator.
    """
    N, K = rng.integers(10, 100, size=(2,))
    t_test = rng.random(
        K,
    ).cumsum()
    A_test = rng.random((N, K))

    bands = _find_confidence_bands(t_test, A_test)
    upper = bands["upper"]
    lower = bands["lower"]

    # Check if at least 95% of the functions are within the bands at each breakpoint
    assert np.all(np.isclose(lower, A_test).sum(0) == 1)
    assert np.all(np.isclose(upper, A_test).sum(0) == 1)
    lb = (lower < A_test) | np.isclose(lower, A_test)
    ub = (upper > A_test) | np.isclose(upper, A_test)
    n_within = np.all(lb & ub, axis=1).sum()
    assert (
        n_within >= 0.95 * N
    ), "Less than 95% of functions are entirely within the bands"
