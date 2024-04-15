import numpy as np
from jax import vmap

from phlash.size_history import SizeHistory
from phlash.util import tree_stack


def confidence_band(
    posterior: list[SizeHistory],
    confidence_level: float = 0.95,
    solver=None,
    approx: bool = True,
) -> tuple[SizeHistory, SizeHistory]:
    """
    Compute a confidence band for a posterior sample of size history functions.

    This function calculates the lower and upper bounds of size histories such that
    a specified percentage of the posterior sample falls within these bounds at all
    time points.

    Args:
        posterior: A list of sample size history functions.
        confidence_level: The desired level of confidence for the bands (default 0.95).
        solver: PuLP solver class. Defaults to pulp.GUROBI if None. It might not be
                available on every machine.
        approx: If True, approximate each function at a dense grid of points. If False,
                take the union of all unique timepoints as the grid.

    Returns:
        tuple[SizeHistory, SizeHistory]: A pair of size histories (lower, upper) with
        the property that 100 * confidence_level percent of the posterior sample are
        between the upper and lower bounds for all time points.

    Notes:
        Setting approx=False greatly increases the size of the optimization
        problem that must be solved, so it is recommended that you leave this
        at the default value of True.

        If the function is still too slow, consider reducing the number of posterior
        samples.
    """
    eta = tree_stack(posterior)
    if approx:
        if isinstance(approx, int):
            M = approx
        else:
            M = 200
        t1 = eta.t[:, 1].min()
        tM = eta.t[:, -1].max()
        t = np.geomspace(t1, tM, M)
        t = np.insert(t, 0, 0.0)
    else:
        t = np.unique(eta.t.reshape(-1))
        np.sort(t)
    A = 1 / 2 / np.asarray(vmap(SizeHistory.__call__, (0, None))(eta, t))
    d = _find_confidence_bands(t, A, confidence_level, solver)
    return (
        SizeHistory(t=t, c=1 / 2 / d["upper"]),
        SizeHistory(t=t, c=1 / 2 / d["lower"]),
    )


def _find_confidence_bands(
    t: np.ndarray, A: np.ndarray, confidence_level=0.95, solver=None
):
    """Finds the optimal confidence bands for a set of piecewise constant functions.

    Args:
        t: A vector of common breakpoints for the piecewise constant
           functions. Shape should be (K,) where K is the number of breakpoints.
        A: A 2D array of function values corresponding to the breakpoints in t. Shape
           should be (N, K) where N is the number of piecewise constant functions.
        confidence_level: The desired level of confidence for the bands.
        solver: PuLP solver class. Defaults to pulp.GUROBI, but may not be available on
                every machine.

    Returns:
        dict: A dictionary containing the optimal upper and lower bounds at each
              breakpoint. Keys are 'upper' and 'lower', each mapping to a list of
              bounds.

    Raises:
        ValueError: If T and A have different shapes.
    """
    # defer the import to here as it prints some annoying warning messages that
    # most users don't need to see
    import pulp as pl

    N, K = A.shape
    if t.shape != (K,):
        raise ValueError("A and t have incompatible shapes")

    # Setting up the problem
    prob = pl.LpProblem("Confidence_Bands", pl.LpMinimize)
    u = pl.LpVariable.dicts("u", range(K), cat="Continuous")
    ell = pl.LpVariable.dicts("l", range(K), cat="Continuous")
    y = pl.LpVariable.dicts(
        "y", range(N), cat="Binary"
    )  # y[i]=1 if function i is within the band
    z = pl.LpVariable.dicts(
        "z", [(i, k) for i in range(N) for k in range(K)], cat="Binary"
    )

    # Objective function
    prob += pl.lpSum([u[k] - ell[k] for k in range(K)])

    M = A.max() - A.min() + 1

    # Adding constraints
    for i in range(N):
        for k in range(K):
            prob += ell[k] <= A[i, k] + M * (1 - y[i])
            prob += u[k] >= A[i, k] - M * (1 - y[i])

    for i in range(N):
        prob += pl.lpSum([z[(i, k)] for k in range(K)]) == K * y[i]

    prob += pl.lpSum([y[i] for i in range(N)]) >= confidence_level * N

    # Solving the problem
    prob.solve(solver)

    # Extracting the solution
    if pl.LpStatus[prob.status] == "Optimal":
        u_solution = np.array([pl.value(u[k]) for k in range(K)])
        l_solution = np.array([pl.value(ell[k]) for k in range(K)])
        # z_solution = np.array([[pl.value(z[i, k]) for k in range(K)]
        #                        for i in range(N)])
        return {"upper": u_solution, "lower": l_solution}
    else:
        raise RuntimeError("No optimal solution found")
