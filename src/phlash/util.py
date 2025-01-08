import jax
import jax.numpy as jnp
import jax.tree_util
import matplotlib
import numpy as np
import scienceplots  # noqa: F401
from scipy.interpolate import PPoly

import phlash.size_history


class Pattern:
    """Class for parsing PSMC-style pattern strings."""

    def __init__(self, pattern: str):
        try:
            epochs = self._epochs = []
            for s in pattern.split("+"):
                if "*" in s:
                    k, width = map(int, s.split("*"))
                else:
                    k = 1
                    width = int(s)
                epochs += [width] * k
        except Exception:
            raise ValueError("could not parse pattern")
        if len(epochs) == 0:
            raise ValueError("pattern must contain at least one epoch")
        if any(e <= 0 for e in epochs):
            raise ValueError("epochs must be positive")

    @property
    def M(self):
        return sum(self._epochs)

    def __len__(self) -> int:
        return len(self._epochs)

    def expand(self, x):
        assert len(x) == len(self)
        return sum([e * [xx] for e, xx in zip(self._epochs, x)], [])

    def left_endpoints(self):
        x = []
        for e in self._epochs:
            x.extend([True] + [False] * (e - 1))
        return np.array(x)


def tree_stack(trees):
    return jax.tree.map(lambda *v: jnp.stack(v), *trees)


def tree_unstack(tree):
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    return [treedef.unflatten(leaf) for leaf in zip(*leaves, strict=True)]


def softplus_inv(y):
    # y > 0
    return y + jnp.log1p(-jnp.exp(-y))


def invert_cpwli(R: PPoly):
    """Invert a continuous piecewise-linear increasing function."""
    x = R.x
    assert np.isclose(x[0], 0.0)
    b, a = R.c
    # the inverse func interpolates (R(x[i]), x[i]) for i = 0, 1, ..., n - 1
    return PPoly(x=np.append(a, np.inf), c=np.array([1 / b, x[:-1]]), extrapolate=False)


def plot_posterior(
    dms: list["phlash.size_history.DemographicModel"],
    ax: "matplotlib.axes.Axes" = None,
    credible_width: float = 0.95,
    **kwargs,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Plot the posterior distribution of the effective population size through time.

    Args:
        dms: list of DemographicModel objects
        ax: matplotlib axes object
        credible_width: credible interval width. If None, do not plot credible bands.
        **kwargs: keyword arguments to pass to matplotlib.pyplot.plot

    Returns:
        t: time points
        median: posterior median
        credible_interval: credible interval (None if credible_bands is None)
    """
    if ax is None:
        import matplotlib.pyplot as plt

        plt.style.use("science")
        ax = plt.gca()
    dms = tree_stack(dms)
    t1, tM = jnp.quantile(dms.eta.t[:, 1:], jnp.array([0.025, 0.975]))
    t = jnp.geomspace(t1, tM, 1000)
    Ne = jax.vmap(phlash.size_history.SizeHistory.__call__, (0, None, None))(
        dms.eta, t, True
    )
    m = jnp.median(Ne, axis=0)
    ax.plot(t, m, **kwargs)
    if credible_width is not None:
        alpha = (1 - credible_width) / 2
        q_lb, q_ub = jnp.quantile(Ne, jnp.array([alpha, 1 - alpha]), axis=0)
        ax.fill_between(t, q_lb, q_ub, alpha=0.1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    return t, m, jnp.array([q_lb, q_ub]) if credible_width is not None else None
