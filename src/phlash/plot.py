import jax
import jax.numpy as jnp
import scienceplots  # noqa: F401

from phlash.size_history import DemographicModel, SizeHistory
from phlash.util import tree_stack

def plot_posterior(
    dms: list[DemographicModel],
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
    Ne = jax.vmap(SizeHistory.__call__, (0, None, None))(dms.eta, t, True)
    m = jnp.median(Ne, axis=0)
    ax.plot(t, m, **kwargs)
    if credible_width is not None:
        alpha = (1 - credible_width) / 2
        q_lb, q_ub = jnp.quantile(Ne, jnp.array([alpha, 1 - alpha]), axis=0)
        ax.fill_between(t, q_lb, q_ub, alpha=0.1)
    return t, m, jnp.array([q_lb, q_ub]) if credible_width is not None else None
