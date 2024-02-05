import jax.numpy as jnp
from jax import vmap

from phlash.size_history import DemographicModel, SizeHistory
from phlash.util import tree_stack


def plot_posterior(dms: list[DemographicModel], ax: "matplotlib.axes.Axes" = None):
    if ax is None:
        import matplotlib.pyplot as plt

        ax = plt.gca()
    dms = tree_stack(dms)
    t1, tM = jnp.quantile(dms.eta.t[:, 1:], jnp.array([0.025, 0.975]))
    t = jnp.geomspace(t1, tM, 1000)
    Ne = vmap(SizeHistory.__call__, (0, None, None))(dms.eta, t, True)
    q025, m, q975 = jnp.quantile(Ne, jnp.array([0.025, 0.5, 0.975]), axis=0)
    ax.plot(t, m)
    ax.fill_between(t, q025, q975, alpha=0.1)
    for d in "top", "right":
        ax.spines[d].set_visible(False)
