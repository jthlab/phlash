import itertools as it

import numpy as np
from jax import jit
from jax import numpy as jnp
from jax import vmap

from eastbay.log import getLogger
from eastbay.size_history import DemographicModel, SizeHistory
from eastbay.util import tree_stack

logger = getLogger(__name__)


def _is_running_in_jupyter():
    try:
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:  # Check for Jupyter kernel
            return False
    except (ImportError, AttributeError):
        return False
    return True


def style_axis(ax: "matplotlib.axes.Axes"):
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Generations")
    ax.set_ylabel("$N_e$")


class _IPythonLivePlot:
    def __init__(
        self,
        mcp0: "eastbay.mcmc.MCMCParams",
        truth: "eastbay.size_history.DemographicModel" = None,
    ):
        import matplotlib.pyplot as plt
        from IPython.display import display, set_matplotlib_formats

        set_matplotlib_formats("svg")
        self._fig, self._ax = plt.subplots()
        ax = self._ax
        style_axis(ax)
        theta = 1.0
        if truth is not None:
            theta = truth.theta
            t1, tM = truth.eta.t[[1, -1]]
            ax.plot(
                np.append(truth.eta.t, 5 * truth.eta.t[-1]),
                np.append(truth.eta.Ne, truth.eta.Ne[-1]),
                color="black",
            )
            ax.set_ylim(0.2 * truth.eta.Ne.min(), 5 * truth.eta.Ne.max())
        else:
            t1, tM = mcp0.to_dm().eta.t[[1, -1]]
        x = np.geomspace(0.2 * t1, 5 * tM, 1000)
        (self._line,) = ax.plot([], [], color="tab:blue")
        self._poly = [ax.fill_between([], 0, 0, color="tab:blue")]
        self._display = display(self._fig, display_id=True)

        def qtiles(mcp):
            def f(mcp):
                dm = mcp.to_dm().rescale(100 * theta)
                return dm.eta(x)

            ys = vmap(f)(mcp)
            Ne = 1 / 2 / ys
            return x, jnp.quantile(Ne, jnp.array([0.025, 0.5, 0.975]), axis=0)

        self._qtiles = jit(qtiles)

    def __call__(self, state, **kw):
        self._plot(state.particles)

    def _plot(self, mcp: "eastbay.mcmc.MCMCParams"):
        x, (q025, m, q975) = self._qtiles(mcp)
        self._line.set_data(x, m)
        self._poly[0].remove()
        self._poly[0] = self._ax.fill_between(
            x, q025, q975, color="tab:blue", alpha=0.1
        )
        self._display.update(self._fig)


def liveplot_cb(
    mcp0, truth: "eastbay.size_history.DemographicModel" = None, plot_every: int = 1
):
    def f(state, **kw):
        return None

    try:
        if _is_running_in_jupyter():
            counter = it.count()
            cb = _IPythonLivePlot(mcp0, truth)

            def f(state, **kw):  # noqa: F811
                if next(counter) % plot_every == 0:
                    cb(state, **kw)

    except Exception as e:
        logger.debug("Live plot init failed: %s", str(e))
    return f


def plot_posterior(dms: list[DemographicModel], ax: "matplotlib.axes.Axes" = None):
    if ax is None:
        import matplotlib.pyplot as plt

        ax = plt.gca()
    dms = tree_stack(dms)
    t1, tM = jnp.quantile(dms.eta.t[:, 1:], jnp.array([0.025, 0.975]))
    t = jnp.geomspace(t1, tM, 1000)
    y = vmap(SizeHistory.__call__, (0, None))(dms.eta, t)
    Ne = 1.0 / 2.0 / y
    q025, m, q975 = jnp.quantile(Ne, jnp.array([0.025, 0.5, 0.975]), axis=0)
    ax.plot(t, m, color="tab:blue")
    ax.fill_between(t, q025, q975, color="tab:blue", alpha=0.1)
    style_axis(ax)
