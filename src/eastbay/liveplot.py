import jax.numpy as jnp
import numpy as np
import plotly.graph_objs as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from jax import jit, vmap

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
    ax.set_xlabel("Time (generations)")
    ax.set_ylabel("$N_e$")


class _IPythonLivePlotDash:
    def __init__(self, truth: DemographicModel = None):
        # enable latex display
        self.app = Dash(__name__)
        self._x = None
        self._fig = go.Figure()
        self._fig.update_layout(
            template="simple_white", xaxis_title="Time", yaxis_title="$N_e$"
        )
        self._fig.update_xaxes(type="log")
        self._fig.update_yaxes(type="log")

        if truth is not None:
            self._x = np.geomspace(truth.eta.t[1], 2 * truth.eta.t[-1], 1000)
            self._fig.add_trace(
                go.Scatter(
                    x=np.append(truth.eta.t, 5 * truth.eta.t[-1]),
                    y=np.append(truth.eta.Ne, truth.eta.Ne[-1]),
                    mode="lines",
                    line=dict(color="black"),
                    name="Truth",
                )
            )
            self._fig.update_yaxes(
                type="log",
                range=np.log10([0.2 * truth.eta.Ne.min(), 5 * truth.eta.Ne.max()]),
            )

        self._fig.add_trace(
            go.Scatter(
                x=np.array([]),
                y=np.array([]),
                mode="lines",
                line=dict(color="rgb(31,119,164)"),
                name="Estimate",
            )
        )

        self._fig.add_trace(
            go.Scatter(
                x=np.array([]),
                y=np.array([]),
                mode="none",
                fill="toself",
                name="95% ci",
                # #1f77b4
                fillcolor="rgba(31, 119, 164, 0.3)",
            )
        )

        self.app.layout = html.Div(
            [
                dcc.Graph(id="live-graph", figure=self._fig, mathjax=True),
                dcc.Interval(
                    id="interval-component",
                    interval=1 * 1000,  # in milliseconds
                    n_intervals=0,
                ),
            ]
        )

        @self.app.callback(
            Output("live-graph", "figure"), [Input("interval-component", "n_intervals")]
        )
        def update_graph_live(n):
            return self._fig

        def qtiles(dms):
            def f(dm):
                return dm.eta(self._x)

            ys = vmap(f)(dms)
            Ne = 1 / 2 / ys
            return jnp.quantile(Ne, jnp.array([0.025, 0.5, 0.975]), axis=0)

        self._qtiles = jit(qtiles)
        self.app.run(jupyter_mode="inline")

    def __call__(self, dms: DemographicModel):
        if self._x is None:
            self._x = np.geomspace(dms.eta.t[:, 1].min(), dms.eta.t[:, -1].max(), 1000)
        q025, m, q975 = self._qtiles(dms)

        # Update the line trace for the mean
        self._fig.update_traces(
            selector=dict(type="scatter", mode="lines", name="Estimate"),
            overwrite=True,
            x=self._x,
            y=m,
        )

        # Update the fill trace for the quantiles
        x_combined = np.concatenate([self._x, self._x[::-1]])  # x, then x reversed
        y_combined = np.concatenate([q025, q975[::-1]])  # upper, then lower reversed
        self._fig.update_traces(
            selector=dict(type="scatter", mode="none", fill="toself", name="95% ci"),
            overwrite=True,
            x=x_combined,
            y=y_combined,
        )

        self._fig.update_yaxes(
            type="log", range=np.log10([0.5 * q025.min(), 2 * q975.max()])
        )


class _IPythonLivePlot:
    def __init__(
        self,
        truth: DemographicModel = None,
    ):
        import matplotlib.pyplot as plt
        from IPython.display import display, set_matplotlib_formats

        set_matplotlib_formats("svg")
        self._fig, self._ax = plt.subplots()
        ax = self._ax
        style_axis(ax)
        self._x = None
        if truth is not None:
            ax.plot(
                np.append(truth.eta.t, 5 * truth.eta.t[-1]),
                np.append(truth.eta.Ne, truth.eta.Ne[-1]),
                color="black",
            )
            ax.set_ylim(0.2 * truth.eta.Ne.min(), 5 * truth.eta.Ne.max())
            self._x = np.geomspace(truth.eta.t[1], 5 * truth.eta.t[-1], 1000)
        (self._line,) = ax.plot([], [], color="tab:blue")
        self._poly = [ax.fill_between([], 0, 0, color="tab:blue")]
        self._display = display(self._fig, display_id=True)

        def qtiles(dms):
            def f(dm):
                return dm.eta(self._x)

            ys = vmap(f)(dms)
            Ne = 1 / 2 / ys
            return jnp.quantile(Ne, jnp.array([0.025, 0.5, 0.975]), axis=0)

        self._qtiles = jit(qtiles)

    def __call__(self, dms: DemographicModel):
        if self._x is None:
            self._x = np.geomspace(dms.eta.t[:, 1].min(), dms.eta.t[:, -1].max(), 1000)
        q025, m, q975 = self._qtiles(dms)
        self._line.set_data(self._x, m)
        self._poly[0].remove()
        self._poly[0] = self._ax.fill_between(
            self._x, q025, q975, color="tab:blue", alpha=0.1
        )
        self._display.update(self._fig)


def liveplot_cb(
    truth: "eastbay.size_history.DemographicModel" = None,
    plot_every: int = 1,
):
    def f(dms):
        return None

    try:
        if _is_running_in_jupyter():
            f = _IPythonLivePlotDash(truth)  # noqa: F811
            # enable plotly rendering of latex equations
    except Exception as e:
        logger.debug("Live plot init failed: %s", str(e))
        raise

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
