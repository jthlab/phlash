import json

import jax.numpy as jnp
import numpy as np
import plotly.graph_objs as go
import plotly.io._renderers
from IPython import get_ipython
from IPython.display import Javascript, display
from jax import jit, vmap
from loguru import logger

from phlash.size_history import DemographicModel

_js_update = """
window.updatePhlashPlot = function(x, m, q025, q975, hasTruth) {
    // Update the line trace for the median
    Plotly.restyle('{plot_id}', {
        'x': [x], // Update x for the mean line
        'y': [m]  // Update y for the mean line
    // the line trace is the first or second (depending on if truth is present)
    }, [0 + hasTruth]);

    // Update the fill trace for the quantiles
    var xCombined = x.concat(x.slice().reverse());
    var yCombined = q025.concat(q975.slice().reverse());
    Plotly.restyle('{plot_id}', {
        'x': [xCombined], // Update x for the fill area
        'y': [yCombined]  // Update y for the fill area
    }, [1 + hasTruth]); // second or third
};
"""
try:
    # this api is not documented and could change in a future release.
    plotly.io._renderers.renderers[
        "PhlashRenderer"
    ] = plotly.io._renderers.NotebookRenderer(
        config={}, connected=True, post_script=_js_update
    )
except Exception:
    logger.debug(
        "Couldn't add a custom renderer using the (undocumented) "
        "plotly.io._renderers interface. Live-plotting will be disabled. "
        "Please notify the package maintainer."
    )
    raise ImportError


def _is_running_in_jupyter():
    try:
        if "IPKernelApp" not in get_ipython().config:  # Check for Jupyter kernel
            return False

    except (ImportError, AttributeError):
        return False
    return True


def liveplot_cb(
    truth: DemographicModel = None,
    plot_every: int = 1,
):
    def f(dms):
        return None

    try:
        if _is_running_in_jupyter():
            f = _IPythonLivePlot(truth)  # noqa: F811
    except Exception as e:
        logger.debug("Live plot init failed: {}", e)
        raise

    return f


class _IPythonLivePlot:
    def __init__(self, truth: DemographicModel = None):
        # instance variables
        self._x = None
        self._truth = truth
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
                    x=np.append(truth.eta.t, 2 * truth.eta.t[-1]),
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

        def qtiles(dms):
            def f(dm):
                return dm.eta(self._x)

            ys = vmap(f)(dms)
            Ne = 1 / 2 / ys
            return jnp.quantile(Ne, jnp.array([0.025, 0.5, 0.975]), axis=0)

        self._qtiles = jit(qtiles)
        self._fig.show("PhlashRenderer")
        self._handle = display(display_id=True)

    def __call__(self, dms: DemographicModel):
        if self._x is None:
            self._x = np.geomspace(dms.eta.t[:, 1].min(), dms.eta.t[:, -1].max(), 1000)
        q025, m, q975 = self._qtiles(dms)
        args = [a.tolist() for a in [self._x, m, q025, q975]] + [
            self._truth is not None
        ]
        js_args = ", ".join(map(json.dumps, args))
        js = f"window.updatePhlashPlot({js_args})"
        self._handle.update(Javascript(js))
