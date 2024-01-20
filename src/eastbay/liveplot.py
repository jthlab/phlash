import socket
import time
import uuid

import jax.numpy as jnp
import numpy as np
import plotly.graph_objs as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
from jax import jit, vmap

from eastbay.log import getLogger
from eastbay.size_history import DemographicModel

logger = getLogger(__name__)


def _is_running_in_jupyter():
    try:
        from IPython import get_ipython

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
            f = _IPythonLivePlotDash(truth)  # noqa: F811
    except Exception as e:
        logger.debug("Live plot init failed: %s", str(e))
        raise

    return f


def _random_port() -> int:
    sock = socket.socket()
    sock.bind(("", 0))
    return sock.getsockname()[1]


class _IPythonLivePlotDash:
    def __init__(self, truth: DemographicModel = None):
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

        # can configure download option to be svg instead
        # config = {
        #   'toImageButtonOptions': {
        #     'format': 'svg', # one of png, svg, jpeg, webp
        #     'filename': 'custom_image',
        #     'height': 500,
        #     'width': 700,
        #     'scale': 1 # Multiply title/legend/axis/canvas sizes by this factor
        #   }
        # }
        timestamp = uuid.uuid1()
        self._port = _random_port()
        self._app = Dash(__name__, requests_pathname_prefix=f"/proxy/{self._port}/")
        self._app.layout = html.Div(
            [
                dcc.Graph(id=f"live-graph-{timestamp}", figure=self._fig, mathjax=True),
                dcc.Interval(
                    id=f"interval-component-{timestamp}",
                    interval=1 * 1000,  # in milliseconds
                    n_intervals=0,
                    disabled=False,
                ),
            ]
        )

        self._changed = False
        self.finished = self.did_finish = False

        @self._app.callback(
            Output(f"live-graph-{timestamp}", "figure"),
            [Input(f"interval-component-{timestamp}", "n_intervals")],
        )
        def update_graph_live(n):
            if not self._changed:
                raise PreventUpdate
            self._changed = False
            return self._fig

        @self._app.callback(
            Output(f"interval-component-{timestamp}", "disabled"),
            [Input(f"interval-component-{timestamp}", "n_intervals")],
        )
        def is_finished(n):
            if self.finished:
                self.did_finish = True
            return self.finished

        self._app.run(
            host="127.0.0.1",
            port=self._port,
            jupyter_mode="inline",
            jupyter_server_url="/",
        )

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

        # allow the user to zoom in or out instead
        # self._fig.update_yaxes(
        #     type="log", range=np.log10([0.5 * q025.min(), 2 * q975.max()])
        # )
        self._changed = True

    def finish(self):
        self.finished = True
        # wait for the callback to disable
        while not self.did_finish:
            time.sleep(0.1)
        logger.debug("finished successfully")
