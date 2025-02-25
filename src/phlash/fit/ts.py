import itertools as it

import jax
import jax.numpy as jnp
import numpy as np
import tskit
from jax import vmap
from jax.scipy.special import xlogy
from loguru import logger

from ..afs import fold_transform
from ..transition import _expQ
from .base import BaseFitter


def fit(data: list[tskit.TreeSequence], test_data=None, **options):
    global _fitter
    _fitter = TreeSeqFitter(data, test_data, **options)
    return _fitter.fit()


class TreeSeqFitter(BaseFitter):
    def __init__(self, data, test_data=None, **options):
        options.setdefault("minibatch_size", 5)
        super().__init__(data, test_data, **options)

        if test_data is not None:
            ts = test_data
            samples = list(it.islice(it.pairwise(ts.samples()), 0, 10))
            td = [[] for _ in samples]
            for tree in test_data.trees():
                for a, (i, j) in zip(td, samples):
                    t = tree.get_tmrca(i, j) / 2 / self.N0
                    if a and a[-1][0] == t:
                        a[-1][1] += tree.span
                    else:
                        a.append([t, tree.span])
            M = max(len(a) for a in td)
            for a in td:
                a += [[1.0, -1.0]] * (M - len(a))
            td = np.array(td)
            afs = ts.allele_frequency_spectrum(
                mode="branch", span_normalise=True, polarised=False
            )
            self.test_data = {"data": td, "afs": {ts.num_samples: afs}}

    @property
    def sequence_length(self):
        """
        Get the sequence length.
        """
        return sum(ts.get_sequence_length() for ts in self.data)

    @property
    def num_samples(self):
        return self.data[0].num_samples

    def setup_gpu_kernel(self):
        pass

    def load_data(self):
        self.afs = afs = {}
        for ts in self.data:
            n = ts.num_samples
            afs.setdefault(n, np.zeros(n + 1))
            afs[n] += ts.allele_frequency_spectrum(
                mode="branch", span_normalise=True, polarised=False
            )

    def calculate_watterson(self):
        def f(ts):
            i = np.arange(1, ts.num_samples)
            L = ts.get_sequence_length()
            return ts.get_num_mutations() / L / (1 / i).sum(), L

        thetas, weights = np.transpose([f(ts) for ts in self.data])
        return np.average(thetas, weights=weights)

    def data_iterator(self):
        # find the longest tree seq. we will pad the shorter ones to avoid recompiles
        M = self.options.get("ts_length", 500)
        nodes = self.options.get("nodes")
        if nodes is not None:
            nodes = np.array(nodes)
            logger.debug("Using custom nodes")

        def helper():
            while True:
                # pick a random tree seq
                i = jax.random.choice(self.get_key(), len(self.data), shape=())
                ts = self.data[i]
                # pick a random starting tree in the interval [0, ts.num_trees - M]
                j = jax.random.choice(self.get_key(), ts.num_trees - M, shape=())
                # pick a random pair of nodes to sample
                ns = nodes if nodes is not None else ts.samples()
                if ns.ndim == 1:
                    j, k = jax.random.choice(
                        self.get_key(), ns, shape=(2,), replace=False
                    )
                else:
                    assert ns.ndim == 2
                    assert ns.shape[1] == 2
                    j, k = jax.random.choice(self.get_key(), ns)
                ret = []
                tree = ts.first()
                tree.seek_index(j)
                while len(ret) < M:
                    t = tree.get_tmrca(j, k) / 2 / self.N0
                    if ret and t == ret[-1][0]:
                        ret[-1][1] += tree.span
                    else:
                        ret.append([t, tree.span])
                    if not tree.next():
                        break
                ret += [[0.0, -1.0]] * (M - len(ret))
                yield ret

        bh = it.batched(helper(), self.minibatch_size)
        yield from map(np.array, bh)

    def elpd(self, particles, weights, **kwargs):
        @jax.jit
        @jax.vmap
        def f(p):
            return self.log_density(p, **self.test_data)

        return f(particles).mean()

    def log_density(self, particle, **kwargs):
        """
        Compute the log density.
        """
        data = kwargs["data"]
        dm = particle.to_dm()
        r = dm.rho
        eta = dm.eta
        R = eta.R
        times = data[..., 0].reshape(-1)
        times = jnp.sort(times)
        cs = jax.vmap(eta)(times)
        dt = jnp.diff(times)

        def eQ(dt, c):
            dt_safe = jnp.where(dt > 0.0, dt, 1.0)
            ret = _expQ(dt_safe * r, dt_safe * c, 2)
            return jnp.where(dt > 0.0, ret, jnp.eye(3))

        P = jax.vmap(eQ)(dt, cs[:-1])

        def f(accum, Pi):
            A = accum @ Pi
            return A, A[0, :2]

        _, Pcum = jax.lax.scan(f, jnp.eye(3), P)
        log_Pcum = jnp.log(Pcum)
        # Pcum = jax.lax.associative_scan(jax.remat(jnp.matmul), P)[:, 0, :2]

        @vmap
        def g(s, t, span):
            i, j = jnp.searchsorted(times, jnp.array([s, t]), side="right")
            log_eta_t = jnp.log(eta(t))
            log_p_span = span * log_Pcum[i - 1, 0]
            log_p_trans = jnp.select(
                [jnp.isclose(s, t), t > s, t < s],
                [
                    log_Pcum[i - 1, 0],
                    log_eta_t + log_Pcum[i - 1, 1] - (R(t) - R(s)),
                    log_eta_t + log_Pcum[j - 1, 1],
                ],
            )
            return jnp.where(span > 0, log_p_trans + log_p_span, 0.0)

        @vmap
        def f(seq):
            tt, ss = seq.T
            return g(tt[:-1], tt[1:], ss[:-1]).sum()

        ll_seq = f(data).sum()

        # afs
        ll_afs = 0.0
        afs = kwargs.get("afs", {})
        for n in afs:
            T = fold_transform(n)
            e = eta.etbl(n)
            e /= e.sum()
            ll_afs += xlogy(T @ afs[n][1:-1], T @ e).sum()

        return ll_seq + ll_afs
