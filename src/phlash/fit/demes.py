import itertools as it
from collections import Counter
from collections.abc import Callable, Collection
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pprint import pformat

import demes
import jax
import jax.numpy as jnp
import momi3
import numpy as np
from jax import vmap
from jax.flatten_util import ravel_pytree
from loguru import logger
from momi3.common import Path, get_path, set_path

import phlash.model
import phlash.params
import phlash.util
from phlash.data import Contig
from phlash.params import SinglePopMCMCParams

from .base import BaseFitter


@jax.tree_util.register_dataclass
@dataclass(kw_only=True)
class DemesMCMCParams(phlash.params.MCMCParams):
    x: jax.Array
    populations: tuple[str] = field(metadata=dict(static=True))
    f: Callable[[jax.Array], dict[Path, float]] = field(metadata=dict(static=True))
    iicr: Callable[[dict[Path, float]], float] = field(metadata=dict(static=True))
    sfs: dict[tuple[int], Callable] = field(default=None, metadata=dict(static=True))

    def bind(self, num_samples) -> SinglePopMCMCParams:
        t = self.times
        c, s = vmap(
            lambda u: self.iicr(num_samples=num_samples, t=u, params=self.params)
        )(t)
        # unlike the panmixia case, we can have c=0 in epochs (e.g. structured
        # populations, no migration. We need to clip c to avoid nans in the log
        # likelihood.
        # p = -jnp.diff(s)
        # eta = SizeHistory(t=t, c=c)
        # c = jnp.where(s < 1e-5, c, eta.c)
        return SinglePopMCMCParams.from_linear(
            c=c,
            t1=self.t1,
            tM=self.tM,
            theta=self.theta,
            rho=self.rho,
            window_size=self.window_size,
            N0=self.N0,
        )

    @property
    def M(self):
        return 16

    @property
    def params(self):
        return self.f(self.x)

    @classmethod
    def default(cls, theta: float, x, f, iicr, sfs, populations):
        t1 = 1e-4
        tM = 15.0
        dtM = tM - t1
        t_tr = jnp.array([jnp.log(t1), jnp.log(dtM)])
        return cls(
            t_tr=t_tr,
            log_rho_over_theta=0.0,
            theta=theta,
            window_size=100,
            N0=1e4,
            x=x,
            f=f,
            iicr=iicr,
            sfs=sfs,
            populations=populations,
        )


class DemesFitter(BaseFitter):
    def __init__(
        self, g: demes.Graph, paths: Collection[Path], data, test_data, **options
    ):
        # storage
        self._g = g
        self._paths = paths

        # validation
        assert isinstance(g, demes.Graph)
        if g.time_units != "generations":
            raise ValueError("Demes graph must be in units of generations.")

        samples = {
            frozenset(Counter([c.populations[i] for i in row]).items())
            for c in data
            for row in c.pop_indices
        }
        ns = {sum(dict(c).values()) for c in samples}

        # all poulations have same number of samples, 2
        assert ns == {2}

        # make life simple by assuming all the contigs are defined over the same set of
        # populations and pairs
        try:
            for x, y in it.pairwise(data):
                assert x.populations == y.populations
                assert np.all(x.pop_indices == y.pop_indices)
                if test_data is not None:
                    assert x.populations == test_data.populations
                    assert np.all(x.pop_indices == test_data.pop_indices)
        except AssertionError as e:
            raise ValueError(
                "All data must have the same populations and indices."
            ) from e

        # ensure that the data populations are a subset of the deme populations
        data_populations = {p for c in data for p in c.populations}
        deme_populations = {d.name for d in g.demes}
        if not data_populations <= deme_populations:
            raise ValueError(
                "Data populations must be a subset of the demography populations. "
                f"Data populations: {data_populations}; "
                f"Demography populations: {deme_populations}"
            )

        super().__init__(data, test_data, **options)

    def initialize_model(self):
        theta = self.calculate_watterson()
        if self.mutation_rate:
            N0 = theta / 4 / self.mutation_rate
        else:
            d0 = self._g.demes[0]
            assert np.isinf(d0.start_time)
            e0 = d0.epochs[0]
            assert e0.size_function == "constant"
            assert e0.start_size == e0.end_size
            N0 = e0.start_size
        # rescale everything in terms of N0
        g = deepcopy(self._g)
        g = demes.Graph.fromdict(phlash.util.rescale_demography(g.asdict(), 2 * N0))
        self._g = g
        # initialize momi components: sfs and iicr
        m3 = momi3.Momi3(self._g)
        self._iicr = m3.iicr_nd(2)
        # use sfs information too if provided
        self._sfs = None
        if self.afs:
            top_n = max((a.sum(), n) for n, a in self.afs.items())[1]
            # only use the top_n populations
            self.afs = {top_n: self.afs[top_n]}
            self.afs_transform = self.test_afs_transform = None
            num_samples = dict(zip(self.populations, top_n))
            self._sfs = {tuple(top_n): m3.sfs(num_samples)}
        init = self.options.get("init")
        if init is None:
            # initialize model based on learned params and the initial values from the
            # graph
            params = self._g.asdict()
            f, finv = self._iicr.reparameterize(self._paths)
            x0 = finv(params)
            init = DemesMCMCParams.default(
                x=x0,
                theta=theta,
                f=f,
                iicr=self._iicr,
                sfs=self._sfs,
                populations=self.populations,
            )
            init.N0 = N0
        return init

    def initialize_callback(self):
        super().initialize_callback()
        old_cb = self.callback

        def callback(state):
            if len(self.populations) == 1:
                p = self.populations[0]
                ns = {p: 2}
                ps = vmap(lambda p: p.bind(ns))(state.particles)
                return old_cb(self.state._replace(particles=ps))
            params = self._g.asdict()
            med = jax.tree.map(jnp.median, state.particles.params)
            for path, val in med.items():
                set_path(params, path, val)
            g = phlash.util.rescale_demography(params, 1 / state.particles.N0)
            pd = {path: float(get_path(g, path)) for path in self._paths}
            logger.debug("Median parameters:\n{}", pformat(pd))

        self.callback = callback

    def load_data(self):
        super().load_data()
        pc = {}
        ci = []
        for i, row in enumerate(self.pop_indices):
            counts = Counter([self.populations[r] for r in row])
            norm_counts = tuple([counts[p] for p in self.populations])
            pc.setdefault(norm_counts, []).append(i)
            ci.append(list(pc).index(norm_counts))
        self._count_indices = jnp.array(ci)
        self._population_counts = jnp.array(list(pc))
        # check that the chunks correspond to the correct count indices
        N, C = self.chunks.shape[:2]
        assert self._count_indices.shape == (N,)

    def log_density(self, particle, **kwargs):
        """
        Compute the log density.
        """

        @vmap
        def f(counts):
            ns = dict(zip(particle.populations, counts))
            return particle.bind(num_samples=ns)

        inds = kwargs["inds"]
        ci = self._count_indices[inds[0]]
        pc = self._population_counts
        warmup = kwargs["warmup"]
        mcps = jax.tree.map(lambda a: a[ci], f(pc))

        @vmap
        def g(mcp, inds, warmup):
            return phlash.model.log_density(
                mcp,
                weights=kwargs["weights"],
                inds=inds,
                warmup=warmup,
                kern=kwargs["kern"],
                alpha=self.options.get("alpha", 0.0),
                beta=self.options.get("beta", 0.0),
                afs=None,
                ld=None,
            )

        l1 = g(mcps, inds, warmup).sum()
        # manually process afs
        l2 = 0.0
        for n in particle.sfs:
            obsfs = kwargs["afs"][n]
            # the rate of mutation per unit time in phlash is theta/2.
            # (that's because the # hmm emission probability is
            # `theta * E(TMRCA)` = theta * branch_length / 2`.)
            l2 += particle.sfs[n].loglik(particle.params, obsfs)
        return l1 + l2

    def optimization_step(self, data, **kwargs):
        """
        Perform a single optimization step.
        """
        # Placeholder for actual optimization logic.
        inds = data
        kwargs["inds"] = inds
        kwargs["kern"] = self.train_kern
        kwargs["warmup"] = self.warmup_chunks[inds]
        old_st = self.state
        new_st = super().optimization_step(data, **kwargs)

        @vmap
        def f(mcp, old_mcp):
            return replace(
                mcp,
                t_tr=old_mcp.t_tr,
            )

        new_particles = f(new_st.particles, old_st.particles)
        new_st = new_st._replace(particles=new_particles)
        return new_st

    def initialize_particles(self):
        # initialized raveled representation of state
        init0 = self.initialize_model()

        def f(key):
            x0, unravel = ravel_pytree(init0)
            key1, key2 = jax.random.split(key)
            x1 = x0 + self.sigma * jax.random.normal(key1, shape=x0.shape)
            init = unravel(x1)
            return init

        keys = jax.random.split(self.get_key(), self.options.get("num_particles", 500))
        self.particles = jax.vmap(f)(keys)

    def _finalize(self, state):
        def f(p):
            d = self._g.asdict()
            paths = p.f(p.params)
            for path, val in paths.items():
                set_path(d, path, val)
            d = phlash.util.rescale_demography(d, 1 / 2 / p.N0)
            return d

        ps = phlash.util.tree_unstack(state.particles)
        return list(map(f, ps))


def fit_demes(
    g: demes.Graph,
    paths: Collection[Path],
    data: list[Contig],
    test_data: Contig = None,
    **options,
) -> list[demes.Graph]:
    return DemesFitter(g, paths, data, test_data, **options).fit()
