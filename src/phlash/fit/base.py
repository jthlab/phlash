import functools
from dataclasses import replace

import blackjax
import jax
import numpy as np
import optax
import tqdm.auto as tqdm
from jax import grad, jit
from jax import numpy as jnp
from jax import vmap
from jax.flatten_util import ravel_pytree
from loguru import logger

import phlash.data
import phlash.model
from phlash.afs import default_afs_transform
from phlash.kernel import get_kernel
from phlash.model import PhlashMCMCParams
from phlash.size_history import DemographicModel, SizeHistory
from phlash.util import Pattern, tree_unstack


def _check_jax_gpu():
    if jax.local_devices()[0].platform != "gpu":
        logger.warning(
            "Detected that Jax is not running on GPU; you appear to have "
            "CPU-mode Jax installed. Performance may be improved by installing "
            "Jax-GPU instead. For installation instructions visit:\n\n\t{}\n",
            "https://github.com/google/jax?tab=readme-ov-file#installation",
        )


_particles = None


class BaseFitter:
    def __init__(self, data, test_data=None, **options):
        """
        Initialize the fitting procedure with data and configuration options.
        """
        self.data = data

        # FIXME: assumes all datasets have the same number of samples
        try:
            self.num_samples = next(
                d.hets.shape[0] for d in self.data if d.hets is not None
            )
        except StopIteration:
            raise ValueError("No data found")

        self.test_data = test_data
        self.options = options
        self.M = options.get("M", 16)
        self.key = options.get("key", jax.random.PRNGKey(1))
        self.state = None
        self.init = None
        self.afs = None
        self.ld = None
        self.chunks = None
        self.initialize()

    def initialize(self):
        """
        Perform initial checks and load data.
        """
        _check_jax_gpu()
        self.load_data()
        self.setup_gpu_kernel()

    def get_key(self):
        """
        Get the random key.
        """
        self.key, subkey = jax.random.split(self.key)
        return subkey

    def load_data(self):
        """
        Load and preprocess input data.
        """
        # the bin size in base pairs. observations are grouped into bins of this width.
        # recombination occurs between bins but not within bins. default 100 is same as
        # psmc.
        # the amount overlap between adjacent windows, used to break long sequential hmm
        # observations into parallelizable batches. this should generally be left at the
        # default 500. see manuscript for more information on this parameter.
        logger.info("Loading data...")
        overlap = self.options.get("overlap", 500)
        # the size of each "chunk", see manuscript. this is estimated from data.
        chunk_size = self.options.get("chunk_size", 50_000)
        max_samples = self.options.get("max_samples", 20)
        num_workers = self.options.get("num_workers")

        # process afs and ld first, because init_chunks destroys data!
        self.afs = phlash.data.init_afs(self.data)
        self.ld = phlash.data.init_ld(self.data)

        (self.chunks, self.populations, self.pop_indices) = phlash.data.init_chunks(
            self.data, self.window_size, overlap, chunk_size, max_samples, num_workers
        )
        # in the one population case, all the chunks are exchangeable, so we can just
        # combine the first two dimensions
        logger.debug("chunks.shape={}", self.chunks.shape)

        # avoid storing a huge array on gpu if we're only going to use a small part of
        # it in expectation, we will sample S * niter rows of the data.
        logger.debug("minibatch size: {}", self.minibatch_size)

        if self.test_data:
            self.test_afs = phlash.data.init_afs([self.test_data])
            self.test_ld = phlash.data.init_ld([self.test_data])
        else:
            self.test_afs = self.test_ld = None

    def setup_gpu_kernel(self):
        """
        Initialize the GPU kernel for computations.
        """
        overlap = self.options.get("overlap", 500)
        self.warmup_chunks, data_chunks = np.split(self.chunks, [overlap], axis=2)

        self.train_kern = get_kernel(
            M=self.M,
            data=np.ascontiguousarray(data_chunks),
            double_precision=self.options.get("double_precision", False),
        )

        if not self.test_data:
            return

        max_samples = self.options.get("max_samples", 20)
        # add a chunk axis, not used here
        test_hets = self.test_data.hets[:max_samples, None]
        test_afs = self.test_afs
        test_ld = self.test_ld
        N_test = test_hets.shape[0]
        self.test_kernel = get_kernel(
            M=self.M,
            data=np.ascontiguousarray(test_hets),
            double_precision=False,
        )

        def elpd(mcps, weights):
            @vmap
            def _elpd_ll(mcp):
                return self.log_density(
                    mcp,
                    weights=weights,
                    inds=(jnp.arange(N_test), jnp.full(N_test, 0)),
                    kern=self.test_kernel,
                    warmup=None,
                    afs=test_afs,
                    afs_transform=self.test_afs_transform,
                    ld=test_ld,
                    _components=True,
                )

            e = _elpd_ll(mcps)
            return e.mean(0)

        self.elpd = jit(elpd)

    def _initialize_model(self):
        """
        Initialize the MCMC model parameters and optimizer.
        """
        if self.init is not None:
            return self.init
        init = self.options.get("init")
        theta = self.theta
        # although we could work in the per-generation scaling if 'mutation_rate' is
        # passed, it seems to be numerically better (estimates are more accurate) to
        # work in the coalescent scaling. perhaps because all the calculations are
        # "O(1)" instead of "O(huge number) * O(tiny number)" ...
        logger.info("Scaled mutation rate Θ={:.4g}", theta)
        logger.info("Initializing model")
        if init is None:
            N0 = None
            # If there are n samples coalescing at rate c then the rate of first
            # coalescence is n * c.
            # so first coalescence X ~ Exp(n/2N0). Then find t such that p(X<=t) = 1/M:
            # 1 - exp(-(n/2N)t) = 1/M => t = -log(1 - 1/M) / (n / 2N)
            if self.mutation_rate is not None:
                N0 = theta / 4 / self.mutation_rate
                logger.debug("N0={}", N0)
            t1 = self.options.get("t1", -jnp.log1p(-1.0 / 16) / (2 * self.num_samples))
            tM = self.options.get("tM", 15.0)
            assert t1 < tM
            logger.debug("t1={:g} tM={:f}", t1, tM)
            rho = self.options.get("rho_over_theta", 1.0) * theta
            # this pattern is similar to the psmc default, but we have fewer params
            # (16) to use, so are a little more conservative with parameter tying
            pat = "16*1"
            M = len(Pattern(pat))
            init = PhlashMCMCParams.from_linear(
                pattern_str=pat,
                rho=rho,
                t1=t1,
                tM=tM,
                c=jnp.ones(M),
                theta=theta,
                N0=N0,
                window_size=self.window_size,
            )
        assert isinstance(init, PhlashMCMCParams)
        self.init = init
        return init

    def initialize_particles(self):
        # initialized raveled representation of state
        init0 = self._initialize_model()

        def f(key):
            fl = functools.partial(
                PhlashMCMCParams.from_linear,
                pattern_str="16*1",
                rho=init0.rho,
                t1=init0.t1,
                tM=init0.tM,
                theta=init0.theta,
                N0=init0.N0,
                window_size=self.window_size,
            )
            init = fl(c=jnp.ones_like(init0.c))
            x0, unravel = ravel_pytree(init)
            sd = self.options.get("sigma", 0.5)
            key1, key2 = jax.random.split(key)
            x1 = x0 + sd * jax.random.normal(key1, shape=x0.shape)
            init = unravel(x1)
            return init

        keys = jax.random.split(self.get_key(), self.num_particles)
        self.particles = jax.vmap(f)(keys)

    def data_iterator(self):
        """
        Iterate over the data.
        """
        while True:
            N, C = self.chunks.shape[:2]
            inds0 = jax.random.choice(self.get_key(), N, shape=(self.minibatch_size,))
            inds1 = jax.random.choice(self.get_key(), C, shape=(self.minibatch_size,))
            yield (inds0, inds1)

    def initialize_callback(self):
        # build the plot callback
        cb = self.options.get("callback")
        if not cb:
            try:
                from phlash.liveplot import liveplot_cb

                cb = liveplot_cb(truth=self.options.get("truth"))
            except ImportError:
                # if necessary libraries aren't installed, just initialize a dummy
                # callback
                logger.debug("liveplot_cb not available")

                def cb(*a, **kw):
                    pass

        def callback(state):
            return cb(self._dms(state))

        self.callback = callback

    def _dms(self, state):
        params_cls = state.particles.__class__
        ret = vmap(params_cls.to_dm)(state.particles)
        if self.mutation_rate:
            ret = vmap(DemographicModel.rescale, (0, None))(ret, self.mutation_rate)
        return ret

    def fit(self):
        """
        Run the optimization loop.
        """
        global _particles
        self.initialize_particles()
        self.initialize_optimizer()
        self.initialize_callback()
        progress = self.options.get("progress", True)

        # begin iteration over data points
        di = self.data_iterator()

        # weights to multiply terms in log density
        # to have unbiased gradient estimates, need to pre-multiply the chunk term by
        # ratio
        weights = self.options.get("weights", (None,) * 4)
        default_weights = (1.0, self.num_chunks / self.minibatch_size, 1.0, 1.0)
        weights = self.weights = jnp.array(
            [dw if w is None else w for dw, w in zip(default_weights, weights)]
        )
        logger.debug("weights: {}", weights)

        kw = dict(
            kern=self.train_kern,
            weights=weights,
            alpha=self.options.get("alpha", 1e-5),
            beta=self.options.get("beta", 1e-5),
        )

        logger.info("Starting optimization...")

        # with tqdm.trange(
        #     100, disable=not progress, desc="Pre-fitting"
        # ) as self.pbar:
        #     for i in self.pbar:
        #         self.state = jax.tree.map(
        #             lambda a: jnp.astype(a, jnp.float64), self.state
        #         )
        #         self.state = self._optimization_step(next(di), **kw)
        #         self.callback(self.state)
        #         _particles = self.state.particles

        kw["afs"] = self.afs
        kw["ld"] = self.ld

        self.converged_state = None

        with tqdm.trange(
            self.niter, disable=not progress, desc="Fitting model"
        ) as self.pbar:
            for i in self.pbar:
                # this prevents an extra recompile after the first iteration
                # due to type promotion stuff
                self.state = jax.tree.map(
                    lambda a: jnp.astype(a, jnp.float64), self.state
                )
                self.state = self._optimization_step(next(di), **kw)
                if i == 100:
                    global _debug
                    _debug = True

                conv = self.converged(i)
                if conv is not False:
                    self.state = conv[2]
                    break

                self.callback(self.state)
                self.particles = _particles = self.state.particles

        logger.info("MCMC finished successfully")
        return tree_unstack(self._dms(self.state))

    def converged(self, i: int):
        # if there is a test set, check elpd() function for computing expected
        if not self.test_data:
            return False

        patience = self.options.get("patience", 100)

        if self.converged_state is None:
            self.converged_state = {
                "ema": None,
                "best_elpd": None,
                "a": 0,
                "last_elpd": None,
            }

        cs = self.converged_state

        if i % 10 == 0:
            w = self.weights.at[0].set(0.0).at[1].set(1.0)
            es = np.asarray(self.elpd(self.state.particles, w))
            e = es.sum()
            last_e = cs["ema"] or 0.0
            if cs["ema"] is None:
                cs["ema"] = e
            else:
                cs["ema"] = 0.9 * cs["ema"] + 0.1 * e
            if cs["best_elpd"] is None or cs["ema"] > cs["best_elpd"][1]:
                cs["a"] = 0
                cs["best_elpd"] = (i, cs["ema"], self.state)
            else:
                cs["a"] += 1
                # delta symbol: ∆
            self.pbar.set_description(
                f"∆={e - last_e:.0f} selpd={cs['ema']:.0f} a={cs['a']}"
            )
            if i - cs["best_elpd"][0] > patience:
                logger.info(
                    "The expected log-predictive density has not improved in "
                    f"the last {patience} iterations; exiting."
                )
                return cs["best_elpd"]

        # catch-all return if not converged or not checked
        return False

    def initialize_optimizer(self):
        """
        Initialize the optimizer.
        """
        lr = self.options.get("learning_rate", 1e-2)
        opt = optax.nadam(learning_rate=lr)
        # df = jit(grad(self.log_density), static_argnames=["kern"])
        df = grad(self.log_density)
        self.svgd = blackjax.svgd(df, opt)
        self.state = self.svgd.init(self.particles)
        self.step = jit(self.svgd.step, static_argnames=["kern"])

    def log_density(self, particle, **kwargs):
        """
        Compute the log density.
        """

        @vmap
        def f(mcp, inds, warmup):
            return phlash.model.log_density(
                mcp=mcp,
                weights=kwargs["weights"],
                inds=inds,
                warmup=warmup,
                ld=kwargs.get("ld"),
                afs_transform=self.afs_transform,
                afs=kwargs.get("afs"),
                kern=kwargs["kern"],
                alpha=self.options.get("alpha", 0.0),
                beta=self.options.get("beta", 0.0),
                _components=kwargs.get("_components"),
            )

        mcps = vmap(lambda _: particle)(kwargs["inds"])
        return f(mcps, kwargs["inds"], kwargs["warmup"]).sum(0)

    def _optimization_step(self, inds, **kwargs):
        """
        Perform a single optimization step.
        """
        # Placeholder for actual optimization logic.
        kwargs["inds"] = inds
        kwargs["warmup"] = self.warmup_chunks[inds]
        new_st = self.step(self.state, **kwargs)
        return new_st

    def _calculate_watterson(self):
        """
        Compute Watterson's estimator for mutation rate.
        """
        overlap = self.options.get("overlap", 500)
        ch0 = self.chunks[:, :, overlap:]
        w = ch0.sum(axis=(0, 1, 2))
        # this is fairly wrong if compositing many chunks together.
        return w[1] / w[0]

    ## some useful properties
    @property
    def theta(self):
        return self.options.get("theta", self._calculate_watterson())

    @property
    def window_size(self):
        """
        Get the window size.
        """
        return self.options.get("window_size", 100)

    @property
    def mutation_rate(self):
        """
        Get the mutation rate.
        """
        if self.options.get("truth"):
            if self.options.get("mutation_rate"):
                raise ValueError("mutation rate is already known from truth")
            return self.options["truth"].theta
        return self.options.get("mutation_rate")

    @property
    def niter(self):
        """
        Get the number of iterations.
        """
        return self.options.get("niter", 1000)

    @property
    def num_chunks(self):
        """
        Get the number of chunks.
        """
        s = self.chunks.shape
        return s[0] * s[1]

    @property
    def minibatch_size(self):
        """
        Get the minibatch size.
        """
        # on average, we'd like to visit every data point once. but we don't want it to
        # be too huge because that slows down computation, and usually isn't doesn't
        # lead to a big improvement. For now it's capped at 5.
        S = self.options.get("minibatch_size")
        if not S:
            S = max(1, min(5, int(self.num_chunks / self.niter)))
        return S

    @property
    def num_particles(self):
        """
        Get the number of particles.
        """
        return self.options.get("num_particles", 500)


def l2_kernel(p1: PhlashMCMCParams, p2: PhlashMCMCParams, length_scale=1.0):
    eta1 = p1.to_dm().eta
    eta2 = p2.to_dm().eta
    l2_2 = eta1.squared_l2(eta2, t_max=15.0, log=True)
    ret = jnp.exp(-l2_2 / length_scale)
    return ret


class PhlashFitter(BaseFitter):
    def load_data(self):
        # in the one population case, all the chunks are exchangeable, so we can just
        # merge all the chunks
        super().load_data()
        # convert afs to standard vector representation in the 1-pop case
        daft = self.options.get("afs_transform", default_afs_transform)
        if self.afs:
            self.afs = {k: v.todense()[1:-1] for k, v in self.afs.items()}
            self.afs_transform = {n: daft(self.afs[n]) for n in self.afs}
            for n in self.afs:
                logger.debug(
                    "transformed afs[{}]:{}", n, self.afs_transform[n](self.afs[n])
                )
        else:
            self.afs_transform = None
        if self.test_afs:
            self.test_afs = {k: v.todense()[1:-1] for k, v in self.test_afs.items()}
            self.test_afs_transform = {n: daft(self.test_afs[n]) for n in self.test_afs}
        else:
            self.test_afs_transform = None

        # massage the chunks
        self.chunks = self.chunks.reshape(-1, *self.chunks.shape[2:])[:, None]
        # if too many chunks, downsample so as not to use up all the gpu memory
        if self.num_chunks > 5 * self.minibatch_size * self.niter:
            # important: use numpy to do this _not_ jax. (jax will put it on the gpu
            # which causes the very problem we are trying to solve.)
            old_size = self.chunks.size
            rng = np.random.default_rng(np.asarray(self.get_key()))
            self.chunks = rng.choice(
                self.chunks, size=(5 * self.minibatch_size * self.niter,), replace=False
            )
            gb = 1024**3
            logger.debug(
                "Downsampled chunks from {:.2f}Gb to {:.2f}Gb",
                old_size / gb,
                self.chunks.size / gb,
            )
        logger.debug("after merging: chunks.shape={}", self.chunks.shape)

    def initialize_optimizer(self):
        """
        Initialize the optimizer.
        """
        lr = self.options.get("learning_rate", 1e-2)
        opt = optax.nadam(learning_rate=lr)
        # df = jit(grad(self.log_density), static_argnames=["kern"])
        df = grad(self.log_density)
        self.svgd = blackjax.svgd(df, opt)
        self.state = self.svgd.init(self.particles)
        self.step = jit(self.svgd.step, static_argnames=["kern"])
        # self.step = self.svgd.step

    def _optimization_step(self, inds, **kwargs):
        """
        Perform a single optimization step.
        """
        # Placeholder for actual optimization logic.
        kwargs["afs_transform"] = self.afs_transform
        old_st = self.state
        new_st = super()._optimization_step(inds, **kwargs)
        if not self.options.get("learn_t", False):
            new_particles = replace(new_st.particles, t_tr=old_st.particles.t_tr)
            new_st = new_st._replace(particles=new_particles)

        # compute ld averaged across all particles because it's slow :-)
        @jax.jit
        @jax.value_and_grad
        def f(ps):
            t1 = ps.t1.min()
            tM = ps.tM.max()
            t = jnp.geomspace(t1, tM, 100)
            cs = vmap(lambda p: p.to_dm().eta(t))(ps)
            eta = SizeHistory(t=t, c=cs.mean(0))
            dm = DemographicModel(eta=eta, theta=new_st.particles.theta, rho=None)
            N0 = ps.N0
            return phlash.model._loglik_ld(dm, N0, self.ld)

        f, df = f(new_st.particles)
        M = abs(df.c_tr).max()
        c_tr_prime = new_st.particles.c_tr + 0.1 * df.c_tr / M
        new_particles = replace(new_st.particles, c_tr=c_tr_prime)
        new_st = new_st._replace(particles=new_particles)
        return new_st


# for post-mortem/debugging...
_fitter = None


def fit(data, test_data=None, **options):
    global _fitter
    _fitter = PhlashFitter(data, test_data, **options)
    return _fitter.fit()
