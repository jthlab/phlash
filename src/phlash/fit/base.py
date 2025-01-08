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
from phlash.ld.data import LdStats
from phlash.model import PhlashMCMCParams
from phlash.size_history import DemographicModel
from phlash.util import Pattern, tree_unstack


def _check_jax_gpu():
    if jax.local_devices()[0].platform != "gpu":
        logger.warning(
            "Detected that Jax is not running on GPU; you appear to have "
            "CPU-mode Jax installed. Performance may be improved by installing "
            "Jax-GPU instead. For installation instructions visit:\n\n\t{}\n",
            "https://github.com/google/jax?tab=readme-ov-file#installation",
        )


class BaseFitter:
    def __init__(self, data, test_data=None, **options):
        """
        Initialize the fitting procedure with data and configuration options.
        """
        self.data = data
        self.test_data = test_data
        self.options = options
        self.M = options.get("M", 16)
        self.key = options.get("key", jax.random.PRNGKey(1))
        self.state = None
        self.afs = None
        self.ld = None
        self.chunks = None
        self.kernel = None
        self._particles = None  # for debugging
        self.initialize()

    def initialize(self):
        """
        Perform initial checks and load data.
        """
        _check_jax_gpu()
        self.load_data()
        self.setup_gpu_kernel()
        self.initialize_particles()
        self.initialize_callback()  # for plotting
        self.initialize_optimizer()

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
        overlap = self.options.get("overlap", 500)
        # the size of each "chunk", see manuscript. this is estimated from data.
        chunk_size = self.options.get("chunk_size")
        max_samples = self.options.get("max_samples", 20)
        num_workers = self.options.get("num_workers")
        logger.info("Loading data...")
        (self.chunks, self.populations, self.pop_indices) = phlash.data.init_chunks(
            self.data, self.window_size, overlap, chunk_size, max_samples, num_workers
        )
        # collapse the first two dimensions
        logger.debug("chunks.shape={}", self.chunks.shape)

        # avoid storing a huge array on gpu if we're only going to use a small part of
        # it in expectation, we will sample at most S * niter rows of the data.
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
        self.afs = phlash.data.init_afs(self.data)
        if self.afs:
            self.afs_transform = {
                n: default_afs_transform(self.afs[n]) for n in self.afs
            }
        else:
            self.afs_transform = None
        self.ld = phlash.data.init_ld(self.data)

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
        test_hets = self.test_data.hets[:max_samples]
        test_afs = self.test_data.afs
        test_ld = self.test_data.ld
        if test_ld is not None:
            test_ld = {k: LdStats.summarize(v) for k, v in test_ld.items() if v}
        N_test = test_hets.shape[0]
        self.test_kernel = get_kernel(
            M=self.M,
            data=np.ascontiguousarray(test_hets),
            double_precision=False,
        )

        def elpd(mcps):
            @vmap
            def _elpd_ll(mcp):
                return self.log_density(
                    mcp,
                    c=jnp.array([0.0, 1.0, 1.0, 1.0]),
                    inds=jnp.arange(N_test),
                    kern=self.test_kernel,
                    warmup=None,
                    afs=test_afs,
                    ld=test_ld,
                    afs_transform=self.afs_transform,
                )

            return _elpd_ll(mcps).mean()

        self.elpd = jit(elpd)

    def _initialize_model(self):
        """
        Initialize the MCMC model parameters and optimizer.
        """
        init = self.options.get("init")
        theta = self.options.get("theta", self._calculate_watterson())
        # although we could work in the per-generation scaling if 'mutation_rate' is
        # passed, it seems to be numerically better (estimates are more accurate) to
        # work in the coalescent scaling. perhaps because all the calculations are
        # "O(1)" instead of "O(huge number) * O(tiny number)" ...
        logger.info("Scaled mutation rate Θ={:.4g}", theta)
        if init is None:
            N0 = None
            # If there are n samples coalescing at rate c then the rate of first
            # coalescence is n * c.
            # so first coalescence X ~ Exp(n/2N0). Then find t such that p(X<=t) = 1/M:
            # 1 - exp(-(n/2N)t) = 1/M => t = -log(1 - 1/M) / (n / 2N)
            if self.mutation_rate is not None:
                N0 = theta / 4 / self.mutation_rate
                logger.debug("N0={}", N0)
            t1 = self.options.get("t1", -jnp.log1p(-1.0 / 16) / self.num_samples)
            tM = self.options.get("tM", 15.0)
            assert t1 < tM
            logger.debug("t1={:g} tM={:f}", t1, tM)
            rho = self.options.get("rho_over_theta", 1.0) * theta
            # this pattern is similar to the psmc default, but we have fewer params
            # (16) to use, so are a little more conservative with parameter tying
            pat = "14*1+1*2"
            init = PhlashMCMCParams.from_linear(
                pattern_str=pat,
                rho=rho,
                t1=t1,
                tM=tM,
                c=jnp.ones(len(Pattern(pat))),  # len(c)==len(Pattern(pattern))
                theta=theta,
                alpha=self.options.get("alpha", 0.0),
                beta=self.options.get("beta", 0.0),
                N0=N0,
                window_size=self.window_size,
            )
        assert isinstance(init, PhlashMCMCParams)
        return init

    def initialize_particles(self):
        # initialized raveled representation of state
        init = self._initialize_model()
        x0, unravel = ravel_pytree(init)
        prior_mu = x0
        prior_prec = self.options.get("sigma", 0.5)
        self.particles = vmap(unravel)(
            prior_mu
            + prior_prec
            * jax.random.normal(
                self.get_key(),
                shape=(self.options.get("num_particles", 500), len(prior_mu)),
            )
        )

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
        progress = self.options.get("progress", True)

        # begin iteration over data points
        di = self.data_iterator()

        # weights to multiply terms in log density
        # to have unbiased gradient estimates, need to pre-multiply the chunk term by
        # ratio
        weights = self.options.get(
            "weights", np.array([1.0, self.num_chunks / self.minibatch_size, 1.0, 1.0])
        )
        kw = dict(
            kern=self.train_kern,
            weights=weights,
            afs=self.afs,
            ld=self.ld,
            afs_transform=self.afs_transform,
        )

        logger.info("Starting optimization...")
        with tqdm.trange(
            self.niter, disable=not progress, desc="Fitting model"
        ) as self.pbar:
            for i in self.pbar:
                self.state = self._optimization_step(next(di), **kw)
                if self.converged(i):
                    break

                self.callback(self.state)

        logger.info("MCMC finished successfully")
        return tree_unstack(self._dms(self.state))

    def converged(self, i: int):
        # if there is a test set, check elpd() function for computing expected
        if not self.test_data:
            return False

        patience = self.options.get("patience", 100)

        if not hasattr(self, "ema"):
            self.ema = None
            self.best_elpd = None
            self.a = 0

        if i % 10 == 0:
            e = self.elpd(self.state.particles)
            if self.ema is None:
                self.ema = e
            else:
                self.ema = 0.9 * self.ema + 0.1 * e
            if self.best_elpd is None or self.ema > self.best_elpd[1]:
                self.a = 0
                self.best_elpd = (i, self.ema, self.state)
            else:
                self.a += 1
            self.pbar.set_description(f"elpd={self.ema:.2f} a={self.a}")
            if i - self.best_elpd[0] > patience:
                logger.info(
                    "The expected log-predictive density has not improved in "
                    f"the last {patience} iterations; exiting."
                )
                return True

        # catch-all return if not converged or not checked
        return False

    def initialize_optimizer(self):
        """
        Initialize the optimizer.
        """
        opt = optax.nadamw(learning_rate=self.options.get("learning_rate", 0.1))
        df = grad(self.log_density)
        self.svgd = blackjax.svgd(df, opt)
        self.state = self.svgd.init(self.particles)
        self.step = jit(self.svgd.step, static_argnames=["kern"])
        # self.step = self.svgd.step

    def log_density(self, particle, **kwargs):
        """
        Compute the log density.
        """
        return phlash.model.log_density(particle, **kwargs)

    def _optimization_step(self, inds, **kwargs):
        """
        Perform a single optimization step.
        """
        # Placeholder for actual optimization logic.
        kwargs["inds"] = inds
        kwargs["warmup"] = self.warmup_chunks[inds]
        return self.step(self.state, **kwargs)

    def _calculate_watterson(self):
        """
        Compute Watterson's estimator for mutation rate.
        """
        overlap = self.options.get("overlap", 500)
        ch0 = self.chunks[:, :, overlap:]
        w = ch0.sum(axis=(0, 1, 2))
        return w[1] / w[0]

    ## some useful properties
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
    def num_samples(self):
        """
        Get the number of samples.
        """
        # FIXME: assumes all datasets have the same number of samples
        try:
            return next(d.hets.shape[0] for d in self.data if d.hets is not None)
        except StopIteration:
            raise ValueError("No data found")

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


def fit(data, test_data=None, **options):
    return BaseFitter(data, test_data, **options).fit_model()
