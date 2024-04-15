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

from phlash.afs import bws_transform, fold_transform
from phlash.data import Contig, init_mcmc_data
from phlash.kernel import get_kernel
from phlash.model import log_density
from phlash.params import MCMCParams
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


_particles = None  # for debugging


def fit(
    data: list[Contig],
    test_data: Contig = None,
    **options,
) -> list[DemographicModel]:
    """
    Sample demographic models from posterior.

    Args:
        data: A list of Contig objects representing the datasets.
        test_data: A Contig object for computing the expected log-predictive density.
            Used for assessing convergence and preventing overfitting. Usage is
            recommended.
        **options: options for the fitting procedure. Includes parameters like number
            of iterations, window size, overlap, chunk size, and random seed.

    Returns:
        A list of posterior samples.

    Raises:
        ValueError: If there are issues with the input data or options.

    Examples:
        fit([contig1, contig2], test_data=contig3)

    Notes:
        - Check the 'DemographicModel' documentation for more details on the output
          models.
        - The options dictionary can be used to control the behavior of the fitting
          procedure in various ways. See the source code of this function for more
          information.
    """
    _check_jax_gpu()
    # some defaults pulled from the options dict
    key = options.get("key", jax.random.PRNGKey(1))
    # the number of svgd iterations. if a test dataset is provided, the optimizer
    # might perform fewer if convergence criteria are met.
    niter = options.get("niter", 1000)
    # the bin size in base pairs. observations are grouped into bins of this width.
    # recombination occurs between bins but not within bins. default 100 is same as
    # psmc.
    window_size = options.get("window_size", 100)
    # the amount overlap between adjacent windows, used to break long sequential hmm
    # observations into parallelizable batches. this should generally be left at the
    # default 500. see manuscript for more information on this parameter.
    overlap = options.get("overlap", 500)
    # the size of each "chunk", see manuscript. this is estimated from data.
    chunk_size = options.get("chunk_size")
    max_samples = options.get("max_samples", 20)

    # data loading routines. this can take a little while.
    # the number of parallel workers. by default, use all cores, but this can take up
    # too much memory. set num_workers=1 to process the data sequentially.
    num_workers = options.get("num_workers")
    logger.info("Loading data")
    afs, chunks = init_mcmc_data(
        data, window_size, overlap, chunk_size, max_samples, num_workers
    )
    # to conserve memory, we get rid of data at this point
    del data
    # the mutation rate per generation, if known.
    mutation_rate = options.get("mutation_rate")
    # if we know the true dm, just use its mutation rate
    if options.get("truth"):
        if mutation_rate:
            raise ValueError("mutation rate is already known from truth")
        mutation_rate = options["truth"].theta
    # if the elpd does not improve for this many iterations, exit the training loop
    elpd_cutoff = options.get("elpd_cutoff", 100)
    # The user can specify an arbitrary linear transform for the afs--this allows things
    # like binning and/or folding the AFS. The transform needs to act like a stochastic
    # matrix (though not necessarily square) -- it must send probability distributions
    # to other probability distributions, potentially in a lower-dimensional space.
    if options.get("afs_transform"):
        afs_transform = options["afs_transform"]
    else:
        # by default, fold the afs and apply a 90% binning strategy as in
        # Bhaskar-Wang-Song
        T1 = fold_transform(len(afs) + 1)
        T2 = bws_transform(T1 @ afs)
        afs_transform = T2 @ T1

    # on average, we'd like to visit every data point once. but we don't want it to be
    # too huge because that slows down computation, and usually isn't doesn't lead to
    # a big improvement. For now it's capped at 5.
    S = options.get("minibatch_size")
    if not S:
        S = max(1, min(5, int(len(chunks) / niter)))
    logger.debug("Minibatch size: {}", S)

    # avoid storing a huge array on gpu if we're only going to use a small part of it
    # in expectation, we will sample at most S * niter rows of the data.
    if len(chunks) > 5 * S * niter:
        key, subkey = jax.random.split(key)
        # important: use numpy to do this _not_ jax. (jax will put it on the gpu which
        # causes the very problem we are trying to solve.)
        old_size = chunks.size
        chunks = np.random.default_rng(np.asarray(subkey)).choice(
            chunks, size=(5 * S * niter,), replace=False
        )
        gb = 1024**3
        logger.debug(
            "Downsampled chunks from {:.2f}Gb to {:.2f}Gb",
            old_size / gb,
            chunks.size / gb,
        )
    N = len(chunks)

    # initialize the model
    init = options.get("init")
    # watterson's estimator of the mutation rate
    ch0 = chunks[:, overlap:]
    watterson = ch0[ch0 > -1].mean() / window_size
    # User can override theta if they want -- mainly useful for getting aligned
    # beginning/end time points across different populations.
    watterson = options.get("theta", watterson)
    # although we could work in the per-generation scaling if 'mutation_rate' is passed,
    # it seems to be numerically better (estimates are more accurate) to work in the
    # coalescent scaling. perhaps because all the calculations are "O(1)" instead of
    # "O(huge number) * O(tiny number)" ...
    theta = watterson / 4  # i.e., N0=1
    logger.info("Scaled mutation rate Θ={:.4g}", theta)
    if init is None:
        if mutation_rate is not None:
            N0 = theta / mutation_rate
            options.setdefault("t1", 1e1 / 2 / N0)
            options.setdefault("tM", 1e6 / 2 / N0)
        t1 = options.get("t1", 1e-4)
        tM = options.get("tM", 15.0)
        rho = options.get("rho_over_theta", 1.0) * theta
        # this pattern is similar to the psmc default, but we have fewer params
        # (16) to use, so are a little more conservative with parameter tying
        pat = "14*1+1*2"
        init = MCMCParams.from_linear(
            pattern=pat,
            rho=rho * window_size,
            t1=t1,
            tM=tM,
            c=jnp.ones(len(Pattern(pat))),  # len(c)==len(Pattern(pattern))
            theta=theta * window_size,
            alpha=options.get("alpha", 0.0),
            beta=options.get("beta", 0.0),
        )
    assert isinstance(init, MCMCParams)
    opt = optax.amsgrad(learning_rate=options.get("learning_rate", 0.1))
    svgd = blackjax.svgd(grad(log_density), opt)

    # set up the particles and add noise
    M = init.M
    x0, unravel = ravel_pytree(init)
    ndim = len(x0)
    prior_mu = x0
    key, rng_key_init = jax.random.split(key, 2)
    prior_prec = options.get("sigma", 1.0) * jnp.eye(ndim)
    initial_particles = vmap(unravel)(
        jax.random.multivariate_normal(
            rng_key_init,
            prior_mu,
            prior_prec,
            shape=(options.get("num_particles", 500),),
        )
    )
    state = svgd.init(initial_particles)
    # this function takes gradients steps.
    step = jit(svgd.step, static_argnames=["kern"])

    # the warmup chunks and data chunks are analyzed differently; the data chunks load
    # onto the GPU whereas the warmup chunks are processed by native jax.
    warmup_chunks, data_chunks = np.split(chunks, [overlap], axis=1)

    train_kern = get_kernel(
        M=M,
        data=np.ascontiguousarray(data_chunks),
        double_precision=options.get("double_precision", False),
    )

    # if there is a test set, define elpd() function for computing expected
    # log-predictive density. used to gauge convergence.
    if test_data:
        d = test_data.get_data(window_size)
        test_afs = d["afs"]
        test_data = d["het_matrix"][:max_samples]
        N_test = test_data.shape[0]
        test_kern = get_kernel(
            M=M,
            data=np.ascontiguousarray(d["het_matrix"]),
            double_precision=False,
        )

        @jit
        def elpd(mcps):
            @vmap
            def _elpd_ll(mcp):
                return log_density(
                    mcp,
                    c=jnp.array([0.0, 1.0, 1.0]),
                    inds=jnp.arange(N_test),
                    kern=test_kern,
                    warmup=jnp.full([N_test, 1], -1, dtype=jnp.int8),
                    afs=test_afs,
                    afs_transform=afs_transform,
                )

            return _elpd_ll(mcps).mean()

    # to have unbiased gradient estimates, need to pre-multiply the chunk term by ratio
    # (dataset size) / (minibatch size) = N / S.
    kw = dict(
        kern=train_kern,
        c=jnp.array([1.0, N / S, 1.0]),
        afs=afs,
        afs_transform=afs_transform,
    )

    # build the plot callback
    cb = options.get("callback")
    if not cb:
        try:
            from phlash.liveplot import liveplot_cb

            cb = liveplot_cb(truth=options.get("truth"))
        except ImportError:
            # if necessary libraries aren't installed, just initialize a dummy callback
            def cb(*a, **kw):
                pass

    def dms():
        ret = vmap(MCMCParams.to_dm)(state.particles)
        # rates are per window, so we have to scale up to get the per-base-pair rates.
        ret = ret._replace(theta=ret.theta / window_size, rho=ret.rho / window_size)
        if mutation_rate:
            ret = vmap(DemographicModel.rescale, (0, None))(ret, mutation_rate)
        return ret

    ema = best_elpd = None
    a = 0  # tracks the number of iterations since the elpd went up
    global _particles  # for debugging
    with tqdm.trange(
        niter, disable=not options.get("progress", True), desc="Fitting model"
    ) as pbar:
        for i in pbar:
            key, subkey = jax.random.split(key, 2)
            inds = kw["inds"] = jax.random.choice(subkey, N, shape=(S,))
            kw["warmup"] = warmup_chunks[inds]
            state1 = step(state, **kw)

            def f(x):
                assert jnp.isfinite(x).all()
                return x

            state = jax.tree_map(f, state1)
            _particles = state.particles
            if test_data is not None and i % 10 == 0:
                e = elpd(state.particles)
                if ema is None:
                    ema = e
                else:
                    ema = 0.9 * ema + 0.1 * e
                if best_elpd is None or ema > best_elpd[1]:
                    a = 0
                    best_elpd = (i, ema, state)
                else:
                    a += 1
                if i - best_elpd[0] > elpd_cutoff:
                    logger.info(
                        "The expected log-predictive density has not improved in "
                        f"the last {elpd_cutoff} iterations; exiting."
                    )
                    break
                pbar.set_description(f"elpd={ema:.2f} a={a}")
            cb(dms())

    # notify the live plot that we are done. fails if we are not using liveplot.
    try:
        plotter.finish()
    except Exception:
        pass

    # convert to list of dms, easier for the end user who doesn't know jax
    return tree_unstack(dms())
