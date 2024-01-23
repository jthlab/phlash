from concurrent.futures import ProcessPoolExecutor, as_completed

import blackjax
import jax
import numpy as np
import optax
import tqdm.auto as tqdm
from jax import grad, jit
from jax import numpy as jnp
from jax import vmap
from jax.flatten_util import ravel_pytree
from jaxlib.xla_extension import XlaRuntimeError

import eastbay.hmm
import eastbay.liveplot
from eastbay.data import Contig
from eastbay.log import getLogger
from eastbay.model import log_density
from eastbay.params import MCMCParams
from eastbay.size_history import DemographicModel
from eastbay.util import tree_unstack

logger = getLogger(__name__)


def _init_data(
    data: list[Contig],
    window_size: int,
    overlap: int,
    chunk_size: int = None,
    max_samples: int = 20,
):
    """Chunk up the data. If chunk_size is missing, set it to ~1/5th of the shortest
    contig. (This may not be optimal)."""
    afss = []
    chunk_size = int(min(0.2 * ds.L / window_size for ds in data))
    if chunk_size < 10 * overlap:
        logger.warning(
            "The chunk size is %dbp, which is less than 10 times the overlap (%dbp).",
            chunk_size,
            overlap,
        )
    chunks = []
    with ProcessPoolExecutor() as pool:
        futs = []
        for ds in data:
            futs.append(
                pool.submit(
                    ds.to_chunked,
                    overlap=overlap,
                    chunk_size=chunk_size,
                    window_size=window_size,
                )
            )
        for f in as_completed(futs):
            d = f.result()
            if d.afs is not None:
                afss.append(d.afs)
            if d.chunks is not None:
                chunks.append(d.chunks)

    assert all(a.ndim == 1 for a in afss)
    assert len({a.shape for a in afss}) == 1
    # all afs have same dimension
    assert len({ch.shape[-1] for ch in chunks}) == 1
    assert all(ch.ndim == 2 for ch in chunks)
    return np.sum(afss, 0), np.concatenate(chunks, 0)


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
    afs, chunks = _init_data(data, window_size, overlap, chunk_size, max_samples)
    # the mutation rate per generation, if known.
    mutation_rate = options.get("mutation_rate")
    # if the elpd does not improve for this many iterations, exit the training loop
    elpd_cutoff = options.get("elpd_cutoff", 100)

    # on average, we'd like to visit every data point once. but we don't want it to be
    # too huge because that slows down computation, and usually isn't doesn't lead to
    # a big improvement. For now it's capped at 5.
    S = options.get("minibatch_size")
    if not S:
        S = max(1, min(10, int(len(chunks) / niter)))

    # avoid storing a huge array on gpu if we're only going to use a small part of it
    if len(chunks) > 10 * S * niter:
        key, subkey = jax.random.split(key)
        # important: use numpy to do this _not_ jax. (jax will put it on the gpu which
        # causes the very problem we are trying to solve.)
        chunks = np.random.default_rng(np.asarray(subkey)).choice(
            chunks, size=(10 * S * niter,), replace=False
        )
    N = len(chunks)

    # keyboard interrupt handler and callback. This is used inside the gpu code in case
    # ctrl-c was caught while the gpu kernel was executing.
    kbi = False

    def kbi_cb(*args):
        nonlocal kbi
        kbi = True

    # initialize the model
    init = options.get("init")
    if init is None:
        ch0 = chunks[:, overlap:]
        theta = ch0[ch0 > -1].mean()
        logger.info("Scaled mutation rate Θ=%f", theta)
        if mutation_rate:
            # if mutation rate is known then pick t1, tM such that in rescaled time,
            # t1=10, tM=1e4 generations. (seems like a good default?)
            N0 = theta / window_size / 4 / mutation_rate
            t1, tM = (10**x / 2 / N0 for x in [1, 4])
        else:
            t1 = 1e-4
            tM = 15.0
        # or, the user can override
        t1 = options.get("t1", t1)
        tM = options.get("tM", tM)
        init = MCMCParams.from_linear(
            # this pattern is similar to the psmc default, but we have fewer params
            # (16) to use, so are a little more conservative with parameter tying
            pattern="14*1+1*2",
            rho=options.get("rho_over_theta", 1.0) * theta,
            t1=t1,
            tM=tM,
            c=jnp.ones(15),
            theta=theta,
            alpha=options.get("alpha", 0.0),
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
    # construct the GPU kernel, load the data onto it

    # defer loading the gpu module until necessary, to keep from having to initialize
    # CUDA on overall package load.
    from eastbay.gpu import PSMCKernel

    train_kern = PSMCKernel(
        M=M,
        data=np.ascontiguousarray(data_chunks),
        double_precision=options.get("double_precision", False),
        kbi_cb=kbi_cb,
    )

    # if there is a test set, define elpd() function for computing expected
    # log-predictive density. used to gauge convergence.
    if test_data:
        d = test_data.get_data(window_size)
        test_afs = d["afs"]
        test_data = d["het_matrix"][:max_samples]
        N_test = test_data.shape[0]
        test_kern = PSMCKernel(
            M=M,
            data=np.ascontiguousarray(d["het_matrix"]),
            double_precision=False,
            kbi_cb=kbi_cb,
        )

        @jit
        def elpd(mcps):
            @vmap
            def ll(mcp):
                return log_density(
                    mcp,
                    c=jnp.array([0.0, 1.0, 1.0]),
                    inds=jnp.arange(N_test),
                    kern=test_kern,
                    warmup=jnp.full([N_test, 1], -1, dtype=jnp.int8),
                    afs=test_afs,
                )

            return jax.scipy.special.logsumexp(ll(mcps)).mean()

    # to have unbiased gradient estimates, need to pre-multiply the chunk term by ratio
    # (dataset size) / (minibatch size) = N / S.
    kw = dict(
        kern=train_kern,
        c=jnp.array([1.0, N / S, 1.0]),
        afs=afs,
    )

    # if specified, use the mutation rate for plotting
    if mutation_rate is None and "truth" in options:
        mutation_rate = options["truth"].theta

    # build the plot callback
    cb = options.get("callback")
    if not cb:
        cb = eastbay.liveplot.liveplot_cb(truth=options.get("truth"))

    def dms():
        ret = vmap(MCMCParams.to_dm)(state.particles)
        # the learned rates are per window, so we have to scale up the mutation and
        # recombination to get the per-base-pair rates.
        ret = ret._replace(theta=ret.theta / window_size, rho=ret.rho / window_size)
        if mutation_rate:
            ret = vmap(DemographicModel.rescale, (0, None))(ret, mutation_rate)
        return ret

    try:
        ema = best_elpd = None
        a = 0
        with tqdm.trange(
            niter, disable=not options.get("progress", True), desc="Fitting model"
        ) as pbar:
            for i in pbar:
                key, subkey = jax.random.split(key, 2)
                inds = kw["inds"] = jax.random.choice(subkey, N, shape=(S,))
                kw["warmup"] = warmup_chunks[inds]
                state = step(state, **kw)
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
                    pbar.set_description(
                        f"elpd={ema:.2f} %best={ema/best_elpd[1]:.3f} a={a}"
                    )
                cb(dms())
                if kbi:
                    raise KeyboardInterrupt

    except (KeyboardInterrupt, XlaRuntimeError) as e:
        if isinstance(e, XlaRuntimeError):
            # this might be due to keyboard interrupt while we were inside the gpu
            # callback. unfortunately, the jax exception appears to just contain a big
            # string and is not more structured, so there is no better way to check for
            # KeyboardException than grepping the error message.
            (msg,) = e.args
            if not msg.find("CpuCallback error: KeyboardInterrupt"):
                raise
        logger.warning("Caught Ctrl-C; returning current estimates")

    # notify the live plot that we are done. fails if we are not using liveplot.
    try:
        plotter.finish()
    except Exception:
        pass

    # convert to list of dms, easier for the end user who doesn't know jax
    return tree_unstack(dms())
