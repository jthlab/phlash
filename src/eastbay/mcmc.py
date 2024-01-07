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
from jaxtyping import Array, Float64, Int8, Int64

import eastbay.liveplot
import eastbay.model
from eastbay.gpu import PSMCKernel
from eastbay.log import getLogger
from eastbay.params import MCMCParams, PSMCParams
from eastbay.util import tree_unstack

logger = getLogger(__name__)


# prior and likelihood
def _log_prior(mcp: MCMCParams) -> float:
    dm = mcp.to_dm()
    ret = sum(
        jax.scipy.stats.norm.logpdf(a, loc=mu, scale=sigma).sum()
        for (a, mu, sigma) in [
            # (jnp.log(dm.eta.c), 0.0, 3.0),
            (mcp.log_rho, jnp.log(dm.theta), 1.0),
        ]
    )
    ret -= mcp.alpha * jnp.sum(jnp.diff(mcp.c_tr) ** 2)
    return ret


def _log_density(
    mcp: MCMCParams,
    c: Float64[Array, "3"],
    inds: Int64[Array, "batch"],
    warmup: Int8[Array, "c ell"],
    kern: PSMCKernel,
    afs: Int64[Array, "n"],
) -> float:
    r"""
    Computes the log density of a statistical model by combining the contributions from
    the prior, the hidden Markov model (HMM), and the allele frequency spectrum (AFS)
    model, weighted by given coefficients.

    Args:
        mcp: The Markov Chain Monte Carlo parameters used to specify the model.
        c: Weights for each component of the density - prior, HMM model, and AFS model.
        inds: Mini-batch indices for selecting subsets of the data.
        data: Data matrix used in the model computation.
        kern: An instantiated PSMC Kernel used in the computation.
        afs: The allele frequency spectrum data.

    Returns:
        The log density, or negative infinity where the result is not finite.
    """
    dm = mcp.to_dm()
    pp = PSMCParams.from_dm(dm)
    pis = vmap(lambda pp, d: eastbay.model.psmc_ll(pp, d)[0], (None, 0))(
        pp, warmup
    )  # (I, M)
    pps = vmap(lambda pi: pp._replace(pi=pi))(pis)
    l1 = _log_prior(mcp)
    l2 = vmap(kern.loglik, (0, 0))(pps, inds).sum()
    if afs is not None:
        n = len(afs) + 1
        etbl = dm.eta.etbl(n)
        esfs = etbl / etbl.sum()
        l3 = jax.scipy.special.xlogy(afs, esfs).sum()
    else:
        l3 = 0.0
    ll = jnp.array([l1, l2, l3])
    ret = jnp.dot(c, ll)
    return jnp.where(jnp.isfinite(ret), ret, -jnp.inf)


def _chunk_het_matrix(
    het_matrix: np.ndarray, overlap: int, chunk_size: int, pad: bool = True
) -> np.ndarray:
    data = het_matrix.clip(-1, 1)
    assert data.ndim == 2
    data = np.ascontiguousarray(data)
    assert data.data.c_contiguous
    N, L = data.shape
    S = chunk_size + overlap
    if L < S:
        logger.warn("Chromosome length=%d is less than chunk size+overlap=%d", L, S)
        return np.empty([0, S], dtype=np.int8)
    if pad:
        data = np.pad(data, [[0, 0], [0, S - (L % S)]], constant_values=-1)
    L = data.shape[1]
    num_chunks = (L - S) // chunk_size
    # note that if we don't pad, we are throwing away data here.
    data = data[:, : L - S + num_chunks * chunk_size]
    new_shape = (N, 1 + num_chunks, S)
    new_strides = (data.strides[0], data.strides[1] * chunk_size, data.strides[1])
    chunked = np.lib.stride_tricks.as_strided(
        data, shape=new_shape, strides=new_strides
    )
    return np.copy(chunked.reshape(-1, S))


def _init_data(
    data: list["eastbay.dataset.Dataset"],
    window_size: int,
    overlap: int,
    chunk_size: int = None,
    max_samples: int = 20,
):
    """Chunk up the data. If chunk_size is missing, set it to ~1/5th of the shortest
    contig. (This may not be optimal)."""
    N = data[0].N
    afss = []
    chunk_size = int(min(0.2 * ds.L / window_size for ds in data))
    if chunk_size < 10 * overlap:
        logger.warn(
            "The chunk size is %dbp, which is less than 10 times the overlap (%dbp).",
            chunk_size,
            overlap,
        )
    chunks = []
    with ProcessPoolExecutor() as pool:
        futs = []
        for ds in data:
            if ds.N != N:
                raise ValueError("All datasets must have the same number of samples")
            futs.append(pool.submit(ds.get_data, window_size))
        for f in as_completed(futs):
            d = f.result()
            afss.append(d["afs"])
            ch = _chunk_het_matrix(d["het_matrix"][:max_samples], overlap, chunk_size)
            chunks.append(ch)

    assert all(a.ndim == 1 for a in afss)
    assert len({a.shape for a in afss}) == 1
    # all afs have same dimension
    assert len({ch.shape[-1] for ch in chunks}) == 1
    assert all(ch.ndim == 2 for ch in chunks)
    return np.sum(afss, 0), np.concatenate(chunks, 0)


def fit(
    data: list["eastbay.dataset.Dataset"],
    test_data: "eastbay.dataset.Dataset" = None,
    options: dict = {},
) -> list["eastbay.size_history.DemographicModel"]:
    """Add docstring."""
    # some defaults pulled from the options dict
    key = options.get("key", jax.random.PRNGKey(1))
    niter = options.get("niter", 1000)
    window_size = options.get("window_size", 100)
    overlap = options.get("overlap", 500)
    chunk_size = options.get("chunk_size")
    max_samples = options.get("max_samples", 20)
    afs, chunks = _init_data(data, window_size, overlap, chunk_size, max_samples)

    # on average, we'd like to visit every data point once. but we don't want it to be
    # too huge because that slows down computation, and usually isn't doesn't lead to
    # a big improvement. For now it's capped at 5.
    S = options.get("minibatch_size")
    if not S:
        S = min(5, int(len(chunks) / niter))

    # avoid storing a huge array on gpu if we're only going to use a small part of it
    if len(chunks) > 10 * S * niter:
        key, subkey = jax.random.split(key)
        # important: use numpy to do this _not_ jax. (jax will put it on the gpu which
        # causes the very problem we are trying to solve.)
        chunks = np.random.default_rng(np.asarray(subkey)).choice(
            chunks, size=(10 * S * niter,), replace=False
        )
    N = len(chunks)

    # initialize the model
    init = options.get("init")
    if init is None:
        ch0 = chunks[:, overlap:]
        theta = ch0[ch0 > -1].mean()
        logger.info("Scaled mutation rate Θ=%f", theta)
        init = MCMCParams.from_linear(
            pattern="14*1+1*2",
            rho=options.get("rho_over_theta", 1.0) * theta,
            t1=1e-4,
            tM=15.0,
            c=jnp.ones(15),
            theta=theta,
            alpha=options.get("alpha", 0.0),
        )
    assert isinstance(init, MCMCParams)
    opt = optax.amsgrad(learning_rate=options.get("learning_rate", 0.1))
    svgd = blackjax.svgd(grad(_log_density), opt)

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
    train_kern = PSMCKernel(
        M=M,
        data=np.ascontiguousarray(data_chunks),
        double_precision=options.get("double_precision", False),
    )

    # if there is a test set, define elpd() function for computing expected
    # log-predictive density
    # used to gauge convergence.
    if test_data:
        d = test_data.get_data()
        test_afs = d["afs"]
        test_data = d["het_matrix"][:max_samples]
        N_test = test_data.shape[0]
        test_kern = PSMCKernel(
            M=M, data=np.ascontiguousarray(d["het_matrix"]), double_precision=False
        )

        @jit
        def elpd(mcps):
            @vmap
            def ll(mcp):
                return _log_density(
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
    if not options.get("callback"):
        cb = eastbay.liveplot.liveplot_cb(
            state, truth=options.get("truth"), plot_every=options.get("plot_every")
        )
    else:
        cb = options["callback"]
    try:
        ema = best_elpd = None
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
                        best_elpd = (i, ema, state)
                    if i - best_elpd[0] > 100:
                        logger.info(
                            "The expected log-predictive density has not improved in "
                            "the last 100 iterations; exiting."
                        )
                        state = best_elpd[2]
                        break
                    pbar.set_description(
                        f"elpd={ema:.0f} best={best_elpd[1]:.0f} "
                        f"age={i - best_elpd[0]:d}"
                    )
                cb(state)
    except KeyboardInterrupt:
        pass  # if we break out, just return the most recent state

    dms = vmap(MCMCParams.to_dm)(state.particles)
    # the learned rates are per window, so we have to scale up the
    # mutation and recombination to get the per-base-pair rates.
    dms = dms._replace(theta=dms.theta / window_size, rho=dms.rho / window_size)
    # convert to list of dms, easier for the end user who doesn't know jax
    return tree_unstack(dms)
