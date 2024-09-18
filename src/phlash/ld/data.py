import itertools as it
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import jax
import jax.numpy as jnp
import numba
import numpy as np
import tqdm.auto as tqdm

import phlash.ld.stats


def calc_ld(genetic_pos, genotypes, r_buckets):
    genotypes = np.asarray(genotypes)
    genetic_pos = np.asarray(genetic_pos)
    # filter to only biallelic genotypes since that is what is modeled by the LD moments
    # functions
    bi = genotypes.max(axis=(1, 2)) == 1
    genotypes = genotypes[bi]
    genetic_pos = genetic_pos[bi]

    # shared_arrays = []
    # for array, dtype in [(genotypes, np.int8), (genetic_pos, np.float32)]:
    #     array = array[bi].astype(dtype)
    #     shm = shared_memory.SharedMemory(create=True, size=array.nbytes)
    #     np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)[:] = array[:]
    #     shared_arrays.append((shm.name, dtype, array.shape))

    # compute counts
    futs = {}
    ret = {}
    with ThreadPoolExecutor() as pool:
        for a, b in it.pairwise(r_buckets):
            f = pool.submit(_helper, genotypes, genetic_pos, a, b)
            futs[f] = (a, b)
        for f in tqdm.tqdm(
            futs,
            desc="Calculating LD",
            unit="bucket",
        ):
            d = f.result()
            if d is not None:
                a, b = futs[f]
                ret[(a, b)] = d
        return ret


def _helper(genotypes, genetic_pos, a, b):
    # have to hold references to the shared memory objects for the whole function call
    # shms = [
    #     shared_memory.SharedMemory(name=shm_name) for shm_name, _, _ in shared_arrays
    # ]
    # genotypes, genetic_pos = (
    #     np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    #     for (_, dtype, shape), shm in zip(shared_arrays, shms)
    # )

    @jax.jit
    @partial(jax.vmap, in_axes=(0, None, None))
    def annulus(i, a, b):
        p = jnp.array(genetic_pos)[i]
        u = p + a
        v = p + b
        uv = jnp.sort(jnp.array([u, v]))
        ret = jnp.searchsorted(genetic_pos, uv)
        return jnp.insert(ret, 0, i)

    A = jnp.arange(len(genetic_pos))
    ijk = np.asarray(annulus(A, a, b))
    i, j, k = ijk.T
    mask = j < k
    ijk = ijk[mask]
    i, j, k = ijk.T
    assert i.size == j.size == k.size
    if not i.size:
        return
    stats = _compute_stats(genotypes, ijk)
    ret = dict(zip(["Dhat", "D2", "Dz", "pi2"], stats))
    # ret["n"] = len(inds)
    return ret


@numba.jit(nogil=True, nopython=True, cache=True)
def _compute_stats(genotypes, ijk):
    assert genotypes.shape[2] == 2
    ret = np.zeros(4)
    counts = np.zeros(9, dtype=numba.int16)
    # inds = np.concatenate(
    #     [
    #         np.stack([np.full(kk - jj, ii), np.arange(jj, kk)], 1)
    #         for ii, jj, kk in zip(i, j, k)
    #     ]
    # )
    n = 0
    for a in range(len(ijk)):
        i = ijk[a, 0]
        j = ijk[a, 1]
        k = ijk[a, 2]
        g_x = genotypes[i]
        if np.any(g_x == -1):
            continue
        s_x = 3 * np.sum(g_x, 1)
        for y in range(j, k):
            g_y = genotypes[y]
            if np.any(g_y == -1):
                continue
            r = s_x + np.sum(g_y, 1)
            counts[:] = 0
            for z in range(len(r)):
                counts[r[z]] += 1
            ret[0] += (phlash.ld.stats.Dhat(counts) - ret[0]) / (n + 1)
            ret[1] += (phlash.ld.stats.D2(counts) - ret[1]) / (n + 1)
            ret[2] += (phlash.ld.stats.Dz(counts) - ret[2]) / (n + 1)
            ret[3] += (phlash.ld.stats.pi2(counts) - ret[3]) / (n + 1)
            n += 1
    return ret
