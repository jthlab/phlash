import itertools as it
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numba
import numpy as np
import tqdm.auto as tqdm

import phlash.ld.stats


class LdStats(NamedTuple):
    D2: jax.Array
    Dz: jax.Array
    pi2: jax.Array

    def norm(self):
        return jnp.array([self.D2, self.Dz]) / self.pi2

    @classmethod
    def summarize(
        cls, lds: list["LdStats"], key: int | jax.Array = 1, B: int = 10_000
    ) -> dict[str, jax.Array]:
        if isinstance(key, int):
            key = jax.random.PRNGKey(key)

        # convert to stacked pytree
        N = len(lds)
        lds = jax.tree.map(lambda *a: jnp.array(a), *lds)
        # normalize
        lds = jax.vmap(cls.norm)(lds)
        reps = jax.random.choice(key, lds, shape=(B, N), replace=True)
        assert reps.shape == (B, N, 2)  # Dz / pi2, D2 / pi2
        Sigma_boot = jnp.cov(reps.mean(1), rowvar=False)
        return dict(mu=jnp.mean(lds, axis=0), Sigma=Sigma_boot)


def calc_ld(
    physical_pos, genetic_pos, genotypes, r_buckets, region_size
) -> dict[tuple[float, float], list[LdStats]]:
    genotypes = np.asarray(genotypes)
    L, N, _ = genotypes.shape
    assert genotypes.shape == (L, N, 2)
    if N < 4:
        raise ValueError(
            "At least N=4 diploid samples are needed to compute LD "
            "statistics (and preferably lots more)"
        )
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
    regions = np.searchsorted(
        physical_pos, np.arange(physical_pos.min(), physical_pos.max(), region_size)
    )
    with ThreadPoolExecutor() as pool:
        for a, b in it.pairwise(r_buckets):
            ret[(a, b)] = []
            for i, j in it.pairwise(regions):
                assert i < j
                f = pool.submit(_helper, genotypes[i:j], genetic_pos[i:j], a, b)
                futs[f] = (a, b)
        for f in tqdm.tqdm(
            futs,
            desc="Calculating LD",
            unit="bucket",
        ):
            ld = f.result()
            if ld is not None:
                k = futs[f]
                ret.setdefault(k, []).append(ld)
        return ret


def _helper(genotypes, genetic_pos, a, b) -> LdStats:
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
    d = dict(zip(["D2", "Dz", "pi2"], stats))
    return LdStats(**d)


@numba.jit(nogil=True, nopython=True, cache=True)
def _compute_stats(genotypes, ijk):
    assert genotypes.shape[2] == 2
    ret = np.zeros(3)
    counts = np.zeros(9, dtype=numba.int16)
    # inds = np.concatenate(
    #     [
    #         np.stack([np.full(kk - jj, ii), np.arange(jj, kk)], 1)
    #         for ii, jj, kk in zip(i, j, k)
    #     ]
    # )
    n = 0
    for a in range(len(ijk)):
        if n > 5_000_000:
            break
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
            # ret[0] += (phlash.ld.stats.Dhat(counts) - ret[0]) / (n + 1)
            ret[0] += (phlash.ld.stats.D2(counts) - ret[0]) / (n + 1)
            ret[1] += (phlash.ld.stats.Dz(counts) - ret[1]) / (n + 1)
            ret[2] += (phlash.ld.stats.pi2(counts) - ret[2]) / (n + 1)
            n += 1
    return ret
