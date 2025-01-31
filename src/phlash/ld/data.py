import itertools as it
import os
from concurrent.futures import ThreadPoolExecutor
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
        lds = jnp.array(lds)
        N = len(lds)
        reps = jax.random.choice(key, lds, shape=(B, N), replace=True)
        assert reps.shape == (B, N, 2)  # Dz / pi2, D2 / pi2
        Sigma_boot = jnp.cov(reps.mean(1), rowvar=False)
        return dict(mu=jnp.mean(lds, axis=0), Sigma=Sigma_boot)


def calc_ld(
    physical_pos, genetic_pos, genotypes, r_buckets, ld_region_size
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
    physical_pos = physical_pos[bi]

    regions = np.arange(physical_pos[0], physical_pos[-1], ld_region_size).astype(int)
    # compute counts
    futs = {}
    ret = {}
    try:
        num_threads = int(os.environ.get("PHLASH_LD_NUM_THREADS"))
    except (TypeError, ValueError):
        num_threads = None
    with ThreadPoolExecutor(num_threads) as pool:
        for a, b in it.pairwise(r_buckets):
            ret[(a, b)] = []
            for i, j in it.pairwise(regions):
                u, v = np.maximum(
                    0, np.searchsorted(physical_pos, [i, j], side="right") - 1
                )
                f = pool.submit(_helper, genotypes, genetic_pos, a, b, u, v)
                futs[f] = (a, b)
        for f in tqdm.tqdm(
            futs,
            desc="Calculating LD",
            unit="bucket",
        ):
            ld = f.result()
            if ld is not None:
                k = futs[f]
                ret.setdefault(k, []).append(ld.norm())
        return ret


def _helper(genotypes, genetic_pos, a, b, u, v) -> LdStats:
    # arrays can not be created or returned in nopython mode
    ret = np.zeros(4)
    counts = np.zeros(9)
    _compute_stats(genotypes, genetic_pos, a, b, u, v, counts, ret)
    d = dict(zip(["D2", "Dz", "pi2"], ret[:3]))
    return LdStats(**d)


@numba.jit(nogil=True, nopython=True)
def _compute_stats(genotypes, genetic_pos, a, b, u, v, counts, ret):
    n = 0
    i = u
    while i < v:
        g_x = genotypes[i]
        p_x = genetic_pos[i]
        s_x = 3 * np.sum(g_x, 1)
        if np.any(g_x == -1):
            continue
        j = i
        while j < v and genetic_pos[j] - p_x < a:
            j += 1
        while j < v and genetic_pos[j] - p_x < b:
            g_y = genotypes[j]
            if np.any(g_y == -1):
                continue
            r = s_x + np.sum(g_y, 1)
            counts[:] = 0
            for z in range(len(r)):
                counts[r[z]] += 1
            # ret[0] += (phlash.ld.stats.Dhat(counts) - ret[0]) / (n + 1)
            ret[0] += (phlash.ld.stats.nD2(counts) - ret[0]) / (n + 1)
            ret[1] += (phlash.ld.stats.nDz(counts) - ret[1]) / (n + 1)
            ret[2] += (phlash.ld.stats.nPi2(counts) - ret[2]) / (n + 1)
            j += 1
            n += 1
            if n > 1e7:
                # cannot return inside loops
                # instead force break
                i = v
                j = v
        i += 1
    ret[3] = n
