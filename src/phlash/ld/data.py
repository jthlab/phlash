import itertools as it
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import tqdm.auto as tqdm

import phlash.ld.stats


def calc_ld(genetic_pos, genotypes, r_buckets):
    genotypes = np.asarray(genotypes)
    genetic_pos = np.asarray(genetic_pos)

    @jax.vmap
    def count_pairs(gts_x, gts_y):
        def c(gt_x, gt_y):
            return 3 * gt_x.sum() + gt_y.sum()
            miss = jnp.any((gt_x == -1) | (gt_y == -1))
            return jnp.where(miss, 9, ret)

        r = jax.vmap(c, (0, 0))(gts_x, gts_y)
        return jnp.bincount(r, length=10)[:-1]

    @jax.jit
    @partial(jax.vmap, in_axes=(0, None, None))
    def annulus(i, a, b):
        p = jnp.array(genetic_pos)[i]
        u = p + a
        v = p + b
        uv = jnp.sort(jnp.array([u, v]))
        ret = jnp.searchsorted(genetic_pos, uv)
        return jnp.insert(ret, 0, i)

    # compute counts
    ret = {}
    A = jnp.arange(len(genetic_pos))
    for a, b in tqdm.tqdm(
        it.pairwise(r_buckets),
        total=len(r_buckets) - 1,
        desc="Calculating LD",
        unit="bucket",
    ):
        ijk = np.asarray(annulus(A, a, b))
        i, j, k = ijk.T
        mask = j < k
        i, j, k = ijk[mask].T
        assert i.size == j.size == k.size
        if not i.size:
            continue
        inds = np.concatenate(
            [
                np.stack([np.full(kk - jj, ii), np.arange(jj, kk)], 1)
                for ii, jj, kk in zip(i, j, k)
            ]
        )
        counts = count_pairs(genotypes[inds[:, 0]], genotypes[inds[:, 1]])
        d = ret[(a, b)] = {}
        for stat in ["Dhat", "D2", "Dz", "pi2"]:
            f_stat = jax.vmap(getattr(phlash.ld.stats, stat))
            d[stat] = f_stat(counts)
        d["n"] = len(counts)

    return ret
