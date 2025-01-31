from typing import NamedTuple

import jax
import numpy as np
import scipy


class LinearAfsTransform(NamedTuple):
    A: jax.Array

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.A @ x


class IdentityAfsTransform(NamedTuple):
    def __call__(self, x: jax.Array) -> jax.Array:
        return x


def fold_transform(n):
    rows = (n - 1) // 2 + (n - 1) % 2
    cols = n - 1
    T = np.eye(N=rows, M=cols)
    T += T[:, ::-1]
    # divide to ensure that every allele size is counted at most once -- if n is odd the
    # middle entry is now counted twice
    T /= T.sum(0)
    return T


def project_transform(n, m) -> np.ndarray:
    """Project an afs on n samples (i.e. an (n-1)-dimensional count vector) to an afs on
    m samples (i.e. an (m-1)-dimensional count vector)."""
    assert n >= m
    i, j = np.ogrid[1:m, 1:n]
    return scipy.stats.hypergeom.pmf(M=n, N=m, n=j, k=i)


def bws_transform(afs, alpha: float = 0.2) -> np.ndarray:
    # Bhaskar-Wang-Song transform:
    n = len(afs) + 1
    p = np.cumsum(afs) / np.sum(afs)
    i = np.searchsorted(p, 1.0 - alpha, "right") + 1
    ret = np.eye(N=i, M=n - 1)
    if i < n - 1:
        j = np.arange(n - 1)[None]
        ret = np.concatenate([ret, (i <= j).astype(float)])
    return ret


def default_afs_transform(afs: np.ndarray) -> np.ndarray:
    if afs.ndim > 1:
        return IdentityAfsTransform()

    T1 = fold_transform(len(afs) + 1)
    assert (T1 @ afs).sum() == afs.sum()
    T2 = bws_transform(T1 @ afs)
    return LinearAfsTransform(T2 @ T1)
