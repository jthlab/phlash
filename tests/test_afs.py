import numpy as np

from phlash.afs import bws_transform, fold_transform, project_transform


def test_fold():
    for x, y in [
        ([], []),
        ([1], [1]),
        ([1, 2], [3]),
        (np.arange(5), [4, 4, 2]),
        (np.arange(6), [5, 5, 5]),
    ]:
        n = len(x) + 1
        T = fold_transform(n)
        np.testing.assert_allclose(T @ x, y)


def test_project_transform(rng):
    def afs(n):
        return 2 / np.arange(1, n)

    m, n = sorted(rng.integers(2, 100, size=(2,)))
    afs_n = afs(n)
    afs_m = afs(m)
    T = project_transform(n, m)
    proj_m = T @ afs_n
    np.testing.assert_allclose(afs_m, proj_m)


def test_bws():
    s = np.array([1])
    np.testing.assert_allclose(bws_transform(s), np.eye(1))
    s = np.array([100000, 1])
    np.testing.assert_allclose(bws_transform(s), np.eye(2))
    s = np.array([100000, 200, 1])
    np.testing.assert_allclose(bws_transform(s), [[1, 0, 0], [0, 1, 1]])
