import numpy as np

from phlash.util import fold_afs, project_afs


def test_fold():
    for x, y in [
        ([], []),
        ([1], [1]),
        ([1, 2], [3]),
        (np.arange(5), [4, 4, 2]),
        (np.arange(6), [5, 5, 5]),
    ]:
        np.testing.assert_allclose(fold_afs(x), y)


def test_project_afs(rng):
    def afs(n):
        return 2 / np.arange(1, n)

    m, n = sorted(rng.integers(2, 100, size=(2,)))
    afs_n = afs(n)
    afs_m = afs(m)
    proj_m = project_afs(afs_n, m)
    np.testing.assert_allclose(afs_m, proj_m)
