import numpy as np

from phlash.util import fold_afs


def test_fold():
    for x, y in [
        ([], []),
        ([1], [1]),
        ([1, 2], [3]),
        (np.arange(5), [4, 4, 2]),
        (np.arange(6), [5, 5, 5]),
    ]:
        np.testing.assert_allclose(fold_afs(x), y)
