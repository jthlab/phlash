import numpy as np
from scipy.interpolate import PPoly

from phlash.util import invert_cpwli, tree_stack, tree_unstack


def test_tree_stack():
    tree = {"a": np.array([1, 2]), "b": np.array([3, 4])}
    ts = tree_stack([tree, tree])
    assert ts["a"].shape == (2, 2)
    assert ts["b"].shape == (2, 2)


def test_tree_unstack():
    tree = {"a": np.array([[1, 2], [1, 2]]), "b": np.array([[3, 4], [3, 4]])}
    tu = tree_unstack(tree)
    assert len(tu) == 2
    assert tu[0]["a"].shape == (2,)
    assert tu[0]["b"].shape == (2,)
    assert tu[1]["a"].shape == (2,)
    assert tu[1]["b"].shape == (2,)


def test_invert_cpwli(rng):
    # construct continuous piecewise linear function increasing function
    x = np.array([0.0, 1.0, 2.0, 3.0, np.inf])
    c = rng.uniform(size=(1, 4))
    R = PPoly(x=x, c=c).antiderivative()
    Q = invert_cpwli(R)
    # check that Q(R(x)) = x for random x
    x = np.concatenate([rng.uniform(size=10), [0.0, 1.0, 2.0, 3.0, 100]])
    np.testing.assert_allclose(Q(R(x)), x)
    # check that inv(inv(R)) = R
    R2 = invert_cpwli(Q)
    np.testing.assert_allclose(R.x, R2.x)
    np.testing.assert_allclose(R.c, R2.c)
    # check that it returns nan outside the domain
    assert np.isnan(Q(-1))
