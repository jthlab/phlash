import numpy as np

from phlash.util import tree_stack, tree_unstack


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
