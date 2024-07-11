import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np
from scipy.interpolate import PPoly


class Pattern:
    """Class for parsing PSMC-style pattern strings."""

    def __init__(self, pattern: str):
        try:
            epochs = self._epochs = []
            for s in pattern.split("+"):
                if "*" in s:
                    k, width = map(int, s.split("*"))
                else:
                    k = 1
                    width = int(s)
                epochs += [width] * k
        except Exception:
            raise ValueError("could not parse pattern")
        if len(epochs) == 0:
            raise ValueError("pattern must contain at least one epoch")
        if any(e <= 0 for e in epochs):
            raise ValueError("epochs must be positive")

    @property
    def M(self):
        return sum(self._epochs)

    def __len__(self) -> int:
        return len(self._epochs)

    def expand(self, x):
        assert len(x) == len(self)
        return sum([e * [xx] for e, xx in zip(self._epochs, x)], [])


def tree_stack(trees):
    return jax.tree.map(lambda *v: jnp.stack(v), *trees)


def tree_unstack(tree):
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    return [treedef.unflatten(leaf) for leaf in zip(*leaves, strict=True)]


def softplus_inv(y):
    # y > 0
    return y + jnp.log1p(-jnp.exp(-y))


def invert_cpwli(R: PPoly):
    """Invert a continuous piecewise-linear increasing function."""
    x = R.x
    assert np.isinf(x[-1])
    assert np.isclose(x[0], 0.0)
    b, a = R.c
    # the inverse func interpolates (R(x[i]), x[i]) for i = 0, 1, ..., n - 1
    return PPoly(x=np.append(a, np.inf), c=np.array([1 / b, x[:-1]]), extrapolate=False)
