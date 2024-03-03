import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import scipy


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
    return jtu.tree_map(lambda *v: jnp.stack(v), *trees)


def tree_unstack(tree):
    leaves, treedef = jtu.tree_flatten(tree)
    return [treedef.unflatten(leaf) for leaf in zip(*leaves, strict=True)]


def fold_afs(afs):
    afs = jnp.array(afs)
    n = len(afs)
    if n % 2 == 1:
        m = n // 2
        return jnp.append(fold_afs(jnp.delete(afs, m)), afs[m])
    return afs[: n // 2] + afs[-1 : -1 - n // 2 : -1]


def project_afs(afs, m):
    """Project an n-dimensional afs (i.e. an (n-1)-dimensional count vector) to an
    m-dimensional afs (i.e. an (m-1)-dimensional count vector)."""
    assert afs.ndim == 1
    n = afs.size + 1
    assert n > m
    i, j = np.ogrid[1:m, 1:n]
    B = scipy.stats.hypergeom.pmf(M=n, N=m, n=j, k=i)
    return B @ afs


def softplus_inv(y):
    # y > 0
    return y + jnp.log1p(-jnp.exp(-y))
