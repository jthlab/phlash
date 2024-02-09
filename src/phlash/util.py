import jax.numpy as jnp
import jax.tree_util as jtu


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
