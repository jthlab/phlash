import jax.numpy as jnp
import jax.tree_util as jtu


def tree_stack(trees):
    return jtu.tree_map(lambda *v: jnp.stack(v), *trees)


def tree_unstack(tree):
    leaves, treedef = jtu.tree_flatten(tree)
    return [treedef.unflatten(leaf) for leaf in zip(*leaves, strict=True)]
