import jax
import jax.numpy as jnp

TREE = jax.numpy.array([
    [(0.0, 0.8), (0.125, 0.5), (-0.125, 0.5)],
    [(0.2, 0.25), (-0.2, 0.25), (0.0, 27 / 44)],
    [(0.35, 0.0), (-0.35, 0.0), (0.0, 0.35)],
    [(0.075, 0.0), (0.075, -0.2), (-0.075, -0.2)],
    [(0.075, 0.0), (-0.075, -0.2), (-0.075, 0.0)]
])


def _rotate_point(point, alpha):
    c, s = jax.numpy.cos(alpha), jax.numpy.sin(alpha)
    R = jax.numpy.array([[c, -s], [s, c]])
    point = R @ point
    return point


def _move_point(params, point):
    offset, alpha = params
    return offset + _rotate_point(point, alpha)


_move_triangle = jax.vmap(_move_point, in_axes=(None, 0))
_move_figure = jax.vmap(_move_triangle, in_axes=(None, 0))


def params_to_tree(params):
    tree = _move_figure(params, TREE)
    return tree


params_to_trees = jax.vmap(params_to_tree)


def get_tree_centers(params):
    """Tree centers using the small-radius proxy."""
    delta = 0.28
    pos, ang = params
    offsets = jnp.stack([-jnp.sin(ang) * delta, jnp.cos(ang) * delta], axis=-1)
    return pos + offsets


THR = 2 * 0.52
THR2 = THR ** 2
