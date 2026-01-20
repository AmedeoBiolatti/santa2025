import jax
import jax.numpy as jnp

TREE = jax.numpy.array([
    [(0.0, 0.8), (-0.125, 0.5), (0.125, 0.5)],
    [(0.2, 0.25), (0.0, 27 / 44), (-0.2, 0.25)],
    [(0.35, 0.0), (0.0, 0.35), (-0.35, 0.0)],
    [(0.075, 0.0), (-0.075, -0.2), (0.075, -0.2)],
    [(0.075, 0.0), (-0.075, 0.0), (-0.075, -0.2)],
])

TREE_NORMALS = jnp.array([
    [(0.9230769230769231, -0.3846153846153846), (-0.0, 1.0), (-0.9230769230769231, -0.3846153846153846)],
    [(-0.8762160353251135, -0.4819187497721559), (0.8762160353251135, -0.4819187497721559), (-0.0, 1.0)],
    [(-0.7071067811865476, -0.7071067811865476), (0.7071067811865476, -0.7071067811865476), (-0.0, 1.0)],
    [(0.8, -0.6), (-0.0, 1.0), (-1.0, 0.0)],
    [(-0.0, -1.0), (1.0, 0.0), (-0.8, 0.6)],
])

TREE_DISJOINT = jax.numpy.array([
    [(0.0, 0.8), (-0.125, 0.5), (0.125, 0.5)],
    [(0.2, 0.25), (-0.0625, 0.5), (-0.2, 0.25)],
    [(0.2, 0.25), (0.0625, 0.5), (-0.0625, 0.5)],

    [(0.35, 0.0), (-0.1, 0.25), (-0.35, 0.0)],
    [(0.35, 0.0), (0.1, 0.25), (-0.1, 0.25)],

    [(-0.075, -0.2), (0.075, -0.2), (-0.075, 0.0)],
    [(0.075, 0.0), (-0.075, 0.0), (0.075, -0.2)],
])


def _rotate_point(point, alpha):
    assert point.ndim == 1
    c, s = jax.numpy.cos(alpha), jax.numpy.sin(alpha)
    R = jax.numpy.array([[c, -s], [s, c]])
    point = R @ point
    return point


def _move_point(params, point):
    offset, alpha = params
    return offset + _rotate_point(point, alpha)


_move_triangle = jax.vmap(_move_point, in_axes=(None, 0))
_move_figure = jax.vmap(_move_triangle, in_axes=(None, 0))


def params_to_tree(params, disjoint: bool = False) -> jax.Array:
    if disjoint:
        return _move_figure(params, TREE_DISJOINT)
    return _move_figure(params, TREE)


def params_to_trees(params, disjoint: bool = False) -> jax.Array:
    return jax.vmap(params_to_tree, (0, None))(params, disjoint)


def get_tree_centers(params):
    """Tree centers using the small-radius proxy."""
    delta = CENTER_Y
    pos, ang = params
    offsets = jnp.stack([-jnp.sin(ang) * delta, jnp.cos(ang) * delta], axis=-1)
    return pos + offsets


CENTER_Y = 0.2972
CENTER_R = 0.5029

THR = 2 * CENTER_R
THR2 = THR ** 2
