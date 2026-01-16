import jax
import jax.numpy as jnp


def _edge_normals(tri):
    # tri: (3, 2)
    tri_next = jnp.roll(tri, shift=-1, axis=0)
    edges = tri_next - tri
    # Perpendiculars (normals); normalization is unnecessary for SAT.
    return jnp.stack([-edges[:, 1], edges[:, 0]], axis=1)


def triangles_intersect(t0, t1, eps=1e-12):
    """
    Strict triangle intersection:
    - returns True for overlaps with positive area (including containment),
    - returns False for "touching only" (shared edge/vertex contact) and separated cases.
    """
    axes = jnp.concatenate([_edge_normals(t0), _edge_normals(t1)], axis=0)  # (6, 2)

    proj0 = t0 @ axes.T
    proj1 = t1 @ axes.T

    min0 = proj0.min(axis=0)
    max0 = proj0.max(axis=0)
    min1 = proj1.min(axis=0)
    max1 = proj1.max(axis=0)

    overlap_strict = (max0 > min1 + eps) & (max1 > min0 + eps)
    return jnp.all(overlap_strict)


def triangles_intersection_score(t0, t1, eps=1e-12):
    """
    Strict triangle intersection score:
    - positive when triangles overlap,
    - <= 0 when separated or only touching.
    """
    axes = jnp.concatenate([_edge_normals(t0), _edge_normals(t1)], axis=0)  # (6, 2)

    proj0 = t0 @ axes.T
    proj1 = t1 @ axes.T

    min0 = proj0.min(axis=0)
    max0 = proj0.max(axis=0)
    min1 = proj1.min(axis=0)
    max1 = proj1.max(axis=0)

    score = (max0 - min1 - eps) * (max1 - min0 - eps)
    return jnp.min(score, axis=-1)


_figure_intersection_score = triangles_intersection_score
_figure_intersection_score = jax.vmap(_figure_intersection_score, (0, None, None))
_figure_intersection_score = jax.vmap(_figure_intersection_score, (None, 0, None))


def figure_intersection_score(t0, t1, eps=1e-12):
    """
    Given two figures (each figure is a union of triangles)
    """
    score = _figure_intersection_score(t0, t1, eps)
    score = jax.nn.relu(score)
    score = jax.numpy.sum(score)
    return score
