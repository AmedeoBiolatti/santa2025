import jax
import jax.numpy as jnp
from jax import lax

MAXV = 7  # <= 7 verts after clipping a triangle by an AABB


def cross2(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _segment_line_intersection_t(n: jnp.ndarray, c: jnp.ndarray,
                                 s: jnp.ndarray, e: jnp.ndarray,
                                 eps: float = 1e-12) -> jnp.ndarray:
    """Vectorized t for intersection of segment s->e with line n·p = c."""
    d = e - s
    denom = jnp.einsum("k,ik->i", n, d)          # (MAXV,)
    numer = c - jnp.einsum("k,ik->i", n, s)      # (MAXV,)
    t = jnp.where(jnp.abs(denom) > eps, numer / denom, jnp.zeros_like(denom))
    return jnp.clip(t, 0.0, 1.0)


def clip_poly_halfplane_fast(poly: jnp.ndarray, count: jnp.ndarray,
                             n: jnp.ndarray, c: jnp.ndarray):
    """
    Sutherland–Hodgman for one half-plane n·p >= c.
    Static buffers, but vectorized; emits <= MAXV points.
    """
    idx = jnp.arange(MAXV, dtype=jnp.int32)
    valid = idx < count  # (MAXV,)

    # e are the current vertices (only first 'count' are valid).
    e = poly  # (MAXV,2)

    # s are previous vertices; for i=0, s is last valid vertex at count-1.
    s_idx = jnp.where(idx == 0, count - 1, idx - 1)
    s = poly[s_idx]  # (MAXV,2)

    # In/out tests (masked by valid so invalid edges emit nothing).
    s_in = (jnp.einsum("k,ik->i", n, s) >= c) & valid
    e_in = (jnp.einsum("k,ik->i", n, e) >= c) & valid

    # Intersection for all edges; only used when crossing.
    t = _segment_line_intersection_t(n, c, s, e)
    inter = s + t[:, None] * (e - s)

    # Each edge emits:
    # - inter if crossing (s_in xor e_in)
    # - e if e_in
    inter_flag = s_in ^ e_in
    e_flag = e_in

    k = inter_flag.astype(jnp.int32) + e_flag.astype(jnp.int32)  # points per edge (0/1/2)
    out_count = jnp.minimum(jnp.sum(k), jnp.int32(MAXV))

    # Compute write positions (prefix sum).
    start = jnp.cumsum(k) - k  # (MAXV,)

    # Indices for the emitted points (flattened to length 2*MAXV).
    # For each edge i:
    #   slot 0: inter goes to start[i]         if inter_flag
    #   slot 1: e goes to start[i]+inter_flag  if e_flag
    idx_inter = start
    idx_e = start + inter_flag.astype(jnp.int32)

    flat_idx = jnp.concatenate([idx_inter, idx_e], axis=0)  # (2*MAXV,)
    flat_pts = jnp.concatenate([inter, e], axis=0)          # (2*MAXV,2)
    flat_mask = jnp.concatenate([inter_flag, e_flag], axis=0)  # (2*MAXV,)

    # Compaction into (MAXV,2) via tiny dense op:
    # one_hot(row) * mask gives (2*MAXV, MAXV). Then transpose @ points => (MAXV,2).
    row = jnp.where(flat_mask, flat_idx, jnp.int32(0))
    oh = jax.nn.one_hot(row, MAXV, dtype=poly.dtype) * flat_mask[:, None].astype(poly.dtype)
    out_poly = oh.T @ flat_pts  # (MAXV,2)

    return out_poly, out_count


def polygon_area_fast(poly: jnp.ndarray, count: jnp.ndarray) -> jnp.ndarray:
    """Vectorized shoelace for up to MAXV vertices; returns 0 if count < 3."""
    idx = jnp.arange(MAXV, dtype=jnp.int32)
    valid = idx < count

    j = (idx + 1) % jnp.maximum(count, 1)  # avoid %0; masked anyway when count<1
    pi = poly[idx]
    pj = poly[j]

    s = jnp.sum(cross2(pi, pj) * valid.astype(poly.dtype))
    area = 0.5 * jnp.abs(s)
    return jnp.where(count >= 3, area, jnp.array(0.0, dtype=poly.dtype))


def triangle_aabb_intersection_area(tri_ccw: jnp.ndarray,
                                         xmin: jnp.ndarray, ymin: jnp.ndarray,
                                         xmax: jnp.ndarray, ymax: jnp.ndarray) -> jnp.ndarray:
    dtype = tri_ccw.dtype

    # Initialize polygon = triangle (padded)
    poly = jnp.zeros((MAXV, 2), dtype=dtype).at[:3].set(tri_ccw)
    count = jnp.int32(3)

    # Clip by 4 AABB half-planes
    poly, count = clip_poly_halfplane_fast(poly, count, jnp.array([ 1.0,  0.0], dtype=dtype), jnp.array(xmin,  dtype=dtype))
    poly, count = clip_poly_halfplane_fast(poly, count, jnp.array([-1.0,  0.0], dtype=dtype), jnp.array(-xmax, dtype=dtype))
    poly, count = clip_poly_halfplane_fast(poly, count, jnp.array([ 0.0,  1.0], dtype=dtype), jnp.array(ymin,  dtype=dtype))
    poly, count = clip_poly_halfplane_fast(poly, count, jnp.array([ 0.0, -1.0], dtype=dtype), jnp.array(-ymax, dtype=dtype))

    return polygon_area_fast(poly, count)


__all__ = ["triangle_aabb_intersection_area"]
