from __future__ import annotations

import jax
import jax.numpy as jnp

from flax.struct import dataclass

from santa.tree_packing import area

__all__ = ['Grid2D']

_deltas = jnp.array([[0, -1], [-1, -1], [+1, -1], [0, 0], [-1, 0], [+1, 0], [0, +1], [-1, +1], [+1, +1]])


def compute_figure_aabb_area(figure, aabb):
    triangle_aabb_fn = lambda t, a: area.triangle_aabb_intersection_area(t, a[0], a[1], a[2], a[3])
    figure_aabb_fn = jax.vmap(triangle_aabb_fn, (0, None))
    figure_aabb_list_fn = jax.vmap(figure_aabb_fn, (None, 0))
    areas = figure_aabb_list_fn(figure, aabb).sum(-1)
    assert len(areas.shape) == 1
    return areas


def compute_figures_aabb_area(figures, aabb_grid):
    triangle_aabb_fn = lambda t, a: area.triangle_aabb_intersection_area(t, a[0], a[1], a[2], a[3])
    figures_aabb_fn = jax.vmap(jax.vmap(triangle_aabb_fn, (0, None)), (0, None))
    figures_aabb_grid_fn = jax.vmap(jax.vmap(figures_aabb_fn, (None, 0)), (None, 0))
    areas = figures_aabb_grid_fn(figures, aabb_grid).sum(-1)
    assert len(areas.shape) == 3
    return areas


@dataclass
class Grid2D:
    """
    added padding to the grid to avoid validity checks
    """
    ij2k: jax.Array
    ij2n: jax.Array
    k2ij: jax.Array
    ijk2a: jax.Array | None = None  # cell, object -> intersection area

    n: jax.Array | int = 20
    size: jax.Array | float = 1.04
    center: jax.Array | float = 0.0

    @property
    def N(self):
        return self.ij2n.shape[0]

    @property
    def capacity(self):
        return self.ij2k.shape[2]

    def get_cell_centers(self, ij):
        return self.size / 2 + ij * self.size

    def compute_ij(self, points: jax.Array) -> jax.Array:
        is_nan = jnp.isnan(points).any(-1, keepdims=True)
        n_half = self.n // 2
        points = points - self.center
        ij = (jnp.clip(points // self.size, -n_half, n_half - 1) + n_half).astype(jnp.uint16)
        return (ij + 1) * (~is_nan)

    @staticmethod
    def empty(num, n=20, size=1.04, capacity=16, center=0.0, dtype=int):
        N = n + 2
        ij2n = jnp.zeros((N, N), dtype=dtype)
        ij2k = jnp.zeros((N, N, capacity), dtype=dtype) - 1
        k2ij = jnp.zeros((num, 2), dtype=dtype)
        return Grid2D(
            ij2k=ij2k,
            ij2n=ij2n,
            k2ij=k2ij,
            ijk2a=None,
            n=n,
            size=size,
            center=center
        )

    @staticmethod
    def init(
            centers: jax.Array,
            figures: jax.Array | None = None,
            *,
            n=16,
            capacity=8,
            size=1.04,
            center=0.0
    ) -> Grid2D:
        num = centers.shape[0]
        grid = Grid2D.empty(num, n=n, size=size, capacity=capacity, center=center)
        ij = grid.compute_ij(centers)

        ij2k = grid.ij2k
        ij2n = grid.ij2n
        k2ij = grid.k2ij

        def step(k, state):
            ij2k, ij2n, k2ij = state
            ij_k = ij[k]
            c = jnp.argmin(ij2k[*ij_k])
            ij2k = ij2k.at[*ij_k, c].set(k)
            ij2n = ij2n.at[*ij_k].add(1)
            k2ij = k2ij.at[k].set(ij_k)

            return ij2k, ij2n, k2ij

        ij2k, ij2n, k2ij = jax.lax.fori_loop(0, num, step, (ij2k, ij2n, k2ij))

        # init area intersection
        if figures is not None:
            aabb_grid = grid.comput_aabb()
            ijk2a = compute_figures_aabb_area(figures, aabb_grid)
        else:
            ijk2a = None

        return grid.replace(
            ij2k=ij2k,
            ij2n=ij2n,
            k2ij=k2ij,
            ijk2a=ijk2a
        )

    def update(
            self: Grid2D,
            k: int,
            center: jax.Array,
            figure: jax.Array | None = None,
    ) -> Grid2D:
        ij_k_new = self.compute_ij(center)
        ij_k_old = self.k2ij[k]
        c = jnp.argmin(self.ij2k[*ij_k_new])

        _upd = jnp.where(self.ij2k[*ij_k_old] == k, -1, self.ij2k[*ij_k_old])
        ij2k = self.ij2k.at[*ij_k_old].set(_upd).at[*ij_k_new, c].set(k)
        ij2n = self.ij2n.at[*ij_k_old].add(-1).at[*ij_k_new].add(1)
        k2ij = self.k2ij.at[k].set(ij_k_new)

        ijk2a = self.ijk2a
        if figure is not None:
            ijk2a = ijk2a.at[ij_k_old + _deltas, k].set(0)
            indexes_new = ij_k_new + _deltas
            aabb = jax.vmap(lambda idx: self.compute_cell_aabb(idx[0], idx[1]))(indexes_new)
            area_k = compute_figure_aabb_area(figure, aabb)
            ijk2a = ijk2a.at[indexes_new[..., 0], indexes_new[..., 1], k].set(area_k)

        return self.replace(
            ij2k=ij2k,
            ij2n=ij2n,
            k2ij=k2ij,
            ijk2a=ijk2a
        )

    def propose_candidates(self: Grid2D, k: int) -> jax.Array:
        ij = self.k2ij[k]
        candidates = jax.vmap(lambda delta: self.ij2k[*(ij + delta)])(_deltas)
        candidates = candidates.ravel()
        candidates = jnp.where(candidates == k, -1, candidates)
        candidates = jnp.sort(candidates, descending=True)
        return candidates

    def propose_candidates_by_pos(self, pos: jax.Array) -> jax.Array:
        ij = self.compute_ij(pos)
        candidates = jax.vmap(lambda delta: self.ij2k[*(ij + delta)])(_deltas)
        candidates = candidates.ravel()
        candidates = jnp.sort(candidates, descending=True)
        return candidates

    def propose_candidates_by_cell(self, i, j):
        ij = jnp.array([i, j])
        candidates = jax.vmap(lambda delta: self.ij2k[*(ij + delta)])(_deltas)
        candidates = candidates.ravel()
        candidates = jnp.sort(candidates, descending=True)
        return candidates

    def compute_cell_aabb(self, i, j) -> jax.Array:
        ij = jnp.array([i, j])
        xy_min = -self.size * (self.n // 2) + (ij - 1) * self.size
        xy_max = -self.size * (self.n // 2) + (ij) * self.size
        return jnp.concatenate((xy_min, xy_max), -1)

    def comput_aabb(self):
        i_ = jnp.arange(self.N)
        j_ = jnp.arange(self.N)
        fn = lambda i, j: self.compute_cell_aabb(i, j)
        fn = jax.vmap(fn, (0, None))
        fn = jax.vmap(fn, (None, 0))
        return fn(i_, j_)
