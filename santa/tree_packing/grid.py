from __future__ import annotations

import jax
import jax.numpy as jnp

from flax.struct import dataclass

_deltas = jnp.array([[0, -1], [-1, -1], [+1, -1], [0, 0], [-1, 0], [+1, 0], [0, +1], [-1, +1], [+1, +1]])


@dataclass
class Grid2D:
    """
    added padding to the grid to avoid validity checks
    """
    ij2k: jax.Array
    ij2n: jax.Array
    k2ij: jax.Array
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
            n=n,
            size=size,
            center=center
        )

    @staticmethod
    def init(centers: jax.Array, *, n=16, capacity=8, size=1.04, center=0.0) -> Grid2D:
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
        return grid.replace(
            ij2k=ij2k,
            ij2n=ij2n,
            k2ij=k2ij,
        )

    def update(self: Grid2D, centers: jax.Array, k: int) -> Grid2D:
        ij_k_new = self.compute_ij(centers[k])
        ij_k_old = self.k2ij[k]
        c = jnp.argmin(self.ij2k[*ij_k_new])

        _upd = jnp.where(self.ij2k[*ij_k_old] == k, -1, self.ij2k[*ij_k_old])
        ij2k = self.ij2k.at[*ij_k_old].set(_upd).at[*ij_k_new, c].set(k)
        ij2n = self.ij2n.at[*ij_k_old].add(-1).at[*ij_k_new].add(1)
        k2ij = self.k2ij.at[k].set(ij_k_new)

        return self.replace(
            ij2k=ij2k,
            ij2n=ij2n,
            k2ij=k2ij,
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
