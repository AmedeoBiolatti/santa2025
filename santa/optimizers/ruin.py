import jax
import jax.numpy as jnp

from santa.core import Optimizer, Solution, SolutionEval, Any, GlobalState


class RandomRuin(Optimizer):
    def __init__(self, n_remove: int = 1):
        super(RandomRuin, self).__init__()
        self.n_remove = n_remove

    def _select_indices(self, n_trees: int, rng: jax.Array) -> jax.Array:
        if self.n_remove <= 0:
            return jnp.zeros((0,), dtype=jnp.int32)
        perm = jax.random.permutation(rng, n_trees)
        return perm[: self.n_remove]

    def _set_positions_nan(self, pos: jax.Array, indices: jax.Array) -> jax.Array:
        def body(i, current_pos):
            idx = indices[i]
            return current_pos.at[idx].set(jnp.nan)

        return jax.lax.fori_loop(0, indices.shape[0], body, pos)

    def apply(
            self,
            solution: SolutionEval,
            state: Any,
            global_state: GlobalState,
            rng: jax.Array,
    ) -> tuple[Solution | SolutionEval, Any]:
        rng, idx_rng = jax.random.split(rng)
        pos, ang = solution.params

        indices = self._select_indices(pos.shape[0], idx_rng)
        new_pos = self._set_positions_nan(pos, indices)
        new_solution = self.problem.to_solution_update((new_pos, ang), solution, indices)
        new_solution = self.problem.eval_update(new_solution, solution, indexes=indices)

        return new_solution, state


class SpatialRuin(Optimizer):
    def __init__(self, n_remove: int = 1):
        super(SpatialRuin, self).__init__()
        self.n_remove = n_remove

    def _random_point(self, pos: jax.Array, rng: jax.Array) -> jax.Array:
        if pos.shape[0] == 0:
            return jnp.zeros((2,), dtype=pos.dtype)
        valid = jnp.isfinite(pos).all(axis=1)
        has_valid = jnp.any(valid)
        valid_pos_min = jnp.where(valid[:, None], pos, jnp.inf)
        valid_pos_max = jnp.where(valid[:, None], pos, -jnp.inf)
        min_coords = jnp.min(valid_pos_min, axis=0)
        max_coords = jnp.max(valid_pos_max, axis=0)
        min_coords = jnp.where(has_valid, min_coords, jnp.zeros((2,), dtype=pos.dtype))
        max_coords = jnp.where(has_valid, max_coords, jnp.zeros((2,), dtype=pos.dtype))
        return jax.random.uniform(rng, shape=(2,), minval=min_coords, maxval=max_coords)

    def _select_indices(self, pos: jax.Array, rng: jax.Array) -> jax.Array:
        if self.n_remove <= 0 or pos.shape[0] == 0:
            return jnp.zeros((0,), dtype=jnp.int32)
        point = self._random_point(pos, rng)
        diff = pos - point
        sq_distances = jnp.sum(diff ** 2, axis=1)
        sq_distances = jnp.where(jnp.isnan(sq_distances), jnp.inf, sq_distances)
        sorted_indices = jnp.argsort(sq_distances)
        return sorted_indices[: self.n_remove]

    def _set_positions_nan(self, pos: jax.Array, indices: jax.Array) -> jax.Array:
        def body(i, current_pos):
            idx = indices[i]
            return current_pos.at[idx].set(jnp.nan)

        return jax.lax.fori_loop(0, indices.shape[0], body, pos)

    def apply(
            self,
            solution: SolutionEval,
            state: Any,
            global_state: GlobalState,
            rng: jax.Array,
    ) -> tuple[Solution | SolutionEval, Any]:
        pos, ang = solution.params
        indices = self._select_indices(pos, rng)
        new_pos = self._set_positions_zero(pos, indices)
        new_solution = self.problem.to_solution_update((new_pos, ang), solution, indices)
        new_solution = self.problem.eval_update(new_solution, solution, indexes=indices)
        return new_solution, state
