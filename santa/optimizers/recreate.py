import jax
import jax.numpy as jnp

from santa.core import Optimizer, Solution, SolutionEval, Any, GlobalState
from santa.tree_packing import tree


class RandomRecreate(Optimizer):
    def __init__(self, box_size: float = 5.0, max_recreate: int = 1, delta: float = 0.35):
        super(RandomRecreate, self).__init__()
        self.box_size = box_size
        self.max_recreate = max_recreate
        self.delta = delta

    def init_state(self, solution: Solution | SolutionEval) -> Any:
        return {"iteration": jnp.array(0, dtype=jnp.int32)}

    def _removed_mask(self, pos: jax.Array) -> jax.Array:
        return jnp.all(jnp.isnan(pos), axis=-1)

    def _select_indices(
            self,
            removed_mask: jax.Array,
            rng: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        n_trees = removed_mask.shape[0]
        if self.max_recreate > n_trees:
            raise ValueError("max_recreate must be <= number of trees")

        noise = jax.random.uniform(rng, (n_trees,))
        key = jnp.where(removed_mask, 1.0, 0.0) + noise * 1e-3
        sorted_indices = jnp.argsort(-key)

        indices = sorted_indices[: self.max_recreate]
        valid_mask = removed_mask[indices]
        return indices, valid_mask

    def apply(
            self,
            solution: SolutionEval,
            state: Any,
            global_state: GlobalState,
            rng: jax.Array,
    ) -> tuple[Solution | SolutionEval, Any]:
        pos, ang = solution.params
        trees = solution.solution.trees
        removed_mask = self._removed_mask(pos)

        rng, idx_rng, pos_rng, ang_rng = jax.random.split(rng, 4)
        indices, valid_mask = self._select_indices(removed_mask, idx_rng)

        max_abs = jnp.nanmax(jnp.abs(trees))
        max_abs = jnp.where(jnp.isfinite(max_abs), max_abs, 0.0)
        minval = -max_abs - self.delta
        maxval = max_abs + self.delta

        random_pos = jax.random.uniform(
            pos_rng,
            (indices.shape[0], pos.shape[1]),
            minval=minval,
            maxval=maxval,
        )
        pos_at_indices = pos[indices]
        new_pos_values = jnp.where(valid_mask[:, None], random_pos, pos_at_indices)
        new_pos = pos.at[indices].set(new_pos_values)
        random_ang = jax.random.uniform(
            ang_rng,
            (indices.shape[0],),
            minval=-jnp.pi,
            maxval=jnp.pi,
        )
        ang_at_indices = ang[indices]
        new_ang_values = jnp.where(valid_mask, random_ang, ang_at_indices)
        new_ang = ang.at[indices].set(new_ang_values)

        new_solution = self.problem.to_solution_update((new_pos, new_ang), solution, indices)
        new_solution = self.problem.eval_update(new_solution, solution, indexes=indices)

        new_state = {"iteration": state["iteration"] + 1}
        return new_solution, new_state
