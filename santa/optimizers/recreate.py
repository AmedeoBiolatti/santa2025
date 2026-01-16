import jax
import jax.numpy as jnp

from santa.core import Optimizer, Solution, SolutionEval, Any, GlobalState


class RandomRecreate(Optimizer):
    def __init__(self, box_size: float = 5.0, max_recreate: int = 1):
        super(RandomRecreate, self).__init__()
        self.box_size = box_size
        self.max_recreate = max_recreate

    def init_state(self, solution: Solution | SolutionEval) -> Any:
        return {"iteration": jnp.array(0, dtype=jnp.int32)}

    def _removed_mask(self, pos: jax.Array) -> jax.Array:
        return jnp.all(pos == 0.0, axis=-1)

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

    def _build_target_mask(
            self,
            n_trees: int,
            indices: jax.Array,
            valid_mask: jax.Array,
    ) -> jax.Array:
        def body(i, current_mask):
            idx = indices[i]
            valid = valid_mask[i]
            return current_mask.at[idx].set(valid)

        init = jnp.zeros((n_trees,), dtype=jnp.bool_)
        return jax.lax.fori_loop(0, indices.shape[0], body, init)

    def apply(
            self,
            solution: SolutionEval,
            state: Any,
            global_state: GlobalState,
            rng: jax.Array,
    ) -> tuple[Solution | SolutionEval, Any]:
        pos, ang = solution.params
        removed_mask = self._removed_mask(pos)

        rng, idx_rng, pos_rng, ang_rng = jax.random.split(rng, 4)
        indices, valid_mask = self._select_indices(removed_mask, idx_rng)
        target_mask = self._build_target_mask(pos.shape[0], indices, valid_mask)

        random_pos = jax.random.uniform(
            pos_rng,
            pos.shape,
            minval=-self.box_size,
            maxval=self.box_size,
        )
        new_pos = jnp.where(target_mask[:, None], random_pos, pos)
        random_ang = jax.random.uniform(
            ang_rng,
            ang.shape,
            minval=-jnp.pi,
            maxval=jnp.pi,
        )
        new_ang = jnp.where(target_mask, random_ang, ang)

        new_solution = self.problem.to_solution_update((new_pos, new_ang), solution, indices)
        new_solution = self.problem.eval_update(new_solution, solution, indexes=indices)

        new_state = {"iteration": state["iteration"] + 1}
        return new_solution, new_state
