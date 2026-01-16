import jax
import jax.numpy as jnp

from santa.core import Optimizer, Solution, SolutionEval, Any, GlobalState


class AdaptiveLargeNeighborhoodSearch(Optimizer):
    def __init__(
            self,
            ruin_optimizers: list[Optimizer],
            recreate_optimizers: list[Optimizer],
            full_optimizers: list[Optimizer] | None = None,
            reaction_factor: float = 0.01,
            reward_improve: float = 1.0,
            reward_no_improve: float = 0.0,
            min_weight: float = 1e-3,
    ):
        super(AdaptiveLargeNeighborhoodSearch, self).__init__()
        self.ruin_optimizers = ruin_optimizers
        self.recreate_optimizers = recreate_optimizers
        self.full_optimizers = full_optimizers or []
        self.reaction_factor = reaction_factor
        self.reward_improve = reward_improve
        self.reward_no_improve = reward_no_improve
        self.min_weight = min_weight

    def set_problem(self, problem):
        super().set_problem(problem)
        for opt in self.ruin_optimizers:
            opt.set_problem(problem)
        for opt in self.recreate_optimizers:
            opt.set_problem(problem)
        for opt in self.full_optimizers:
            opt.set_problem(problem)

    def init_state(self, solution: Solution | SolutionEval) -> Any:
        ruin_states = tuple(opt.init_state(solution) for opt in self.ruin_optimizers)
        recreate_states = tuple(opt.init_state(solution) for opt in self.recreate_optimizers)
        state = {
            "iteration": jnp.array(0, dtype=jnp.int32),
            "ruin_states": ruin_states,
            "recreate_states": recreate_states,
            "ruin_weights": jnp.ones((len(self.ruin_optimizers),), dtype=jnp.float32),
            "recreate_weights": jnp.ones((len(self.recreate_optimizers),), dtype=jnp.float32),
        }
        if self.full_optimizers:
            state["full_states"] = tuple(opt.init_state(solution) for opt in self.full_optimizers)
            state["full_weights"] = jnp.ones((len(self.full_optimizers),), dtype=jnp.float32)
            state["mode_weights"] = jnp.ones((2,), dtype=jnp.float32)
        return state

    def _select_index(self, weights: jax.Array, rng: jax.Array) -> jax.Array:
        probs = weights / jnp.sum(weights)
        return jax.random.choice(rng, weights.shape[0], p=probs)

    def _update_weights(
            self,
            weights: jax.Array,
            index: jax.Array,
            reward: jax.Array,
    ) -> jax.Array:
        updated = (1.0 - self.reaction_factor) * weights
        updated = updated.at[index].add(self.reaction_factor * reward)
        return jnp.maximum(updated, self.min_weight)

    def apply(
            self,
            solution: SolutionEval,
            state: Any,
            global_state: GlobalState,
            rng: jax.Array,
    ) -> tuple[Solution | SolutionEval, Any]:
        if not self.full_optimizers:
            rng, ruin_rng, recreate_rng = jax.random.split(rng, 3)
            ruin_idx = self._select_index(state["ruin_weights"], ruin_rng)
            recreate_idx = self._select_index(state["recreate_weights"], recreate_rng)

            def apply_ruin(i):
                opt = self.ruin_optimizers[i]
                opt_state = state["ruin_states"][i]
                new_solution, new_state = opt.apply(solution, opt_state, global_state, rng)
                new_states = list(state["ruin_states"])
                new_states[i] = new_state
                return new_solution, tuple(new_states)

            ruined_solution, new_ruin_states = jax.lax.switch(
                ruin_idx,
                [lambda i=i: apply_ruin(i) for i in range(len(self.ruin_optimizers))],
            )

            def apply_recreate(i):
                opt = self.recreate_optimizers[i]
                opt_state = state["recreate_states"][i]
                new_solution, new_state = opt.apply(ruined_solution, opt_state, global_state, rng)
                new_states = list(state["recreate_states"])
                new_states[i] = new_state
                return new_solution, tuple(new_states)

            recreated_solution, new_recreate_states = jax.lax.switch(
                recreate_idx,
                [lambda i=i: apply_recreate(i) for i in range(len(self.recreate_optimizers))],
            )

            recreated_solution = self.problem.eval(recreated_solution)
            current_score = self.problem.score(solution, global_state)
            new_score = self.problem.score(recreated_solution, global_state)
            improved = new_score < current_score
            reward = jax.lax.cond(
                improved,
                lambda: jnp.array(self.reward_improve, dtype=jnp.float32),
                lambda: jnp.array(self.reward_no_improve, dtype=jnp.float32),
            )

            new_ruin_weights = self._update_weights(state["ruin_weights"], ruin_idx, reward)
            new_recreate_weights = self._update_weights(state["recreate_weights"], recreate_idx, reward)

            new_state = {
                "iteration": state["iteration"] + 1,
                "ruin_states": new_ruin_states,
                "recreate_states": new_recreate_states,
                "ruin_weights": new_ruin_weights,
                "recreate_weights": new_recreate_weights,
            }
            return recreated_solution, new_state

        rng, mode_rng, ruin_rng, recreate_rng, full_rng, step_rng = jax.random.split(rng, 6)
        mode_idx = self._select_index(state["mode_weights"], mode_rng)

        def apply_ruin_recreate(_):
            ruin_idx = self._select_index(state["ruin_weights"], ruin_rng)
            recreate_idx = self._select_index(state["recreate_weights"], recreate_rng)

            def apply_ruin(i):
                opt = self.ruin_optimizers[i]
                opt_state = state["ruin_states"][i]
                new_solution, new_state = opt.apply(solution, opt_state, global_state, rng)
                new_states = list(state["ruin_states"])
                new_states[i] = new_state
                return new_solution, tuple(new_states)

            ruined_solution, new_ruin_states = jax.lax.switch(
                ruin_idx,
                [lambda i=i: apply_ruin(i) for i in range(len(self.ruin_optimizers))],
            )

            def apply_recreate(i):
                opt = self.recreate_optimizers[i]
                opt_state = state["recreate_states"][i]
                new_solution, new_state = opt.apply(ruined_solution, opt_state, global_state, rng)
                new_states = list(state["recreate_states"])
                new_states[i] = new_state
                return new_solution, tuple(new_states)

            recreated_solution, new_recreate_states = jax.lax.switch(
                recreate_idx,
                [lambda i=i: apply_recreate(i) for i in range(len(self.recreate_optimizers))],
            )
            return (
                recreated_solution,
                new_ruin_states,
                new_recreate_states,
                state["full_states"],
                ruin_idx,
                recreate_idx,
                jnp.array(-1, dtype=jnp.int32),
            )

        def apply_full(_):
            full_idx = self._select_index(state["full_weights"], full_rng)

            def apply_full_opt(i):
                opt = self.full_optimizers[i]
                opt_state = state["full_states"][i]
                new_solution, new_state = opt.apply(solution, opt_state, global_state, step_rng)
                new_states = list(state["full_states"])
                new_states[i] = new_state
                return new_solution, tuple(new_states)

            full_solution, new_full_states = jax.lax.switch(
                full_idx,
                [lambda i=i: apply_full_opt(i) for i in range(len(self.full_optimizers))],
            )
            return (
                full_solution,
                state["ruin_states"],
                state["recreate_states"],
                new_full_states,
                jnp.array(-1, dtype=jnp.int32),
                jnp.array(-1, dtype=jnp.int32),
                full_idx,
            )

        (
            new_solution,
            new_ruin_states,
            new_recreate_states,
            new_full_states,
            ruin_idx,
            recreate_idx,
            full_idx,
        ) = jax.lax.switch(mode_idx, [apply_ruin_recreate, apply_full], None)

        new_solution = self.problem.eval(new_solution)
        current_score = self.problem.score(solution, global_state)
        new_score = self.problem.score(new_solution, global_state)
        improved = new_score < current_score
        reward = jax.lax.cond(
            improved,
            lambda: jnp.array(self.reward_improve, dtype=jnp.float32),
            lambda: jnp.array(self.reward_no_improve, dtype=jnp.float32),
        )

        new_mode_weights = self._update_weights(state["mode_weights"], mode_idx, reward)

        def update_ruin_recreate_weights():
            new_ruin_weights = self._update_weights(state["ruin_weights"], ruin_idx, reward)
            new_recreate_weights = self._update_weights(state["recreate_weights"], recreate_idx, reward)
            return new_ruin_weights, new_recreate_weights, state["full_weights"]

        def update_full_weights():
            new_full_weights = self._update_weights(state["full_weights"], full_idx, reward)
            return state["ruin_weights"], state["recreate_weights"], new_full_weights

        new_ruin_weights, new_recreate_weights, new_full_weights = jax.lax.cond(
            mode_idx == 0,
            update_ruin_recreate_weights,
            update_full_weights,
        )

        new_state = {
            "iteration": state["iteration"] + 1,
            "ruin_states": new_ruin_states,
            "recreate_states": new_recreate_states,
            "full_states": new_full_states,
            "ruin_weights": new_ruin_weights,
            "recreate_weights": new_recreate_weights,
            "full_weights": new_full_weights,
            "mode_weights": new_mode_weights,
        }
        return new_solution, new_state
