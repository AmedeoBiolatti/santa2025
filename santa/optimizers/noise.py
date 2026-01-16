import jax
import jax.numpy as jnp

from santa.core import Optimizer, Solution, SolutionEval, Any, GlobalState


class NoiseOptimizer(Optimizer):
    def __init__(self, noise_level=0.01):
        super(NoiseOptimizer, self).__init__()
        self.noise_level = noise_level

    def apply(
            self,
            solution: SolutionEval,
            state: Any,
            global_state: GlobalState,
            rng: jax.Array,
    ) -> tuple[Solution | SolutionEval, Any]:
        pos, ang = solution.params
        num = pos.shape[0]
        rng, rng_idx = jax.random.split(rng)
        rng_pos, rng_ang = jax.random.split(rng)

        idx = jax.random.randint(rng_idx, (), minval=0, maxval=num)

        pos_noise = self.noise_level * jax.random.uniform(rng_pos, pos[idx].shape, minval=-1, maxval=+1)
        ang_noise = self.noise_level * jax.random.uniform(rng_ang, ang[idx].shape, minval=-1, maxval=+1)

        new_pos = pos.at[idx].add(pos_noise)
        new_ang = ang.at[idx].add(ang_noise)

        new_solution = Solution((new_pos, new_ang))

        # evaluate new_solution
        new_solution = self.problem.eval_update(new_solution, solution, indexes=idx)

        return new_solution, state


class BisectionNoiseOptimizer(Optimizer):
    def __init__(self, noise_level: float = 0.01, n_steps: int = 5):
        super(BisectionNoiseOptimizer, self).__init__()
        self.noise_level = noise_level
        self.n_steps = n_steps

    def apply(
            self,
            solution: SolutionEval,
            state: Any,
            global_state: GlobalState,
            rng: jax.Array,
    ) -> tuple[Solution | SolutionEval, Any]:
        def _select_tree(mask: jax.Array, left, right):
            return jax.tree_util.tree_map(lambda a, b: jnp.where(mask, a, b), left, right)

        pos, ang = solution.params
        num = pos.shape[0]
        rng, rng_idx = jax.random.split(rng)
        rng_pos, rng_ang = jax.random.split(rng)

        idx = jax.random.randint(rng_idx, (), minval=0, maxval=num)

        pos_noise = self.noise_level * jax.random.uniform(rng_pos, pos[idx].shape, minval=-1, maxval=+1)
        ang_noise = self.noise_level * jax.random.uniform(rng_ang, ang[idx].shape, minval=-1, maxval=+1)

        def _evaluate(alpha: jax.Array) -> SolutionEval:
            new_pos = pos.at[idx].add(alpha * pos_noise)
            new_ang = ang.at[idx].add(alpha * ang_noise)
            new_solution = Solution((new_pos, new_ang))
            return self.problem.eval_update(new_solution, solution, indexes=idx)

        low_alpha = jnp.array(0.0)
        high_alpha = jnp.array(1.0)
        low_solution = solution
        low_score = global_state.score(low_solution)
        high_solution = _evaluate(high_alpha)
        high_score = global_state.score(high_solution)

        def body(_, carry):
            low_alpha, high_alpha, low_solution, high_solution, low_score, high_score = carry
            mid_alpha = (low_alpha + high_alpha) / 2.0
            mid_solution = _evaluate(mid_alpha)
            mid_score = global_state.score(mid_solution)

            compare_low = low_score <= high_score
            keep_left = (compare_low & (mid_score < low_score)) | (~compare_low & (mid_score >= high_score))

            next_low_alpha = jnp.where(keep_left, low_alpha, mid_alpha)
            next_high_alpha = jnp.where(keep_left, mid_alpha, high_alpha)

            next_low_solution = _select_tree(keep_left, low_solution, mid_solution)
            next_high_solution = _select_tree(keep_left, mid_solution, high_solution)

            next_low_score = jnp.where(keep_left, low_score, mid_score)
            next_high_score = jnp.where(keep_left, mid_score, high_score)

            return (
                next_low_alpha,
                next_high_alpha,
                next_low_solution,
                next_high_solution,
                next_low_score,
                next_high_score,
            )

        low_alpha, high_alpha, low_solution, high_solution, low_score, high_score = jax.lax.fori_loop(
            0,
            self.n_steps,
            body,
            (low_alpha, high_alpha, low_solution, high_solution, low_score, high_score),
        )

        best_solution = jax.lax.cond(
            low_score <= high_score,
            lambda _: low_solution,
            lambda _: high_solution,
            operand=None,
        )

        return best_solution, state
