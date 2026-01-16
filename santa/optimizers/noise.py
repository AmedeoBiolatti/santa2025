import jax

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
