import jax
from santa.core import Optimizer, Solution, SolutionEval, Any, GlobalState


class GradientDescent(Optimizer):
    def __init__(self, learning_rate: float = 1e-2, n_steps: int = 1):
        super(GradientDescent, self).__init__()
        if n_steps <= 0:
            raise ValueError("n_steps must be >= 1")
        self.learning_rate = float(learning_rate)
        self.n_steps = int(n_steps)

    def apply(
            self,
            solution: SolutionEval,
            state: Any,
            global_state: GlobalState,
            rng: jax.Array,
    ) -> tuple[Solution | SolutionEval, Any]:
        def score_from_params(params):
            cand = self.problem.to_solution(params)
            cand = self.problem.eval(cand)
            return self.problem.score(cand, global_state)

        def step(_, params):
            _, grads = jax.value_and_grad(score_from_params)(params)
            return jax.tree.map(lambda p, g: p - self.learning_rate * g, params, grads)

        new_params = jax.lax.fori_loop(0, self.n_steps, step, solution.params)
        new_solution = self.problem.to_solution(new_params)
        new_solution = self.problem.eval(new_solution)
        return new_solution, state
