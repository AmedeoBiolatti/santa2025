import jax
import jax.numpy as jnp

from santa.core import Optimizer, Solution, SolutionEval, Any, GlobalState


class RuinAndRecreate(Optimizer):
    def __init__(
            self,
            ruin: Optimizer,
            recreate: Optimizer,
            evaluate_after_ruin: bool = True,
    ):
        super(RuinAndRecreate, self).__init__()
        self.ruin = ruin
        self.recreate = recreate
        self.evaluate_after_ruin = evaluate_after_ruin

    def set_problem(self, problem):
        super().set_problem(problem)
        self.ruin.set_problem(problem)
        self.recreate.set_problem(problem)

    def init_state(self, solution: Solution | SolutionEval) -> Any:
        return {
            "iteration": jnp.array(0, dtype=jnp.int32),
            "ruin_state": self.ruin.init_state(solution),
            "recreate_state": self.recreate.init_state(solution),
        }

    def apply(
            self,
            solution: SolutionEval,
            state: Any,
            global_state: GlobalState,
            rng: jax.Array,
    ) -> tuple[Solution | SolutionEval, Any]:
        rng, ruin_rng, recreate_rng = jax.random.split(rng, 3)
        ruin_state = state["ruin_state"]
        recreate_state = state["recreate_state"]

        ruined_solution, new_ruin_state = self.ruin.apply(
            solution,
            ruin_state,
            global_state,
            ruin_rng,
        )
        if self.evaluate_after_ruin:
            ruined_solution = self.problem.eval(ruined_solution)

        recreated_solution, new_recreate_state = self.recreate.apply(
            ruined_solution,
            recreate_state,
            global_state,
            recreate_rng,
        )

        new_state = {
            "iteration": state["iteration"] + 1,
            "ruin_state": new_ruin_state,
            "recreate_state": new_recreate_state,
        }
        return recreated_solution, new_state
