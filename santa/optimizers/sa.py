import jax
import jax.numpy as jnp

from santa.core import Optimizer, Solution, SolutionEval, Any, GlobalState


class SimulatedAnnealing(Optimizer):
    def __init__(
            self,
            optimizer: Optimizer,
            initial_temp: float = 1.0,
            min_temp: float = 1e-6,
            cooling_schedule: str = "exponential",
            cooling_rate: float = 0.995,
            patience: int | None = None,
            verbose: bool = False,
    ):
        super(SimulatedAnnealing, self).__init__()
        self.optimizer = optimizer
        self.initial_temp = initial_temp
        self.min_temp = min_temp
        self.cooling_schedule = cooling_schedule
        self.cooling_rate = cooling_rate
        self.patience = patience
        self.verbose = verbose

    def set_problem(self, problem):
        super().set_problem(problem)
        self.optimizer.set_problem(problem)

    def init_state(self, solution: Solution | SolutionEval) -> Any:
        inner_state = self.optimizer.init_state(solution)
        return {
            "iteration": jnp.array(0, dtype=jnp.int32),
            "counter": jnp.array(0, dtype=jnp.int32),
            "temperature": jnp.array(self.initial_temp),
            "inner_state": inner_state,
            "n_accepted": jnp.array(0, dtype=jnp.int32),
            "n_rejected": jnp.array(0, dtype=jnp.int32),
        }

    def _compute_temperature(self, state: dict, global_state: GlobalState) -> tuple[jax.Array, jax.Array, jax.Array]:
        iteration = state["iteration"]
        counter = state["counter"]

        if self.patience is not None:
            counter = counter * (global_state.iters_since_last_improvement > 1)
            iteration = jnp.where(counter > self.patience, 0, iteration)
            counter = jnp.where(counter > self.patience, 0, counter)

        if self.cooling_schedule == "exponential":
            temp = self.initial_temp * jnp.power(self.cooling_rate, iteration)
        elif self.cooling_schedule == "linear":
            temp = self.initial_temp - self.cooling_rate * iteration
        elif self.cooling_schedule == "logarithmic":
            temp = self.initial_temp / jnp.log(iteration + jnp.e)
        else:
            raise ValueError(f"Unknown cooling_schedule: {self.cooling_schedule}")
        return jnp.maximum(temp, self.min_temp), iteration, counter

    def apply(
            self,
            solution: SolutionEval,
            state: Any,
            global_state: GlobalState,
            rng: jax.Array,
    ) -> tuple[Solution | SolutionEval, Any]:
        if self.patience is not None:
            state["temperature"] = jnp.where(
                global_state.iters_since_last_improvement > self.patience,
                self.initial_temp,
                state["temperature"]
            )

        rng, accept_rng = jax.random.split(rng)
        inner_state = state["inner_state"]

        candidate_solution, candidate_state = self.optimizer.apply(
            solution,
            inner_state,
            global_state,
            rng,
        )
        candidate_solution = self.problem.eval(candidate_solution)

        current_score = self.problem.score(solution, global_state)
        candidate_score = self.problem.score(candidate_solution, global_state)
        delta = candidate_score - current_score

        temperature, iteration, counter = self._compute_temperature(state, global_state)
        accept_prob = jnp.minimum(jnp.exp(-delta / temperature), 1.0)
        accept = jax.random.uniform(accept_rng) < accept_prob

        new_solution, new_inner_state = jax.lax.cond(
            accept,
            lambda: (candidate_solution, candidate_state),
            lambda: (solution, inner_state),
        )

        new_state = {
            "iteration": iteration + 1,
            "counter": counter + 1,
            "temperature": temperature,
            "inner_state": new_inner_state,
            "n_accepted": state["n_accepted"] + accept.astype(jnp.int32),
            "n_rejected": state["n_rejected"] + (~accept).astype(jnp.int32),
        }

        if self.verbose:
            total = new_state["n_accepted"] + new_state["n_rejected"]
            rate = jax.lax.cond(
                total > 0,
                lambda: new_state["n_accepted"] / total,
                lambda: jnp.array(0.0, dtype=jnp.float32),
            )
            jax.debug.print("    SA: temp={temp:.2f} acc-rate={rate:.4f}", temp=temperature, rate=rate)

        return new_solution, new_state
