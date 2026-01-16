import jax
import jax.numpy as jnp

from santa.core import Optimizer, Solution, SolutionEval, Any, GlobalState


class Chain(Optimizer):
    def __init__(self, *optimizers: Optimizer):
        super(Chain, self).__init__()
        self.optimizers = optimizers

    def set_problem(self, problem):
        super().set_problem(problem)
        for opt in self.optimizers:
            opt.set_problem(problem)

    def init_state(self, solution: Solution | SolutionEval) -> Any:
        return tuple(opt.init_state(solution) for opt in self.optimizers)

    def apply(
            self,
            solution: SolutionEval,
            state: Any,
            global_state: GlobalState,
            rng: jax.Array,
    ) -> tuple[Solution | SolutionEval, Any]:
        new_states = []
        current_solution = solution
        current_rng = rng
        for opt, opt_state in zip(self.optimizers, state):
            if current_rng is None:
                step_rng = None
            else:
                current_rng, step_rng = jax.random.split(current_rng)
            current_solution, new_state = opt.apply(
                current_solution,
                opt_state,
                global_state,
                step_rng,
            )
            new_states.append(new_state)
        return current_solution, tuple(new_states)


class Repeat(Optimizer):
    def __init__(self, optimizer: Optimizer, n: int):
        super(Repeat, self).__init__()
        if n < 0:
            raise ValueError("n must be >= 0")
        self.optimizer = optimizer
        self.n = int(n)

    def set_problem(self, problem):
        super().set_problem(problem)
        self.optimizer.set_problem(problem)

    def init_state(self, solution: Solution | SolutionEval) -> Any:
        return self.optimizer.init_state(solution)

    def apply(
            self,
            solution: SolutionEval,
            state: Any,
            global_state: GlobalState,
            rng: jax.Array,
    ) -> tuple[Solution | SolutionEval, Any]:
        if self.n == 0:
            return solution, state

        if rng is None:
            def body(_, carry):
                current_solution, current_state = carry
                new_solution, new_state = self.optimizer.apply(
                    current_solution,
                    current_state,
                    global_state,
                    None,
                )
                return new_solution, new_state

            return jax.lax.fori_loop(0, self.n, body, (solution, state))

        def body(_, carry):
            current_solution, current_state, current_rng = carry
            current_rng, step_rng = jax.random.split(current_rng)
            new_solution, new_state = self.optimizer.apply(
                current_solution,
                current_state,
                global_state,
                step_rng,
            )
            return new_solution, new_state, current_rng

        new_solution, new_state, _ = jax.lax.fori_loop(0, self.n, body, (solution, state, rng))
        return new_solution, new_state


class RestoreBest(Optimizer):
    def __init__(self, optimizer: Optimizer, patience: int):
        super(RestoreBest, self).__init__()
        if patience < 0:
            raise ValueError("patience must be >= 0")
        self.optimizer = optimizer
        self.patience = int(patience)

    def set_problem(self, problem):
        super().set_problem(problem)
        self.optimizer.set_problem(problem)

    def init_state(self, solution: Solution | SolutionEval) -> Any:
        return self.optimizer.init_state(solution)

    def apply(
            self,
            solution: SolutionEval,
            state: Any,
            global_state: GlobalState,
            rng: jax.Array,
    ) -> tuple[Solution | SolutionEval, Any]:
        should_restore = global_state.iters_since_last_improvement >= self.patience
        solution = jax.lax.cond(
            should_restore,
            lambda: global_state.best_solution,
            lambda: solution,
        )
        return self.optimizer.apply(solution, state, global_state, rng)


class RandomSelect(Optimizer):
    def __init__(
            self,
            optimizers: list[Optimizer],
            probabilities: list[float] | None = None,
    ):
        super(RandomSelect, self).__init__()
        self.optimizers = optimizers
        if probabilities is None:
            self.probabilities = None
        else:
            if len(probabilities) != len(optimizers):
                raise ValueError("probabilities must match optimizers length")
            if abs(sum(probabilities) - 1.0) > 1e-6:
                raise ValueError("probabilities must sum to 1.0")
            self.probabilities = jnp.array(probabilities)

    def set_problem(self, problem):
        super().set_problem(problem)
        for opt in self.optimizers:
            opt.set_problem(problem)

    def init_state(self, solution: Solution | SolutionEval) -> Any:
        return tuple(opt.init_state(solution) for opt in self.optimizers)

    def apply(
            self,
            solution: SolutionEval,
            state: Any,
            global_state: GlobalState,
            rng: jax.Array,
    ) -> tuple[Solution | SolutionEval, Any]:
        if rng is None:
            raise ValueError("RandomSelect requires rng")

        rng, select_rng, step_rng = jax.random.split(rng, 3)
        probs = self.probabilities
        if probs is None:
            probs = jnp.ones((len(self.optimizers),)) / len(self.optimizers)

        index = jax.random.choice(select_rng, len(self.optimizers), p=probs)

        def apply_opt(i):
            opt = self.optimizers[i]
            opt_state = state[i]
            new_solution, new_state = opt.apply(solution, opt_state, global_state, step_rng)
            new_states = list(state)
            new_states[i] = new_state
            return new_solution, tuple(new_states)

        branches = [lambda i=i: apply_opt(i) for i in range(len(self.optimizers))]
        return jax.lax.switch(index, branches)


class RandomChoice(RandomSelect):
    pass


class Parallel(Optimizer):
    def __init__(self, optimizers: list[Optimizer]):
        super(Parallel, self).__init__()
        self.optimizers = optimizers

    def set_problem(self, problem):
        super().set_problem(problem)
        for opt in self.optimizers:
            opt.set_problem(problem)

    def init_state(self, solution: Solution | SolutionEval) -> Any:
        return tuple(opt.init_state(solution) for opt in self.optimizers)

    def apply(
            self,
            solution: SolutionEval,
            state: Any,
            global_state: GlobalState,
            rng: jax.Array,
    ) -> tuple[Solution | SolutionEval, Any]:
        if rng is None:
            rngs = [None] * len(self.optimizers)
        else:
            rngs = list(jax.random.split(rng, len(self.optimizers)))

        candidates = []
        new_states = []
        for opt, opt_state, opt_rng in zip(self.optimizers, state, rngs):
            cand_solution, cand_state = opt.apply(solution, opt_state, global_state, opt_rng)
            cand_solution = self.problem.eval(cand_solution)
            candidates.append(cand_solution)
            new_states.append(cand_state)

        scores = jnp.stack([global_state.score(sol) for sol in candidates])
        best_idx = jnp.argmin(scores)
        branches = [lambda c=c: c for c in candidates]
        best_solution = jax.lax.switch(best_idx, branches)

        return best_solution, tuple(new_states)


class Vmap(Optimizer):
    def __init__(self, optimizer: Optimizer, n: int):
        super(Vmap, self).__init__()
        if n <= 0:
            raise ValueError("n must be >= 1")
        self.optimizer = optimizer
        self.n = int(n)

    def set_problem(self, problem):
        super().set_problem(problem)
        self.optimizer.set_problem(problem)

    def init_state(self, solution: Solution | SolutionEval) -> Any:
        return self.optimizer.init_state(solution)

    def apply(
            self,
            solution: SolutionEval,
            state: Any,
            global_state: GlobalState,
            rng: jax.Array,
    ) -> tuple[Solution | SolutionEval, Any]:
        if rng is None:
            raise ValueError("Vmap requires rng")
        rngs = jax.random.split(rng, self.n)

        def apply_once(step_rng):
            return self.optimizer.apply(solution, state, global_state, step_rng)

        candidates, candidate_states = jax.vmap(apply_once)(rngs)
        candidates = jax.vmap(self.problem.eval)(candidates)
        scores = jax.vmap(global_state.score)(candidates)
        best_idx = jnp.argmin(scores)
        new_solution, new_state = jax.tree.map(lambda t: t[best_idx], (candidates, candidate_states))
        return new_solution, new_state
