from __future__ import annotations

from typing import Callable, Any
from numbers import Integral

import jax
import jax.numpy as jnp
from flax.struct import dataclass


@dataclass
class ConstraintEval:
    violation: jax.Array
    aux_data: dict[str, Any] | None = None


class Constraint:
    def eval(self, solution: Solution) -> ConstraintEval:
        raise NotImplementedError()

    def eval_update(
            self,
            solution: Solution,
            prev_sol: Solution | SolutionEval,
            prev_eval: ConstraintEval,
            indexes
    ) -> ConstraintEval:
        class_name = self.__class__.__name__
        print("eval_update not implemented for class '%s', fallback :(" % class_name)
        return self.eval(solution)

    def eval_partial(self, solution: Solution, indexes) -> ConstraintEval:
        raise NotImplementedError()


@dataclass
class Solution:
    params: Any
    aux_data: dict[str, Any] | None = None

    def n_missing(self) -> int:
        n_nan = jax.tree.reduce(jax.numpy.add, jax.tree.map(jax.numpy.sum, jax.tree.map(jax.numpy.isnan, self.params)))
        return n_nan

    def reg(self) -> float:
        return 0.0

    @staticmethod
    def init(params, *args, **kwargs) -> Solution:
        return Solution(params)

    def update(self, params, indicies: jax.Array) -> Solution:
        return Solution(params)

    @staticmethod
    def init_params(*args, **kwargs):
        raise NotImplementedError


@dataclass
class SolutionEval:
    solution: Solution
    objective: jax.Array
    constraints: dict[str, ConstraintEval]

    @property
    def params(self):
        return self.solution.params

    @property
    def aux_data(self):
        return self.solution.aux_data

    def n_missing(self) -> int:
        return self.solution.n_missing()

    def reg(self):
        return self.solution.reg()

    def total_violation(self):
        return sum([v.violation for v in self.constraints.values()])


@dataclass
class GlobalState:
    rng: jax.Array
    best_solution: SolutionEval
    best_feasible_solution: SolutionEval
    t: jax.Array
    iters_since_last_improvement: jax.Array
    iters_since_last_feasible_improvement: jax.Array
    best_score: float = jax.numpy.inf
    best_feasible_score: float = jax.numpy.inf
    mu: float = 1e6
    tol: float = 1e-12

    def split_rng(self) -> tuple[GlobalState, jax.Array]:
        rng, rng_ = jax.random.split(self.rng, 2)
        return self.replace(rng=rng), rng_

    def next(self) -> GlobalState:
        return self.replace(
            t=self.t + 1,
            iters_since_last_improvement=self.iters_since_last_improvement + 1,
            iters_since_last_feasible_improvement=self.iters_since_last_feasible_improvement + 1,
        )

    def maybe_update_best(self, problem: Problem, solution: SolutionEval) -> GlobalState:
        is_feasible = solution.total_violation() < self.tol
        score = problem.score(solution, self)
        improved = score < self.best_score
        feasible_improved = is_feasible & (score < self.best_feasible_score)
        best_solution, best_score = jax.lax.cond(
            improved,
            lambda: (solution, score),
            lambda: (self.best_solution, self.best_score),
        )
        best_feasible_solution, best_feasible_score = jax.lax.cond(
            feasible_improved,
            lambda: (solution, score),
            lambda: (self.best_feasible_solution, self.best_feasible_score),
        )
        iters_since_last_improvement = self.iters_since_last_improvement * (~improved)
        iters_since_last_feasible_improvement = self.iters_since_last_feasible_improvement * (~feasible_improved)
        return self.replace(
            best_solution=best_solution,
            best_score=best_score,
            iters_since_last_improvement=iters_since_last_improvement,

            best_feasible_solution=best_feasible_solution,
            best_feasible_score=best_feasible_score,
            iters_since_last_feasible_improvement=iters_since_last_feasible_improvement,
        )


class Problem:
    objective_fn: Callable
    constraints: dict[str, Constraint]
    solution_cls: type[Solution] = Solution

    def __init__(self, objective_fn: Any, constraints: dict[str, Constraint], solution_cls=Solution) -> None:
        self.objective_fn = objective_fn
        self.constraints = constraints
        self.solution_cls = solution_cls

    def eval(self, solution: Solution) -> SolutionEval:
        if isinstance(solution, SolutionEval):
            return solution
        objective = self.objective_fn(solution)
        constraints = {k: c.eval(solution) for k, c in self.constraints.items()}
        return SolutionEval(
            solution=solution,
            objective=objective,
            constraints=constraints
        )

    def eval_update(self, solution: Solution, prev_solution: SolutionEval, indexes) -> SolutionEval:
        objective = self.objective_fn(solution)
        constraints = {
            k: c.eval_update(solution, prev_solution, prev_solution.constraints[k], indexes)
            for k, c in self.constraints.items()
        }
        return SolutionEval(
            solution=solution,
            objective=objective,
            constraints=constraints
        )

    def score(self, solution: SolutionEval, global_state: GlobalState) -> jax.Array:
        violation = solution.total_violation()
        n_missing = solution.n_missing()
        reg = solution.reg()
        return solution.objective + global_state.mu * violation + 1.0 * n_missing + 1e-6 * reg

    def init_solution(self, *args, **kwargs):
        params = self.solution_cls.init_params(*args, **kwargs)
        return self.solution_cls.init(params)

    def to_solution(self, params: Any) -> Solution:
        return self.solution_cls.init(params)

    def to_solution_update(self, params: Any, prev_solution: Solution | SolutionEval, indicies: jax.Array) -> Solution:
        if isinstance(prev_solution, SolutionEval):
            prev_solution = prev_solution.solution
        return prev_solution.update(params, indicies)

    def init_global_state(self, seed: int, *args, **kwargs) -> GlobalState:
        solution = self.eval(self.init_solution(*args, **kwargs))
        return GlobalState(
            t=jnp.zeros(0, dtype=jnp.uint32),
            rng=jax.random.PRNGKey(seed) if isinstance(seed, int) else seed,
            best_solution=solution,
            best_feasible_solution=solution,
            iters_since_last_improvement=jnp.zeros((), dtype=jnp.uint32),
            iters_since_last_feasible_improvement=jnp.zeros((), dtype=jnp.uint32),
        )


class Optimizer:
    def set_problem(self, problem: Problem):
        self.problem = problem

    def init_state(self, solution: Solution | SolutionEval) -> Any:
        return ()

    def apply(
            self,
            solution: SolutionEval,
            state: Any,
            global_state: GlobalState,
            rng: jax.Array
    ) -> tuple[Solution | SolutionEval, Any]:
        raise NotImplementedError()

    def step(
            self,
            solution: SolutionEval,
            state: Any,
            global_state: GlobalState
    ) -> tuple[SolutionEval, Any, GlobalState]:
        global_state, rng = global_state.split_rng()
        solution, state = self.apply(solution, state, global_state, rng)
        global_state = global_state.maybe_update_best(self.problem, solution)
        solution = self.problem.eval(solution)
        return solution, state, global_state

    def __and__(self, other):
        if not isinstance(other, Optimizer):
            return NotImplemented
        from santa.optimizers.combine import Chain
        if isinstance(self, Chain) and isinstance(other, Chain):
            return Chain(*self.optimizers, *other.optimizers)
        if isinstance(self, Chain):
            return Chain(*self.optimizers, other)
        if isinstance(other, Chain):
            return Chain(self, *other.optimizers)
        return Chain(self, other)

    def __rand__(self, other):
        if not isinstance(other, Optimizer):
            return NotImplemented
        return other.__and__(self)

    def __rshift__(self, other):
        return self.__and__(other)

    def __rrshift__(self, other):
        return self.__rand__(other)

    def _repeat(self, n):
        if not isinstance(n, Integral):
            return NotImplemented
        from santa.optimizers.combine import Repeat
        return Repeat(self, int(n))

    def __mul__(self, other):
        return self._repeat(other)

    def __rmul__(self, other):
        return self._repeat(other)

    def __or__(self, other):
        if not isinstance(other, Optimizer):
            return NotImplemented
        from santa.optimizers.combine import RandomChoice
        if isinstance(self, RandomChoice) and isinstance(other, RandomChoice):
            return RandomChoice([*self.optimizers, *other.optimizers])
        if isinstance(self, RandomChoice):
            return RandomChoice([*self.optimizers, other])
        if isinstance(other, RandomChoice):
            return RandomChoice([self, *other.optimizers])
        return RandomChoice([self, other])

    def __ror__(self, other):
        if not isinstance(other, Optimizer):
            return NotImplemented
        return other.__or__(self)
