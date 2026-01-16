import jax
import jax.numpy as jnp

from santa.core import Constraint, ConstraintEval, Solution
from santa.tree_packing.grid import Grid2D
from santa.tree_packing.intersection import figure_intersection_score
from santa.tree_packing import tree

_figure_intersection_score_matrix = figure_intersection_score
_figure_intersection_score_matrix = jax.vmap(_figure_intersection_score_matrix, (0, None))
_figure_intersection_score_matrix = jax.vmap(_figure_intersection_score_matrix, (None, 0))


class IntersectionConstraint(Constraint):
    def eval(self, solution: Solution) -> ConstraintEval:
        trees = tree.params_to_trees(solution.params)
        matrix = _figure_intersection_score_matrix(trees, trees)
        matrix = jnp.fill_diagonal(matrix, 0, inplace=False)
        matrix = jnp.maximum(matrix, matrix.T)

        return ConstraintEval(
            violation=matrix.sum(),
            aux_data={"matrix": matrix}
        )

    def eval_update(self, solution: Solution, prev_eval: ConstraintEval, indexes) -> ConstraintEval:
        if indexes.ndim == 0:
            indexes = jax.numpy.expand_dims(indexes, 0)

        matrix = prev_eval.aux_data["matrix"]
        trees = tree.params_to_trees(solution.params)

        trees_modified = trees[indexes]
        update = _figure_intersection_score_matrix(trees_modified, trees)
        matrix = matrix.at[indexes, :].set(update.T)
        matrix = matrix.at[:, indexes].set(update)
        matrix = matrix.at[indexes, indexes].set(0.0)

        return ConstraintEval(
            violation=matrix.sum(),
            aux_data={"matrix": matrix}
        )


class IntersectionConstraintV2(Constraint):
    def __init__(self, n=20, capacity=16):
        self.n = n
        self.capacity = capacity

    def eval(self, solution: Solution) -> ConstraintEval:
        from santa.tree_packing.grid import Grid2D

        trees = tree.params_to_trees(solution.params)
        matrix = _figure_intersection_score_matrix(trees, trees)
        matrix = jnp.fill_diagonal(matrix, 0, inplace=False)
        matrix = jnp.maximum(matrix, matrix.T)
        matrix = jnp.pad(matrix, ((0, 1), (0, 1)), constant_values=0)

        centers = tree.get_tree_centers(solution.params)

        return ConstraintEval(
            violation=matrix.sum(),
            aux_data={
                "matrix": matrix,
                "grid": Grid2D.init(centers, size=tree.THR, n=self.n, capacity=self.capacity)
            }
        )

    def eval_update(self, solution: Solution, prev_eval: ConstraintEval, indexes) -> ConstraintEval:
        if indexes.ndim == 0:
            indexes = jax.numpy.expand_dims(indexes, 0)

        matrix = prev_eval.aux_data["matrix"]
        grid: Grid2D = prev_eval.aux_data["grid"]

        # set to zero
        def update(matrix, i):
            def cond(carry):
                matrix, j = carry
                return (j < candidates.shape[0]) & (candidates[j] >= 0)

            def body(carry):
                matrix, j = carry
                c = candidates[j]
                matrix = matrix.at[[i, c], [c, i]].set(0.0)
                return matrix, j + 1

            candidates = grid.propose_candidates(i)
            matrix, j = jax.lax.while_loop(cond, body, (matrix, 0))
            return matrix, None

        matrix, _ = jax.lax.scan(update, matrix, indexes)

        # update structures
        trees = tree.params_to_trees(solution.params)
        centers = tree.get_tree_centers(solution.params)
        for i in indexes:
            grid = grid.update(centers, i)

        # compute distance for candidates
        def update(matrix, i):
            def compute(i, c):
                ti = trees[i]
                tc = trees[c]
                ci = centers[i]
                cc = centers[c]

                diff = ci - cc
                dist2 = jnp.sum(diff * diff)
                near = dist2 < tree.THR2

                value = jax.lax.cond(
                    near,
                    lambda: figure_intersection_score(ti, tc),
                    lambda: 0.0
                )
                return value

            def cond(carry):
                matrix, j = carry
                return (j < candidates.shape[0]) & (candidates[j] >= 0)

            def body(carry):
                matrix, j = carry
                c = candidates[j]
                v = compute(i, c)
                matrix = matrix.at[[i, c], [c, i]].set(v)
                return matrix, j + 1

            candidates = grid.propose_candidates(i)
            matrix, j = jax.lax.while_loop(cond, body, (matrix, 0))
            return matrix, None

        matrix, _ = jax.lax.scan(update, matrix, indexes)

        return ConstraintEval(
            violation=matrix.sum(),
            aux_data={
                "matrix": matrix,
                "grid": grid
            }
        )


class BoundConstraint(Constraint):
    def __init__(self, min_pos=-10.0, max_pos=10.0):
        self.min_pos = min_pos
        self.max_pos = max_pos

    def eval(self, solution: Solution) -> ConstraintEval:
        pos, ang = solution.params

        r = jax.nn.relu(pos - self.max_pos).sum()
        l = jax.nn.relu(self.min_pos - pos).sum()

        return ConstraintEval(
            violation=r + l,
        )

    def eval_update(self, solution: Solution, prev_eval: ConstraintEval, indexes) -> ConstraintEval:
        return self.eval(solution)
