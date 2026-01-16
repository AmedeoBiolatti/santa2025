import jax
import jax.numpy as jnp

from santa.core import Problem
from santa.tree_packing import tree, constraints
from santa.tree_packing.solution import TreePackingSolution


def objective_fn(solution):
    params = solution.params
    trees = tree.params_to_trees(params)
    num_trees = trees.shape[0]

    delta_x = trees[..., 0].max() - trees[..., 0].min()
    delta_y = trees[..., 1].max() - trees[..., 1].min()
    length = jnp.maximum(delta_x, delta_y)
    score = length * length / num_trees
    return score


def create_tree_packing_problem(
        *,
        intersection_version: int = 2
):
    side = 16 * tree.THR
    if intersection_version == 1:
        intersection_constraint = constraints.IntersectionConstraint()
    elif intersection_version == 2:
        intersection_constraint = constraints.IntersectionConstraintV2()
    else:
        raise ValueError(f"Invalid intersection version: {intersection_version}")

    problem = Problem(
        objective_fn=objective_fn,
        constraints={
            "intersection": intersection_constraint,
            "bounds": constraints.BoundConstraint(min_pos=-side / 2, max_pos=side / 2),
        },
        solution_cls=TreePackingSolution
    )
    return problem
