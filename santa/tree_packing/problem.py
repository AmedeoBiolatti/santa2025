import jax.numpy as jnp

from santa.core import Problem, Solution
from santa.tree_packing import tree, constraints


def objective_fn(solution):
    params = solution.params
    trees = tree.params_to_trees(params)
    num_trees = trees.shape[0]

    delta_x = trees[..., 0].max() - trees[..., 0].min()
    delta_y = trees[..., 1].max() - trees[..., 1].min()
    length = jnp.maximum(delta_x, delta_y)
    score = length * length / num_trees
    return score


def get_init_solution_fn(num_trees):
    def init_solution():
        params = jnp.zeros((num_trees, 2)), jnp.zeros((num_trees,))
        return Solution(params)

    return init_solution


def create_tree_packing_problem(
        num_trees: int,
        intersection_version: int = 2
):
    n = 20
    side = n * tree.THR
    if intersection_version == 1:
        intersection_constraint = constraints.IntersectionConstraint()
    elif intersection_version == 2:
        intersection_constraint = constraints.IntersectionConstraintV2(n=n, capacity=8)
    else:
        raise ValueError(f"Invalid intersection version: {intersection_version}")

    problem = Problem(
        objective_fn=objective_fn,
        constraints={
            "intersection": intersection_constraint,
            "bounds": constraints.BoundConstraint(min_pos=-side / 2, max_pos=side / 2),
        }
    )
    problem.init_solution = get_init_solution_fn(num_trees)
    return problem
