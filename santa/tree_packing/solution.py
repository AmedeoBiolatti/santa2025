import jax
import jax.numpy as jnp
from flax.struct import dataclass

from santa.core import Solution
from santa.tree_packing import tree, grid

ENABLE_AREA_COMPUTATION = False


@dataclass
class TreePackingSolution(Solution):
    def reg(self):
        pos, _ = self.params
        return jnp.max(jnp.abs(pos))

    @staticmethod
    def init_params(num_trees: int, method="random", rng=None, side=10.):
        if method == "random":
            rng_pos, rng_ang = jax.random.split(rng)
            pos = jax.random.uniform(rng_pos, shape=(num_trees, 2), minval=-side / 2, maxval=+side / 2)
            ang = jax.random.uniform(rng_ang, shape=(num_trees,), minval=-jnp.pi, maxval=+jnp.pi)
        elif method == "empty":
            pos = jnp.zeros((num_trees, 2)) + jnp.nan
            ang = jnp.zeros((num_trees,)) + jnp.nan
        else:
            raise ValueError("Unknown method: {}".format(method))
        return pos, ang

    @staticmethod
    def init(params, n=16, size=tree.THR, capacity=8):
        centers = tree.get_tree_centers(params)
        g = grid.Grid2D.init(
            centers,
            figures=tree.params_to_trees(params, disjoint=True) if ENABLE_AREA_COMPUTATION else None,
            n=n,
            size=size,
            capacity=capacity
        )
        return TreePackingSolution(params, aux_data={"grid": g})

    def update(self, params, indicies: jax.Array):
        g: grid.Grid2D = self.aux_data["grid"]
        centers = tree.get_tree_centers(params)
        figures = tree.params_to_trees(params, disjoint=True)
        for i in indicies:
            g = g.update(
                i,
                centers[i],
                figure=figures[i] if ENABLE_AREA_COMPUTATION else None
            )
        return TreePackingSolution(params, aux_data={"grid": g})

    @property
    def trees(self):
        return tree.params_to_trees(self.params)

    @property
    def centers(self):
        return tree.get_tree_centers(self.params)
