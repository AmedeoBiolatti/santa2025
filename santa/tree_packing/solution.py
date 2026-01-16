import jax
import jax.numpy as jnp
from flax.struct import dataclass

from santa.core import Solution
from santa.tree_packing import tree, grid


@dataclass
class TreePackingSolution(Solution):
    def reg(self):
        pos, _ = self.params
        return jnp.max(jnp.abs(pos))

    @staticmethod
    def init_params(num_trees: int, method="random", rng=None):
        if method == "random":
            rng_pos, rng_ang = jax.random.split(rng)
            pos = jax.random.uniform(rng_pos, shape=(num_trees, 2), minval=-5, maxval=+5)
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
        g = grid.Grid2D.init(centers, n=n, size=size, capacity=capacity)
        return TreePackingSolution(params, aux_data={"grid": g})

    def update(self, params, indicies: jax.Array):
        g: grid.Grid2D = self.aux_data["grid"]
        centers = tree.get_tree_centers(params)
        for i in indicies:  # TODO
            g = g.update(centers, i)
        return TreePackingSolution(params, aux_data={"grid": g})

    @property
    def trees(self):
        return tree.params_to_trees(self.params)

    @property
    def centers(self):
        return tree.get_tree_centers(self.params)
