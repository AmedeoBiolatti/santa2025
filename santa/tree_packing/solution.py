import jax
import jax.numpy as jnp
from flax.struct import dataclass

from santa.core import Solution


@dataclass
class TreePackingSolution(Solution):
    def reg(self):
        pos, _ = self.params
        return jnp.max(jnp.abs(pos))


def init_solution(
        num_trees: int,
        rng: jax.Array,
        length: float = 10.0
) -> TreePackingSolution:
    rng_pos, rng_ang = jax.random.split(rng)
    pos = jax.random.uniform(rng_pos, minval=-length, maxval=length, shape=(num_trees, 2))
    ang = jax.random.uniform(rng_ang, minval=0.0, maxval=2 * jnp.pi, shape=(num_trees,))
    params = pos, ang
    return TreePackingSolution(params=params)
