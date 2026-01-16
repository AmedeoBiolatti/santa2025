import jax
import jax.numpy as jnp

from santa.core import Solution


def init_solution(
        num_trees: int,
        rng: jax.Array,
        length: float = 10.0
) -> Solution:
    rng_pos, rng_ang = jax.random.split(rng)
    pos = jax.random.uniform(rng_pos, minval=-length, maxval=length, shape=(num_trees, 2))
    ang = jax.random.uniform(rng_ang, minval=0.0, maxval=2 * jnp.pi, shape=(num_trees,))
    params = pos, ang
    return Solution(
        params=params,
    )
