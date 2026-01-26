import jax
import jax.numpy as jnp

from santa.tree_packing import intersection, tree


def _triangles_intersection_scores(t0, t1, n0=None, n1=None, eps=1e-12):
    """
    Strict triangle intersection score:
    - positive when triangles overlap,
    - <= 0 when separated or only touching.
    """
    if n0 is None:
        n0 = intersection._edge_normals(t0)
    if n1 is None:
        n1 = intersection._edge_normals(t1)

    axes = jnp.concatenate([n0, n1], axis=0)  # (6, 2)

    proj0 = t0 @ axes.T
    proj1 = t1 @ axes.T

    min0 = proj0.min(axis=0)
    max0 = proj0.max(axis=0)
    min1 = proj1.min(axis=0)
    max1 = proj1.max(axis=0)

    score = (max0 - min1 - eps) * (max1 - min0 - eps)
    return score


def figure_intersection_scores(f0, f1):
    fn = _triangles_intersection_scores
    fn = jax.vmap(fn, (0, None))
    fn = jax.vmap(fn, (None, 0))
    return fn(f0, f1)


# ---- SCORING ----
def intersection_score(figure0, figure1):
    assert figure0.ndim == 3
    assert figure0.shape[1:] == (3, 2)
    assert figure1.ndim == 3
    assert figure1.shape[1:] == (3, 2)

    score = figure_intersection_scores(figure0, figure1)
    assert score.ndim == 3

    score = jnp.min(score, axis=-1)
    score = jax.nn.relu(score)
    score = jnp.sum(score)
    return score


def signed_intersection_score(figure0, figure1):
    assert figure0.ndim == 3
    assert figure0.shape[1:] == (3, 2)
    assert figure1.ndim == 3
    assert figure1.shape[1:] == (3, 2)

    score = figure_intersection_scores(figure0, figure1)
    assert score.ndim == 3

    score = jnp.min(score, axis=-1)
    score = jnp.max(score)
    return score


def distance_score(figure0, figure1):
    assert figure0.ndim == 3
    assert figure0.shape[1:] == (3, 2)
    assert figure1.ndim == 3
    assert figure1.shape[1:] == (3, 2)

    score = figure_intersection_scores(figure0, figure1)
    assert score.ndim == 3

    score = jax.nn.relu(-score)
    score = jnp.mean(score, axis=-1)
    score = jnp.log(1 + score)
    score = jnp.mean(score)
    return score


# ---- UTILS ----
def apply_scoring_fn(
        score_fn,
        params0: tuple[jax.Array, jax.Array],
        params: tuple[jax.Array, jax.Array],
        candidates: jax.Array | None = None,
        method="scan"
):
    figure = tree.params_to_tree(params0)

    if not isinstance(score_fn, list | tuple):
        score_fn = [score_fn]

    if candidates is None:
        candidates = jnp.arange(params[0].shape[0])

    def fn(i):
        params_i = jax.tree.map(lambda p: p[i], params)
        figure_i = tree.params_to_tree(params_i)

        score = sum([s(figure, figure_i) for s in score_fn])
        return score

    def maybe_fn(i):
        return jax.lax.cond(i >= 0, fn, lambda i: 0.0, i)

    if method == "scan":
        def body_fn(k, _):
            i = candidates[k]
            return k + 1, maybe_fn(i)

        _, scores = jax.lax.scan(body_fn, 0, length=candidates.shape[0])
    elif method == "while":
        def cond_fn(carry):
            _, k = carry
            i = candidates[k]
            return (i >= 0) & (k < candidates.shape[0])

        def body_fn(carry):
            scores, k = carry
            i = candidates[k]
            scores = scores.at[k].set(fn(i))
            carry = scores, k + 1
            return carry

        scores = jnp.zeros((len(candidates),))
        (scores, _) = jax.lax.while_loop(cond_fn, body_fn, (scores, 0))
    elif method == "vmap":
        scores = jax.vmap(maybe_fn)(candidates)
    else:
        raise ValueError(f"Unknown score method: {method}")
    return scores
