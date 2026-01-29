#!/usr/bin/env python3
"""
Analyze which branches of the tree go deep and why.
Find if we can restructure to reduce maximum depth.
"""

import numpy as np
from pathlib import Path
from collections import defaultdict


def load_model(path: str):
    with open(path, 'r') as f:
        lines = f.readlines()

    depth, num_features, num_targets = None, None, None
    values = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if depth is None:
            depth, num_features, num_targets = int(parts[0]), int(parts[1]), int(parts[2])
        else:
            values.extend(float(x) for x in parts)

    num_internal = (1 << depth) - 1
    num_leaves = 1 << depth
    w_size = num_internal * num_features
    b_size = num_internal

    W = np.array(values[:w_size]).reshape(num_internal, num_features)
    B = np.array(values[w_size:w_size + b_size])
    leaf_pred = np.array(values[w_size + b_size:], dtype=np.uint8).reshape(num_leaves, num_targets)

    return depth, num_features, num_targets, W, B, leaf_pred


def get_node_depth(node_idx: int) -> int:
    """Get depth of a node (root = 0)."""
    if node_idx == 0:
        return 0
    return 1 + get_node_depth((node_idx - 1) // 2)


def get_leaves_under(node_idx: int, num_internal: int) -> list:
    """Get all leaf indices under a node."""
    if node_idx >= num_internal:
        return [node_idx - num_internal]
    leaves = []
    stack = [node_idx]
    while stack:
        n = stack.pop()
        if n >= num_internal:
            leaves.append(n - num_internal)
        else:
            stack.append(2*n + 1)
            stack.append(2*n + 2)
    return leaves


def is_uniform(node_idx: int, num_internal: int, leaf_pred: np.ndarray) -> bool:
    """Check if all leaves under node have same prediction."""
    leaves = get_leaves_under(node_idx, num_internal)
    if len(leaves) <= 1:
        return True
    first = leaf_pred[leaves[0]]
    return all(np.array_equal(leaf_pred[l], first) for l in leaves[1:])


def find_effective_depth(node_idx: int, num_internal: int, leaf_pred: np.ndarray, current_depth: int = 0) -> int:
    """Find the effective depth needed for this subtree (after considering uniform subtrees)."""
    if node_idx >= num_internal:
        # Leaf
        return current_depth

    if is_uniform(node_idx, num_internal, leaf_pred):
        # This subtree is uniform - can be collapsed to current depth
        return current_depth

    # Need to check both children
    left = 2 * node_idx + 1
    right = 2 * node_idx + 2

    left_depth = find_effective_depth(left, num_internal, leaf_pred, current_depth + 1)
    right_depth = find_effective_depth(right, num_internal, leaf_pred, current_depth + 1)

    return max(left_depth, right_depth)


def find_deep_paths(node_idx: int, num_internal: int, leaf_pred: np.ndarray,
                    current_depth: int, min_depth: int, path: list) -> list:
    """Find all paths that go to at least min_depth."""
    if node_idx >= num_internal:
        # Leaf
        if current_depth >= min_depth:
            leaf_idx = node_idx - num_internal
            return [(path.copy(), leaf_pred[leaf_idx])]
        return []

    if is_uniform(node_idx, num_internal, leaf_pred):
        # Uniform subtree
        if current_depth >= min_depth:
            leaves = get_leaves_under(node_idx, num_internal)
            return [(path.copy(), leaf_pred[leaves[0]])]
        return []

    # Recurse
    results = []
    left = 2 * node_idx + 1
    right = 2 * node_idx + 2

    results.extend(find_deep_paths(left, num_internal, leaf_pred, current_depth + 1, min_depth, path + ['L']))
    results.extend(find_deep_paths(right, num_internal, leaf_pred, current_depth + 1, min_depth, path + ['R']))

    return results


def main():
    data_path = "data/intersection_tree_filter.txt"
    if not Path(data_path).exists():
        data_path = "../data/intersection_tree_filter.txt"

    print(f"Loading model from: {data_path}")
    depth, num_features, num_targets, W, B, leaf_pred = load_model(data_path)

    num_internal = (1 << depth) - 1

    print(f"\nOriginal depth: {depth}")
    print(f"Leaves: {1 << depth}")

    # Find effective depth
    eff_depth = find_effective_depth(0, num_internal, leaf_pred)
    print(f"\nEffective depth (after collapsing uniform subtrees): {eff_depth}")

    # Analyze depth distribution
    print(f"\n{'='*60}")
    print("DEPTH DISTRIBUTION OF NON-UNIFORM NODES")
    print(f"{'='*60}")

    depth_counts = defaultdict(int)
    for node_idx in range(num_internal):
        if not is_uniform(node_idx, num_internal, leaf_pred):
            d = get_node_depth(node_idx)
            depth_counts[d] += 1

    for d in sorted(depth_counts.keys()):
        print(f"  Depth {d}: {depth_counts[d]} non-uniform nodes")

    # Find paths that go deep
    print(f"\n{'='*60}")
    print("DEEP PATHS (depth >= 6)")
    print(f"{'='*60}")

    deep_paths = find_deep_paths(0, num_internal, leaf_pred, 0, 6, [])

    # Group by prediction pattern
    by_pattern = defaultdict(list)
    for path, pred in deep_paths:
        key = tuple(pred.tolist())
        by_pattern[key].append(''.join(path))

    print(f"\nFound {len(deep_paths)} deep paths leading to {len(by_pattern)} unique patterns:")

    for pattern, paths in sorted(by_pattern.items(), key=lambda x: -len(x[1])):
        active = [i for i, v in enumerate(pattern) if v == 1]
        print(f"\n  Pattern (targets {active}):")
        for p in paths[:5]:
            print(f"    {p}")
        if len(paths) > 5:
            print(f"    ... and {len(paths)-5} more")

    # Analyze if deep patterns could be moved
    print(f"\n{'='*60}")
    print("ANALYSIS: CAN WE REDUCE DEPTH?")
    print(f"{'='*60}")

    all_ones = tuple([1] * num_targets)
    all_zeros = tuple([0] * num_targets)

    # Count patterns that need depth > 5
    deep_unique = set()
    for path, pred in deep_paths:
        if len(path) > 5:  # Goes past depth 5
            deep_unique.add(tuple(pred.tolist()))

    print(f"\nPatterns requiring depth > 5: {len(deep_unique)}")
    for pattern in deep_unique:
        if pattern != all_ones:
            active = [i for i, v in enumerate(pattern) if v == 1]
            print(f"  targets {active}")

    # The challenge: these patterns exist at depth 6-8 because the splits
    # at earlier depths don't separate them well.

    print(f"\n{'='*60}")
    print("RECOMMENDATION")
    print(f"{'='*60}")

    if eff_depth <= 6:
        print(f"\nTree can be reduced to depth {eff_depth} with 100% accuracy.")
    else:
        non_trivial = len([p for p in deep_unique if p != all_ones])
        print(f"\nTree requires depth {eff_depth} due to {non_trivial} selective patterns at deep levels.")
        print(f"\nOptions to reduce depth:")
        print(f"  1. Accept ~{100*len(deep_paths)/(1<<depth):.1f}% prediction changes (merge deep patterns to 'all ones')")
        print(f"  2. Retrain tree with depth constraint")
        print(f"  3. Use variable-depth representation in C++ (saves memory but same speed)")


if __name__ == "__main__":
    main()