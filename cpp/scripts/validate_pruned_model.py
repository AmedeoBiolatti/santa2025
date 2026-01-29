#!/usr/bin/env python3
"""
Validate that the pruned model produces the same results as the original.
"""

import numpy as np
from pathlib import Path


def load_model(path: str):
    """Load tree model from text file."""
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


def predict_leaf(features: np.ndarray, depth: int, W: np.ndarray, B: np.ndarray) -> int:
    """Predict leaf index for given features."""
    num_internal = (1 << depth) - 1
    node = 0

    while node < num_internal:
        row = W[node]
        bias = B[node]
        score = np.dot(row, features) + bias
        if score <= 0:
            node = 2 * node + 1
        else:
            node = 2 * node + 2

    return node - num_internal


def main():
    original_path = "data/intersection_tree_filter.txt"
    pruned_path = "data/intersection_tree_filter_pruned.txt"

    if not Path(original_path).exists() or not Path(pruned_path).exists():
        print("Error: Could not find model files")
        return

    print("Loading models...")
    orig_depth, orig_nf, orig_nt, orig_W, orig_B, orig_pred = load_model(original_path)
    prune_depth, prune_nf, prune_nt, prune_W, prune_B, prune_pred = load_model(pruned_path)

    print(f"Original: depth={orig_depth}, features={orig_nf}, targets={orig_nt}")
    print(f"Pruned:   depth={prune_depth}, features={prune_nf}, targets={prune_nt}")

    # Generate random test cases
    np.random.seed(42)
    num_tests = 10000

    print(f"\nRunning {num_tests} random tests...")
    mismatches = 0

    for i in range(num_tests):
        # Generate random features (similar to actual feature range)
        x = np.random.uniform(-2, 2)
        y = np.random.uniform(-2, 2)
        r = np.sqrt(x*x + y*y)
        ang = np.random.uniform(0, 2*np.pi)
        c = np.cos(ang)
        s = np.sin(ang)
        features = np.array([x, y, r, c, s, x*c, y*c, x*s, y*s, r*c], dtype=np.float32)

        # Predict with both models
        orig_leaf = predict_leaf(features, orig_depth, orig_W, orig_B)
        prune_leaf = predict_leaf(features, prune_depth, prune_W, prune_B)

        orig_targets = orig_pred[orig_leaf]
        prune_targets = prune_pred[prune_leaf]

        if not np.array_equal(orig_targets, prune_targets):
            mismatches += 1
            if mismatches <= 5:
                print(f"\nMismatch at test {i}:")
                print(f"  Features: {features}")
                print(f"  Original leaf {orig_leaf}: {orig_targets}")
                print(f"  Pruned leaf {prune_leaf}: {prune_targets}")

    print(f"\n{'='*60}")
    print(f"Results: {mismatches} mismatches out of {num_tests} tests")
    if mismatches == 0:
        print("SUCCESS: Pruned model produces identical results!")
    else:
        print(f"FAILURE: {100*mismatches/num_tests:.2f}% mismatch rate")


if __name__ == "__main__":
    main()
