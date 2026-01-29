#!/usr/bin/env python3
"""
Restructure the intersection tree filter to reduce depth.

Strategy:
1. Find all unique leaf prediction patterns that need to be preserved
2. Build a new decision tree that separates these patterns with minimal depth
3. Use a greedy approach: at each node, find the best split that separates patterns
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple
import random


@dataclass
class TreeNode:
    """Node in the restructured tree."""
    is_leaf: bool = False
    # For leaves:
    prediction: Optional[np.ndarray] = None
    # For internal nodes:
    weights: Optional[np.ndarray] = None
    bias: Optional[float] = None
    left: Optional['TreeNode'] = None
    right: Optional['TreeNode'] = None


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


def generate_samples(num_samples: int = 100000, seed: int = 42) -> np.ndarray:
    """Generate random feature samples similar to actual tree pair features."""
    np.random.seed(seed)

    samples = []
    for _ in range(num_samples):
        # Random positions and angles
        x = np.random.uniform(-2, 2)
        y = np.random.uniform(-2, 2)
        r = np.sqrt(x*x + y*y)
        ang = np.random.uniform(0, 2*np.pi)
        c = np.cos(ang)
        s = np.sin(ang)

        # Feature vector
        feat = [x, y, r, c, s, x*c, y*c, x*s, y*s, r*c]
        samples.append(feat)

    return np.array(samples, dtype=np.float32)


def predict_original(features: np.ndarray, depth: int, W: np.ndarray, B: np.ndarray,
                     leaf_pred: np.ndarray) -> Tuple[int, np.ndarray]:
    """Predict with original model, returns (leaf_idx, prediction)."""
    num_internal = (1 << depth) - 1
    node = 0

    while node < num_internal:
        score = np.dot(W[node], features) + B[node]
        node = 2 * node + 1 if score <= 0 else 2 * node + 2

    leaf_idx = node - num_internal
    return leaf_idx, leaf_pred[leaf_idx]


def collect_samples_per_pattern(samples: np.ndarray, depth: int, W: np.ndarray,
                                 B: np.ndarray, leaf_pred: np.ndarray) -> dict:
    """Group samples by their prediction pattern."""
    pattern_samples = {}

    for features in samples:
        _, pred = predict_original(features, depth, W, B, leaf_pred)
        key = tuple(pred.tolist())

        if key not in pattern_samples:
            pattern_samples[key] = []
        pattern_samples[key].append(features)

    return {k: np.array(v) for k, v in pattern_samples.items()}


def find_best_split(samples_by_pattern: dict, num_features: int) -> Tuple[np.ndarray, float, float]:
    """
    Find the best hyperplane split that separates patterns.
    Returns (weights, bias, score).
    """
    all_samples = []
    all_labels = []

    patterns = list(samples_by_pattern.keys())
    if len(patterns) <= 1:
        return None, None, 0.0

    for pattern, samples in samples_by_pattern.items():
        label = patterns.index(pattern)
        for s in samples:
            all_samples.append(s)
            all_labels.append(label)

    all_samples = np.array(all_samples)
    all_labels = np.array(all_labels)

    best_weights = None
    best_bias = None
    best_score = -1

    # Try random hyperplanes
    for _ in range(500):
        # Random weights
        weights = np.random.randn(num_features).astype(np.float32)
        weights = weights / (np.linalg.norm(weights) + 1e-8)

        # Project samples
        projections = all_samples @ weights

        # Try multiple bias values
        sorted_proj = np.sort(np.unique(projections))
        if len(sorted_proj) < 2:
            continue

        # Try splitting at percentiles
        for percentile in [25, 50, 75]:
            bias = -np.percentile(projections, percentile)

            # Evaluate split
            left_mask = projections + bias <= 0
            right_mask = ~left_mask

            if not np.any(left_mask) or not np.any(right_mask):
                continue

            # Score: how well does this split separate patterns?
            left_patterns = set(all_labels[left_mask])
            right_patterns = set(all_labels[right_mask])

            # Prefer splits that put each pattern entirely on one side
            left_pure = sum(1 for p in left_patterns if not np.any(all_labels[right_mask] == p))
            right_pure = sum(1 for p in right_patterns if not np.any(all_labels[left_mask] == p))

            score = (left_pure + right_pure) / len(patterns)

            if score > best_score:
                best_score = score
                best_weights = weights.copy()
                best_bias = bias

    # Also try axis-aligned splits (often work well)
    for feat_idx in range(num_features):
        weights = np.zeros(num_features, dtype=np.float32)
        weights[feat_idx] = 1.0

        projections = all_samples @ weights
        sorted_proj = np.sort(np.unique(projections))

        if len(sorted_proj) < 2:
            continue

        for percentile in [25, 50, 75]:
            bias = -np.percentile(projections, percentile)

            left_mask = projections + bias <= 0
            right_mask = ~left_mask

            if not np.any(left_mask) or not np.any(right_mask):
                continue

            left_patterns = set(all_labels[left_mask])
            right_patterns = set(all_labels[right_mask])

            left_pure = sum(1 for p in left_patterns if not np.any(all_labels[right_mask] == p))
            right_pure = sum(1 for p in right_patterns if not np.any(all_labels[left_mask] == p))

            score = (left_pure + right_pure) / len(patterns)

            if score > best_score:
                best_score = score
                best_weights = weights.copy()
                best_bias = bias

    return best_weights, best_bias, best_score


def split_samples(samples_by_pattern: dict, weights: np.ndarray, bias: float) -> Tuple[dict, dict]:
    """Split samples into left and right based on hyperplane."""
    left = {}
    right = {}

    for pattern, samples in samples_by_pattern.items():
        projections = samples @ weights + bias
        left_mask = projections <= 0
        right_mask = ~left_mask

        if np.any(left_mask):
            left[pattern] = samples[left_mask]
        if np.any(right_mask):
            right[pattern] = samples[right_mask]

    return left, right


def build_tree_recursive(samples_by_pattern: dict, num_features: int,
                         max_depth: int, current_depth: int = 0) -> TreeNode:
    """Recursively build a decision tree."""
    patterns = list(samples_by_pattern.keys())

    # Base case: only one pattern or max depth reached
    if len(patterns) == 1 or current_depth >= max_depth:
        # Return most common pattern (or any if all same)
        if len(patterns) == 1:
            pred = np.array(patterns[0], dtype=np.uint8)
        else:
            # Take the pattern with most samples
            best_pattern = max(patterns, key=lambda p: len(samples_by_pattern[p]))
            pred = np.array(best_pattern, dtype=np.uint8)
        return TreeNode(is_leaf=True, prediction=pred)

    # Find best split
    weights, bias, score = find_best_split(samples_by_pattern, num_features)

    if weights is None or score == 0:
        # Can't find a good split, make a leaf
        best_pattern = max(patterns, key=lambda p: len(samples_by_pattern[p]))
        pred = np.array(best_pattern, dtype=np.uint8)
        return TreeNode(is_leaf=True, prediction=pred)

    # Split and recurse
    left_samples, right_samples = split_samples(samples_by_pattern, weights, bias)

    # Handle edge cases
    if not left_samples:
        left_samples = {patterns[0]: samples_by_pattern[patterns[0]][:1]}
    if not right_samples:
        right_samples = {patterns[0]: samples_by_pattern[patterns[0]][:1]}

    left_node = build_tree_recursive(left_samples, num_features, max_depth, current_depth + 1)
    right_node = build_tree_recursive(right_samples, num_features, max_depth, current_depth + 1)

    return TreeNode(
        is_leaf=False,
        weights=weights,
        bias=bias,
        left=left_node,
        right=right_node
    )


def tree_depth(node: TreeNode) -> int:
    """Compute depth of tree."""
    if node.is_leaf:
        return 0
    return 1 + max(tree_depth(node.left), tree_depth(node.right))


def count_nodes(node: TreeNode) -> Tuple[int, int]:
    """Count (internal, leaves)."""
    if node.is_leaf:
        return 0, 1
    li, ll = count_nodes(node.left)
    ri, rl = count_nodes(node.right)
    return 1 + li + ri, ll + rl


def flatten_to_complete_tree(root: TreeNode, target_depth: int, num_features: int, num_targets: int):
    """Flatten variable-depth tree to complete binary tree arrays."""
    num_internal = (1 << target_depth) - 1
    num_leaves = 1 << target_depth

    W = np.zeros((num_internal, num_features), dtype=np.float32)
    B = np.full(num_internal, -1.0, dtype=np.float32)  # Default: always go left
    leaf_pred = np.ones((num_leaves, num_targets), dtype=np.uint8)  # Default: all ones

    def fill(node: TreeNode, idx: int, default_pred: np.ndarray):
        if idx >= num_internal:
            # Leaf position
            leaf_idx = idx - num_internal
            if node.is_leaf:
                leaf_pred[leaf_idx] = node.prediction
            else:
                leaf_pred[leaf_idx] = default_pred
            return

        if node.is_leaf:
            # This node is a leaf but we need to fill subtree
            W[idx] = 0
            B[idx] = -1  # Always go left
            fill(node, 2*idx + 1, node.prediction)
            fill(node, 2*idx + 2, node.prediction)
        else:
            W[idx] = node.weights
            B[idx] = node.bias
            fill(node.left, 2*idx + 1, default_pred)
            fill(node.right, 2*idx + 2, default_pred)

    default = np.ones(num_targets, dtype=np.uint8)
    fill(root, 0, default)

    return W, B, leaf_pred


def save_model(path: str, depth: int, num_features: int, num_targets: int,
               W: np.ndarray, B: np.ndarray, leaf_pred: np.ndarray):
    """Save model to text file."""
    with open(path, 'w') as f:
        f.write("# Restructured intersection tree filter model\n")
        f.write(f"# Depth reduced from 8 to {depth}\n\n")
        f.write(f"{depth} {num_features} {num_targets}\n\n")

        f.write(f"# W matrix ({W.shape[0]} x {W.shape[1]} = {W.size} values)\n")
        for row in W:
            f.write(" ".join(f"{x}" for x in row) + "\n")

        f.write(f"\n# B vector ({B.size} values)\n")
        for i in range(0, len(B), 10):
            f.write(" ".join(f"{x}" for x in B[i:i+10]) + "\n")

        f.write(f"\n# Leaf predictions ({leaf_pred.shape[0]} x {leaf_pred.shape[1]} = {leaf_pred.size} values)\n")
        for row in leaf_pred:
            f.write(" ".join(str(x) for x in row) + "\n")


def validate_accuracy(samples: np.ndarray, orig_depth: int, orig_W: np.ndarray,
                      orig_B: np.ndarray, orig_pred: np.ndarray,
                      new_depth: int, new_W: np.ndarray, new_B: np.ndarray,
                      new_pred: np.ndarray) -> float:
    """Calculate accuracy of new model vs original."""
    correct = 0
    total = len(samples)

    for features in samples:
        _, orig_p = predict_original(features, orig_depth, orig_W, orig_B, orig_pred)
        _, new_p = predict_original(features, new_depth, new_W, new_B, new_pred)

        if np.array_equal(orig_p, new_p):
            correct += 1

    return correct / total


def main():
    data_path = "data/intersection_tree_filter.txt"
    if not Path(data_path).exists():
        data_path = "../data/intersection_tree_filter.txt"
    if not Path(data_path).exists():
        print("Error: Could not find data file")
        return

    print(f"Loading original model from: {data_path}")
    orig_depth, num_features, num_targets, orig_W, orig_B, orig_pred = load_model(data_path)

    print(f"\nOriginal model: depth={orig_depth}, {(1<<orig_depth)-1} internal nodes, {1<<orig_depth} leaves")

    # Generate training samples
    print("\nGenerating samples...")
    samples = generate_samples(50000)

    # Collect samples by pattern
    print("Collecting samples per prediction pattern...")
    samples_by_pattern = collect_samples_per_pattern(samples, orig_depth, orig_W, orig_B, orig_pred)

    print(f"Found {len(samples_by_pattern)} unique patterns")
    for pattern, s in sorted(samples_by_pattern.items(), key=lambda x: -len(x[1]))[:5]:
        active = [i for i, v in enumerate(pattern) if v == 1]
        print(f"  {len(s)} samples -> targets {active}")

    # Try different target depths
    best_tree = None
    best_accuracy = 0
    best_depth = orig_depth

    for target_depth in range(4, 8):
        print(f"\n{'='*60}")
        print(f"Building tree with max depth {target_depth}...")

        # Build new tree
        tree = build_tree_recursive(samples_by_pattern, num_features, target_depth)

        actual_depth = tree_depth(tree)
        internal, leaves = count_nodes(tree)
        print(f"  Actual depth: {actual_depth}, internal: {internal}, leaves: {leaves}")

        # Flatten and validate
        new_W, new_B, new_pred = flatten_to_complete_tree(tree, target_depth, num_features, num_targets)

        # Validate
        test_samples = generate_samples(10000, seed=123)
        accuracy = validate_accuracy(test_samples, orig_depth, orig_W, orig_B, orig_pred,
                                    target_depth, new_W, new_B, new_pred)
        print(f"  Accuracy: {100*accuracy:.2f}%")

        if accuracy > best_accuracy or (accuracy == best_accuracy and target_depth < best_depth):
            best_accuracy = accuracy
            best_depth = target_depth
            best_tree = (target_depth, new_W, new_B, new_pred)

    if best_tree is None:
        print("\nFailed to build a good tree!")
        return

    print(f"\n{'='*60}")
    print(f"BEST RESULT: depth={best_depth}, accuracy={100*best_accuracy:.2f}%")

    # Save best model
    output_path = data_path.replace('.txt', f'_depth{best_depth}.txt')
    target_depth, new_W, new_B, new_pred = best_tree
    save_model(output_path, target_depth, num_features, num_targets, new_W, new_B, new_pred)
    print(f"Saved to: {output_path}")

    # Summary
    orig_nodes = (1 << orig_depth) - 1 + (1 << orig_depth)
    new_nodes = (1 << target_depth) - 1 + (1 << target_depth)
    print(f"\nNode reduction: {orig_nodes} -> {new_nodes} ({100*(orig_nodes-new_nodes)/orig_nodes:.1f}% smaller)")


if __name__ == "__main__":
    main()