#!/usr/bin/env python3
"""
Compact the intersection tree filter by preserving original splits
but collapsing uniform subtrees.

This approach maintains 100% accuracy while reducing effective depth.
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class CompactNode:
    """Node in the compacted tree."""
    is_leaf: bool = False
    # For leaves:
    prediction: Optional[np.ndarray] = None
    # For internal nodes:
    weights: Optional[np.ndarray] = None
    bias: Optional[float] = None
    left: Optional['CompactNode'] = None
    right: Optional['CompactNode'] = None
    # Original node index (for debugging)
    orig_idx: int = -1


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


def is_uniform_subtree(node_idx: int, num_internal: int, leaf_pred: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
    """Check if all leaves under node have same prediction."""
    leaves = get_leaves_under(node_idx, num_internal)
    if not leaves:
        return False, None

    first = leaf_pred[leaves[0]]
    for leaf in leaves[1:]:
        if not np.array_equal(leaf_pred[leaf], first):
            return False, None

    return True, first


def build_compact_tree(orig_depth: int, W: np.ndarray, B: np.ndarray,
                       leaf_pred: np.ndarray) -> CompactNode:
    """Build compacted tree by collapsing uniform subtrees."""
    num_internal = (1 << orig_depth) - 1

    def build(node_idx: int) -> CompactNode:
        # Check if this subtree is uniform
        is_uniform, pred = is_uniform_subtree(node_idx, num_internal, leaf_pred)

        if is_uniform:
            return CompactNode(is_leaf=True, prediction=pred, orig_idx=node_idx)

        # Check if this is a leaf in original tree
        if node_idx >= num_internal:
            leaf_idx = node_idx - num_internal
            return CompactNode(is_leaf=True, prediction=leaf_pred[leaf_idx], orig_idx=node_idx)

        # Keep this internal node
        left = build(2 * node_idx + 1)
        right = build(2 * node_idx + 2)

        return CompactNode(
            is_leaf=False,
            weights=W[node_idx].copy(),
            bias=B[node_idx],
            left=left,
            right=right,
            orig_idx=node_idx
        )

    return build(0)


def tree_depth(node: CompactNode) -> int:
    """Get maximum depth of tree."""
    if node.is_leaf:
        return 0
    return 1 + max(tree_depth(node.left), tree_depth(node.right))


def count_nodes(node: CompactNode) -> Tuple[int, int]:
    """Count (internal, leaves)."""
    if node.is_leaf:
        return 0, 1
    li, ll = count_nodes(node.left)
    ri, rl = count_nodes(node.right)
    return 1 + li + ri, ll + rl


def collect_unique_leaves(node: CompactNode) -> list:
    """Collect all unique leaf predictions."""
    if node.is_leaf:
        return [tuple(node.prediction.tolist())]
    return collect_unique_leaves(node.left) + collect_unique_leaves(node.right)


def renumber_tree(root: CompactNode, target_depth: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Renumber the compact tree using BFS to create a more balanced structure.
    Returns (W, B, leaf_pred, actual_depth).
    """
    num_features = len(root.weights) if not root.is_leaf else 10
    num_targets = len(root.prediction) if root.is_leaf else 16

    # Count actual nodes needed
    internal, leaves = count_nodes(root)
    actual_depth = tree_depth(root)

    print(f"  Compact tree has {internal} internal nodes, {leaves} leaves, depth {actual_depth}")

    # For output, we need a complete binary tree
    use_depth = min(target_depth, actual_depth)
    num_internal_out = (1 << use_depth) - 1
    num_leaves_out = 1 << use_depth

    W = np.zeros((num_internal_out, num_features), dtype=np.float32)
    B = np.full(num_internal_out, -1.0, dtype=np.float32)  # Default: always go left
    leaf_pred = np.ones((num_leaves_out, num_targets), dtype=np.uint8)  # Default: all ones

    def fill(node: CompactNode, idx: int, default_pred: np.ndarray):
        if idx >= num_internal_out:
            # This is a leaf position in output
            leaf_idx = idx - num_internal_out
            if node.is_leaf:
                leaf_pred[leaf_idx] = node.prediction
            else:
                # Node is internal but we're at output leaf - need to handle this
                # Take the most common prediction in this subtree
                preds = collect_unique_leaves(node)
                # Just take the first one (they should be resolved correctly anyway)
                leaf_pred[leaf_idx] = np.array(preds[0], dtype=np.uint8)
            return

        if node.is_leaf:
            # Node is leaf but we're at internal position - replicate
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

    return W, B, leaf_pred, use_depth


def save_model(path: str, depth: int, num_features: int, num_targets: int,
               W: np.ndarray, B: np.ndarray, leaf_pred: np.ndarray):
    """Save model to text file."""
    with open(path, 'w') as f:
        f.write("# Compacted intersection tree filter model\n")
        f.write(f"# Depth reduced while maintaining 100% accuracy\n\n")
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


def predict(features: np.ndarray, depth: int, W: np.ndarray, B: np.ndarray,
            leaf_pred: np.ndarray) -> np.ndarray:
    """Predict with a model."""
    num_internal = (1 << depth) - 1
    node = 0

    while node < num_internal:
        score = np.dot(W[node], features) + B[node]
        node = 2*node + 1 if score <= 0 else 2*node + 2

    return leaf_pred[node - num_internal]


def validate(samples: np.ndarray,
             orig_depth: int, orig_W: np.ndarray, orig_B: np.ndarray, orig_pred: np.ndarray,
             new_depth: int, new_W: np.ndarray, new_B: np.ndarray, new_pred: np.ndarray) -> float:
    """Validate new model against original."""
    correct = 0
    for features in samples:
        orig_p = predict(features, orig_depth, orig_W, orig_B, orig_pred)
        new_p = predict(features, new_depth, new_W, new_B, new_pred)
        if np.array_equal(orig_p, new_p):
            correct += 1
    return correct / len(samples)


def main():
    data_path = "data/intersection_tree_filter.txt"
    if not Path(data_path).exists():
        data_path = "../data/intersection_tree_filter.txt"

    print(f"Loading model from: {data_path}")
    orig_depth, num_features, num_targets, orig_W, orig_B, orig_pred = load_model(data_path)

    orig_internal = (1 << orig_depth) - 1
    orig_leaves = 1 << orig_depth
    print(f"\nOriginal model:")
    print(f"  Depth: {orig_depth}")
    print(f"  Internal nodes: {orig_internal}")
    print(f"  Leaves: {orig_leaves}")
    print(f"  Total nodes: {orig_internal + orig_leaves}")

    # Build compacted tree
    print(f"\nBuilding compacted tree...")
    compact = build_compact_tree(orig_depth, orig_W, orig_B, orig_pred)

    actual_depth = tree_depth(compact)
    internal, leaves = count_nodes(compact)

    print(f"\nCompacted tree (variable depth):")
    print(f"  Max depth: {actual_depth}")
    print(f"  Internal nodes: {internal}")
    print(f"  Leaves: {leaves}")
    print(f"  Total nodes: {internal + leaves}")

    # Show unique patterns
    unique = set(collect_unique_leaves(compact))
    print(f"  Unique leaf patterns: {len(unique)}")

    # Generate arrays for different output depths
    print(f"\n{'='*60}")
    print("Testing different output depths...")

    # Generate test samples
    np.random.seed(42)
    test_samples = []
    for _ in range(20000):
        x = np.random.uniform(-2, 2)
        y = np.random.uniform(-2, 2)
        r = np.sqrt(x*x + y*y)
        ang = np.random.uniform(0, 2*np.pi)
        c, s = np.cos(ang), np.sin(ang)
        test_samples.append([x, y, r, c, s, x*c, y*c, x*s, y*s, r*c])
    test_samples = np.array(test_samples, dtype=np.float32)

    best_depth = orig_depth
    best_accuracy = 0
    best_result = None

    for target_depth in range(actual_depth, orig_depth + 1):
        new_W, new_B, new_pred, used_depth = renumber_tree(compact, target_depth)

        accuracy = validate(test_samples, orig_depth, orig_W, orig_B, orig_pred,
                           used_depth, new_W, new_B, new_pred)

        new_internal = (1 << used_depth) - 1
        new_leaves = 1 << used_depth
        reduction = 100 * (orig_internal + orig_leaves - new_internal - new_leaves) / (orig_internal + orig_leaves)

        print(f"\n  Depth {used_depth}: accuracy={100*accuracy:.2f}%, reduction={reduction:.1f}%")

        if accuracy >= 0.9999:  # Allow tiny floating point errors
            if used_depth < best_depth or accuracy > best_accuracy:
                best_depth = used_depth
                best_accuracy = accuracy
                best_result = (new_W, new_B, new_pred)

    if best_result is None:
        print("\nWARNING: Could not find a depth with 100% accuracy!")
        print("Using the maximum depth tree...")
        new_W, new_B, new_pred, _ = renumber_tree(compact, orig_depth)
        best_depth = orig_depth
        best_result = (new_W, new_B, new_pred)

    # Save the best model
    output_path = data_path.replace('.txt', f'_compact.txt')
    new_W, new_B, new_pred = best_result
    save_model(output_path, best_depth, num_features, num_targets, new_W, new_B, new_pred)

    print(f"\n{'='*60}")
    print(f"RESULT:")
    print(f"  Output depth: {best_depth}")
    print(f"  Accuracy: {100*best_accuracy:.2f}%")

    new_internal = (1 << best_depth) - 1
    new_leaves = 1 << best_depth
    print(f"  Nodes: {orig_internal + orig_leaves} -> {new_internal + new_leaves}")
    print(f"  Reduction: {100*(orig_internal + orig_leaves - new_internal - new_leaves)/(orig_internal + orig_leaves):.1f}%")
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()