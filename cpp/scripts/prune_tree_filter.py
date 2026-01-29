#!/usr/bin/env python3
"""
Prune the intersection tree filter to remove redundant subtrees.

A subtree can be pruned if all its leaves have identical predictions.
The pruned tree replaces such subtrees with a single leaf.
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class TreeModel:
    depth: int
    num_features: int
    num_targets: int
    W: np.ndarray  # (num_internal, num_features)
    B: np.ndarray  # (num_internal,)
    leaf_pred: np.ndarray  # (num_leaves, num_targets)

    @property
    def num_internal(self) -> int:
        return (1 << self.depth) - 1

    @property
    def num_leaves(self) -> int:
        return 1 << self.depth


def load_model(path: str) -> TreeModel:
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

    return TreeModel(depth, num_features, num_targets, W, B, leaf_pred)


def get_leaves_under_node(node: int, num_internal: int) -> list[int]:
    """Get all leaf indices under a given node."""
    if node >= num_internal:
        return [node - num_internal]

    leaves = []
    stack = [node]
    while stack:
        n = stack.pop()
        if n >= num_internal:
            leaves.append(n - num_internal)
        else:
            stack.append(2 * n + 1)
            stack.append(2 * n + 2)
    return leaves


def find_prunable_nodes(model: TreeModel) -> dict[int, np.ndarray]:
    """
    Find nodes whose subtrees can be pruned.
    Returns dict mapping node index to the common prediction for that subtree.
    Only returns the highest (closest to root) prunable nodes.
    """
    prunable = {}

    # Process nodes from root to leaves (BFS order)
    for node in range(model.num_internal):
        # Skip if an ancestor is already prunable
        ancestor_prunable = False
        n = node
        while n > 0:
            n = (n - 1) // 2
            if n in prunable:
                ancestor_prunable = True
                break
        if ancestor_prunable:
            continue

        leaves = get_leaves_under_node(node, model.num_internal)
        if len(leaves) <= 1:
            continue

        predictions = model.leaf_pred[leaves]
        first_pred = predictions[0]

        if np.all(predictions == first_pred):
            prunable[node] = first_pred

    return prunable


@dataclass
class PrunedNode:
    """Represents a node in the pruned tree."""
    is_leaf: bool
    # For internal nodes:
    weights: Optional[np.ndarray] = None
    bias: Optional[float] = None
    left: Optional['PrunedNode'] = None
    right: Optional['PrunedNode'] = None
    # For leaf nodes:
    prediction: Optional[np.ndarray] = None


def build_pruned_tree(model: TreeModel, prunable: dict[int, np.ndarray]) -> PrunedNode:
    """Build a pruned tree structure."""

    def build_node(node_idx: int) -> PrunedNode:
        # Check if this node should be pruned to a leaf
        if node_idx in prunable:
            return PrunedNode(is_leaf=True, prediction=prunable[node_idx])

        # Check if this is already a leaf in the original tree
        if node_idx >= model.num_internal:
            leaf_idx = node_idx - model.num_internal
            return PrunedNode(is_leaf=True, prediction=model.leaf_pred[leaf_idx])

        # This is an internal node - keep it
        left_child = 2 * node_idx + 1
        right_child = 2 * node_idx + 2

        return PrunedNode(
            is_leaf=False,
            weights=model.W[node_idx].copy(),
            bias=model.B[node_idx],
            left=build_node(left_child),
            right=build_node(right_child),
        )

    return build_node(0)


def count_nodes(node: PrunedNode) -> tuple[int, int]:
    """Count internal nodes and leaves in the pruned tree."""
    if node.is_leaf:
        return 0, 1
    else:
        left_int, left_leaf = count_nodes(node.left)
        right_int, right_leaf = count_nodes(node.right)
        return 1 + left_int + right_int, left_leaf + right_leaf


def compute_max_depth(node: PrunedNode, current_depth: int = 0) -> int:
    """Compute the maximum depth of the pruned tree."""
    if node.is_leaf:
        return current_depth
    else:
        return max(
            compute_max_depth(node.left, current_depth + 1),
            compute_max_depth(node.right, current_depth + 1)
        )


def flatten_pruned_tree(root: PrunedNode, target_depth: int, num_features: int, num_targets: int):
    """
    Flatten pruned tree to arrays for a complete binary tree of given depth.
    Missing internal nodes get zero weights, missing leaves inherit from pruned ancestors.
    """
    num_internal = (1 << target_depth) - 1
    num_leaves = 1 << target_depth

    W = np.zeros((num_internal, num_features), dtype=np.float32)
    B = np.zeros(num_internal, dtype=np.float32)
    leaf_pred = np.ones((num_leaves, num_targets), dtype=np.uint8)  # Default to all 1s

    def fill_node(node: PrunedNode, idx: int, inherited_pred: Optional[np.ndarray]):
        if idx >= num_internal:
            # This is a leaf position
            leaf_idx = idx - num_internal
            if node.is_leaf:
                leaf_pred[leaf_idx] = node.prediction
            elif inherited_pred is not None:
                leaf_pred[leaf_idx] = inherited_pred
            return

        if node.is_leaf:
            # This node was pruned - fill all descendants with this prediction
            W[idx] = 0
            B[idx] = -1  # Always go left (to be consistent)
            # Fill left subtree with this prediction
            fill_node(node, 2 * idx + 1, node.prediction)
            fill_node(node, 2 * idx + 2, node.prediction)
        else:
            # Normal internal node
            W[idx] = node.weights
            B[idx] = node.bias
            fill_node(node.left, 2 * idx + 1, inherited_pred)
            fill_node(node.right, 2 * idx + 2, inherited_pred)

    fill_node(root, 0, None)
    return W, B, leaf_pred


def save_model(path: str, depth: int, num_features: int, num_targets: int,
               W: np.ndarray, B: np.ndarray, leaf_pred: np.ndarray):
    """Save model to text file."""
    with open(path, 'w') as f:
        f.write("# Pruned intersection tree filter model\n")
        f.write(f"# Format: depth num_features num_targets\n\n")
        f.write(f"{depth} {num_features} {num_targets}\n\n")

        f.write(f"# W matrix ({W.shape[0]} * {W.shape[1]} = {W.size} values)\n")
        for row in W:
            f.write(" ".join(f"{x}" for x in row) + "\n")

        f.write(f"\n# B vector ({B.size} values)\n")
        for i in range(0, len(B), 10):
            f.write(" ".join(f"{x}" for x in B[i:i+10]) + "\n")

        f.write(f"\n# Leaf predictions ({leaf_pred.shape[0]} * {leaf_pred.shape[1]} = {leaf_pred.size} values)\n")
        for row in leaf_pred:
            f.write(" ".join(str(x) for x in row) + "\n")


def main():
    # Find the data file
    possible_paths = [
        "data/intersection_tree_filter.txt",
        "../data/intersection_tree_filter.txt",
        "cpp/data/intersection_tree_filter.txt",
    ]

    data_path = None
    for p in possible_paths:
        if Path(p).exists():
            data_path = p
            break

    if data_path is None:
        print("Error: Could not find intersection_tree_filter.txt")
        return

    print(f"Loading model from: {data_path}")
    model = load_model(data_path)

    print(f"\nOriginal tree:")
    print(f"  Depth: {model.depth}")
    print(f"  Internal nodes: {model.num_internal}")
    print(f"  Leaves: {model.num_leaves}")

    # Find prunable nodes
    prunable = find_prunable_nodes(model)
    print(f"\nFound {len(prunable)} prunable subtrees")

    if not prunable:
        print("No pruning possible!")
        return

    # Show largest prunable subtrees
    print("\nLargest prunable subtrees:")
    sorted_prunable = sorted(prunable.items(), key=lambda x: -len(get_leaves_under_node(x[0], model.num_internal)))
    for node, pred in sorted_prunable[:10]:
        leaves = get_leaves_under_node(node, model.num_internal)
        active = [i for i, v in enumerate(pred) if v == 1]
        depth = 0
        n = node
        while n > 0:
            n = (n - 1) // 2
            depth += 1
        print(f"  Node {node} (depth {depth}): {len(leaves)} leaves -> targets {active}")

    # Build pruned tree
    pruned = build_pruned_tree(model, prunable)
    internal_count, leaf_count = count_nodes(pruned)
    max_depth = compute_max_depth(pruned)

    print(f"\nPruned tree (variable depth):")
    print(f"  Max depth: {max_depth}")
    print(f"  Internal nodes: {internal_count}")
    print(f"  Leaves: {leaf_count}")

    # Calculate savings
    original_total = model.num_internal + model.num_leaves
    pruned_total = internal_count + leaf_count
    print(f"\nSavings:")
    print(f"  Original: {original_total} nodes")
    print(f"  Pruned: {pruned_total} nodes")
    print(f"  Reduction: {100 * (original_total - pruned_total) / original_total:.1f}%")

    # Determine output depth - use the minimum depth that can represent the pruned tree
    output_depth = max_depth
    print(f"\nOutput depth: {output_depth}")

    # Flatten to arrays
    W, B, leaf_pred = flatten_pruned_tree(pruned, output_depth, model.num_features, model.num_targets)

    # Count actual non-trivial nodes
    non_trivial = np.sum(~np.all(W == 0, axis=1) | (B != -1))
    print(f"Non-trivial internal nodes: {non_trivial}")

    # Save pruned model
    output_path = data_path.replace('.txt', '_pruned.txt')
    save_model(output_path, output_depth, model.num_features, model.num_targets, W, B, leaf_pred)
    print(f"\nSaved pruned model to: {output_path}")

    # Also print summary of unique patterns
    unique_patterns = np.unique(leaf_pred, axis=0)
    print(f"\nUnique leaf patterns: {len(unique_patterns)}")

    # Verify all leaves have valid predictions
    all_ones_count = np.sum(np.all(leaf_pred == 1, axis=1))
    all_zeros_count = np.sum(np.all(leaf_pred == 0, axis=1))
    print(f"Leaves with all 1s: {all_ones_count}")
    print(f"Leaves with all 0s: {all_zeros_count}")


if __name__ == "__main__":
    main()
