#!/usr/bin/env python3
"""
Analyze the intersection tree filter for pruning opportunities.

This script checks:
1. Nodes where both children have identical leaf predictions
2. Subtrees that can be collapsed (all leaves predict the same)
3. Leaves that predict "all 1s" (could potentially be pruned)
4. Statistics about tree structure and prediction patterns
"""

import numpy as np
from pathlib import Path
from collections import defaultdict
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

    # Parse header
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
    pred_size = num_leaves * num_targets

    W = np.array(values[:w_size]).reshape(num_internal, num_features)
    B = np.array(values[w_size:w_size + b_size])
    leaf_pred = np.array(values[w_size + b_size:w_size + b_size + pred_size], dtype=np.uint8).reshape(num_leaves, num_targets)

    return TreeModel(depth, num_features, num_targets, W, B, leaf_pred)


def get_children(node: int) -> tuple[int, int]:
    """Get left and right children of a node."""
    return 2 * node + 1, 2 * node + 2


def get_leaves_under_node(node: int, num_internal: int, num_leaves: int) -> list[int]:
    """Get all leaf indices under a given node."""
    if node >= num_internal:
        # It's a leaf
        return [node - num_internal]

    leaves = []
    stack = [node]
    while stack:
        n = stack.pop()
        if n >= num_internal:
            leaves.append(n - num_internal)
        else:
            left, right = get_children(n)
            stack.append(left)
            stack.append(right)
    return leaves


def analyze_node_pruning(model: TreeModel) -> dict:
    """Analyze each internal node for pruning opportunities."""
    results = {
        'identical_children': [],  # Nodes where left and right subtrees have same predictions
        'all_ones_subtrees': [],   # Nodes where all leaves predict all 1s
        'all_zeros_subtrees': [],  # Nodes where all leaves predict all 0s
        'single_pattern_subtrees': [],  # Nodes where all leaves have same prediction pattern
    }

    for node in range(model.num_internal):
        leaves = get_leaves_under_node(node, model.num_internal, model.num_leaves)
        predictions = model.leaf_pred[leaves]

        # Check if all leaves under this node have identical predictions
        if len(leaves) > 1:
            first_pred = predictions[0]
            if np.all(predictions == first_pred):
                results['single_pattern_subtrees'].append({
                    'node': node,
                    'num_leaves': len(leaves),
                    'prediction': first_pred.tolist(),
                })

                # Subcategorize
                if np.all(first_pred == 1):
                    results['all_ones_subtrees'].append(node)
                elif np.all(first_pred == 0):
                    results['all_zeros_subtrees'].append(node)

        # Check if left and right children have same aggregated behavior
        if node < model.num_internal:
            left, right = get_children(node)
            left_leaves = get_leaves_under_node(left, model.num_internal, model.num_leaves)
            right_leaves = get_leaves_under_node(right, model.num_internal, model.num_leaves)

            left_preds = model.leaf_pred[left_leaves]
            right_preds = model.leaf_pred[right_leaves]

            # Check if both subtrees have all identical predictions
            if len(left_leaves) > 0 and len(right_leaves) > 0:
                left_unique = np.unique(left_preds, axis=0)
                right_unique = np.unique(right_preds, axis=0)

                if len(left_unique) == 1 and len(right_unique) == 1:
                    if np.array_equal(left_unique[0], right_unique[0]):
                        results['identical_children'].append({
                            'node': node,
                            'left_leaves': len(left_leaves),
                            'right_leaves': len(right_leaves),
                            'prediction': left_unique[0].tolist(),
                        })

    return results


def analyze_leaf_patterns(model: TreeModel) -> dict:
    """Analyze leaf prediction patterns."""
    results = {
        'all_ones': [],
        'all_zeros': [],
        'unique_patterns': {},
        'pattern_counts': defaultdict(list),
    }

    for leaf_idx in range(model.num_leaves):
        pred = model.leaf_pred[leaf_idx]
        pred_tuple = tuple(pred.tolist())

        results['pattern_counts'][pred_tuple].append(leaf_idx)

        if np.all(pred == 1):
            results['all_ones'].append(leaf_idx)
        elif np.all(pred == 0):
            results['all_zeros'].append(leaf_idx)

    results['unique_patterns'] = len(results['pattern_counts'])

    return results


def analyze_zero_weight_nodes(model: TreeModel) -> list[int]:
    """Find nodes with all-zero weights (no split)."""
    zero_nodes = []
    for node in range(model.num_internal):
        if np.allclose(model.W[node], 0) and np.isclose(model.B[node], 0):
            zero_nodes.append(node)
    return zero_nodes


def compute_pruning_savings(model: TreeModel, prunable_nodes: list[dict]) -> dict:
    """Compute potential savings from pruning."""
    # Count how many nodes/leaves could be eliminated
    eliminated_internals = 0
    eliminated_leaves = 0

    visited = set()
    for item in prunable_nodes:
        node = item['node']
        if node in visited:
            continue

        # Count nodes in this subtree
        stack = [node]
        while stack:
            n = stack.pop()
            if n in visited:
                continue
            visited.add(n)

            if n < model.num_internal:
                eliminated_internals += 1
                left, right = get_children(n)
                stack.append(left)
                stack.append(right)
            else:
                eliminated_leaves += 1

    # We keep one "leaf" for each pruned subtree
    kept_leaves = len(prunable_nodes)

    return {
        'eliminated_internal_nodes': eliminated_internals,
        'eliminated_leaves': eliminated_leaves - kept_leaves,
        'original_internal_nodes': model.num_internal,
        'original_leaves': model.num_leaves,
        'reduction_percentage': 100 * (eliminated_internals + eliminated_leaves - kept_leaves) / (model.num_internal + model.num_leaves),
    }


def print_tree_structure(model: TreeModel, max_depth: int = 3):
    """Print tree structure up to a certain depth."""
    print(f"\nTree structure (depth {model.depth}, showing first {max_depth} levels):")
    print("=" * 60)

    for level in range(min(max_depth, model.depth)):
        start = (1 << level) - 1
        end = (1 << (level + 1)) - 1
        print(f"\nLevel {level} (nodes {start}-{end-1}):")

        for node in range(start, min(end, model.num_internal)):
            leaves = get_leaves_under_node(node, model.num_internal, model.num_leaves)
            preds = model.leaf_pred[leaves]
            unique = np.unique(preds, axis=0)

            # Check if this is a zero-weight node
            is_zero = np.allclose(model.W[node], 0) and np.isclose(model.B[node], 0)
            zero_marker = " [ZERO]" if is_zero else ""

            print(f"  Node {node}: {len(leaves)} leaves, {len(unique)} unique patterns{zero_marker}")


def main():
    # Find the data file
    possible_paths = [
        "data/intersection_tree_filter.txt",
        "../data/intersection_tree_filter.txt",
        "cpp/data/intersection_tree_filter.txt",
        "../cpp/data/intersection_tree_filter.txt",
    ]

    data_path = None
    for p in possible_paths:
        if Path(p).exists():
            data_path = p
            break

    if data_path is None:
        print("Error: Could not find intersection_tree_filter.txt")
        print("Searched:", possible_paths)
        return

    print(f"Loading model from: {data_path}")
    model = load_model(data_path)

    print(f"\n{'='*60}")
    print("TREE MODEL STATISTICS")
    print(f"{'='*60}")
    print(f"Depth: {model.depth}")
    print(f"Internal nodes: {model.num_internal}")
    print(f"Leaves: {model.num_leaves}")
    print(f"Features: {model.num_features}")
    print(f"Targets: {model.num_targets}")

    # Analyze leaf patterns
    print(f"\n{'='*60}")
    print("LEAF PATTERN ANALYSIS")
    print(f"{'='*60}")

    leaf_analysis = analyze_leaf_patterns(model)
    print(f"Unique prediction patterns: {leaf_analysis['unique_patterns']}")
    print(f"Leaves predicting all 1s: {len(leaf_analysis['all_ones'])} ({100*len(leaf_analysis['all_ones'])/model.num_leaves:.1f}%)")
    print(f"Leaves predicting all 0s: {len(leaf_analysis['all_zeros'])} ({100*len(leaf_analysis['all_zeros'])/model.num_leaves:.1f}%)")

    # Show most common patterns
    print("\nMost common patterns:")
    sorted_patterns = sorted(leaf_analysis['pattern_counts'].items(), key=lambda x: -len(x[1]))
    for pattern, leaves in sorted_patterns[:10]:
        active_targets = [i for i, v in enumerate(pattern) if v == 1]
        print(f"  {len(leaves)} leaves: targets {active_targets}")

    # Analyze zero-weight nodes
    print(f"\n{'='*60}")
    print("ZERO-WEIGHT NODE ANALYSIS")
    print(f"{'='*60}")

    zero_nodes = analyze_zero_weight_nodes(model)
    print(f"Nodes with all-zero weights: {len(zero_nodes)}")
    if zero_nodes:
        print(f"  Nodes: {zero_nodes[:20]}{'...' if len(zero_nodes) > 20 else ''}")

    # Analyze pruning opportunities
    print(f"\n{'='*60}")
    print("PRUNING OPPORTUNITY ANALYSIS")
    print(f"{'='*60}")

    node_analysis = analyze_node_pruning(model)

    print(f"\nNodes with identical left/right subtree predictions: {len(node_analysis['identical_children'])}")
    for item in node_analysis['identical_children'][:10]:
        print(f"  Node {item['node']}: {item['left_leaves']}+{item['right_leaves']} leaves -> {sum(item['prediction'])} active targets")

    print(f"\nSubtrees with single prediction pattern: {len(node_analysis['single_pattern_subtrees'])}")
    # Filter to show only subtrees with > 1 leaf
    multi_leaf = [x for x in node_analysis['single_pattern_subtrees'] if x['num_leaves'] > 1]
    for item in multi_leaf[:10]:
        active = sum(item['prediction'])
        print(f"  Node {item['node']}: {item['num_leaves']} leaves -> {active} active targets")

    print(f"\nAll-ones subtrees (root nodes): {len(node_analysis['all_ones_subtrees'])}")
    print(f"All-zeros subtrees (root nodes): {len(node_analysis['all_zeros_subtrees'])}")

    # Compute savings
    if node_analysis['single_pattern_subtrees']:
        savings = compute_pruning_savings(model, node_analysis['single_pattern_subtrees'])
        print(f"\n{'='*60}")
        print("POTENTIAL PRUNING SAVINGS")
        print(f"{'='*60}")
        print(f"Could eliminate {savings['eliminated_internal_nodes']} internal nodes")
        print(f"Could eliminate {savings['eliminated_leaves']} leaves")
        print(f"Total reduction: {savings['reduction_percentage']:.1f}%")

    # Print tree structure
    print_tree_structure(model)

    # Detailed analysis of prunable subtrees
    if multi_leaf:
        print(f"\n{'='*60}")
        print("DETAILED PRUNABLE SUBTREES (with >1 leaf)")
        print(f"{'='*60}")

        # Sort by number of leaves (larger subtrees = more savings)
        multi_leaf_sorted = sorted(multi_leaf, key=lambda x: -x['num_leaves'])
        for item in multi_leaf_sorted[:20]:
            node = item['node']
            depth = 0
            n = node
            while n > 0:
                n = (n - 1) // 2
                depth += 1

            active = [i for i, v in enumerate(item['prediction']) if v == 1]
            print(f"  Node {node} (depth {depth}): {item['num_leaves']} leaves, targets {active}")


if __name__ == "__main__":
    main()
