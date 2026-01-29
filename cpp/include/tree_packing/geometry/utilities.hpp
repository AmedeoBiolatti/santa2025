#pragma once

#include "../core/types.hpp"
#include <array>

namespace tree_packing {

// Compute per-triangle intersection scores between two trees defined by params.
// Returns a 5x5 matrix where entry [i][j] is the intersection score between
// triangle i of the first tree and triangle j of the second tree.
[[nodiscard]] std::array<std::array<float, TREE_NUM_TRIANGLES>, TREE_NUM_TRIANGLES>
triangle_pair_intersection_scores(
    const TreeParams& a,
    const TreeParams& b,
    float eps = EPSILON
);

}  // namespace tree_packing

