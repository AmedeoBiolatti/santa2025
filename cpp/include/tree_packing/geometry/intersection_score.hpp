#pragma once

#include "../core/types.hpp"
#include <array>

namespace tree_packing {

// Number of targets: 5x5 triangle pairs
constexpr size_t NUM_SCORE_TARGETS = TREE_NUM_TRIANGLES * TREE_NUM_TRIANGLES;  // 25

// Signed intersection score function g(x, y, angle)
// Tree A is at origin (0, 0, 0), Tree B is at (x, y, angle)
// Returns 25 values (5x5 triangles), positive means overlap
//
// Output layout (row-major 5x5):
//   out[i * 5 + j] = score between triangle i of tree A and triangle j of tree B
void compute_intersection_scores(
    float x, float y, float angle,
    std::array<float, NUM_SCORE_TARGETS>& out,
    float eps = EPSILON
);

// Batch version for efficiency
void compute_intersection_scores_batch(
    const float* x,      // [N]
    const float* y,      // [N]
    const float* angle,  // [N]
    size_t n,
    float* out,          // [N, 25]
    float eps = EPSILON
);

}  // namespace tree_packing