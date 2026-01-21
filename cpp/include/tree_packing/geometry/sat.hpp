#pragma once

#include "../core/types.hpp"
#include <array>

namespace tree_packing {

// Compute edge normals for a triangle (SAT axes)
// Returns 3 normalized perpendicular vectors to the triangle edges
[[nodiscard]] std::array<Vec2, 3> compute_edge_normals(const Triangle& tri);

// Project triangle vertices onto an axis and return min/max
struct Projection {
    float min;
    float max;
};

[[nodiscard]] Projection project_triangle(const Triangle& tri, const Vec2& axis);

// Check if two triangles strictly intersect (overlap with positive area)
// Returns false for touching-only cases
[[nodiscard]] bool triangles_intersect(const Triangle& t0, const Triangle& t1, float eps = EPSILON);

// Compute intersection score between two triangles
// - Positive when triangles overlap
// - Zero or negative when separated or only touching
[[nodiscard]] float triangles_intersection_score(
    const Triangle& t0,
    const Triangle& t1,
    float eps = EPSILON,
    bool assume_valid = false
);

// Compute intersection score between two figures (unions of triangles)
// Sums up all pairwise triangle intersection scores
[[nodiscard]] float figure_intersection_score(
    const Figure& f0,
    const Figure& f1,
    float eps = EPSILON,
    bool allow_negative = false,
    bool assume_valid = false
);

// Compute full intersection matrix between all pairs of figures
// matrix[i * n + j] = intersection score between figure i and j
void compute_intersection_matrix(
    const std::vector<Figure>& figures,
    std::vector<float>& matrix
);

// Update rows/columns of intersection matrix for modified indices
void update_intersection_matrix(
    const std::vector<Figure>& figures,
    std::vector<float>& matrix,
    const std::vector<int>& modified_indices
);

}  // namespace tree_packing
