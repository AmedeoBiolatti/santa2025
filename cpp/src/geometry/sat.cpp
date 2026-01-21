#include "tree_packing/geometry/sat.hpp"
#include <algorithm>
#include <cmath>

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

namespace tree_packing {

std::array<Vec2, 3> compute_edge_normals(const Triangle& tri) {
    std::array<Vec2, 3> normals;

    // Edge from v0 to v1
    Vec2 e0 = tri.v1 - tri.v0;
    normals[0] = e0.perpendicular().normalized();

    // Edge from v1 to v2
    Vec2 e1 = tri.v2 - tri.v1;
    normals[1] = e1.perpendicular().normalized();

    // Edge from v2 to v0
    Vec2 e2 = tri.v0 - tri.v2;
    normals[2] = e2.perpendicular().normalized();

    return normals;
}

Projection project_triangle(const Triangle& tri, const Vec2& axis) {
    float p0 = tri.v0.dot(axis);
    float p1 = tri.v1.dot(axis);
    float p2 = tri.v2.dot(axis);

    return Projection{
        std::min({p0, p1, p2}),
        std::max({p0, p1, p2})
    };
}

bool triangles_intersect(const Triangle& t0, const Triangle& t1, float eps) {
    // Get normals from both triangles
    auto n0 = compute_edge_normals(t0);
    auto n1 = compute_edge_normals(t1);

    // Check all 6 axes
    for (int i = 0; i < 3; ++i) {
        auto proj0 = project_triangle(t0, n0[i]);
        auto proj1 = project_triangle(t1, n0[i]);

        // Strict overlap check (not just touching)
        if (!(proj0.max > proj1.min + eps && proj1.max > proj0.min + eps)) {
            return false;  // Separating axis found
        }
    }

    for (int i = 0; i < 3; ++i) {
        auto proj0 = project_triangle(t0, n1[i]);
        auto proj1 = project_triangle(t1, n1[i]);

        if (!(proj0.max > proj1.min + eps && proj1.max > proj0.min + eps)) {
            return false;
        }
    }

    return true;  // No separating axis found, triangles intersect
}

float triangles_intersection_score(const Triangle& t0, const Triangle& t1, float eps, bool assume_valid) {
    if (!assume_valid) {
        if (t0.is_nan() || t1.is_nan()) {
            return 0.0f;
        }
    }

    auto n0 = compute_edge_normals(t0);
    auto n1 = compute_edge_normals(t1);

    float min_score = std::numeric_limits<float>::max();

    // Check all 6 axes
    for (int i = 0; i < 3; ++i) {
        auto proj0 = project_triangle(t0, n0[i]);
        auto proj1 = project_triangle(t1, n0[i]);

        // score = (max0 - min1 - eps) * (max1 - min0 - eps)
        float score = (proj0.max - proj1.min - eps) * (proj1.max - proj0.min - eps);
        min_score = std::min(min_score, score);
    }

    for (int i = 0; i < 3; ++i) {
        auto proj0 = project_triangle(t0, n1[i]);
        auto proj1 = project_triangle(t1, n1[i]);

        float score = (proj0.max - proj1.min - eps) * (proj1.max - proj0.min - eps);
        min_score = std::min(min_score, score);
    }

    return min_score;
}

float figure_intersection_score(const Figure& f0, const Figure& f1, float eps, bool allow_negative, bool assume_valid) {
    if (!assume_valid) {
        if (f0.is_nan() || f1.is_nan()) {
            return 0.0f;
        }
    }

    float total = 0.0f;

    // Compute all pairwise triangle intersection scores
    for (size_t i = 0; i < TREE_NUM_TRIANGLES; ++i) {
        for (size_t j = 0; j < TREE_NUM_TRIANGLES; ++j) {
            float score = triangles_intersection_score(f0.triangles[i], f1.triangles[j], eps, assume_valid);
            if (allow_negative) {
                total += score;
            } else {
                total += std::max(0.0f, score);  // ReLU
            }
        }
    }

    return total;
}

void compute_intersection_matrix(
    const std::vector<Figure>& figures,
    std::vector<float>& matrix
) {
    size_t n = figures.size();
    matrix.resize(n * n, 0.0f);

#ifdef ENABLE_OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            float score = figure_intersection_score(figures[i], figures[j]);
            matrix[i * n + j] = score;
            matrix[j * n + i] = score;
        }
    }
}

void update_intersection_matrix(
    const std::vector<Figure>& figures,
    std::vector<float>& matrix,
    const std::vector<int>& modified_indices
) {
    size_t n = figures.size();

    for (int idx : modified_indices) {
        if (idx < 0 || static_cast<size_t>(idx) >= n) continue;

        // Update row and column for this index
#ifdef ENABLE_OPENMP
        #pragma omp parallel for
#endif
        for (size_t j = 0; j < n; ++j) {
            if (j == static_cast<size_t>(idx)) {
                matrix[idx * n + j] = 0.0f;  // Diagonal
            } else {
                float score = figure_intersection_score(figures[idx], figures[j]);
                matrix[idx * n + j] = score;
                matrix[j * n + idx] = score;
            }
        }
    }
}

}  // namespace tree_packing
