#include "tree_packing/geometry/sat.hpp"
#include <algorithm>
#include <cmath>

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

namespace tree_packing {
namespace {
inline void project_triangle_pair(
    const Triangle& t0,
    const Triangle& t1,
    const Vec2& axis,
    float& min0,
    float& max0,
    float& min1,
    float& max1
) {
    float t0p0 = t0.v0.dot(axis);
    float t0p1 = t0.v1.dot(axis);
    float t0p2 = t0.v2.dot(axis);
    min0 = t0p0;
    max0 = t0p0;
    if (t0p1 < min0) min0 = t0p1;
    if (t0p1 > max0) max0 = t0p1;
    if (t0p2 < min0) min0 = t0p2;
    if (t0p2 > max0) max0 = t0p2;

    float t1p0 = t1.v0.dot(axis);
    float t1p1 = t1.v1.dot(axis);
    float t1p2 = t1.v2.dot(axis);
    min1 = t1p0;
    max1 = t1p0;
    if (t1p1 < min1) min1 = t1p1;
    if (t1p1 > max1) max1 = t1p1;
    if (t1p2 < min1) min1 = t1p2;
    if (t1p2 > max1) max1 = t1p2;
}
}  // namespace

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
        float min0 = 0.0f;
        float max0 = 0.0f;
        float min1 = 0.0f;
        float max1 = 0.0f;
        project_triangle_pair(t0, t1, n0[i], min0, max0, min1, max1);

        // Strict overlap check (not just touching)
        if (!(max0 > min1 + eps && max1 > min0 + eps)) {
            return false;  // Separating axis found
        }
    }

    for (int i = 0; i < 3; ++i) {
        float min0 = 0.0f;
        float max0 = 0.0f;
        float min1 = 0.0f;
        float max1 = 0.0f;
        project_triangle_pair(t0, t1, n1[i], min0, max0, min1, max1);

        if (!(max0 > min1 + eps && max1 > min0 + eps)) {
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
        float min0 = 0.0f;
        float max0 = 0.0f;
        float min1 = 0.0f;
        float max1 = 0.0f;
        project_triangle_pair(t0, t1, n0[i], min0, max0, min1, max1);

        // score = (max0 - min1 - eps) * (max1 - min0 - eps)
        float score = (max0 - min1 - eps) * (max1 - min0 - eps);
        min_score = std::min(min_score, score);
    }

    for (int i = 0; i < 3; ++i) {
        float min0 = 0.0f;
        float max0 = 0.0f;
        float min1 = 0.0f;
        float max1 = 0.0f;
        project_triangle_pair(t0, t1, n1[i], min0, max0, min1, max1);

        float score = (max0 - min1 - eps) * (max1 - min0 - eps);
        min_score = std::min(min_score, score);
    }

    return min_score;
}

float triangles_intersection_score_from_normals(
    const Triangle& t0,
    const Triangle& t1,
    const std::array<Vec2, 3>& n0,
    const std::array<Vec2, 3>& n1,
    float eps,
    bool assume_valid
) {
    if (!assume_valid) {
        if (t0.is_nan() || t1.is_nan()) {
            return 0.0f;
        }
    }

    float min_score = std::numeric_limits<float>::max();

    for (int i = 0; i < 3; ++i) {
        float min0 = 0.0f;
        float max0 = 0.0f;
        float min1 = 0.0f;
        float max1 = 0.0f;
        project_triangle_pair(t0, t1, n0[i], min0, max0, min1, max1);

        float score = (max0 - min1 - eps) * (max1 - min0 - eps);
        min_score = std::min(min_score, score);
    }

    for (int i = 0; i < 3; ++i) {
        float min0 = 0.0f;
        float max0 = 0.0f;
        float min1 = 0.0f;
        float max1 = 0.0f;
        project_triangle_pair(t0, t1, n1[i], min0, max0, min1, max1);

        float score = (max0 - min1 - eps) * (max1 - min0 - eps);
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

float figure_intersection_score_from_normals(
    const Figure& f0,
    const Figure& f1,
    const std::array<std::array<Vec2, 3>, TREE_NUM_TRIANGLES>& n0,
    const std::array<std::array<Vec2, 3>, TREE_NUM_TRIANGLES>& n1,
    float eps,
    bool allow_negative,
    bool assume_valid
) {
    if (!assume_valid) {
        if (f0.is_nan() || f1.is_nan()) {
            return 0.0f;
        }
    }

    float total = 0.0f;

    for (size_t i = 0; i < TREE_NUM_TRIANGLES; ++i) {
        for (size_t j = 0; j < TREE_NUM_TRIANGLES; ++j) {
            float score = triangles_intersection_score_from_normals(
                f0.triangles[i], f1.triangles[j], n0[i], n1[j], eps, assume_valid
            );
            if (allow_negative) {
                total += score;
            } else {
                total += std::max(0.0f, score);
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
