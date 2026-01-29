#include "tree_packing/geometry/intersection_score.hpp"
#include "tree_packing/geometry/sat.hpp"
#include "tree_packing/core/tree.hpp"

namespace tree_packing {

void compute_intersection_scores(
    float x, float y, float angle,
    std::array<float, NUM_SCORE_TARGETS>& out,
    float eps
) {
    // Tree A: at origin, use precomputed shape and normals
    const auto& tree_a = TREE_SHAPE;
    const auto& normals_a = TREE_NORMALS;

    // Tree B: transform shape by (x, y, angle)
    Figure tree_b;
    for (size_t i = 0; i < TREE_NUM_TRIANGLES; ++i) {
        tree_b.triangles[i] = transform_triangle(Vec2{x, y}, angle, TREE_SHAPE[i]);
    }

    // Compute rotated normals for tree B
    std::array<std::array<Vec2, 3>, TREE_NUM_TRIANGLES> normals_b;
    get_tree_normals(angle, normals_b);

    // Compute all 5x5 scores
    for (size_t i = 0; i < TREE_NUM_TRIANGLES; ++i) {
        for (size_t j = 0; j < TREE_NUM_TRIANGLES; ++j) {
            out[i * TREE_NUM_TRIANGLES + j] = triangles_intersection_score_from_normals(
                tree_a[i], tree_b[j],
                normals_a[i], normals_b[j],
                eps
            );
        }
    }
}

void compute_intersection_scores_batch(
    const float* x,
    const float* y,
    const float* angle,
    size_t n,
    float* out,
    float eps
) {
    // Tree A is constant: at origin with precomputed normals
    const auto& tree_a = TREE_SHAPE;
    const auto& normals_a = TREE_NORMALS;

    for (size_t k = 0; k < n; ++k) {
        const float xk = x[k];
        const float yk = y[k];
        const float ak = angle[k];

        // Transform tree B
        Figure tree_b;
        for (size_t i = 0; i < TREE_NUM_TRIANGLES; ++i) {
            tree_b.triangles[i] = transform_triangle(Vec2{xk, yk}, ak, TREE_SHAPE[i]);
        }

        // Compute rotated normals
        std::array<std::array<Vec2, 3>, TREE_NUM_TRIANGLES> normals_b;
        get_tree_normals(ak, normals_b);

        // Compute all 5x5 scores
        float* o = out + k * NUM_SCORE_TARGETS;
        for (size_t i = 0; i < TREE_NUM_TRIANGLES; ++i) {
            for (size_t j = 0; j < TREE_NUM_TRIANGLES; ++j) {
                o[i * TREE_NUM_TRIANGLES + j] = triangles_intersection_score_from_normals(
                    tree_a[i], tree_b[j],
                    normals_a[i], normals_b[j],
                    eps
                );
            }
        }
    }
}

}  // namespace tree_packing