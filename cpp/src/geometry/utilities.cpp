#include "tree_packing/geometry/utilities.hpp"

#include "tree_packing/core/tree.hpp"
#include "tree_packing/geometry/sat.hpp"

namespace tree_packing {
namespace {

[[nodiscard]] inline AABB triangle_aabb(const Triangle& tri) {
    AABB aabb;
    aabb.expand(tri);
    return aabb;
}

}  // namespace

std::array<std::array<float, TREE_NUM_TRIANGLES>, TREE_NUM_TRIANGLES>
triangle_pair_intersection_scores(const TreeParams& a, const TreeParams& b, float eps) {
    std::array<std::array<float, TREE_NUM_TRIANGLES>, TREE_NUM_TRIANGLES> scores{};

    const Figure fa = params_to_figure(a);
    const Figure fb = params_to_figure(b);

    const Vec2 ca = get_tree_center(a);
    const Vec2 cb = get_tree_center(b);
    const Vec2 diff = ca - cb;
    if (diff.length_squared() >= THR2) {
        return scores;
    }

    const AABB aabb_a = compute_aabb(fa);
    const AABB aabb_b = compute_aabb(fb);
    if (!aabb_a.intersects(aabb_b)) {
        return scores;
    }

    std::array<std::array<Vec2, 3>, TREE_NUM_TRIANGLES> na{};
    std::array<std::array<Vec2, 3>, TREE_NUM_TRIANGLES> nb{};
    get_tree_normals(a.angle, na);
    get_tree_normals(b.angle, nb);

    std::array<AABB, TREE_NUM_TRIANGLES> taabb_a{};
    std::array<AABB, TREE_NUM_TRIANGLES> taabb_b{};
    for (size_t i = 0; i < TREE_NUM_TRIANGLES; ++i) {
        taabb_a[i] = triangle_aabb(fa.triangles[i]);
        taabb_b[i] = triangle_aabb(fb.triangles[i]);
    }

    for (size_t i = 0; i < TREE_NUM_TRIANGLES; ++i) {
        if (!taabb_a[i].intersects(aabb_b)) {
            continue;
        }
        for (size_t j = 0; j < TREE_NUM_TRIANGLES; ++j) {
            if (!taabb_b[j].intersects(aabb_a)) {
                continue;
            }
            if (!taabb_a[i].intersects(taabb_b[j])) {
                continue;
            }
            const float score = triangles_intersection_score_from_normals(
                fa.triangles[i],
                fb.triangles[j],
                na[i],
                nb[j],
                eps
            );
            scores[i][j] = score;
        }
    }

    return scores;
}

}  // namespace tree_packing

