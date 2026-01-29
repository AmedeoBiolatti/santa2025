#pragma once

#include "../core/tree.hpp"

namespace tree_packing {

// Compute per-triangle AABBs and the figure AABB using convex hull vertices.
inline void compute_triangle_aabbs_and_figure_aabb(
    const Figure& figure,
    std::array<AABB, TREE_NUM_TRIANGLES>& tri_aabbs,
    AABB& figure_aabb
) {
    for (size_t i = 0; i < TREE_NUM_TRIANGLES; ++i) {
        const auto& tri = figure[i];
        tri_aabbs[i] = AABB();
        tri_aabbs[i].expand(tri.v0);
        tri_aabbs[i].expand(tri.v1);
        tri_aabbs[i].expand(tri.v2);
    }

    figure_aabb = AABB();
    for (const auto& [ti, vi] : CONVEX_HULL_INDEXES) {
        const auto& tri = figure[static_cast<size_t>(ti)];
        const Vec2& v = (vi == 0) ? tri.v0 : (vi == 1) ? tri.v1 : tri.v2;
        figure_aabb.expand(v);
    }
}

}  // namespace tree_packing

