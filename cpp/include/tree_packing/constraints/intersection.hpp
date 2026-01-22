#pragma once

#include "../core/solution.hpp"
#include "../spatial/grid2d.hpp"
#include <vector>

namespace tree_packing {

// Intersection constraint: penalizes overlapping trees
class IntersectionConstraint {
public:
    IntersectionConstraint() = default;

    // Full evaluation (compute sparse intersection map)
    [[nodiscard]] float eval(
        const Solution& solution,
        SolutionEval::IntersectionMap& map,
        int* out_count = nullptr
    ) const;

    // Incremental evaluation (update map for modified indices)
    [[nodiscard]] float eval_update(
        const Solution& solution,
        SolutionEval::IntersectionMap& map,
        const std::vector<int>& modified_indices,
        float prev_total,
        int prev_count,
        int* out_count = nullptr
    ) const;

private:
    // Compute intersection score between two figures using spatial grid
    [[nodiscard]] float compute_pair_score(
        const Figure& f0,
        const Figure& f1,
        const Vec2& c0,
        const Vec2& c1
    ) const;

    // Compute intersection score using cached triangle normals
    [[nodiscard]] float compute_pair_score_from_normals(
        const Figure& f0,
        const Figure& f1,
        const std::array<std::array<Vec2, 3>, TREE_NUM_TRIANGLES>& n0,
        const std::array<std::array<Vec2, 3>, TREE_NUM_TRIANGLES>& n1,
        const Vec2& c0,
        const Vec2& c1
    ) const;
};

}  // namespace tree_packing
