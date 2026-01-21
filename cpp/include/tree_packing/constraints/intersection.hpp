#pragma once

#include "../core/solution.hpp"
#include "../spatial/grid2d.hpp"
#include <unordered_map>
#include <vector>

namespace tree_packing {

// Intersection constraint: penalizes overlapping trees
class IntersectionConstraint {
public:
    IntersectionConstraint() = default;

    // Full evaluation (compute sparse intersection map)
    [[nodiscard]] float eval(
        const Solution& solution,
        std::vector<std::unordered_map<int, float>>& map
    ) const;

    // Incremental evaluation (update map for modified indices)
    [[nodiscard]] float eval_update(
        const Solution& solution,
        const Solution& prev_solution,
        std::vector<std::unordered_map<int, float>>& map,
        const std::vector<int>& modified_indices,
        float prev_total
    ) const;

private:
    // Compute intersection score between two figures using spatial grid
    [[nodiscard]] float compute_pair_score(
        const Figure& f0,
        const Figure& f1,
        const Vec2& c0,
        const Vec2& c1
    ) const;
};

}  // namespace tree_packing
