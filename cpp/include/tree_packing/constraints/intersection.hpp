#pragma once

#include "../core/solution.hpp"
#include "../spatial/grid2d.hpp"
#include "../spatial/figure_hash2d.hpp"
#include <vector>
#include <set>

namespace tree_packing {

class IntersectionTreeFilter;

// Intersection constraint: penalizes overlapping trees
class IntersectionConstraint {
public:
    IntersectionConstraint() { init(); }

    // Full evaluation (compute sparse intersection map)
    [[nodiscard]] double eval(
        const Solution& solution,
        SolutionEval::IntersectionMap& map,
        int* out_count = nullptr
    ) const;

    // Incremental evaluation (update map for modified indices)
    [[nodiscard]] double eval_update(
        const Solution& solution,
        SolutionEval::IntersectionMap& map,
        const std::vector<int>& modified_indices,
        double prev_total,
        int prev_count,
        int* out_count = nullptr
    ) const;

    // Incremental evaluation for removals only (no recomputation for removed indices)
    [[nodiscard]] double eval_remove(
        const Solution& solution,
        SolutionEval::IntersectionMap& map,
        const std::vector<int>& removed_indices,
        double prev_total,
        int prev_count,
        int* out_count = nullptr
    ) const;

    // ============ Figure Hash Based Methods ============

    // Full evaluation using figure hash (spatial hashing at figure level)
    [[nodiscard]] double eval_figure_hash(
        const Solution& solution,
        SolutionEval::IntersectionMap& map,
        int* out_count = nullptr
    ) const;

    // Incremental evaluation using figure hash
    [[nodiscard]] double eval_update_figure_hash(
        const Solution& solution,
        SolutionEval::IntersectionMap& map,
        const std::vector<int>& modified_indices,
        double prev_total,
        int prev_count,
        int* out_count = nullptr
    ) const;

    // Incremental removal using figure hash
    [[nodiscard]] double eval_remove_figure_hash(
        const Solution& solution,
        SolutionEval::IntersectionMap& map,
        const std::vector<int>& removed_indices,
        double prev_total,
        int prev_count,
        int* out_count = nullptr
    ) const;

    void init();

    void set_tree_filter(const IntersectionTreeFilter* filter) {
        tree_filter_ = filter;
    }

private:
    mutable std::vector<Index> candidates_;
    mutable std::set<size_t> modified_;
    const IntersectionTreeFilter* tree_filter_{nullptr};

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
        const AABB& aabb0,
        const AABB& aabb1,
        const Vec2& c0,
        const Vec2& c1
    ) const;

    [[nodiscard]] float compute_pair_score_from_normals_per_triangle(
        const Solution& solution,
        size_t idx0,
        size_t idx1,
        const Figure& f0,
        const Figure& f1,
        const std::array<std::array<Vec2, 3>, TREE_NUM_TRIANGLES>& n0,
        const std::array<std::array<Vec2, 3>, TREE_NUM_TRIANGLES>& n1,
        const std::array<AABB, TREE_NUM_TRIANGLES>& taabb0,
        const std::array<AABB, TREE_NUM_TRIANGLES>& taabb1,
        const AABB& aabb0,
        const AABB& aabb1,
        const Vec2& c0,
        const Vec2& c1
    ) const;
};

}  // namespace tree_packing
