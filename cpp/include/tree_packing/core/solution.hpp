#pragma once

#include "types.hpp"
#include "tree.hpp"
#include "../spatial/grid2d.hpp"
#include "../spatial/figure_hash2d.hpp"
#include <algorithm>
#include <atomic>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace tree_packing {

// Forward declaration
class Problem;
struct SolutionEval;

// Solution stores tree parameters and cached derived data
class Solution {
public:
    Solution() = default;
    explicit Solution(size_t num_trees);

    // Initialize with random positions and angles
    static Solution init_random(size_t num_trees, float side = 10.0f, uint64_t seed = 42);

    // Initialize with empty (NaN) positions
    static Solution init_empty(size_t num_trees);

    // Initialize from params with grid
    static Solution init(const TreeParamsSoA& params, int grid_n = 16, float grid_size = THR, int grid_capacity = 8);

    // Get/set params
    [[nodiscard]] const TreeParamsSoA& params() const { return params_; }
    TreeParamsSoA& params() { return params_; }

    [[nodiscard]] TreeParams get_params(size_t i) const { return params_.get(i); }
    void set_params(size_t i, const TreeParams& p);

    // Get cached data
    [[nodiscard]] const std::vector<Figure>& figures() const { return figures_; }
    [[nodiscard]] const std::vector<Vec2>& centers() const { return centers_; }
    [[nodiscard]] const std::vector<AABB>& aabbs() const { return aabbs_; }
    [[nodiscard]] const std::vector<std::array<AABB, TREE_NUM_TRIANGLES>>& triangle_aabbs() const {
        return triangle_aabbs_;
    }
    [[nodiscard]] const std::vector<float>& max_abs() const { return max_abs_; }
    [[nodiscard]] const std::vector<std::array<std::array<Vec2, 3>, TREE_NUM_TRIANGLES>>& normals() const { return normals_; }
    [[nodiscard]] float max_abs(size_t i) const { return max_abs_[i]; }
    [[nodiscard]] float max_max_abs() const { return max_max_abs_; }
    [[nodiscard]] size_t max_max_abs_idx() const { return max_max_abs_idx_; }
    [[nodiscard]] const Grid2D& grid() const { return grid_; }
    [[nodiscard]] const FigureHash2D& figure_hash() const { return figure_hash_; }
    [[nodiscard]] bool is_valid(size_t i) const { return valid_[i]; }

    // Update solution with new params for specified indices
    // Returns a new solution with updated cache
    [[nodiscard]] Solution update(const TreeParamsSoA& new_params, const std::vector<int>& indices) const;

    // Number of trees
    [[nodiscard]] size_t size() const { return params_.size(); }

    // Count missing (NaN) trees
    [[nodiscard]] int n_missing() const { return missing_count_; }

    // Get indices of removed (invalid) trees - O(1), no iteration needed
    [[nodiscard]] const std::vector<int>& removed_indices() const { return removed_indices_; }

    // Regularization term (max absolute position)
    [[nodiscard]] float reg() const;

    // Set tree to NaN
    void set_nan(size_t i);

    // Recompute all cached data
    void recompute_cache();

    // Copy data from another solution without changing structure
    void copy_from(const Solution& other);

    // Validate cached data against params (optional grid check)
    [[nodiscard]] bool validate_cache(float tol = 1e-5f, bool check_grid = false) const;

protected:
    friend class Problem;
    TreeParamsSoA params_;
    std::vector<Figure> figures_;
    std::vector<Vec2> centers_;
    std::vector<AABB> aabbs_;
    std::vector<std::array<AABB, TREE_NUM_TRIANGLES>> triangle_aabbs_;
    std::vector<float> max_abs_;
    std::vector<std::array<std::array<Vec2, 3>, TREE_NUM_TRIANGLES>> normals_;
    float max_max_abs_{0.0f};
    size_t max_max_abs_idx_{static_cast<size_t>(-1)};
    int64_t reg_sum_int_{0};  // Cached sum of (int)(1000 * max_abs_) for valid trees
    std::vector<char> valid_;
    std::vector<int> removed_indices_;  // Tracks invalid tree indices (order doesn't matter)
    int missing_count_{0};
    Grid2D grid_;
    FigureHash2D figure_hash_;

    void update_cache_for(size_t i);
    void update_cache_for(size_t i, bool new_valid);
    void update_cache_on_removal_for(size_t i);
    void update_cache_on_insertion_for(size_t i);
    void update_cache_on_update_for(size_t i, const TreeParams& p);
    int update_cache_on_transition(size_t i, bool new_valid);
};

// Evaluated solution with objective and constraints
struct SolutionEval {
    Solution solution;
    float objective{0.0f};

    // Constraint evaluations
    double intersection_violation{0.0};
    double bounds_violation{0.0};
    int intersection_count{0};

    // Intersection map (cached for incremental updates)
    // Stores each pair (i,j) in both directions for O(1) removal
    struct IntersectionEntry {
        Index neighbor{-1};
        float score{0.0f};
        int back_index{-1};  // Index in neighbor's list pointing back
    };
    using IntersectionList = std::vector<IntersectionEntry>;
    using IntersectionMap = std::vector<IntersectionList>;
    IntersectionMap intersection_map;

    // Cached bounds for incremental objective updates
    int16_t valid_count{0};
    float min_x{0.0f};
    float max_x{0.0f};
    float min_y{0.0f};
    float max_y{0.0f};
    Index min_x_idx{-1};
    Index max_x_idx{-1};
    Index min_y_idx{-1};
    Index max_y_idx{-1};
    mutable float score_cache_mu{std::numeric_limits<float>::quiet_NaN()};
    mutable float score_cache_value{0.0f};
    mutable bool score_cache_valid{false};

    void copy_from(const SolutionEval& other) {
        solution.copy_from(other.solution);
        objective = other.objective;
        intersection_violation = other.intersection_violation;
        bounds_violation = other.bounds_violation;
        intersection_count = other.intersection_count;
        valid_count = other.valid_count;
        min_x = other.min_x;
        max_x = other.max_x;
        min_y = other.min_y;
        max_y = other.max_y;
        min_x_idx = other.min_x_idx;
        max_x_idx = other.max_x_idx;
        min_y_idx = other.min_y_idx;
        max_y_idx = other.max_y_idx;

        intersection_map = other.intersection_map;
    }

    [[nodiscard]] double total_violation() const {
        double v = intersection_violation + bounds_violation;
        return v > 0.0 ? v : 0.0;
    }

    [[nodiscard]] int n_missing() const {
        return solution.n_missing();
    }

    [[nodiscard]] float reg() const {
        return solution.reg();
    }
};

}  // namespace tree_packing
