#pragma once

#include "types.hpp"
#include "tree.hpp"
#include "../spatial/grid2d.hpp"
#include <memory>
#include <unordered_map>
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
    [[nodiscard]] const Grid2D& grid() const { return grid_; }
    [[nodiscard]] bool is_valid(size_t i) const { return valid_[i]; }

    // Update solution with new params for specified indices
    // Returns a new solution with updated cache
    [[nodiscard]] Solution update(const TreeParamsSoA& new_params, const std::vector<int>& indices) const;

    // Number of trees
    [[nodiscard]] size_t size() const { return params_.size(); }

    // Count missing (NaN) trees
    [[nodiscard]] int n_missing() const { return params_.count_nan(); }

    // Regularization term (max absolute position)
    [[nodiscard]] float reg() const;

    // Set tree to NaN
    void set_nan(size_t i);

    // Recompute all cached data
    void recompute_cache();

private:
    TreeParamsSoA params_;
    std::vector<Figure> figures_;
    std::vector<Vec2> centers_;
    std::vector<AABB> aabbs_;
    std::vector<char> valid_;
    Grid2D grid_;

    void update_cache_for(size_t i);
};

// Evaluated solution with objective and constraints
struct SolutionEval {
    Solution solution;
    float objective{0.0f};

    // Constraint evaluations
    float intersection_violation{0.0f};
    float bounds_violation{0.0f};

    // Intersection map (cached for incremental updates)
    std::vector<std::unordered_map<int, float>> intersection_map;

    // Cached bounds for incremental objective updates
    int valid_count{0};
    float min_x{0.0f};
    float max_x{0.0f};
    float min_y{0.0f};
    float max_y{0.0f};
    int min_x_idx{-1};
    int max_x_idx{-1};
    int min_y_idx{-1};
    int max_y_idx{-1};

    [[nodiscard]] float total_violation() const {
        return intersection_violation + bounds_violation;
    }

    [[nodiscard]] int n_missing() const {
        return solution.n_missing();
    }

    [[nodiscard]] float reg() const {
        return solution.reg();
    }
};

}  // namespace tree_packing
