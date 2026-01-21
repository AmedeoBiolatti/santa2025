#include "tree_packing/core/solution.hpp"
#include "tree_packing/random/rng.hpp"
#include <cmath>
#include <algorithm>

namespace tree_packing {
namespace {
AABB compute_aabb(const Figure& fig) {
    AABB aabb;
    for (const auto& tri : fig.triangles) {
        aabb.expand(tri);
    }
    return aabb;
}
}  // namespace

Solution::Solution(size_t num_trees) {
    params_.resize(num_trees);
    figures_.resize(num_trees);
    centers_.resize(num_trees);
    aabbs_.resize(num_trees);
    valid_.resize(num_trees, false);
}

Solution Solution::init_random(size_t num_trees, float side, uint64_t seed) {
    Solution sol(num_trees);
    RNG rng(seed);

    float half = side / 2.0f;
    for (size_t i = 0; i < num_trees; ++i) {
        sol.params_.x[i] = rng.uniform(-half, half);
        sol.params_.y[i] = rng.uniform(-half, half);
        sol.params_.angle[i] = rng.uniform(-PI, PI);
        sol.valid_[i] = true;
    }

    sol.recompute_cache();
    return sol;
}

Solution Solution::init_empty(size_t num_trees) {
    Solution sol(num_trees);

    for (size_t i = 0; i < num_trees; ++i) {
        sol.params_.set_nan(i);
    }

    sol.recompute_cache();
    return sol;
}

Solution Solution::init(const TreeParamsSoA& params, int grid_n, float grid_size, int grid_capacity) {
    Solution sol;
    sol.params_ = params;
    sol.figures_.resize(params.size());
    sol.centers_.resize(params.size());
    sol.aabbs_.resize(params.size());
    sol.valid_.resize(params.size(), false);

    // Compute figures and centers
    params_to_figures(sol.params_, sol.figures_);
    get_tree_centers(sol.params_, sol.centers_);
    for (size_t i = 0; i < params.size(); ++i) {
        sol.valid_[i] = !sol.params_.is_nan(i);
        if (sol.valid_[i]) {
            sol.aabbs_[i] = compute_aabb(sol.figures_[i]);
        } else {
            sol.aabbs_[i] = AABB{};
        }
    }

    // Initialize grid
    sol.grid_ = Grid2D::init(sol.centers_, grid_n, grid_capacity, grid_size);

    return sol;
}

void Solution::set_params(size_t i, const TreeParams& p) {
    params_.set(i, p);
    update_cache_for(i);
}

void Solution::set_nan(size_t i) {
    params_.set_nan(i);
    valid_[i] = false;
    aabbs_[i] = AABB{};
    // Update figure to NaN
    for (auto& tri : figures_[i].triangles) {
        tri.v0 = Vec2::nan();
        tri.v1 = Vec2::nan();
        tri.v2 = Vec2::nan();
    }
    centers_[i] = Vec2::nan();
    grid_.update(static_cast<int>(i), Vec2::nan());
}

Solution Solution::update(const TreeParamsSoA& new_params, const std::vector<int>& indices) const {
    Solution sol = *this;
    sol.params_ = new_params;

    for (int idx : indices) {
        if (idx >= 0 && static_cast<size_t>(idx) < sol.size()) {
            sol.update_cache_for(static_cast<size_t>(idx));
        }
    }

    return sol;
}

void Solution::recompute_cache() {
    params_to_figures(params_, figures_);
    get_tree_centers(params_, centers_);
    if (valid_.size() != params_.size()) {
        valid_.resize(params_.size(), false);
    }
    if (aabbs_.size() != params_.size()) {
        aabbs_.resize(params_.size());
    }
    for (size_t i = 0; i < params_.size(); ++i) {
        valid_[i] = !params_.is_nan(i);
        if (valid_[i]) {
            aabbs_[i] = compute_aabb(figures_[i]);
        } else {
            aabbs_[i] = AABB{};
        }
    }
    grid_ = Grid2D::init(centers_, 16, 8, THR);
}

void Solution::update_cache_for(size_t i) {
    TreeParams p = params_.get(i);
    figures_[i] = params_to_figure(p);
    centers_[i] = get_tree_center(p);
    valid_[i] = !params_.is_nan(i);
    if (valid_[i]) {
        aabbs_[i] = compute_aabb(figures_[i]);
    } else {
        aabbs_[i] = AABB{};
    }
    grid_.update(static_cast<int>(i), centers_[i]);
}

float Solution::reg() const {
    float max_abs = 0.0f;
    for (size_t i = 0; i < params_.size(); ++i) {
        if (!params_.is_nan(i)) {
            max_abs = std::max(max_abs, std::abs(params_.x[i]));
            max_abs = std::max(max_abs, std::abs(params_.y[i]));
        }
    }
    return max_abs;
}

}  // namespace tree_packing
