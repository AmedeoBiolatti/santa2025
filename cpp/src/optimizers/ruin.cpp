#include "tree_packing/optimizers/ruin.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace tree_packing {

// RandomRuin implementation
RandomRuin::RandomRuin(int n_remove) : n_remove_(n_remove) {
    if (n_remove_ < 1) n_remove_ = 1;
}

std::pair<SolutionEval, std::any> RandomRuin::apply(
    const SolutionEval& solution,
    std::any& state,
    GlobalState& global_state,
    RNG& rng
) {
    const auto& params = solution.solution.params();
    size_t n = params.size();

    // Select random indices to remove
    auto indices = rng.choice(static_cast<int>(n), n_remove_);

    // Create new params with NaN for removed trees
    TreeParamsSoA new_params = params;
    for (int idx : indices) {
        new_params.set_nan(idx);
    }

    // Create new solution with updated params
    Solution new_sol = solution.solution.update(new_params, indices);

    // Evaluate the new solution incrementally
    SolutionEval new_eval = problem_->eval_update(new_sol, solution, indices);

    return {new_eval, state};
}

OptimizerPtr RandomRuin::clone() const {
    return std::make_unique<RandomRuin>(*this);
}

// SpatialRuin implementation
SpatialRuin::SpatialRuin(int n_remove) : n_remove_(n_remove) {
    if (n_remove_ < 1) n_remove_ = 1;
}

Vec2 SpatialRuin::random_point(const TreeParamsSoA& params, RNG& rng) {
    float min_x = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float min_y = std::numeric_limits<float>::max();
    float max_y = std::numeric_limits<float>::lowest();

    bool has_valid = false;
    for (size_t i = 0; i < params.size(); ++i) {
        if (!params.is_nan(i)) {
            min_x = std::min(min_x, params.x[i]);
            max_x = std::max(max_x, params.x[i]);
            min_y = std::min(min_y, params.y[i]);
            max_y = std::max(max_y, params.y[i]);
            has_valid = true;
        }
    }

    if (!has_valid) {
        return Vec2{0.0f, 0.0f};
    }

    return Vec2{rng.uniform(min_x, max_x), rng.uniform(min_y, max_y)};
}

std::vector<int> SpatialRuin::select_closest(const TreeParamsSoA& params, const Vec2& point) {
    std::vector<std::pair<float, int>> distances;
    distances.reserve(params.size());

    for (size_t i = 0; i < params.size(); ++i) {
        if (!params.is_nan(i)) {
            float dx = params.x[i] - point.x;
            float dy = params.y[i] - point.y;
            float dist2 = dx * dx + dy * dy;
            distances.emplace_back(dist2, static_cast<int>(i));
        }
    }

    // Sort by distance
    std::sort(distances.begin(), distances.end());

    // Select closest n_remove
    std::vector<int> result;
    int count = std::min(n_remove_, static_cast<int>(distances.size()));
    for (int i = 0; i < count; ++i) {
        result.push_back(distances[i].second);
    }

    return result;
}

std::pair<SolutionEval, std::any> SpatialRuin::apply(
    const SolutionEval& solution,
    std::any& state,
    GlobalState& global_state,
    RNG& rng
) {
    const auto& params = solution.solution.params();

    // Select random point and find closest trees
    Vec2 point = random_point(params, rng);
    auto indices = select_closest(params, point);

    // Create new params with NaN for removed trees
    TreeParamsSoA new_params = params;
    for (int idx : indices) {
        new_params.set_nan(idx);
    }

    // Create new solution with updated params
    Solution new_sol = solution.solution.update(new_params, indices);

    // Evaluate the new solution incrementally
    SolutionEval new_eval = problem_->eval_update(new_sol, solution, indices);

    return {new_eval, state};
}

OptimizerPtr SpatialRuin::clone() const {
    return std::make_unique<SpatialRuin>(*this);
}

}  // namespace tree_packing
