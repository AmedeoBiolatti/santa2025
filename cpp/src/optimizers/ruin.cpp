#include "tree_packing/optimizers/ruin.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

namespace tree_packing {

// RandomRuin implementation
RandomRuin::RandomRuin(int n_remove, bool verbose)
    : n_remove_(n_remove), verbose_(verbose) {
    if (n_remove_ < 1) n_remove_ = 1;
}

std::any RandomRuin::init_state(const SolutionEval& solution) {
    RuinState state;
    state.indices.reserve(static_cast<size_t>(n_remove_));
    return state;
}

void RandomRuin::apply(
    const SolutionEval& solution,
    std::any& state,
    GlobalState& global_state,
    RNG& rng,
    SolutionEval& out
) {
    if (out.solution.revision() != solution.solution.revision()) {
        out.solution.copy_from(solution.solution);
    }
    const auto& params = out.solution.params();
    size_t n = params.size();

    // Select random indices to remove
    auto* ruin_state = std::any_cast<RuinState>(&state);
    if (!ruin_state) {
        state = init_state(solution);
        ruin_state = std::any_cast<RuinState>(&state);
    }
    auto& indices = ruin_state->indices;
    rng.choice(static_cast<int>(n), n_remove_, indices);

    // Create new solution with updated params
    for (int idx : indices) {
        out.solution.set_nan(static_cast<size_t>(idx));
    }

    // Evaluate the new solution
    problem_->eval_inplace(out.solution, out);

    if (verbose_) {
        std::cout << "[RandomRuin] removed=" << indices.size() << "\n";
    }
}

OptimizerPtr RandomRuin::clone() const {
    return std::make_unique<RandomRuin>(*this);
}

// SpatialRuin implementation
SpatialRuin::SpatialRuin(int n_remove, bool verbose)
    : n_remove_(n_remove), verbose_(verbose) {
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

void SpatialRuin::select_closest(
    const TreeParamsSoA& params,
    const Vec2& point,
    std::vector<int>& out,
    std::vector<std::pair<float, int>>& distances
) {
    distances.clear();
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
    out.clear();
    int count = std::min(n_remove_, static_cast<int>(distances.size()));
    for (int i = 0; i < count; ++i) {
        out.push_back(distances[i].second);
    }
}

std::any SpatialRuin::init_state(const SolutionEval& solution) {
    RuinState state;
    state.indices.reserve(static_cast<size_t>(n_remove_));
    state.distances.reserve(solution.solution.size());
    return state;
}

void SpatialRuin::apply(
    const SolutionEval& solution,
    std::any& state,
    GlobalState& global_state,
    RNG& rng,
    SolutionEval& out
) {
    if (out.solution.revision() != solution.solution.revision()) {
        out.solution.copy_from(solution.solution);
    }
    const auto& params = out.solution.params();

    // Select random point and find closest trees
    Vec2 point = random_point(params, rng);
    auto* ruin_state = std::any_cast<RuinState>(&state);
    if (!ruin_state) {
        state = init_state(solution);
        ruin_state = std::any_cast<RuinState>(&state);
    }
    auto& indices = ruin_state->indices;
    select_closest(params, point, indices, ruin_state->distances);

    // Create new solution with updated params
    for (int idx : indices) {
        out.solution.set_nan(static_cast<size_t>(idx));
    }

    // Evaluate the new solution
    problem_->eval_inplace(out.solution, out);

    if (verbose_) {
        std::cout << "[SpatialRuin] removed=" << indices.size()
                  << " point=(" << point.x << "," << point.y << ")\n";
    }
}

OptimizerPtr SpatialRuin::clone() const {
    return std::make_unique<SpatialRuin>(*this);
}

}  // namespace tree_packing
