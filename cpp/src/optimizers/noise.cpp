#include "tree_packing/optimizers/noise.hpp"
#include <iostream>
#include <vector>

namespace tree_packing {

NoiseOptimizer::NoiseOptimizer(float noise_level, bool verbose)
    : noise_level_(noise_level)
    , verbose_(verbose)
{}

std::any NoiseOptimizer::init_state(const SolutionEval& solution) {
    NoiseState state;
    state.indices.reserve(solution.solution.size());
    state.last_idx = -1;
    state.has_last = false;
    return state;
}

void NoiseOptimizer::apply(
    SolutionEval& solution,
    std::any& state,
    GlobalState&,
    RNG& rng
) {
    auto* noise_state = std::any_cast<NoiseState>(&state);
    if (!noise_state) {
        state = init_state(solution);
        noise_state = std::any_cast<NoiseState>(&state);
    }

    size_t n = solution.solution.size();
    if (n == 0) {
        return;
    }

    auto& indices = noise_state->indices;
    indices.clear();
    indices.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if (solution.solution.is_valid(i)) {
            indices.push_back(static_cast<int>(i));
        }
    }
    if (indices.empty()) {
        return;
    }
    int choice = rng.randint(0, static_cast<int>(indices.size() - 1));
    int idx = indices[static_cast<size_t>(choice)];

    float dx = noise_level_ * rng.uniform(-1.0f, 1.0f);
    float dy = noise_level_ * rng.uniform(-1.0f, 1.0f);
    float da = noise_level_ * rng.uniform(-1.0f, 1.0f);

    TreeParams p = solution.solution.get_params(static_cast<size_t>(idx));
    noise_state->last_idx = idx;
    noise_state->last_params = p;
    noise_state->has_last = true;
    p.pos.x += dx;
    p.pos.y += dy;
    p.angle += da;

    std::vector<int> modified{idx};
    TreeParamsSoA new_params(1);
    new_params.set(0, p);
    problem_->update_and_eval(solution, modified, new_params);

    if (verbose_) {
        std::cout << "[NoiseOptimizer] idx=" << idx
                  << " dx=" << dx << " dy=" << dy << " da=" << da << "\n";
    }
}

void NoiseOptimizer::rollback(SolutionEval& solution, std::any& state) {
    auto* noise_state = std::any_cast<NoiseState>(&state);
    if (!noise_state || !noise_state->has_last || noise_state->last_idx < 0) {
        return;
    }
    TreeParamsSoA params(1);
    params.set(0, noise_state->last_params);
    std::vector<int> indices{noise_state->last_idx};
    problem_->update_and_eval(solution, indices, params);
    noise_state->has_last = false;
}

OptimizerPtr NoiseOptimizer::clone() const {
    return std::make_unique<NoiseOptimizer>(*this);
}

}  // namespace tree_packing
