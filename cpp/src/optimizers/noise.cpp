#include "tree_packing/optimizers/noise.hpp"
#include <iostream>
#include <vector>

namespace tree_packing {

NoiseOptimizer::NoiseOptimizer(float noise_level, int n_change, bool verbose)
    : noise_level_(noise_level)
    , n_change_(n_change)
    , verbose_(verbose)
{}

std::any NoiseOptimizer::init_state(const SolutionEval& solution) {
    NoiseState state;
    state.indices.reserve(solution.solution.size());
    state.selected.reserve(8);
    state.last_indices.reserve(8);
    state.new_params.reserve(8);
    return state;
}

void NoiseOptimizer::apply(
    SolutionEval& solution,
    std::any& state,
    GlobalState& global_state,
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
    int k = n_change_;
    if (k <= 0) {
        return;
    }
    if (k > static_cast<int>(indices.size())) {
        k = static_cast<int>(indices.size());
    }

    auto& selected = noise_state->selected;
    rng.choice(static_cast<int>(indices.size()), k, selected);

    auto& last_indices = noise_state->last_indices;
    last_indices.clear();
    auto& new_params = noise_state->new_params;
    new_params.resize(static_cast<size_t>(k));

    // Push updates to stack for rollback support
    auto& stack = global_state.update_stack();

    for (int i = 0; i < k; ++i) {
        int idx = indices[static_cast<size_t>(selected[static_cast<size_t>(i)])];
        TreeParams p = solution.solution.get_params(static_cast<size_t>(idx));

        // Push the old params to the update stack
        stack.push_update(idx, p);
        last_indices.push_back(idx);

        float dx = noise_level_ * rng.uniform(-1.0f, 1.0f);
        float dy = noise_level_ * rng.uniform(-1.0f, 1.0f);
        float da = noise_level_ * rng.uniform(-1.0f, 1.0f);
        p.pos.x += dx;
        p.pos.y += dy;
        p.angle += da;
        new_params.set(static_cast<size_t>(i), p);
    }

    problem_->update_and_eval(solution, last_indices, new_params);

    if (verbose_) {
        std::cout << "[NoiseOptimizer] n=" << k << "\n";
    }
}

OptimizerPtr NoiseOptimizer::clone() const {
    return std::make_unique<NoiseOptimizer>(*this);
}

}  // namespace tree_packing
