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

    int n_valid = static_cast<int>(n) - solution.solution.n_missing();
    if (n_valid <= 0) {
        return;
    }
    int k = n_change_;
    if (k <= 0) {
        return;
    }
    if (k > n_valid) {
        k = n_valid;
    }

    auto& selected = noise_state->selected;
    selected.clear();
    selected.reserve(static_cast<size_t>(k));

    // Use rejection sampling O(k) instead of permutation O(n)
    int max_attempts = k * 10 + 20;  // Expected ~k attempts when all valid
    int attempts = 0;
    while (static_cast<int>(selected.size()) < k && attempts < max_attempts) {
        int idx = rng.randint(0, static_cast<int>(n) - 1);
        if (solution.solution.is_valid(static_cast<size_t>(idx))) {
            bool dup = false;
            for (int s : selected) {
                if (s == idx) { dup = true; break; }
            }
            if (!dup) {
                selected.push_back(idx);
            }
        }
        ++attempts;
    }
    if (selected.empty()) {
        return;
    }
    k = static_cast<int>(selected.size());

    auto& last_indices = noise_state->last_indices;
    last_indices.clear();
    auto& new_params = noise_state->new_params;
    new_params.resize(static_cast<size_t>(k));

    // Push updates to stack for rollback support
    auto& stack = global_state.update_stack();

    for (int i = 0; i < k; ++i) {
        int idx = selected[static_cast<size_t>(i)];
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
