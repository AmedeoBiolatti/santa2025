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
    state.scratch.copy_from(solution.solution);
    state.indices.reserve(1);
    return state;
}

void NoiseOptimizer::apply(
    const SolutionEval& solution,
    std::any& state,
    GlobalState&,
    RNG& rng,
    SolutionEval& out
) {
    auto* noise_state = std::any_cast<NoiseState>(&state);
    if (!noise_state) {
        state = init_state(solution);
        noise_state = std::any_cast<NoiseState>(&state);
    }

    const auto& params = solution.solution.params();
    size_t n = params.size();
    if (n == 0) {
        out = solution;
        return;
    }

    int idx = rng.randint(0, static_cast<int>(n - 1));

    float dx = noise_level_ * rng.uniform(-1.0f, 1.0f);
    float dy = noise_level_ * rng.uniform(-1.0f, 1.0f);
    float da = noise_level_ * rng.uniform(-1.0f, 1.0f);

    auto& scratch = noise_state->scratch;
    scratch.copy_from(solution.solution);
    TreeParams p = scratch.get_params(static_cast<size_t>(idx));
    p.pos.x += dx;
    p.pos.y += dy;
    p.angle += da;
    scratch.set_params(static_cast<size_t>(idx), p);

    auto& indices = noise_state->indices;
    indices.clear();
    indices.push_back(idx);
    problem_->eval_update_inplace(scratch, solution, indices, out);

    if (verbose_) {
        std::cout << "[NoiseOptimizer] idx=" << idx
                  << " dx=" << dx << " dy=" << dy << " da=" << da << "\n";
    }
}

OptimizerPtr NoiseOptimizer::clone() const {
    return std::make_unique<NoiseOptimizer>(*this);
}

}  // namespace tree_packing
