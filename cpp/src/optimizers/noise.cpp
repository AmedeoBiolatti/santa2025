#include "tree_packing/optimizers/noise.hpp"
#include <vector>

namespace tree_packing {

NoiseOptimizer::NoiseOptimizer(float noise_level)
    : noise_level_(noise_level)
{}

std::pair<SolutionEval, std::any> NoiseOptimizer::apply(
    const SolutionEval& solution,
    std::any& state,
    GlobalState&,
    RNG& rng
) {
    const auto& params = solution.solution.params();
    size_t n = params.size();
    if (n == 0) {
        return {solution, state};
    }

    int idx = rng.randint(0, static_cast<int>(n - 1));

    float dx = noise_level_ * rng.uniform(-1.0f, 1.0f);
    float dy = noise_level_ * rng.uniform(-1.0f, 1.0f);
    float da = noise_level_ * rng.uniform(-1.0f, 1.0f);

    TreeParamsSoA new_params = params;
    new_params.x[idx] += dx;
    new_params.y[idx] += dy;
    new_params.angle[idx] += da;

    std::vector<int> indices{idx};
    Solution new_sol = solution.solution.update(new_params, indices);
    SolutionEval new_eval = problem_->eval_update(new_sol, solution, indices);

    return {new_eval, state};
}

OptimizerPtr NoiseOptimizer::clone() const {
    return std::make_unique<NoiseOptimizer>(*this);
}

}  // namespace tree_packing
