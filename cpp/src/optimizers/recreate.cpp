#include "tree_packing/optimizers/recreate.hpp"
#include "tree_packing/core/tree.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace tree_packing {

RandomRecreate::RandomRecreate(int max_recreate, float box_size, float delta)
    : max_recreate_(max_recreate), box_size_(box_size), delta_(delta)
{
    if (max_recreate_ < 1) max_recreate_ = 1;
}

std::any RandomRecreate::init_state(const SolutionEval& solution) {
    return RandomRecreateState{0};
}

std::vector<int> RandomRecreate::get_removed_indices(const TreeParamsSoA& params) {
    std::vector<int> indices;
    for (size_t i = 0; i < params.size(); ++i) {
        if (params.is_nan(i)) {
            indices.push_back(static_cast<int>(i));
        }
    }
    return indices;
}

std::pair<SolutionEval, std::any> RandomRecreate::apply(
    const SolutionEval& solution,
    std::any& state,
    GlobalState& global_state,
    RNG& rng
) {
    const auto& params = solution.solution.params();
    const auto& figures = solution.solution.figures();

    // Find removed (NaN) indices
    auto removed = get_removed_indices(params);

    if (removed.empty()) {
        // Nothing to recreate
        auto rec_state = std::any_cast<RandomRecreateState>(state);
        rec_state.iteration++;
        return {solution, rec_state};
    }

    // Compute current bounds from valid trees
    float max_abs = 0.0f;
    for (const auto& fig : figures) {
        if (fig.is_nan()) continue;
        for (const auto& tri : fig.triangles) {
            max_abs = std::max({max_abs,
                std::abs(tri.v0.x), std::abs(tri.v0.y),
                std::abs(tri.v1.x), std::abs(tri.v1.y),
                std::abs(tri.v2.x), std::abs(tri.v2.y)});
        }
    }

    // If no valid trees, use default box
    if (max_abs == 0.0f) {
        max_abs = box_size_ / 2.0f;
    }

    float minval = -max_abs - delta_;
    float maxval = max_abs + delta_;

    // Select indices to recreate (up to max_recreate)
    int n_recreate = std::min(max_recreate_, static_cast<int>(removed.size()));

    // Shuffle removed indices and take first n_recreate
    std::vector<int> indices(removed.begin(), removed.end());
    for (int i = static_cast<int>(indices.size()) - 1; i > 0; --i) {
        int j = rng.randint(0, i);
        std::swap(indices[i], indices[j]);
    }
    indices.resize(n_recreate);

    // Create new params with random positions for recreated trees
    TreeParamsSoA new_params = params;
    for (int idx : indices) {
        new_params.x[idx] = rng.uniform(minval, maxval);
        new_params.y[idx] = rng.uniform(minval, maxval);
        new_params.angle[idx] = rng.uniform(-PI, PI);
    }

    // Create new solution with updated params
    Solution new_sol = solution.solution.update(new_params, indices);

    // Evaluate the new solution incrementally
    SolutionEval new_eval = problem_->eval_update(new_sol, solution, indices);

    // Update state
    auto rec_state = std::any_cast<RandomRecreateState>(state);
    rec_state.iteration++;

    return {new_eval, rec_state};
}

OptimizerPtr RandomRecreate::clone() const {
    return std::make_unique<RandomRecreate>(*this);
}

}  // namespace tree_packing
