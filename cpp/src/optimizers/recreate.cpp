#include "tree_packing/optimizers/recreate.hpp"
#include "tree_packing/core/tree.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

namespace tree_packing {

RandomRecreate::RandomRecreate(int max_recreate, float box_size, float delta, bool verbose)
    : max_recreate_(max_recreate)
    , box_size_(box_size)
    , delta_(delta)
    , verbose_(verbose)
{
    if (max_recreate_ < 1) max_recreate_ = 1;
}

std::any RandomRecreate::init_state(const SolutionEval& solution) {
    RandomRecreateState state;
    state.iteration = 0;
    return state;
}

void RandomRecreate::get_removed_indices(const TreeParamsSoA& params, std::vector<int>& out) {
    out.clear();
    for (size_t i = 0; i < params.size(); ++i) {
        if (params.is_nan(i)) {
            out.push_back(static_cast<int>(i));
        }
    }
}

void RandomRecreate::apply(
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
    float max_abs = out.solution.max_max_abs();

    // Find removed (NaN) indices
    auto* rec_state = std::any_cast<RandomRecreateState>(&state);
    if (!rec_state) {
        state = init_state(solution);
        rec_state = std::any_cast<RandomRecreateState>(&state);
    }
    auto& removed = rec_state->removed;
    get_removed_indices(params, removed);

    if (removed.empty()) {
        // Nothing to recreate
        rec_state->iteration++;
        return;
    }

    // If no valid trees, use default box
    if (max_abs == 0.0f) {
        max_abs = box_size_ / 2.0f;
    }

    float minval = -max_abs - delta_;
    float maxval = max_abs + delta_;

    // Select indices to recreate (up to max_recreate)
    int n_recreate = std::min(max_recreate_, static_cast<int>(removed.size()));

    // Take first n_recreate removed indices (order doesn't matter)
    auto& indices = rec_state->indices;
    indices = removed;
    indices.resize(n_recreate);

    // Create new solution with updated params
    for (int idx : indices) {
        TreeParams p{
            rng.uniform(minval, maxval),
            rng.uniform(minval, maxval),
            rng.uniform(-PI, PI)
        };
        out.solution.set_params(static_cast<size_t>(idx), p);
    }

    // Evaluate the new solution
    problem_->eval_inplace(out.solution, out);

    // Update state
    rec_state->iteration++;

    if (verbose_) {
        std::cout << "[RandomRecreate] iter=" << rec_state->iteration
                  << " removed=" << removed.size()
                  << " recreated=" << n_recreate << "\n";
    }
}

OptimizerPtr RandomRecreate::clone() const {
    return std::make_unique<RandomRecreate>(*this);
}

}  // namespace tree_packing
