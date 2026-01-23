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

void RandomRecreate::get_removed_indices(const Solution& solution, std::vector<int>& out) {
    out.clear();
    for (size_t i = 0; i < solution.size(); ++i) {
        if (!solution.is_valid(i)) {
            out.push_back(static_cast<int>(i));
        }
    }
}

void RandomRecreate::apply(
    SolutionEval& solution,
    std::any& state,
    GlobalState& global_state,
    RNG& rng
) {
    (void)global_state;
    const auto& params = solution.solution.params();
    float max_abs = solution.solution.max_max_abs();

    // Find removed (NaN) indices
    auto* rec_state = std::any_cast<RandomRecreateState>(&state);
    if (!rec_state) {
        state = init_state(solution);
        rec_state = std::any_cast<RandomRecreateState>(&state);
    }
    auto& removed = rec_state->removed;
    get_removed_indices(solution.solution, removed);

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

    rec_state->prev_params.resize(indices.size());
    for (size_t k = 0; k < indices.size(); ++k) {
        int idx = indices[k];
        if (idx < 0 || static_cast<size_t>(idx) >= params.size()) {
            rec_state->prev_params.set_nan(k);
            continue;
        }
        if (solution.solution.is_valid(static_cast<size_t>(idx))) {
            rec_state->prev_params.set(k, solution.solution.get_params(static_cast<size_t>(idx)));
        } else {
            rec_state->prev_params.set_nan(k);
        }
    }

    // Create new solution with updated params
    TreeParamsSoA new_params(static_cast<size_t>(n_recreate));
    for (int i = 0; i < n_recreate; ++i) {
        TreeParams p{
            rng.uniform(minval, maxval),
            rng.uniform(minval, maxval),
            rng.uniform(-PI, PI)
        };
        new_params.set(static_cast<size_t>(i), p);
    }

    // Evaluate the new solution
    problem_->update_and_eval(solution, indices, new_params);

    // Update state
    rec_state->iteration++;

    if (verbose_) {
        std::cout << "[RandomRecreate] iter=" << rec_state->iteration
                  << " removed=" << removed.size()
                  << " recreated=" << n_recreate << "\n";
    }
}

void RandomRecreate::rollback(SolutionEval& solution, std::any& state) {
    auto* rec_state = std::any_cast<RandomRecreateState>(&state);
    if (!rec_state) {
        return;
    }
    const auto& indices = rec_state->indices;
    if (indices.empty()) {
        return;
    }
    std::vector<int> valid_indices;
    std::vector<int> invalid_indices;
    TreeParamsSoA valid_params;
    for (size_t k = 0; k < indices.size(); ++k) {
        int idx = indices[k];
        if (rec_state->prev_params.is_nan(k)) {
            invalid_indices.push_back(idx);
        } else {
            size_t out_idx = valid_indices.size();
            valid_indices.push_back(idx);
            valid_params.resize(valid_indices.size());
            valid_params.set(out_idx, rec_state->prev_params.get(k));
        }
    }
    if (!invalid_indices.empty()) {
        problem_->remove_and_eval(solution, invalid_indices);
    }
    if (!valid_indices.empty()) {
        problem_->update_and_eval(solution, valid_indices, valid_params);
    }
}

OptimizerPtr RandomRecreate::clone() const {
    return std::make_unique<RandomRecreate>(*this);
}

}  // namespace tree_packing
