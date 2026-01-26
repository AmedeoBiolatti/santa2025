#include "tree_packing/optimizers/optimizer.hpp"
#include <utility>

namespace tree_packing {

void Optimizer::step(
    SolutionEval& solution,
    std::any& state,
    GlobalState& global_state
) {
    // Split RNG for this step
    RNG rng(global_state.split_rng());

    // Apply optimizer
    apply(solution, state, global_state, rng);

    // Update global state
    global_state.maybe_update_best(*problem_, solution);

    // Increment iteration
    global_state.next();
}

void Optimizer::run(
    SolutionEval& solution,
    std::any& state,
    GlobalState& global_state,
    int n
) {
    if (n <= 0) {
        return;
    }

    for (int i = 0; i < n; ++i) {
        step(solution, state, global_state);
    }
}

}  // namespace tree_packing
