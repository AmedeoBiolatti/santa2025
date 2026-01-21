#include "tree_packing/optimizers/optimizer.hpp"

namespace tree_packing {

std::tuple<SolutionEval, std::any, GlobalState> Optimizer::step(
    const SolutionEval& solution,
    std::any& state,
    GlobalState& global_state
) {
    // Split RNG for this step
    RNG rng(global_state.split_rng());

    // Apply optimizer
    auto [new_solution, new_state] = apply(solution, state, global_state, rng);

    // Update global state
    global_state.maybe_update_best(*problem_, new_solution);

    // Increment iteration
    global_state.next();

    return {new_solution, new_state, global_state};
}

std::tuple<SolutionEval, std::any, GlobalState> Optimizer::run(
    const SolutionEval& solution,
    std::any& state,
    GlobalState& global_state,
    int n
) {
    if (n <= 0) {
        return {solution, state, global_state};
    }

    SolutionEval current = solution;
    std::any current_state = state;

    for (int i = 0; i < n; ++i) {
        auto [new_solution, new_state, new_global_state] = step(
            current, current_state, global_state
        );
        current = new_solution;
        current_state = new_state;
        global_state = new_global_state;
    }

    state = current_state;
    return {current, current_state, global_state};
}

}  // namespace tree_packing
