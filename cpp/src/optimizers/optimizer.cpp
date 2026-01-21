#include "tree_packing/optimizers/optimizer.hpp"
#include <utility>

namespace tree_packing {

void Optimizer::step(
    const SolutionEval& solution,
    std::any& state,
    GlobalState& global_state,
    SolutionEval& out
) {
    // Split RNG for this step
    RNG rng(global_state.split_rng());

    // Apply optimizer
    apply(solution, state, global_state, rng, out);

    // Update global state
    global_state.maybe_update_best(*problem_, out);

    // Increment iteration
    global_state.next();
}

void Optimizer::run(
    const SolutionEval& solution,
    std::any& state,
    GlobalState& global_state,
    int n,
    SolutionEval& out
) {
    if (n <= 0) {
        out = solution;
        return;
    }

    SolutionEval current = solution;
    SolutionEval next;
    SolutionEval* cur = &current;
    SolutionEval* nxt = &next;

    for (int i = 0; i < n; ++i) {
        step(*cur, state, global_state, *nxt);
        std::swap(cur, nxt);
    }

    out = std::move(*cur);
}

}  // namespace tree_packing
