#include "tree_packing/core/global_state.hpp"
#include "tree_packing/core/problem.hpp"

namespace tree_packing {

GlobalState::GlobalState(uint64_t seed) : rng_(seed) {}

GlobalState::GlobalState(uint64_t seed, const SolutionEval& initial_solution)
    : rng_(seed)
    , best_solution_(initial_solution)
    , best_feasible_solution_(initial_solution)
{
}

uint64_t GlobalState::next_random() {
    return rng_.next();
}

float GlobalState::random_float() {
    return rng_.uniform();
}

float GlobalState::random_float(float min, float max) {
    return rng_.uniform(min, max);
}

int GlobalState::random_int(int min, int max) {
    return rng_.randint(min, max);
}

uint64_t GlobalState::split_rng() {
    return rng_.next();
}

void GlobalState::next() {
    ++t_;
    ++iters_since_improvement_;
    ++iters_since_feasible_improvement_;
}

void GlobalState::maybe_update_best(const Problem& problem, const SolutionEval& solution) {
    float violation = solution.total_violation();
    bool is_feasible = violation < tol_;
    float score = problem.score(solution, *this);

    // Update best overall
    if (score < best_score_) {
        best_score_ = score;
        best_solution_ = solution;
        iters_since_improvement_ = 0;
    }

    // Update best feasible
    if (is_feasible && score < best_feasible_score_) {
        best_feasible_score_ = score;
        best_feasible_solution_ = solution;
        iters_since_feasible_improvement_ = 0;
    }
}

}  // namespace tree_packing
