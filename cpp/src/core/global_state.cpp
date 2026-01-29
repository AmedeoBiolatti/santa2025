#include "tree_packing/core/global_state.hpp"
#include "tree_packing/core/problem.hpp"

namespace tree_packing {

GlobalState::GlobalState(uint64_t seed) : rng_(seed) {}

GlobalState::GlobalState(uint64_t seed, const SolutionEval& initial_solution)
    : rng_(seed)
    , best_params_(initial_solution.solution.params())
    , best_feasible_params_(initial_solution.solution.params())
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
    double violation = solution.total_violation();
    // Best feasible should always respect the default feasibility tolerance.
    bool meets_default_tolerance = violation < static_cast<double>(default_tolerance());
    float score = problem.score(solution, *this);

    // Update best overall (only copy params, not the full solution)
    if (score < best_score_) {
        best_score_ = score;
        best_params_ = solution.solution.params();
        iters_since_improvement_ = 0;
    }

    float feasible_objective = solution.objective;

    // Update best feasible (only copy params)
    if (meets_default_tolerance && feasible_objective < best_feasible_score_) {
        best_feasible_score_ = feasible_objective;
        best_feasible_objective_ = feasible_objective;
        best_feasible_params_ = solution.solution.params();
        iters_since_feasible_improvement_ = 0;
    }
}

}  // namespace tree_packing
