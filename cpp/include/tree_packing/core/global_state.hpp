#pragma once

#include "solution.hpp"
#include "update_stack.hpp"
#include "../random/rng.hpp"
#include <limits>
#include <memory>
#include <optional>

namespace tree_packing {

// Forward declaration
class Problem;

// Global optimization state with best solution tracking
class GlobalState {
public:
    GlobalState() = default;
    explicit GlobalState(uint64_t seed);
    GlobalState(uint64_t seed, const SolutionEval& initial_solution);

    // RNG operations
    [[nodiscard]] uint64_t next_random();
    [[nodiscard]] float random_float();  // [0, 1)
    [[nodiscard]] float random_float(float min, float max);
    [[nodiscard]] int random_int(int min, int max);

    // Split RNG (returns new seed for a sub-operation)
    [[nodiscard]] uint64_t split_rng();

    // Iteration tracking
    void next();

    // Update best solutions if improved
    void maybe_update_best(const Problem& problem, const SolutionEval& solution);

    // Getters
    [[nodiscard]] uint64_t iteration() const { return t_; }
    [[nodiscard]] uint64_t iters_since_improvement() const { return iters_since_improvement_; }
    [[nodiscard]] uint64_t iters_since_feasible_improvement() const { return iters_since_feasible_improvement_; }

    [[nodiscard]] float best_score() const { return best_score_; }
    [[nodiscard]] float best_feasible_score() const { return best_feasible_score_; }

    [[nodiscard]] const TreeParamsSoA* best_params() const {
        return best_params_ ? &(*best_params_) : nullptr;
    }

    [[nodiscard]] const TreeParamsSoA* best_feasible_params() const {
        return best_feasible_params_ ? &(*best_feasible_params_) : nullptr;
    }

    // Penalty multiplier for violations
    [[nodiscard]] float mu() const { return mu_; }
    void set_mu(float mu) { mu_ = mu; }

    // Tolerance for feasibility
    [[nodiscard]] float tolerance() const { return tol_; }
    void set_tolerance(float tol) { tol_ = tol; }

    // Update stack for rollback support
    [[nodiscard]] UpdateStack& update_stack() { return update_stack_; }
    [[nodiscard]] const UpdateStack& update_stack() const { return update_stack_; }

    // Convenience: mark checkpoint on update stack
    [[nodiscard]] size_t mark_checkpoint() { return update_stack_.mark(); }

    // Convenience: rollback to checkpoint
    void rollback_to(const Problem& problem, SolutionEval& solution, size_t checkpoint) {
        apply_rollback(problem, solution, update_stack_, checkpoint);
    }

private:
    RNG rng_;
    uint64_t t_{0};
    uint64_t iters_since_improvement_{0};
    uint64_t iters_since_feasible_improvement_{0};

    float best_score_{std::numeric_limits<float>::infinity()};
    float best_feasible_score_{std::numeric_limits<float>::infinity()};

    std::optional<TreeParamsSoA> best_params_;
    std::optional<TreeParamsSoA> best_feasible_params_;

    float mu_{1e6f};
    float tol_{1e-12f};

    UpdateStack update_stack_{64};  // Pre-allocated with default capacity
};

}  // namespace tree_packing
