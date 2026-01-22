#pragma once

#include "../core/solution.hpp"
#include "../core/problem.hpp"
#include "../core/global_state.hpp"
#include "../random/rng.hpp"
#include <memory>
#include <any>

namespace tree_packing {

class Optimizer;

// Unique pointer alias for optimizers
using OptimizerPtr = std::unique_ptr<Optimizer>;


// Base optimizer interface
class Optimizer {
public:
    virtual ~Optimizer() = default;

    // Set the problem to optimize
    virtual void set_problem(Problem* problem) {
        problem_ = problem;
    }

    // Initialize optimizer state (called once at start)
    virtual std::any init_state(const SolutionEval& solution) {
        return {};
    }

    // Apply one optimization step in-place
    virtual void apply(
        SolutionEval& solution,
        std::any& state,
        GlobalState& global_state,
        RNG& rng
    ) = 0;

    // Rollback the last apply (default: no-op)
    virtual void rollback(SolutionEval& solution, std::any& state) {
        (void)solution;
        (void)state;
    }

    // Clone optimizer (deep copy)
    [[nodiscard]] virtual OptimizerPtr clone() const = 0;

    // Full optimization step (apply + update best)
    void step(
        SolutionEval& solution,
        std::any& state,
        GlobalState& global_state
    );

    // Run multiple steps in sequence
    void run(
        SolutionEval& solution,
        std::any& state,
        GlobalState& global_state,
        int n
    );

protected:
    Problem* problem_{nullptr};
};

}  // namespace tree_packing
