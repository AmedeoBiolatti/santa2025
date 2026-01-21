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

    // Apply one optimization step
    // Returns new solution and updated state
    virtual std::pair<SolutionEval, std::any> apply(
        const SolutionEval& solution,
        std::any& state,
        GlobalState& global_state,
        RNG& rng
    ) = 0;

    // Clone optimizer (deep copy)
    [[nodiscard]] virtual OptimizerPtr clone() const = 0;

    // Full optimization step (apply + update best)
    std::tuple<SolutionEval, std::any, GlobalState> step(
        const SolutionEval& solution,
        std::any& state,
        GlobalState& global_state
    );

    // Run multiple steps in sequence
    std::tuple<SolutionEval, std::any, GlobalState> run(
        const SolutionEval& solution,
        std::any& state,
        GlobalState& global_state,
        int n
    );

protected:
    Problem* problem_{nullptr};
};

}  // namespace tree_packing
