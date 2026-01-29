#pragma once

#include "optimizer.hpp"
#include <vector>
#include <memory>
#include <functional>

namespace tree_packing {

// Type alias for condition functions used by Conditional optimizer
using ConditionFn = std::function<bool(const SolutionEval&, const std::any&, const GlobalState&)>;

// Chain: applies multiple optimizers in sequence
class Chain : public Optimizer {
public:
    explicit Chain(std::vector<OptimizerPtr> optimizers, bool verbose = false);

    void set_problem(Problem* problem) override;

    std::any init_state(const SolutionEval& solution) override;

    void apply(
        SolutionEval& solution,
        std::any& state,
        GlobalState& global_state,
        RNG& rng
    ) override;

    [[nodiscard]] OptimizerPtr clone() const override;

private:
    std::vector<OptimizerPtr> optimizers_;
    bool verbose_;
};

// Repeat: applies an optimizer n times
class Repeat : public Optimizer {
public:
    Repeat(OptimizerPtr optimizer, int n, bool verbose = false);

    void set_problem(Problem* problem) override;

    std::any init_state(const SolutionEval& solution) override;

    void apply(
        SolutionEval& solution,
        std::any& state,
        GlobalState& global_state,
        RNG& rng
    ) override;

    [[nodiscard]] OptimizerPtr clone() const override;

private:
    OptimizerPtr optimizer_;
    int n_;
    bool verbose_;
};

// RestoreBest: restores best solution every n iterations
class RestoreBest : public Optimizer {
public:
    explicit RestoreBest(int interval, bool verbose = false);

    std::any init_state(const SolutionEval& solution) override;

    void apply(
        SolutionEval& solution,
        std::any& state,
        GlobalState& global_state,
        RNG& rng
    ) override;

    [[nodiscard]] OptimizerPtr clone() const override;

private:
    int interval_;
    bool verbose_;
};

// RandomChoice: randomly selects one optimizer to apply
class RandomChoice : public Optimizer {
public:
    explicit RandomChoice(
        std::vector<OptimizerPtr> optimizers,
        std::vector<float> probabilities = {},
        bool verbose = false
    );

    void set_problem(Problem* problem) override;

    std::any init_state(const SolutionEval& solution) override;

    void apply(
        SolutionEval& solution,
        std::any& state,
        GlobalState& global_state,
        RNG& rng
    ) override;

    [[nodiscard]] OptimizerPtr clone() const override;

private:
    std::vector<OptimizerPtr> optimizers_;
    std::vector<float> probabilities_;
    bool verbose_;
};

// Alternate: cycles through optimizers in order
class Alternate : public Optimizer {
public:
    explicit Alternate(
        std::vector<OptimizerPtr> optimizers,
        bool verbose = false
    );

    void set_problem(Problem* problem) override;

    std::any init_state(const SolutionEval& solution) override;

    void apply(
        SolutionEval& solution,
        std::any& state,
        GlobalState& global_state,
        RNG& rng
    ) override;

    [[nodiscard]] OptimizerPtr clone() const override;

private:
    std::vector<OptimizerPtr> optimizers_;
    bool verbose_;
};

// =============================================================================
// Conditional Meta-Optimizer
// =============================================================================

// Conditional: applies optimizer only when ALL conditions are met (logical AND).
// If any condition is false, returns the solution unchanged.
//
// Conditions can be specified via:
// - every_n: run when iteration % every_n == 0
// - min_iters_since_improvement: run when iters_since_improvement >= this value
// - min_iters_since_feasible_improvement: run when iters_since_feasible_improvement >= this value
// - custom_condition: arbitrary ConditionFn for custom logic
//
// Example usage:
//   // Run every 10 iterations AND when stuck for 20 iterations
//   auto opt = std::make_unique<Conditional>(
//       inner_opt->clone(),
//       10,    // every_n
//       20,    // min_iters_since_improvement
//       0      // min_iters_since_feasible_improvement (disabled)
//   );
//
//   // Run only every 100 iterations
//   auto opt = std::make_unique<Conditional>(inner_opt->clone(), 100);
//
//   // Run when stuck for 50 iterations
//   auto opt = std::make_unique<Conditional>(inner_opt->clone(), 0, 50);
class Conditional : public Optimizer {
public:
    // Full constructor with all condition parameters
    Conditional(
        OptimizerPtr optimizer,
        uint64_t every_n = 0,                              // 0 = disabled
        uint64_t min_iters_since_improvement = 0,          // 0 = disabled
        uint64_t min_iters_since_feasible_improvement = 0, // 0 = disabled
        ConditionFn custom_condition = nullptr,            // nullptr = disabled
        bool verbose = false
    );

    void set_problem(Problem* problem) override;

    std::any init_state(const SolutionEval& solution) override;

    void apply(
        SolutionEval& solution,
        std::any& state,
        GlobalState& global_state,
        RNG& rng
    ) override;

    [[nodiscard]] OptimizerPtr clone() const override;

    // Getters for cloning
    [[nodiscard]] uint64_t every_n() const { return every_n_; }
    [[nodiscard]] uint64_t min_iters_since_improvement() const { return min_iters_since_improvement_; }
    [[nodiscard]] uint64_t min_iters_since_feasible_improvement() const { return min_iters_since_feasible_improvement_; }

private:
    bool check_conditions(const SolutionEval& solution, const std::any& state, const GlobalState& global_state) const;

    OptimizerPtr optimizer_;
    uint64_t every_n_;
    uint64_t min_iters_since_improvement_;
    uint64_t min_iters_since_feasible_improvement_;
    ConditionFn custom_condition_;
    bool verbose_;
};

}  // namespace tree_packing
