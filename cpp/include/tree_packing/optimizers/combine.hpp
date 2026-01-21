#pragma once

#include "optimizer.hpp"
#include <vector>
#include <memory>

namespace tree_packing {

// Chain: applies multiple optimizers in sequence
class Chain : public Optimizer {
public:
    explicit Chain(std::vector<OptimizerPtr> optimizers, bool verbose = false);

    void set_problem(Problem* problem) override;

    std::any init_state(const SolutionEval& solution) override;

    void apply(
        const SolutionEval& solution,
        std::any& state,
        GlobalState& global_state,
        RNG& rng,
        SolutionEval& out
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
        const SolutionEval& solution,
        std::any& state,
        GlobalState& global_state,
        RNG& rng,
        SolutionEval& out
    ) override;

    [[nodiscard]] OptimizerPtr clone() const override;

private:
    OptimizerPtr optimizer_;
    int n_;
    bool verbose_;
};

// RestoreBest: restores best solution if no improvement for patience iterations
class RestoreBest : public Optimizer {
public:
    RestoreBest(OptimizerPtr optimizer, int patience, bool verbose = false);

    void set_problem(Problem* problem) override;

    std::any init_state(const SolutionEval& solution) override;

    void apply(
        const SolutionEval& solution,
        std::any& state,
        GlobalState& global_state,
        RNG& rng,
        SolutionEval& out
    ) override;

    [[nodiscard]] OptimizerPtr clone() const override;

private:
    OptimizerPtr optimizer_;
    int patience_;
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
        const SolutionEval& solution,
        std::any& state,
        GlobalState& global_state,
        RNG& rng,
        SolutionEval& out
    ) override;

    [[nodiscard]] OptimizerPtr clone() const override;

private:
    std::vector<OptimizerPtr> optimizers_;
    std::vector<float> probabilities_;
    bool verbose_;
};

}  // namespace tree_packing
