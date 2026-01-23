#pragma once

#include "optimizer.hpp"
#include <utility>
#include <vector>

namespace tree_packing {

// Random ruin: removes n_remove random trees
class RandomRuin : public Optimizer {
public:
    explicit RandomRuin(int n_remove = 1, bool verbose = false);

    std::any init_state(const SolutionEval& solution) override;

    void apply(
        SolutionEval& solution,
        std::any& state,
        GlobalState& global_state,
        RNG& rng
    ) override;

    void rollback(SolutionEval& solution, std::any& state) override;

    [[nodiscard]] OptimizerPtr clone() const override;

private:
    int n_remove_;
    bool verbose_;
};

// Cell ruin: removes n_remove random trees from a random cell
class CellRuin : public Optimizer {
public:
    explicit CellRuin(int n_remove = 1, bool verbose = false);

    std::any init_state(const SolutionEval& solution) override;

    void apply(
        SolutionEval& solution,
        std::any& state,
        GlobalState& global_state,
        RNG& rng
    ) override;

    void rollback(SolutionEval& solution, std::any& state) override;

    [[nodiscard]] OptimizerPtr clone() const override;

private:
    int n_remove_;
    bool verbose_;
};

// State for ruin operators
struct RuinState {
    std::vector<int> indices;
    std::vector<std::pair<float, int>> distances;
    TreeParamsSoA prev_params;
};

}  // namespace tree_packing
