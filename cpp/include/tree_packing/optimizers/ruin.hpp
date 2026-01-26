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

    [[nodiscard]] OptimizerPtr clone() const override;

private:
    int n_remove_;
    bool verbose_;
};

// State for ruin operators (simplified - rollback now uses UpdateStack)
struct RuinState {
    std::vector<int> indices;
    std::vector<std::pair<float, int>> distances;
    // Scratch buffers (reused across calls)
    std::vector<std::pair<int, int>> eligible_cells;
    std::vector<Index> cell_items;
};

}  // namespace tree_packing
