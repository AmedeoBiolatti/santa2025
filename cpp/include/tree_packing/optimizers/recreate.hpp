#pragma once

#include "optimizer.hpp"
#include <vector>

namespace tree_packing {

// Random recreate: places removed trees at random positions
class RandomRecreate : public Optimizer {
public:
    explicit RandomRecreate(
        int max_recreate = 1,
        float box_size = 5.0f,
        float delta = 0.35f,
        bool verbose = false
    );

    std::any init_state(const SolutionEval& solution) override;

    void apply(
        SolutionEval& solution,
        std::any& state,
        GlobalState& global_state,
        RNG& rng
    ) override;

    [[nodiscard]] OptimizerPtr clone() const override;

private:
    int max_recreate_;
    float box_size_;
    float delta_;
    bool verbose_;
};

// GridCellRecreate: recreates trees in low-occupancy grid cells with neighbors
class GridCellRecreate : public Optimizer {
public:
    explicit GridCellRecreate(
        int max_recreate = 1,
        int cell_min = 0,
        int cell_max = 4,
        int neighbor_min = 1,
        int neighbor_max = -1,
        bool verbose = false
    );

    std::any init_state(const SolutionEval& solution) override;

    void apply(
        SolutionEval& solution,
        std::any& state,
        GlobalState& global_state,
        RNG& rng
    ) override;

    [[nodiscard]] OptimizerPtr clone() const override;

private:
    int max_recreate_;
    int cell_min_;
    int cell_max_;
    int neighbor_min_;
    int neighbor_max_;
    bool verbose_;
};

// State for RandomRecreate (simplified - rollback now uses UpdateStack)
struct RandomRecreateState {
    int iteration{0};
    std::vector<int> indices;
    TreeParamsSoA new_params;
};

// State for GridCellRecreate (simplified - rollback now uses UpdateStack)
struct GridCellRecreateState {
    int iteration{0};
    std::vector<int> indices;
    std::vector<std::pair<int, int>> candidate_cells;
    TreeParamsSoA new_params;
};

}  // namespace tree_packing
