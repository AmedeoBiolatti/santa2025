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
        const SolutionEval& solution,
        std::any& state,
        GlobalState& global_state,
        RNG& rng,
        SolutionEval& out
    ) override;

    [[nodiscard]] OptimizerPtr clone() const override;

private:
    int max_recreate_;
    float box_size_;
    float delta_;
    bool verbose_;

    // Find indices of removed (NaN) trees
    void get_removed_indices(const TreeParamsSoA& params, std::vector<int>& out);
};

// State for RandomRecreate
struct RandomRecreateState {
    int iteration{0};
    std::vector<int> removed;
    std::vector<int> indices;
};

}  // namespace tree_packing
