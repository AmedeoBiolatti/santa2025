#pragma once

#include "optimizer.hpp"
#include <vector>

namespace tree_packing {

// Random recreate: places removed trees at random positions
class RandomRecreate : public Optimizer {
public:
    explicit RandomRecreate(int max_recreate = 1, float box_size = 5.0f, float delta = 0.35f);

    std::any init_state(const SolutionEval& solution) override;

    std::pair<SolutionEval, std::any> apply(
        const SolutionEval& solution,
        std::any& state,
        GlobalState& global_state,
        RNG& rng
    ) override;

    [[nodiscard]] OptimizerPtr clone() const override;

private:
    int max_recreate_;
    float box_size_;
    float delta_;

    // Find indices of removed (NaN) trees
    std::vector<int> get_removed_indices(const TreeParamsSoA& params);
};

// State for RandomRecreate
struct RandomRecreateState {
    int iteration{0};
};

}  // namespace tree_packing
