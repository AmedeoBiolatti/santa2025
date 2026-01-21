#pragma once

#include "optimizer.hpp"
#include <vector>
#include <memory>

namespace tree_packing {

// State for ALNS optimizer
struct ALNSState {
    int iteration{0};
    std::vector<std::any> ruin_states;
    std::vector<std::any> recreate_states;
    std::vector<float> ruin_weights;
    std::vector<float> recreate_weights;
};

// Adaptive Large Neighborhood Search optimizer
class ALNS : public Optimizer {
public:
    ALNS(
        std::vector<OptimizerPtr> ruin_operators,
        std::vector<OptimizerPtr> recreate_operators,
        float reaction_factor = 0.01f,
        float reward_improve = 1.0f,
        float reward_no_improve = 0.0f,
        float min_weight = 1e-3f
    );

    void set_problem(Problem* problem) override;

    std::any init_state(const SolutionEval& solution) override;

    std::pair<SolutionEval, std::any> apply(
        const SolutionEval& solution,
        std::any& state,
        GlobalState& global_state,
        RNG& rng
    ) override;

    [[nodiscard]] OptimizerPtr clone() const override;

private:
    std::vector<OptimizerPtr> ruin_operators_;
    std::vector<OptimizerPtr> recreate_operators_;

    float reaction_factor_;
    float reward_improve_;
    float reward_no_improve_;
    float min_weight_;

    // Select operator index based on weights
    int select_index(const std::vector<float>& weights, RNG& rng);

    // Update weights based on reward
    void update_weights(std::vector<float>& weights, int index, float reward);
};

}  // namespace tree_packing
