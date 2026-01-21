#pragma once

#include "optimizer.hpp"

namespace tree_packing {

// Noise optimizer: adds small random perturbations to a single tree
class NoiseOptimizer : public Optimizer {
public:
    explicit NoiseOptimizer(float noise_level = 0.01f);

    std::pair<SolutionEval, std::any> apply(
        const SolutionEval& solution,
        std::any& state,
        GlobalState& global_state,
        RNG& rng
    ) override;

    [[nodiscard]] OptimizerPtr clone() const override;

private:
    float noise_level_;
};

}  // namespace tree_packing
