#pragma once

#include "optimizer.hpp"
#include <vector>

namespace tree_packing {

// Noise optimizer: adds small random perturbations to a single tree
class NoiseOptimizer : public Optimizer {
public:
    explicit NoiseOptimizer(float noise_level = 0.01f, bool verbose = false);

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
    float noise_level_;
    bool verbose_;
};

// State for NoiseOptimizer
struct NoiseState {
    Solution scratch;
    std::vector<int> indices;
};

}  // namespace tree_packing
