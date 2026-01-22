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
        SolutionEval& solution,
        std::any& state,
        GlobalState& global_state,
        RNG& rng
    ) override;

    void rollback(SolutionEval& solution, std::any& state) override;

    [[nodiscard]] OptimizerPtr clone() const override;

private:
    float noise_level_;
    bool verbose_;
};

// State for NoiseOptimizer
struct NoiseState {
    std::vector<int> indices;
    int last_idx{-1};
    TreeParams last_params;
    bool has_last{false};
};

}  // namespace tree_packing
