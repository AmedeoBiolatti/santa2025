#pragma once

#include "optimizer.hpp"
#include <vector>

namespace tree_packing {

// Noise optimizer: adds small random perturbations to a single tree
class NoiseOptimizer : public Optimizer {
public:
    explicit NoiseOptimizer(float noise_level = 0.01f, int n_change = 1, bool verbose = false);

    std::any init_state(const SolutionEval& solution) override;

    void apply(
        SolutionEval& solution,
        std::any& state,
        GlobalState& global_state,
        RNG& rng
    ) override;

    [[nodiscard]] OptimizerPtr clone() const override;

private:
    float noise_level_;
    int n_change_;
    bool verbose_;
};

// State for NoiseOptimizer (simplified - rollback now uses UpdateStack)
struct NoiseState {
    std::vector<int> indices;
    std::vector<int> selected;
    std::vector<int> last_indices;
    TreeParamsSoA new_params;  // scratch buffer
};

}  // namespace tree_packing
