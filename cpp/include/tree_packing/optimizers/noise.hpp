#pragma once

#include "optimizer.hpp"
#include <vector>

namespace tree_packing {

// Noise optimizer: adds small random perturbations to a single tree
class NoiseOptimizer : public Optimizer {
public:
    explicit NoiseOptimizer(float noise_level = 0.01f, bool verbose = false);
    NoiseOptimizer(float noise_level, int n_change, bool verbose);

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
    int n_change_;
    bool verbose_;
};

// State for NoiseOptimizer
struct NoiseState {
    std::vector<int> indices;
    std::vector<int> selected;
    std::vector<int> last_indices;
    TreeParamsSoA last_params;
    TreeParamsSoA new_params;  // scratch buffer
    bool has_last{false};
};

}  // namespace tree_packing
