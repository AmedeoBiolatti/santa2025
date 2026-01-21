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
        const SolutionEval& solution,
        std::any& state,
        GlobalState& global_state,
        RNG& rng,
        SolutionEval& out
    ) override;

    [[nodiscard]] OptimizerPtr clone() const override;

private:
    int n_remove_;
    bool verbose_;
};

// Spatial ruin: removes n_remove trees closest to a random point
class SpatialRuin : public Optimizer {
public:
    explicit SpatialRuin(int n_remove = 1, bool verbose = false);

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
    int n_remove_;
    bool verbose_;

    // Select random point within the bounds of current trees
    Vec2 random_point(const TreeParamsSoA& params, RNG& rng);

    // Find n_remove closest trees to point
    void select_closest(
        const TreeParamsSoA& params,
        const Vec2& point,
        std::vector<int>& out,
        std::vector<std::pair<float, int>>& distances
    );
};

// State for ruin operators
struct RuinState {
    std::vector<int> indices;
    std::vector<std::pair<float, int>> distances;
};

}  // namespace tree_packing
