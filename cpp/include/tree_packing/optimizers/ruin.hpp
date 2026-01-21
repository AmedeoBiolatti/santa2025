#pragma once

#include "optimizer.hpp"
#include <vector>

namespace tree_packing {

// Random ruin: removes n_remove random trees
class RandomRuin : public Optimizer {
public:
    explicit RandomRuin(int n_remove = 1);

    std::pair<SolutionEval, std::any> apply(
        const SolutionEval& solution,
        std::any& state,
        GlobalState& global_state,
        RNG& rng
    ) override;

    [[nodiscard]] OptimizerPtr clone() const override;

private:
    int n_remove_;
};

// Spatial ruin: removes n_remove trees closest to a random point
class SpatialRuin : public Optimizer {
public:
    explicit SpatialRuin(int n_remove = 1);

    std::pair<SolutionEval, std::any> apply(
        const SolutionEval& solution,
        std::any& state,
        GlobalState& global_state,
        RNG& rng
    ) override;

    [[nodiscard]] OptimizerPtr clone() const override;

private:
    int n_remove_;

    // Select random point within the bounds of current trees
    Vec2 random_point(const TreeParamsSoA& params, RNG& rng);

    // Find n_remove closest trees to point
    std::vector<int> select_closest(const TreeParamsSoA& params, const Vec2& point);
};

}  // namespace tree_packing
