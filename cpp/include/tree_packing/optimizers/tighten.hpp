#pragma once

#include "optimizer.hpp"
#include <vector>

namespace tree_packing {

// Global squeeze optimizer: scales all valid tree centers toward the bbox center.
class SqueezeOptimizer : public Optimizer {
public:
    SqueezeOptimizer(
        float min_scale = 0.05f,
        float shrink = 0.92f,
        int bisect_iters = 18,
        int axis_rounds = 3,
        bool verbose = false
    );

    std::any init_state(const SolutionEval& solution) override;

    void apply(
        SolutionEval& solution,
        std::any& state,
        GlobalState& global_state,
        RNG& rng
    ) override;

    [[nodiscard]] OptimizerPtr clone() const override;

private:
    float min_scale_;
    float shrink_;
    int bisect_iters_;
    int axis_rounds_;
    bool verbose_;
};

struct SqueezeState {
    std::vector<int> indices;
    TreeParamsSoA base_params;
    TreeParamsSoA scratch_params;
    std::vector<float> dx;
    std::vector<float> dy;
};

// Compaction optimizer: greedily pulls trees toward the bbox center.
class CompactionOptimizer : public Optimizer {
public:
    explicit CompactionOptimizer(int iters_per_tree = 8, bool verbose = false);

    std::any init_state(const SolutionEval& solution) override;

    void apply(
        SolutionEval& solution,
        std::any& state,
        GlobalState& global_state,
        RNG& rng
    ) override;

    [[nodiscard]] OptimizerPtr clone() const override;

private:
    int iters_per_tree_;
    bool verbose_;
};

struct CompactionState {
    std::vector<int> indices;
    std::vector<int> single_index;
    TreeParamsSoA single_params;
};

// Local search optimizer: fine-grained translate/rotate tightening.
class LocalSearchOptimizer : public Optimizer {
public:
    explicit LocalSearchOptimizer(int iters_per_tree = 18, bool verbose = false);

    std::any init_state(const SolutionEval& solution) override;

    void apply(
        SolutionEval& solution,
        std::any& state,
        GlobalState& global_state,
        RNG& rng
    ) override;

    [[nodiscard]] OptimizerPtr clone() const override;

private:
    int iters_per_tree_;
    bool verbose_;
};

struct LocalSearchState {
    std::vector<int> indices;
    std::vector<int> single_index;
    TreeParamsSoA single_params;
    // Cached minimum AABB gap to any neighbor for each tree
    std::vector<float> min_gaps;
    // Reusable buffer for grid candidate queries (avoids allocations)
    std::vector<Index> candidates_buffer;
};

}  // namespace tree_packing
