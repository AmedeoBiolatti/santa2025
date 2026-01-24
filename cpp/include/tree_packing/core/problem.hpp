#pragma once

#include "solution.hpp"
#include "global_state.hpp"
#include <functional>

namespace tree_packing {

// Forward declarations
class GlobalState;

// Problem definition with objective function and constraints
class Problem {
public:
    Problem() = default;

    // Create the standard tree packing problem
    static Problem create_tree_packing_problem(float side = -1.0f);

    // Evaluate a solution (compute objective and all constraints)
    [[nodiscard]] SolutionEval eval(const Solution& solution) const;

    // Evaluate in-place (reuse buffers in eval)
    void eval_inplace(const Solution& solution, SolutionEval& eval) const;

    // Update solution params in-place and evaluate full objective/constraints.
    void update_and_eval(
        SolutionEval& eval,
        const std::vector<int>& indices,
        const TreeParamsSoA& new_params
    ) const;

    // Specialized path for inserting previously invalid trees
    void insert_and_eval(
        SolutionEval& eval,
        const std::vector<int>& indices,
        const TreeParamsSoA& new_params
    ) const;

    // Mark solution params as NaN in-place and evaluate full objective/constraints.
    void remove_and_eval(
        SolutionEval& eval,
        const std::vector<int>& indices
    ) const;

    // Compute score (objective + penalty * violations + missing penalty)
    [[nodiscard]] float score(const SolutionEval& solution_eval, const GlobalState& global_state) const;

    // Objective function (bounding box area / num_trees)
    [[nodiscard]] float objective(const Solution& solution) const;

    // Individual constraint evaluations
    [[nodiscard]] float intersection_constraint(const Solution& solution) const;
    [[nodiscard]] float bounds_constraint(const Solution& solution) const;

    // Incremental constraint updates
    void update_intersection_matrix(
        const Solution& solution,
        std::vector<float>& matrix,
        const std::vector<int>& modified_indices,
        const Grid2D* old_grid = nullptr
    ) const;

    // Parameters
    [[nodiscard]] float min_pos() const { return min_pos_; }
    [[nodiscard]] float max_pos() const { return max_pos_; }

private:
    float min_pos_{-16.0f * THR / 2.0f};
    float max_pos_{16.0f * THR / 2.0f};
};

}  // namespace tree_packing
