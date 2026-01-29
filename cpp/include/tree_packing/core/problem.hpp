#pragma once

#include "solution.hpp"
#include "global_state.hpp"
#include "../constraints/intersection.hpp"
#include <cmath>
#include <functional>
#include <limits>

namespace tree_packing {

// Forward declarations
class GlobalState;

// Constraint penalty types
enum class ConstraintPenaltyType {
    Linear,      // mu * violation
    Quadratic,   // mu * violation^2
    Tolerant,    // mu_low * violation if violation < tol, else mu * violation
    LogBarrier,  // -mu * log(1 - violation/tol) if violation < tol, else mu * (violation - tol + 1)
    Exponential  // mu * (exp(violation / scale) - 1)
};

// Problem definition with objective function and constraints
class Problem {
public:
    Problem() {
        init();
    };

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

    // Restore solution from params (full re-evaluation)
    void restore_from_params(
        SolutionEval& eval,
        const TreeParamsSoA& params
    ) const;

    // Compute constraint penalty based on penalty type
    [[nodiscard]] float compute_constraint_penalty(float violation, float mu) const {
        if (violation <= 0.0f) {
            return 0.0f;
        }

        switch (constraint_penalty_type_) {
            case ConstraintPenaltyType::Linear:
                return mu * violation;

            case ConstraintPenaltyType::Quadratic:
                return mu * violation * violation;

            case ConstraintPenaltyType::Tolerant:
                if (violation < constraint_tolerance_) {
                    return constraint_mu_low_ * violation;
                }
                return mu * violation;

            case ConstraintPenaltyType::LogBarrier:
                if (violation < constraint_tolerance_) {
                    // -mu * log(1 - violation/tol), grows to infinity as violation -> tol
                    float ratio = violation / constraint_tolerance_;
                    return -mu * std::log(1.0f - ratio);
                }
                // Beyond tolerance: linear penalty from the barrier value
                return mu * (violation - constraint_tolerance_ + 1.0f);

            case ConstraintPenaltyType::Exponential:
                // mu * (exp(violation / scale) - 1)
                return mu * (std::exp(violation / constraint_exp_scale_) - 1.0f);

            default:
                return mu * violation;
        }
    }

    // Compute score (objective + penalty * violations + missing penalty + ceiling penalty)
    [[nodiscard]] float score(const SolutionEval& solution_eval, const GlobalState& global_state) const {
        float mu = global_state.mu();

        float violation = solution_eval.total_violation();
        int n_missing = solution_eval.n_missing();
        float reg = solution_eval.reg();

        float penalty = compute_constraint_penalty(violation, mu);
        float s = solution_eval.objective + penalty + 1.0f * static_cast<float>(n_missing) + 1e-6f * reg;

        // Apply objective ceiling penalty if enabled
        float ceiling = effective_ceiling(global_state);
        if (std::isfinite(ceiling)) {
            float ceiling_violation = solution_eval.objective - ceiling;
            if (ceiling_violation > 0.0f) {
                s += objective_ceiling_mu_ * ceiling_violation;
            }
        }

        return s;
    }

    // Constraint penalty settings
    void set_constraint_penalty_type(ConstraintPenaltyType type) {
        constraint_penalty_type_ = type;
    }
    void set_constraint_tolerance(float tol) {
        constraint_tolerance_ = tol;
    }
    void set_constraint_mu_low(float mu_low) {
        constraint_mu_low_ = mu_low;
    }
    void set_constraint_exp_scale(float scale) {
        constraint_exp_scale_ = scale;
    }

    [[nodiscard]] ConstraintPenaltyType constraint_penalty_type() const { return constraint_penalty_type_; }
    [[nodiscard]] float constraint_tolerance() const { return constraint_tolerance_; }
    [[nodiscard]] float constraint_mu_low() const { return constraint_mu_low_; }
    [[nodiscard]] float constraint_exp_scale() const { return constraint_exp_scale_; }

    // Objective ceiling settings
    void set_objective_ceiling(float ceiling) {
        objective_ceiling_ = ceiling;
    }
    void set_objective_ceiling_delta(float delta) {
        objective_ceiling_delta_ = delta;
    }
    void set_objective_ceiling_mu(float mu) {
        objective_ceiling_mu_ = mu;
    }

    [[nodiscard]] float objective_ceiling() const { return objective_ceiling_; }
    [[nodiscard]] float objective_ceiling_delta() const { return objective_ceiling_delta_; }
    [[nodiscard]] float objective_ceiling_mu() const { return objective_ceiling_mu_; }

    // Compute effective ceiling: if delta is set, use best_feasible_objective + delta; otherwise use ceiling directly
    [[nodiscard]] float effective_ceiling(const GlobalState& global_state) const {
        if (std::isfinite(objective_ceiling_delta_)) {
            float best_feasible_obj = global_state.best_feasible_objective();
            if (std::isfinite(best_feasible_obj)) {
                return best_feasible_obj + objective_ceiling_delta_;
            }
        }
        return objective_ceiling_;
    }

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

    void init() {
        intersection_constraint_.init();
    }
private:
    float min_pos_{-16.0f * THR / 2.0f};
    float max_pos_{16.0f * THR / 2.0f};
    IntersectionConstraint intersection_constraint_;

    // Constraint penalty settings
    ConstraintPenaltyType constraint_penalty_type_{ConstraintPenaltyType::Linear};
    float constraint_tolerance_{1e-6f};   // Tolerance for Tolerant/LogBarrier penalty types
    float constraint_mu_low_{0.0f};       // Low penalty multiplier for Tolerant type (when violation < tol)
    float constraint_exp_scale_{0.1f};    // Scale for Exponential penalty type

    // Objective ceiling penalty settings
    // If delta is finite, ceiling = best_feasible_score + delta
    // Otherwise, use ceiling directly (if finite)
    float objective_ceiling_{std::numeric_limits<float>::infinity()};  // Direct ceiling value
    float objective_ceiling_delta_{std::numeric_limits<float>::infinity()};  // Delta from best feasible
    float objective_ceiling_mu_{1e6f};  // Penalty multiplier
};

}  // namespace tree_packing
