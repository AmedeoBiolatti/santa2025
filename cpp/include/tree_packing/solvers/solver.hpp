#pragma once

#include "../core/types.hpp"
#include "../core/solution.hpp"
#include "../core/problem.hpp"
#include "../spatial/grid2d.hpp"
#include "../random/rng.hpp"
#include <cmath>
#include <memory>
#include <span>
#include <vector>
#include <iostream>

namespace tree_packing {

class Solver;
class Problem;

/**
 * Search region for solvers - either a grid cell or global bounds.
 */
struct SearchRegion {
    float min_x, max_x, min_y, max_y;  // Position bounds
    std::vector<Index> candidates;      // Pre-computed candidate neighbors
    bool has_candidates{false};         // True if candidates are valid
};

// Unique pointer alias for solvers
using SolverPtr = std::unique_ptr<Solver>;

/**
 * Result from a solver containing optimized placements and start/end values.
 */
struct SolverResult {
    TreeParamsSoA params;           // Optimized parameters for the requested indices
    float start_f{0.0f};            // Initial objective value
    float start_g{0.0f};            // Initial constraint violation
    float final_f{0.0f};            // Final objective value
    float final_g{0.0f};            // Final constraint violation (max)
};

/**
 * Base class for placement solvers.
 *
 * A Solver optimizes the placement of one or more trees given the current
 * solution state. Unlike Optimizers which modify the solution in-place,
 * Solvers compute and return optimal placements for specific indices.
 *
 * Typical use:
 *   1. Ruin operator removes some trees (marks as invalid)
 *   2. Solver finds optimal placements for removed trees
 *   3. Recreate operator applies the solver's suggestions
 *
 * The solve() method should find placements that:
 *   - Minimize overlap with existing trees (constraint g <= 0)
 *   - Minimize distance-based objective (f)
 *   - Stay within position bounds
 */
class Solver {
public:
    virtual ~Solver() = default;

    /**
     * Find optimal placements for trees at the given indices.
     *
     * @param eval Current evaluated solution (contains spatial grid, constraint info)
     * @param indices Indices of trees to place (typically 1-4 trees)
     * @param rng Random number generator for stochastic methods
     * @return SolverResult with optimized parameters and optimization history
     */
    [[nodiscard]] virtual SolverResult solve(
        const SolutionEval& eval,
        std::span<const int> indices,
        RNG& rng,
        const GlobalState* global_state = nullptr
    ) const = 0;

    /**
     * Find optimal placement for a single tree.
     * Convenience method that delegates to solve().
     */
    [[nodiscard]] SolverResult solve_single(
        const SolutionEval& eval,
        int index,
        RNG& rng,
        const GlobalState* global_state = nullptr
    ) const {
        std::array<int, 1> indices{index};
        return solve(eval, indices, rng, global_state);
    }

    /**
     * Clone the solver (deep copy for thread safety).
     */
    [[nodiscard]] virtual SolverPtr clone() const = 0;

    /**
     * Maximum supported group size for this solver.
     *
     * Default is 1 (single-tree solvers). Override in joint solvers.
     */
    [[nodiscard]] virtual int max_group_size() const { return 1; }

    /**
     * Whether SolverOptimize should split RNGs before calling this solver.
     *
     * Default is true. Override to consume RNG directly for reproducible
     * matching with other optimizers.
     */
    [[nodiscard]] virtual bool wants_rng_split() const { return true; }

    /**
     * Set the problem (for constraint penalty configuration).
     */
    virtual void set_problem(const Problem* problem) {
        problem_ = problem;
    }

    /**
     * Get the problem pointer.
     */
    [[nodiscard]] const Problem* problem() const { return problem_; }

    /**
     * Get search bounds for positions.
     */
    [[nodiscard]] float min_pos() const { return min_pos_; }
    [[nodiscard]] float max_pos() const { return max_pos_; }

    /**
     * Set search bounds for positions.
     */
    void set_bounds(float min_pos, float max_pos) {
        min_pos_ = min_pos;
        max_pos_ = max_pos;
    }

protected:
    float min_pos_{-8.0f};  // Default bounds
    float max_pos_{8.0f};
    const Problem* problem_{nullptr};

    /**
     * Compute constraint penalty using Problem's settings if available,
     * otherwise use simple linear penalty with given mu.
     */
    [[nodiscard]] float compute_penalty(float violation, float fallback_mu) const {
        if (violation <= 0.0f) {
            return 0.0f;
        }
        if (problem_) {
            return problem_->compute_constraint_penalty(violation, fallback_mu);
        }
        // Fallback: simple linear penalty
        return fallback_mu * violation;
    }

    // Compute objective ceiling penalty using the problem's ceiling settings.
    [[nodiscard]] float compute_objective_ceiling_penalty(
        float objective_value,
        const GlobalState* global_state = nullptr
    ) const {
        if (!problem_) {
            return 0.0f;
        }
        const float ceiling = (global_state != nullptr)
            ? problem_->effective_ceiling(*global_state)
            : problem_->objective_ceiling();
        if (!std::isfinite(ceiling)) {
            return 0.0f;
        }
        const float violation = objective_value - ceiling;
        if (violation <= 0.0f) {
            return 0.0f;
        }
        return problem_->objective_ceiling_mu() * violation;
    }

    /**
     * Compute search region - either constrained to a grid cell or global.
     *
     * @param solution The current solution
     * @param target_index Index of tree being placed (excluded from candidates)
     * @param constrain_to_cell If true, pick a random cell with <4 trees
     * @param prefer_current_cell Legacy knob. When the target is valid and
     *   constrain_to_cell is true, we always anchor to the current cell.
     * @param rng Random number generator
     * @return SearchRegion with bounds and optional pre-computed candidates
     */
    [[nodiscard]] SearchRegion compute_search_region(
        const Solution& solution,
        int target_index,
        bool constrain_to_cell,
        bool prefer_current_cell,
        RNG& rng
    ) const {
        SearchRegion region;
        const auto& grid = solution.grid();

        if (constrain_to_cell) {
            // Anchor the search to the current cell of the target tree when valid.
            // This avoids "teleporting" the search region to unrelated cells,
            // which is especially important for local/noise-based solvers.
            (void)prefer_current_cell;
            if (target_index >= 0 && solution.is_valid(static_cast<size_t>(target_index))) {
                const auto [ci, cj] = grid.get_item_cell(target_index);
                const AABB cell_bounds = grid.cell_bounds(ci, cj);
                region.min_x = cell_bounds.min.x;
                region.max_x = cell_bounds.max.x;
                region.min_y = cell_bounds.min.y;
                region.max_y = cell_bounds.max.y;

                const size_t needed = NEIGHBOR_DELTAS.size() * static_cast<size_t>(grid.capacity());
                region.candidates.resize(needed);
                const size_t n_candidates = grid.get_candidates_by_cell(ci, cj, region.candidates);
                region.candidates.resize(n_candidates);
                region.has_candidates = true;
                return region;
            }

            // Pick a random cell within the solution bounds
            float max_abs = solution.max_max_abs();
            if (max_abs <= 0.0f || !std::isfinite(max_abs)) {
                max_abs = 5.0f;  // Default
            }
            float delta = 0.5f;
            float bound_min = std::max(min_pos_, -max_abs - delta);
            float bound_max = std::min(max_pos_, max_abs + delta);

            // Try to find a cell with fewer than 4 trees (max 10 attempts)
            constexpr int max_cell_trees = 4;
            constexpr int max_attempts = 10;
            for (int attempt = 0; attempt < max_attempts; ++attempt) {
                // Pick random center position for the cell
                float center_x = rng.uniform(bound_min, bound_max);
                float center_y = rng.uniform(bound_min, bound_max);
                Vec2 center{center_x, center_y};

                // Get cell bounds from grid
                auto [ci, cj] = grid.compute_ij(center);
                AABB cell_bounds = grid.cell_bounds(ci, cj);

                // Constrain search to this cell
                region.min_x = cell_bounds.min.x;
                region.max_x = cell_bounds.max.x;
                region.min_y = cell_bounds.min.y;
                region.max_y = cell_bounds.max.y;

                // Pre-compute candidates
                const size_t needed = NEIGHBOR_DELTAS.size() * static_cast<size_t>(grid.capacity());
                region.candidates.resize(needed);
                const size_t n_candidates = grid.get_candidates_by_cell(ci, cj, region.candidates);
                region.candidates.resize(n_candidates);

                // Count valid trees in cell (excluding target)
                int n_trees_in_cell = 0;
                for (int idx : region.candidates) {
                    if (idx >= 0 && idx != target_index && solution.is_valid(idx)) {
                        ++n_trees_in_cell;
                    }
                }

                // Accept cell if it has fewer than max_cell_trees
                if (n_trees_in_cell < max_cell_trees) {
                    break;
                }
            }
            region.has_candidates = true;
        } else {
            // Global search - no pre-computed candidates
            float max_abs = solution.max_max_abs();
            if (max_abs > 0.0f && std::isfinite(max_abs)) {
                float delta = 0.5f;
                region.min_x = region.min_y = std::max(min_pos_, -max_abs - delta);
                region.max_x = region.max_y = std::min(max_pos_, max_abs + delta);
            } else {
                region.min_x = region.min_y = min_pos_;
                region.max_x = region.max_y = max_pos_;
            }
            region.has_candidates = false;
        }

        return region;
    }
};

}  // namespace tree_packing
