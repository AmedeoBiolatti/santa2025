#pragma once

#include "solver.hpp"
#include "fitness_utils.hpp"
#include "../core/tree.hpp"
#include "../geometry/sat.hpp"
#include "../spatial/grid2d.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace tree_packing {

/**
 * Deterministic beam search with short coordinate descent refinement.
 *
 * This solver is designed to be:
 * - Minimal and deterministic (fixed lattice + fixed step schedule)
 * - Cache-friendly (reuses precomputed candidates when available)
 * - Easy to profile (no complex state or allocations in hot loops)
 */
class BeamDescentSolver : public Solver {
public:
    enum class ObjectiveType {
        Distance,  // Distance-based score using neighbor intersections
        Linf,      // L-infinity norm: max abs coordinate of hull vertices
        Zero       // No objective, only constraint satisfaction + ceiling check
    };

    struct Config {
        // Lattice seeding
        int lattice_xy{5};            // Number of lattice points per axis
        int lattice_ang{8};           // Number of lattice angles
        int beam_width{8};            // Number of seeds to keep

        // Coordinate descent refinement
        int descent_levels{4};        // Number of step-size levels
        int max_iters_per_level{8};   // Max improvement steps per level
        float step_xy0{0.5f};         // Initial xy step size
        float step_xy_decay{0.5f};    // Step decay per level
        float step_ang0{PI / 8.0f};   // Initial angle step size
        float step_ang_decay{0.5f};   // Angle step decay per level

        // Penalized objective
        float mu{1e6f};               // Penalty multiplier for constraint violation
        // Isolation penalty for Distance objective: discourages escaping to empty space.
        // We charge an upper-bound per missing neighbor.
        float isolation_penalty_per_missing{std::log(2.0f)}; // Upper bound per missing neighbor
        float target_neighbors{8.0f};                        // Neighbor count that removes isolation penalty

        // Search region behavior
        bool constrain_to_cell{true};    // If true, search within a sampled grid cell
        bool prefer_current_cell{true};  // Kept for API compatibility (current cell is always used when valid)

        // Objective type
        ObjectiveType objective_type{ObjectiveType::Distance};
    };

    BeamDescentSolver() = default;
    explicit BeamDescentSolver(const Config& config) : config_(config) {}

    [[nodiscard]] SolverResult solve(
        const SolutionEval& eval,
        std::span<const int> indices,
        RNG& rng,
        const GlobalState* global_state = nullptr
    ) const override;

    [[nodiscard]] SolverPtr clone() const override {
        auto ptr = std::make_unique<BeamDescentSolver>(config_);
        ptr->set_bounds(min_pos_, max_pos_);
        ptr->set_problem(problem_);
        return ptr;
    }

    [[nodiscard]] const Config& config() const { return config_; }
    Config& config() { return config_; }

private:
    Config config_;

    struct EvalFG {
        float f{0.0f};
        float g{0.0f};
        float cost{0.0f};
    };

    struct EvalScratch {
        std::array<std::array<Vec2, 3>, TREE_NUM_TRIANGLES> normals{};
        std::array<AABB, TREE_NUM_TRIANGLES> tri_aabbs{};
        AABB figure_aabb{};
        std::vector<Index> candidates_storage{};
    };

    [[nodiscard]] static float wrap_angle(float angle) {
        while (angle > PI) angle -= TWO_PI;
        while (angle < -PI) angle += TWO_PI;
        return angle;
    }

    [[nodiscard]] static float lerp(float a, float b, float t) {
        return a + (b - a) * t;
    }

    [[nodiscard]] float compute_objective_linf(const Figure& figure) const;

    [[nodiscard]] std::pair<float, float> compute_objective_and_constraint_distance(
        const Figure& figure,
        const std::array<std::array<Vec2, 3>, TREE_NUM_TRIANGLES>& figure_normals,
        const AABB& figure_aabb,
        const std::array<AABB, TREE_NUM_TRIANGLES>& figure_tri_aabbs,
        const SolutionEval& eval,
        int target_index,
        const std::vector<Index>* candidates_ptr,
        std::vector<Index>& candidates_scratch
    ) const;

    [[nodiscard]] float compute_constraint_only(
        const Figure& figure,
        const std::array<std::array<Vec2, 3>, TREE_NUM_TRIANGLES>& figure_normals,
        const AABB& figure_aabb,
        const std::array<AABB, TREE_NUM_TRIANGLES>& figure_tri_aabbs,
        const SolutionEval& eval,
        int target_index,
        const std::vector<Index>* candidates_ptr,
        std::vector<Index>& candidates_scratch
    ) const;

    [[nodiscard]] EvalFG evaluate_params(
        const TreeParams& params,
        const SolutionEval& eval,
        int target_index,
        const std::vector<Index>* candidates_ptr,
        EvalScratch& scratch,
        const GlobalState* global_state = nullptr
    ) const;

    void insert_into_beam(
        const TreeParams& params,
        float cost,
        std::vector<TreeParams>& beam_params,
        std::vector<float>& beam_costs,
        int& beam_size
    ) const;

    [[nodiscard]] TreeParams clamp_to_region(
        const TreeParams& params,
        const SearchRegion& region
    ) const;
};

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

inline float BeamDescentSolver::compute_objective_linf(const Figure& figure) const {
    float max_abs = 0.0f;
    for (const auto& [ti, vi] : CONVEX_HULL_INDEXES) {
        const auto& tri = figure[static_cast<size_t>(ti)];
        const Vec2& v = (vi == 0) ? tri.v0 : (vi == 1) ? tri.v1 : tri.v2;
        max_abs = std::max(max_abs, std::abs(v.x));
        max_abs = std::max(max_abs, std::abs(v.y));
    }
    return max_abs;
}

inline std::pair<float, float> BeamDescentSolver::compute_objective_and_constraint_distance(
    const Figure& figure,
    const std::array<std::array<Vec2, 3>, TREE_NUM_TRIANGLES>& figure_normals,
    const AABB& figure_aabb,
    const std::array<AABB, TREE_NUM_TRIANGLES>& figure_tri_aabbs,
    const SolutionEval& eval,
    int target_index,
    const std::vector<Index>* candidates_ptr,
    std::vector<Index>& candidates_scratch
) const {
    const auto& solution = eval.solution;
    const auto& grid = solution.grid();

    const std::vector<Index>* candidates = candidates_ptr;
    if (!candidates) {
        Vec2 center{figure[0].centroid()};
        auto [ci, cj] = grid.compute_ij(center);
        const size_t needed = NEIGHBOR_DELTAS.size() * static_cast<size_t>(grid.capacity());
        candidates_scratch.resize(needed);
        const size_t n_candidates = grid.get_candidates_by_cell(ci, cj, candidates_scratch);
        candidates_scratch.resize(n_candidates);
        candidates = &candidates_scratch;
    }

    const auto& all_normals = solution.normals();
    const auto& all_aabbs = solution.aabbs();
    const auto& all_tri_aabbs = solution.triangle_aabbs();

    float total_score = 0.0f;
    int count = 0;
    float max_violation = 0.0f;

    for (int neighbor_idx : *candidates) {
        if (neighbor_idx < 0 || neighbor_idx == target_index) continue;
        if (!solution.is_valid(static_cast<size_t>(neighbor_idx))) continue;

        const auto& neighbor_aabb = all_aabbs[static_cast<size_t>(neighbor_idx)];

        // Conservative figure-level AABB prefilter.
        if (!figure_aabb.intersects(neighbor_aabb)) {
            total_score += std::log(2.0f);
            ++count;
            continue;
        }

        const auto& neighbor_fig = solution.figures()[static_cast<size_t>(neighbor_idx)];
        const auto& neighbor_normals = all_normals[static_cast<size_t>(neighbor_idx)];
        const auto& neighbor_tri_aabbs = all_tri_aabbs[static_cast<size_t>(neighbor_idx)];

        float min_score = std::numeric_limits<float>::max();

        // Triangle-level AABB culling before SAT.
        for (size_t i = 0; i < TREE_NUM_TRIANGLES; ++i) {
            if (!figure_tri_aabbs[i].intersects(neighbor_aabb)) continue;
            for (size_t j = 0; j < TREE_NUM_TRIANGLES; ++j) {
                if (!figure_tri_aabbs[i].intersects(neighbor_tri_aabbs[j])) continue;
                const float score = triangles_intersection_score_from_normals(
                    figure[i], neighbor_fig[j], figure_normals[i], neighbor_normals[j]
                );
                min_score = std::min(min_score, score);
                max_violation = std::max(max_violation, score);
            }
        }

        if (min_score == std::numeric_limits<float>::max()) {
            min_score = -1.0f;
        }

        const float dist_score = std::log(1.0f + std::max(0.0f, -min_score));
        total_score += dist_score;
        ++count;
    }

    const float mean_score = count > 0 ? total_score / static_cast<float>(count) : 0.0f;
    const float tgt = std::max(1.0f, config_.target_neighbors);
    const float missing = std::max(0.0f, tgt - static_cast<float>(count));
    const float isolation = config_.isolation_penalty_per_missing * missing;
    const float f = mean_score + isolation;
    const float g = max_violation;
    return {f, g};
}

inline float BeamDescentSolver::compute_constraint_only(
    const Figure& figure,
    const std::array<std::array<Vec2, 3>, TREE_NUM_TRIANGLES>& figure_normals,
    const AABB& figure_aabb,
    const std::array<AABB, TREE_NUM_TRIANGLES>& figure_tri_aabbs,
    const SolutionEval& eval,
    int target_index,
    const std::vector<Index>* candidates_ptr,
    std::vector<Index>& candidates_scratch
) const {
    const auto& solution = eval.solution;
    const auto& grid = solution.grid();

    const std::vector<Index>* candidates = candidates_ptr;
    if (!candidates) {
        Vec2 center{figure[0].centroid()};
        auto [ci, cj] = grid.compute_ij(center);
        const size_t needed = NEIGHBOR_DELTAS.size() * static_cast<size_t>(grid.capacity());
        candidates_scratch.resize(needed);
        const size_t n_candidates = grid.get_candidates_by_cell(ci, cj, candidates_scratch);
        candidates_scratch.resize(n_candidates);
        candidates = &candidates_scratch;
    }

    const auto& all_normals = solution.normals();
    const auto& all_aabbs = solution.aabbs();
    const auto& all_tri_aabbs = solution.triangle_aabbs();

    float max_violation = 0.0f;
    for (int neighbor_idx : *candidates) {
        if (neighbor_idx < 0 || neighbor_idx == target_index) continue;
        if (!solution.is_valid(static_cast<size_t>(neighbor_idx))) continue;

        const auto& neighbor_aabb = all_aabbs[static_cast<size_t>(neighbor_idx)];
        if (!figure_aabb.intersects(neighbor_aabb)) continue;

        const auto& neighbor_fig = solution.figures()[static_cast<size_t>(neighbor_idx)];
        const auto& neighbor_normals = all_normals[static_cast<size_t>(neighbor_idx)];
        const auto& neighbor_tri_aabbs = all_tri_aabbs[static_cast<size_t>(neighbor_idx)];

        for (size_t i = 0; i < TREE_NUM_TRIANGLES; ++i) {
            if (!figure_tri_aabbs[i].intersects(neighbor_aabb)) continue;
            for (size_t j = 0; j < TREE_NUM_TRIANGLES; ++j) {
                if (!figure_tri_aabbs[i].intersects(neighbor_tri_aabbs[j])) continue;
                const float score = triangles_intersection_score_from_normals(
                    figure[i], neighbor_fig[j], figure_normals[i], neighbor_normals[j]
                );
                max_violation = std::max(max_violation, score);
            }
        }
    }

    return max_violation;
}

inline BeamDescentSolver::EvalFG BeamDescentSolver::evaluate_params(
    const TreeParams& params,
    const SolutionEval& eval,
    int target_index,
    const std::vector<Index>* candidates_ptr,
    EvalScratch& scratch,
    const GlobalState* global_state
) const {
    Figure figure = params_to_figure(params);

    get_tree_normals(params.angle, scratch.normals);
    compute_triangle_aabbs_and_figure_aabb(figure, scratch.tri_aabbs, scratch.figure_aabb);

    EvalFG out{};
    if (config_.objective_type == ObjectiveType::Zero) {
        out.f = 0.0f;
        out.g = compute_constraint_only(
            figure, scratch.normals, scratch.figure_aabb, scratch.tri_aabbs,
            eval, target_index, candidates_ptr, scratch.candidates_storage
        );
    } else if (config_.objective_type == ObjectiveType::Linf) {
        out.f = compute_objective_linf(figure);
        out.g = compute_constraint_only(
            figure, scratch.normals, scratch.figure_aabb, scratch.tri_aabbs,
            eval, target_index, candidates_ptr, scratch.candidates_storage
        );
    } else {
        const auto fg = compute_objective_and_constraint_distance(
            figure, scratch.normals, scratch.figure_aabb, scratch.tri_aabbs,
            eval, target_index, candidates_ptr, scratch.candidates_storage
        );
        out.f = fg.first;
        out.g = fg.second;
    }

    out.cost = out.f + compute_penalty(out.g, config_.mu);
    return out;
}

inline void BeamDescentSolver::insert_into_beam(
    const TreeParams& params,
    float cost,
    std::vector<TreeParams>& beam_params,
    std::vector<float>& beam_costs,
    int& beam_size
) const {
    const int width = std::max(1, config_.beam_width);
    if (beam_params.size() != static_cast<size_t>(width)) {
        beam_params.assign(static_cast<size_t>(width), TreeParams{});
        beam_costs.assign(static_cast<size_t>(width), std::numeric_limits<float>::max());
        beam_size = 0;
    }

    if (beam_size < width) {
        beam_params[static_cast<size_t>(beam_size)] = params;
        beam_costs[static_cast<size_t>(beam_size)] = cost;
        ++beam_size;
    } else if (cost >= beam_costs[static_cast<size_t>(width - 1)]) {
        return;
    } else {
        beam_params[static_cast<size_t>(width - 1)] = params;
        beam_costs[static_cast<size_t>(width - 1)] = cost;
    }

    // Insertion-sort the new tail element into place (small width).
    int i = beam_size - 1;
    while (i > 0) {
        const size_t si = static_cast<size_t>(i);
        const size_t sp = static_cast<size_t>(i - 1);
        if (beam_costs[sp] <= beam_costs[si]) break;
        std::swap(beam_costs[sp], beam_costs[si]);
        std::swap(beam_params[sp], beam_params[si]);
        --i;
    }
}

inline TreeParams BeamDescentSolver::clamp_to_region(
    const TreeParams& params,
    const SearchRegion& region
) const {
    TreeParams out = params;
    out.pos.x = std::clamp(out.pos.x, region.min_x, region.max_x);
    out.pos.y = std::clamp(out.pos.y, region.min_y, region.max_y);
    out.angle = wrap_angle(out.angle);
    return out;
}

inline SolverResult BeamDescentSolver::solve(
    const SolutionEval& eval,
    std::span<const int> indices,
    RNG& rng,
    const GlobalState* global_state
) const {
    SolverResult result;
    result.params.resize(indices.size());
    if (indices.empty()) {
        return result;
    }

    // This solver is primarily tuned for single-tree placement. We solve the
    // first index and copy the result for any additional indices.
    const int target_index = indices[0];
    const auto& solution = eval.solution;

    SearchRegion region = compute_search_region(
        solution, target_index, config_.constrain_to_cell, config_.prefer_current_cell, rng
    );
    const std::vector<Index>* region_candidates = region.has_candidates ? &region.candidates : nullptr;

    const int lattice_xy = std::max(2, config_.lattice_xy);
    const int lattice_ang = std::max(1, config_.lattice_ang);
    const int beam_width = std::max(1, config_.beam_width);

    std::vector<TreeParams> beam_params(static_cast<size_t>(beam_width));
    std::vector<float> beam_costs(static_cast<size_t>(beam_width), std::numeric_limits<float>::max());
    int beam_size = 0;

    EvalScratch scratch{};
    // Pre-reserve candidate scratch to avoid repeated allocations on global search.
    {
        const auto& grid = solution.grid();
        const size_t needed = NEIGHBOR_DELTAS.size() * static_cast<size_t>(grid.capacity());
        scratch.candidates_storage.reserve(needed);
    }

    // Deterministic lattice seeding across (x, y, angle).
    for (int ix = 0; ix < lattice_xy; ++ix) {
        const float tx = static_cast<float>(ix) / static_cast<float>(lattice_xy - 1);
        const float x = lerp(region.min_x, region.max_x, tx);
        for (int iy = 0; iy < lattice_xy; ++iy) {
            const float ty = static_cast<float>(iy) / static_cast<float>(lattice_xy - 1);
            const float y = lerp(region.min_y, region.max_y, ty);
            for (int ia = 0; ia < lattice_ang; ++ia) {
                const float ta = (static_cast<float>(ia) + 0.5f) / static_cast<float>(lattice_ang);
                const float angle = lerp(-PI, PI, ta);
                const TreeParams params{x, y, angle};
                const auto fg = evaluate_params(params, eval, target_index, region_candidates, scratch, global_state);
                insert_into_beam(params, fg.cost, beam_params, beam_costs, beam_size);
            }
        }
    }

    // If seeding produced nothing (should not happen), fall back to region center.
    if (beam_size == 0) {
        const float cx = 0.5f * (region.min_x + region.max_x);
        const float cy = 0.5f * (region.min_y + region.max_y);
        TreeParams fallback{cx, cy, 0.0f};
        const auto fg = evaluate_params(fallback, eval, target_index, region_candidates, scratch, global_state);
        insert_into_beam(fallback, fg.cost, beam_params, beam_costs, beam_size);
    }

    // Compute start metrics from current params when valid; otherwise region center.
    {
        TreeParams start_params{};
        if (solution.is_valid(static_cast<size_t>(target_index))) {
            start_params = solution.get_params(static_cast<size_t>(target_index));
        } else {
            const float cx = 0.5f * (region.min_x + region.max_x);
            const float cy = 0.5f * (region.min_y + region.max_y);
            start_params = TreeParams{cx, cy, 0.0f};
        }
        const auto start_fg = evaluate_params(start_params, eval, target_index, region_candidates, scratch, global_state);
        result.start_f = start_fg.f;
        result.start_g = start_fg.g;
    }

    // Coordinate descent refinement from each beam seed.
    TreeParams best_params = beam_params[0];
    float best_cost = beam_costs[0];
    EvalFG best_fg = evaluate_params(best_params, eval, target_index, region_candidates, scratch, global_state);

    for (int bi = 0; bi < beam_size; ++bi) {
        TreeParams current = beam_params[static_cast<size_t>(bi)];
        EvalFG current_fg = evaluate_params(current, eval, target_index, region_candidates, scratch, global_state);

        for (int level = 0; level < std::max(1, config_.descent_levels); ++level) {
            const float step_xy = config_.step_xy0 * std::pow(config_.step_xy_decay, static_cast<float>(level));
            const float step_ang = config_.step_ang0 * std::pow(config_.step_ang_decay, static_cast<float>(level));

            for (int iter = 0; iter < std::max(1, config_.max_iters_per_level); ++iter) {
                TreeParams best_local = current;
                EvalFG best_local_fg = current_fg;

                const std::array<TreeParams, 6> candidates = {{
                    clamp_to_region(TreeParams{current.pos.x + step_xy, current.pos.y, current.angle}, region),
                    clamp_to_region(TreeParams{current.pos.x - step_xy, current.pos.y, current.angle}, region),
                    clamp_to_region(TreeParams{current.pos.x, current.pos.y + step_xy, current.angle}, region),
                    clamp_to_region(TreeParams{current.pos.x, current.pos.y - step_xy, current.angle}, region),
                    clamp_to_region(TreeParams{current.pos.x, current.pos.y, current.angle + step_ang}, region),
                    clamp_to_region(TreeParams{current.pos.x, current.pos.y, current.angle - step_ang}, region),
                }};

                for (const auto& cand : candidates) {
                    const EvalFG cand_fg = evaluate_params(cand, eval, target_index, region_candidates, scratch, global_state);
                    if (cand_fg.cost < best_local_fg.cost) {
                        best_local = cand;
                        best_local_fg = cand_fg;
                    }
                }

                // No improvement at this step size.
                if (best_local_fg.cost >= current_fg.cost) {
                    break;
                }

                current = best_local;
                current_fg = best_local_fg;
            }
        }

        if (current_fg.cost < best_cost) {
            best_cost = current_fg.cost;
            best_params = current;
            best_fg = current_fg;
        }
    }

    result.params.set(0, best_params);
    result.final_f = best_fg.f;
    result.final_g = best_fg.g;

    // Copy the same placement to any additional requested indices.
    for (size_t i = 1; i < indices.size(); ++i) {
        result.params.set(i, best_params);
    }

    return result;
}

}  // namespace tree_packing
