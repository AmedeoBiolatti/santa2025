#pragma once

#include "solver.hpp"
#include "fitness_utils.hpp"
#include "../core/tree.hpp"
#include "../geometry/sat.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace tree_packing {

/**
 * Random Sampling solver for single tree placement.
 *
 * Samples N random positions and returns the best one based on
 * a fitness function (objective + penalty * constraint violation).
 */
class RandomSamplingSolver : public Solver {
public:
    // Objective function type (same as PSO)
    enum class ObjectiveType {
        Distance,  // Distance-based score using neighbor intersections
        Linf,      // L-infinity norm: max(max(abs(x)), max(abs(y))) of figure vertices
        Zero       // No objective, only constraint satisfaction + ceiling check
    };

    struct Config {
        int n_samples{100};             // Number of random samples
        float mu{1e6f};                 // Penalty multiplier for constraint violation
        // Isolation penalty for Distance objective: discourages escaping to empty space.
        // We charge an upper-bound per missing neighbor.
        float isolation_penalty_per_missing{std::log(2.0f)}; // Upper bound per missing neighbor
        float target_neighbors{8.0f};                        // Neighbor count that removes isolation penalty
        bool constrain_to_cell{true};    // If true, constrain search to a grid cell
        bool prefer_current_cell{true};  // Kept for API compatibility (current cell is always used when valid)
        ObjectiveType objective_type{ObjectiveType::Linf};  // Which objective to use
    };

    RandomSamplingSolver() = default;
    explicit RandomSamplingSolver(const Config& config) : config_(config) {}

    [[nodiscard]] SolverResult solve(
        const SolutionEval& eval,
        std::span<const int> indices,
        RNG& rng,
        const GlobalState* global_state = nullptr
    ) const override;

    [[nodiscard]] SolverPtr clone() const override {
        auto ptr = std::make_unique<RandomSamplingSolver>(config_);
        ptr->set_bounds(min_pos_, max_pos_);
        ptr->set_problem(problem_);
        return ptr;
    }

    // Accessors for config
    [[nodiscard]] const Config& config() const { return config_; }
    Config& config() { return config_; }

private:
    Config config_;

    // Compute fitness for a single sample (lower is better)
    [[nodiscard]] float compute_fitness(
        const TreeParams& params,
        const SolutionEval& eval,
        int target_index,
        const std::vector<Index>* candidates,
        const GlobalState* global_state = nullptr
    ) const;

    // Compute constraint violation (intersection score)
    [[nodiscard]] float compute_constraint(
        const Figure& figure,
        const std::array<std::array<Vec2, 3>, TREE_NUM_TRIANGLES>& figure_normals,
        const AABB& figure_aabb,
        const std::array<AABB, TREE_NUM_TRIANGLES>& figure_tri_aabbs,
        const SolutionEval& eval,
        int target_index,
        const std::vector<Index>* candidates
    ) const;

    // Compute objective (L-infinity norm - max abs coordinate)
    [[nodiscard]] float compute_objective(const Figure& figure) const;
};

// Implementation

inline float RandomSamplingSolver::compute_objective(const Figure& figure) const {
    // L-infinity norm across convex hull vertices
    float max_abs = 0.0f;
    for (const auto& [ti, vi] : CONVEX_HULL_INDEXES) {
        const auto& tri = figure[static_cast<size_t>(ti)];
        const Vec2& v = (vi == 0) ? tri.v0 : (vi == 1) ? tri.v1 : tri.v2;
        max_abs = std::max(max_abs, std::abs(v.x));
        max_abs = std::max(max_abs, std::abs(v.y));
    }
    return max_abs;
}

inline float RandomSamplingSolver::compute_constraint(
    const Figure& figure,
    const std::array<std::array<Vec2, 3>, TREE_NUM_TRIANGLES>& figure_normals,
    const AABB& figure_aabb,
    const std::array<AABB, TREE_NUM_TRIANGLES>& figure_tri_aabbs,
    const SolutionEval& eval,
    int target_index,
    const std::vector<Index>* candidates_ptr
) const {
    const auto& solution = eval.solution;

    // Use provided candidates or query grid
    std::vector<Index> candidates_storage;
    const std::vector<Index>* candidates;
    if (candidates_ptr && !candidates_ptr->empty()) {
        candidates = candidates_ptr;
    } else {
        const auto& grid = solution.grid();
        Vec2 center{figure[0].centroid()};
        auto [ci, cj] = grid.compute_ij(center);
        const size_t needed = NEIGHBOR_DELTAS.size() * static_cast<size_t>(grid.capacity());
        candidates_storage.resize(needed);
        const size_t n_candidates = grid.get_candidates_by_cell(ci, cj, candidates_storage);
        candidates_storage.resize(n_candidates);
        candidates = &candidates_storage;
    }

    const auto& all_normals = solution.normals();
    const auto& all_aabbs = solution.aabbs();
    const auto& all_tri_aabbs = solution.triangle_aabbs();
    float max_violation = 0.0f;

    for (int neighbor_idx : *candidates) {
        if (neighbor_idx < 0 || neighbor_idx == target_index) continue;
        if (!solution.is_valid(neighbor_idx)) continue;

        const auto& neighbor_aabb = all_aabbs[neighbor_idx];

        // Figure-level AABB culling
        if (!figure_aabb.intersects(neighbor_aabb)) continue;

        const auto& neighbor_fig = solution.figures()[neighbor_idx];
        const auto& neighbor_normals = all_normals[neighbor_idx];
        const auto& neighbor_tri_aabbs = all_tri_aabbs[neighbor_idx];

        // Compute max intersection score across triangle pairs
        for (size_t i = 0; i < TREE_NUM_TRIANGLES; ++i) {
            if (!figure_tri_aabbs[i].intersects(neighbor_aabb)) continue;
            for (size_t j = 0; j < TREE_NUM_TRIANGLES; ++j) {
                if (!figure_tri_aabbs[i].intersects(neighbor_tri_aabbs[j])) continue;
                float score = triangles_intersection_score_from_normals(
                    figure[i], neighbor_fig[j],
                    figure_normals[i], neighbor_normals[j]
                );
                max_violation = std::max(max_violation, score);
            }
        }
    }

    return max_violation;
}

inline float RandomSamplingSolver::compute_fitness(
    const TreeParams& params,
    const SolutionEval& eval,
    int target_index,
    const std::vector<Index>* candidates,
    const GlobalState* global_state
) const {
    Figure figure = params_to_figure(params);

    // Compute normals
    std::array<std::array<Vec2, 3>, TREE_NUM_TRIANGLES> figure_normals;
    get_tree_normals(params.angle, figure_normals);

    // Compute AABBs
    AABB figure_aabb;
    std::array<AABB, TREE_NUM_TRIANGLES> figure_tri_aabbs;
    compute_triangle_aabbs_and_figure_aabb(figure, figure_tri_aabbs, figure_aabb);

    // Compute objective based on config
    float f;
    if (config_.objective_type == ObjectiveType::Linf) {
        f = compute_objective(figure);
        float g = compute_constraint(figure, figure_normals, figure_aabb, figure_tri_aabbs, eval, target_index, candidates);
        return f + compute_penalty(g, config_.mu);
    }
    if (config_.objective_type == ObjectiveType::Zero) {
        const float g0 = compute_constraint(
            figure, figure_normals, figure_aabb, figure_tri_aabbs, eval, target_index, candidates
        );
        return compute_penalty(g0, config_.mu);
    }

    // Distance objective: merge objective + constraint in single pass (like PSO)
    const auto& solution = eval.solution;
    std::vector<Index> candidates_storage;
    const std::vector<Index>* cands = candidates;
    if (!cands || cands->empty()) {
        const auto& grid = solution.grid();
        Vec2 center{figure[0].centroid()};
        auto [ci, cj] = grid.compute_ij(center);
        const size_t needed = NEIGHBOR_DELTAS.size() * static_cast<size_t>(grid.capacity());
        candidates_storage.resize(needed);
        const size_t n_candidates = grid.get_candidates_by_cell(ci, cj, candidates_storage);
        candidates_storage.resize(n_candidates);
        cands = &candidates_storage;
    }

    const auto& all_normals = solution.normals();
    const auto& all_aabbs = solution.aabbs();
    const auto& all_tri_aabbs = solution.triangle_aabbs();

    float total_score = 0.0f;
    int count = 0;
    float max_violation = 0.0f;

    for (int neighbor_idx : *cands) {
        if (neighbor_idx < 0 || neighbor_idx == target_index) continue;
        if (!solution.is_valid(neighbor_idx)) continue;

        const auto& neighbor_aabb = all_aabbs[neighbor_idx];

        if (!figure_aabb.intersects(neighbor_aabb)) {
            total_score += std::log(2.0f);
            ++count;
            continue;
        }

        const auto& neighbor_fig = solution.figures()[neighbor_idx];
        const auto& neighbor_normals = all_normals[neighbor_idx];
        const auto& neighbor_tri_aabbs = all_tri_aabbs[neighbor_idx];

        float min_score = std::numeric_limits<float>::max();

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
    f = mean_score + isolation;
    return f + compute_penalty(max_violation, config_.mu);
}

inline SolverResult RandomSamplingSolver::solve(
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

    // For now, only handle single index
    int target_index = indices[0];
    const auto& solution = eval.solution;

    // Compute search region
    SearchRegion region = compute_search_region(
        solution, target_index, config_.constrain_to_cell, config_.prefer_current_cell, rng
    );
    const std::vector<Index>* candidates_ptr = region.has_candidates ? &region.candidates : nullptr;

    // Sample random positions and find the best
    float best_cost = std::numeric_limits<float>::max();
    TreeParams best_params{Vec2{0.0f, 0.0f}, 0.0f};

    for (int i = 0; i < config_.n_samples; ++i) {
        // Random position and angle
        float x = rng.uniform(region.min_x, region.max_x);
        float y = rng.uniform(region.min_y, region.max_y);
        float angle = rng.uniform(-PI, PI);

        TreeParams params{Vec2{x, y}, angle};
        float cost = compute_fitness(params, eval, target_index, candidates_ptr, global_state);

        if (cost < best_cost) {
            best_cost = cost;
            best_params = params;
        }
    }

    // Store result
    result.params.set(0, best_params);

    // Compute final values
    Figure final_figure = params_to_figure(best_params);
    std::array<std::array<Vec2, 3>, TREE_NUM_TRIANGLES> final_normals;
    get_tree_normals(best_params.angle, final_normals);
    AABB final_aabb;
    std::array<AABB, TREE_NUM_TRIANGLES> final_tri_aabbs;
    for (size_t i = 0; i < TREE_NUM_TRIANGLES; ++i) {
        const auto& tri = final_figure[i];
        final_tri_aabbs[i] = AABB();
        final_tri_aabbs[i].expand(tri.v0);
        final_tri_aabbs[i].expand(tri.v1);
        final_tri_aabbs[i].expand(tri.v2);
        final_aabb.expand(tri.v0);
        final_aabb.expand(tri.v1);
        final_aabb.expand(tri.v2);
    }

    if (config_.objective_type == ObjectiveType::Zero) {
        result.final_f = 0.0f;
        result.final_g = compute_constraint(
            final_figure, final_normals, final_aabb, final_tri_aabbs, eval, target_index, candidates_ptr
        );
    } else if (config_.objective_type == ObjectiveType::Linf) {
        result.final_f = compute_objective(final_figure);
        result.final_g = compute_constraint(final_figure, final_normals, final_aabb, final_tri_aabbs, eval, target_index, candidates_ptr);
    } else {
        // Distance objective: compute f and g together.
        const auto& solution_ref = eval.solution;
        std::vector<Index> candidates_storage;
        const std::vector<Index>* cands = candidates_ptr;
        if (!cands || cands->empty()) {
            const auto& grid = solution_ref.grid();
            Vec2 center{final_figure[0].centroid()};
            auto [ci, cj] = grid.compute_ij(center);
            const size_t needed = NEIGHBOR_DELTAS.size() * static_cast<size_t>(grid.capacity());
            candidates_storage.resize(needed);
            const size_t n_candidates = grid.get_candidates_by_cell(ci, cj, candidates_storage);
            candidates_storage.resize(n_candidates);
            cands = &candidates_storage;
        }

        const auto& all_normals = solution_ref.normals();
        const auto& all_aabbs = solution_ref.aabbs();
        const auto& all_tri_aabbs = solution_ref.triangle_aabbs();

        float total_score = 0.0f;
        int count = 0;
        float max_violation = 0.0f;
        for (int neighbor_idx : *cands) {
            if (neighbor_idx < 0 || neighbor_idx == target_index) continue;
            if (!solution_ref.is_valid(static_cast<size_t>(neighbor_idx))) continue;

            const auto& neighbor_aabb = all_aabbs[static_cast<size_t>(neighbor_idx)];
            if (!final_aabb.intersects(neighbor_aabb)) {
                total_score += std::log(2.0f);
                ++count;
                continue;
            }

            const auto& neighbor_fig = solution_ref.figures()[static_cast<size_t>(neighbor_idx)];
            const auto& neighbor_normals = all_normals[static_cast<size_t>(neighbor_idx)];
            const auto& neighbor_tri_aabbs = all_tri_aabbs[static_cast<size_t>(neighbor_idx)];

            float min_score = std::numeric_limits<float>::max();
            for (size_t i = 0; i < TREE_NUM_TRIANGLES; ++i) {
                if (!final_tri_aabbs[i].intersects(neighbor_aabb)) continue;
                for (size_t j = 0; j < TREE_NUM_TRIANGLES; ++j) {
                    if (!final_tri_aabbs[i].intersects(neighbor_tri_aabbs[j])) continue;
                    const float score = triangles_intersection_score_from_normals(
                        final_figure[i], neighbor_fig[j], final_normals[i], neighbor_normals[j]
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
        result.final_f = mean_score + isolation;
        result.final_g = max_violation;
    }

    // Copy result to other indices if needed
    for (size_t i = 1; i < indices.size(); ++i) {
        result.params.set(i, best_params);
    }

    return result;
}

}  // namespace tree_packing
