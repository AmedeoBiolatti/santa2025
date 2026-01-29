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
 * Noise solver: start from the current params, add random perturbations,
 * evaluate N variations, and return the best.
 *
 * This is intentionally simple, deterministic given the RNG seed, and
 * allocation-light in the hot loop.
 */
class NoiseSolver : public Solver {
public:
    enum class ObjectiveType {
        Distance,  // Distance-based score using neighbor intersections
        Linf,      // L-infinity norm: max abs coordinate of hull vertices
        Zero       // Constraint satisfaction only (+ objective ceiling penalty)
    };

    struct Config {
        int n_variations{128};             // Number of noisy candidates
        float pos_sigma{0.25f};            // Position noise scale (uniform in [-sigma, sigma])
        float ang_sigma{PI / 16.0f};       // Angle noise scale (uniform in [-sigma, sigma])
        float mu{1e6f};                    // Penalty multiplier for constraint violation
        // Isolation penalty for Distance objective: upper bound per missing neighbor.
        float isolation_penalty_per_missing{std::log(2.0f)};
        float target_neighbors{8.0f};
        bool constrain_to_cell{true};      // If true, constrain search to a grid cell
        bool prefer_current_cell{true};    // Kept for API compatibility (current cell is always used when valid)
        ObjectiveType objective_type{ObjectiveType::Distance};
    };

    NoiseSolver() = default;
    explicit NoiseSolver(const Config& config) : config_(config) {}

    [[nodiscard]] SolverResult solve(
        const SolutionEval& eval,
        std::span<const int> indices,
        RNG& rng,
        const GlobalState* global_state = nullptr
    ) const override;

    [[nodiscard]] SolverPtr clone() const override {
        auto ptr = std::make_unique<NoiseSolver>(config_);
        ptr->set_bounds(min_pos_, max_pos_);
        ptr->set_problem(problem_);
        return ptr;
    }

    [[nodiscard]] int max_group_size() const override { return 2; }
    [[nodiscard]] bool wants_rng_split() const override { return config_.n_variations != 1; }

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

    struct FigureData {
        Figure figure{};
        std::array<std::array<Vec2, 3>, TREE_NUM_TRIANGLES> normals{};
        std::array<AABB, TREE_NUM_TRIANGLES> tri_aabbs{};
        AABB figure_aabb{};
    };

    [[nodiscard]] static float wrap_angle(float angle) {
        while (angle > PI) angle -= TWO_PI;
        while (angle < -PI) angle += TWO_PI;
        return angle;
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

    [[nodiscard]] std::pair<float, float> compute_objective_and_constraint_distance_with_extra(
        const FigureData& data,
        const SolutionEval& eval,
        int target_index,
        const std::vector<Index>& candidates,
        const FigureData* extra
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

    [[nodiscard]] EvalFG evaluate_params_pair(
        const TreeParams& params,
        const SolutionEval& eval,
        int target_index,
        const std::vector<Index>& candidates,
        const TreeParams* other_params
    ) const;

    [[nodiscard]] TreeParams clamp_to_region(
        const TreeParams& params,
        const SearchRegion& region
    ) const;

    [[nodiscard]] FigureData build_figure_data(const TreeParams& params) const;

    void gather_candidates(
        const Figure& figure,
        const Solution& solution,
        const std::vector<Index>* region_candidates,
        EvalScratch& scratch,
        std::vector<Index>& out
    ) const;
};

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

inline float NoiseSolver::compute_objective_linf(const Figure& figure) const {
    float max_abs = 0.0f;
    for (const auto& [ti, vi] : CONVEX_HULL_INDEXES) {
        const auto& tri = figure[static_cast<size_t>(ti)];
        const Vec2& v = (vi == 0) ? tri.v0 : (vi == 1) ? tri.v1 : tri.v2;
        max_abs = std::max(max_abs, std::abs(v.x));
        max_abs = std::max(max_abs, std::abs(v.y));
    }
    return max_abs;
}

inline std::pair<float, float> NoiseSolver::compute_objective_and_constraint_distance(
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

inline std::pair<float, float> NoiseSolver::compute_objective_and_constraint_distance_with_extra(
    const FigureData& data,
    const SolutionEval& eval,
    int target_index,
    const std::vector<Index>& candidates,
    const FigureData* extra
) const {
    const auto& solution = eval.solution;
    const auto& all_normals = solution.normals();
    const auto& all_aabbs = solution.aabbs();
    const auto& all_tri_aabbs = solution.triangle_aabbs();

    float total_score = 0.0f;
    int count = 0;
    float max_violation = 0.0f;

    auto eval_neighbor = [&](const FigureData& neighbor) {
        if (!data.figure_aabb.intersects(neighbor.figure_aabb)) {
            total_score += std::log(2.0f);
            ++count;
            return;
        }

        float min_score = std::numeric_limits<float>::max();
        for (size_t i = 0; i < TREE_NUM_TRIANGLES; ++i) {
            if (!data.tri_aabbs[i].intersects(neighbor.figure_aabb)) continue;
            for (size_t j = 0; j < TREE_NUM_TRIANGLES; ++j) {
                if (!data.tri_aabbs[i].intersects(neighbor.tri_aabbs[j])) continue;
                const float score = triangles_intersection_score_from_normals(
                    data.figure[i], neighbor.figure[j], data.normals[i], neighbor.normals[j]
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
    };

    for (int neighbor_idx : candidates) {
        if (neighbor_idx < 0 || neighbor_idx == target_index) continue;
        if (!solution.is_valid(static_cast<size_t>(neighbor_idx))) continue;

        const FigureData neighbor{
            solution.figures()[static_cast<size_t>(neighbor_idx)],
            all_normals[static_cast<size_t>(neighbor_idx)],
            all_tri_aabbs[static_cast<size_t>(neighbor_idx)],
            all_aabbs[static_cast<size_t>(neighbor_idx)],
        };
        eval_neighbor(neighbor);
    }

    if (extra != nullptr) {
        eval_neighbor(*extra);
    }

    const float mean_score = count > 0 ? total_score / static_cast<float>(count) : 0.0f;
    const float tgt = std::max(1.0f, config_.target_neighbors);
    const float missing = std::max(0.0f, tgt - static_cast<float>(count));
    const float isolation = config_.isolation_penalty_per_missing * missing;
    const float f = mean_score + isolation;
    const float g = max_violation;
    return {f, g};
}

inline float NoiseSolver::compute_constraint_only(
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

inline NoiseSolver::FigureData NoiseSolver::build_figure_data(const TreeParams& params) const {
    FigureData data{};
    data.figure = params_to_figure(params);
    get_tree_normals(params.angle, data.normals);
    compute_triangle_aabbs_and_figure_aabb(data.figure, data.tri_aabbs, data.figure_aabb);
    return data;
}

inline void NoiseSolver::gather_candidates(
    const Figure& figure,
    const Solution& solution,
    const std::vector<Index>* region_candidates,
    EvalScratch& scratch,
    std::vector<Index>& out
) const {
    out.clear();
    if (region_candidates != nullptr) {
        out = *region_candidates;
        return;
    }

    const auto& grid = solution.grid();
    Vec2 center{figure[0].centroid()};
    auto [ci, cj] = grid.compute_ij(center);
    const size_t needed = NEIGHBOR_DELTAS.size() * static_cast<size_t>(grid.capacity());
    scratch.candidates_storage.resize(needed);
    const size_t n_candidates = grid.get_candidates_by_cell(ci, cj, scratch.candidates_storage);
    scratch.candidates_storage.resize(n_candidates);
    out = scratch.candidates_storage;
}

inline NoiseSolver::EvalFG NoiseSolver::evaluate_params_pair(
    const TreeParams& params,
    const SolutionEval& eval,
    int target_index,
    const std::vector<Index>& candidates,
    const TreeParams* other_params
) const {
    const FigureData data = build_figure_data(params);
    FigureData other_data{};
    const FigureData* other_ptr = nullptr;
    if (other_params != nullptr) {
        other_data = build_figure_data(*other_params);
        other_ptr = &other_data;
    }

    EvalFG out{};
    if (config_.objective_type == ObjectiveType::Zero) {
        out.f = 0.0f;
        std::vector<Index> candidates_scratch = candidates;
        out.g = compute_constraint_only(
            data.figure, data.normals, data.figure_aabb, data.tri_aabbs,
            eval, target_index, &candidates_scratch, candidates_scratch
        );
        if (other_ptr != nullptr && data.figure_aabb.intersects(other_ptr->figure_aabb)) {
            for (size_t i = 0; i < TREE_NUM_TRIANGLES; ++i) {
                if (!data.tri_aabbs[i].intersects(other_ptr->figure_aabb)) continue;
                for (size_t j = 0; j < TREE_NUM_TRIANGLES; ++j) {
                    if (!data.tri_aabbs[i].intersects(other_ptr->tri_aabbs[j])) continue;
                    const float score = triangles_intersection_score_from_normals(
                        data.figure[i], other_ptr->figure[j], data.normals[i], other_ptr->normals[j]
                    );
                    out.g = std::max(out.g, score);
                }
            }
        }
    } else if (config_.objective_type == ObjectiveType::Linf) {
        out.f = compute_objective_linf(data.figure);
        // For Linf, constraints still depend on neighbors (and the paired tree).
        std::vector<Index> candidates_scratch = candidates;
        out.g = compute_constraint_only(
            data.figure, data.normals, data.figure_aabb, data.tri_aabbs,
            eval, target_index, &candidates_scratch, candidates_scratch
        );
        if (other_ptr != nullptr) {
            const auto fg_extra = compute_objective_and_constraint_distance_with_extra(
                data, eval, target_index, candidates, other_ptr
            );
            out.g = std::max(out.g, fg_extra.second);
        }
    } else {
        const auto fg = compute_objective_and_constraint_distance_with_extra(
            data, eval, target_index, candidates, other_ptr
        );
        out.f = fg.first;
        out.g = fg.second;
    }

    out.cost = out.f + compute_penalty(out.g, config_.mu);
    return out;
}

inline NoiseSolver::EvalFG NoiseSolver::evaluate_params(
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

inline TreeParams NoiseSolver::clamp_to_region(
    const TreeParams& params,
    const SearchRegion& region
) const {
    TreeParams out = params;
    out.pos.x = std::clamp(out.pos.x, region.min_x, region.max_x);
    out.pos.y = std::clamp(out.pos.y, region.min_y, region.max_y);
    out.angle = wrap_angle(out.angle);
    return out;
}

inline SolverResult NoiseSolver::solve(
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

    const auto& solution = eval.solution;

    const int n_variations = std::max(1, config_.n_variations);
    const float pos_sigma = std::max(0.0f, config_.pos_sigma);
    const float ang_sigma = std::max(0.0f, config_.ang_sigma);

    // Jointly handle two indices when requested; fall back to single-index logic otherwise.
    if (indices.size() == 2) {
        const int idx0 = indices[0];
        const int idx1 = indices[1];

        SearchRegion region0 = compute_search_region(
            solution, idx0, config_.constrain_to_cell, config_.prefer_current_cell, rng
        );
        SearchRegion region1 = compute_search_region(
            solution, idx1, config_.constrain_to_cell, config_.prefer_current_cell, rng
        );
        const std::vector<Index>* region_candidates0 = region0.has_candidates ? &region0.candidates : nullptr;
        const std::vector<Index>* region_candidates1 = region1.has_candidates ? &region1.candidates : nullptr;

        EvalScratch scratch0{};
        EvalScratch scratch1{};
        {
            const auto& grid = solution.grid();
            const size_t needed = NEIGHBOR_DELTAS.size() * static_cast<size_t>(grid.capacity());
            scratch0.candidates_storage.reserve(needed);
            scratch1.candidates_storage.reserve(needed);
        }

        TreeParams base0{};
        if (solution.is_valid(static_cast<size_t>(idx0))) {
            base0 = solution.get_params(static_cast<size_t>(idx0));
        } else {
            const float cx = 0.5f * (region0.min_x + region0.max_x);
            const float cy = 0.5f * (region0.min_y + region0.max_y);
            base0 = TreeParams{cx, cy, 0.0f};
        }
        base0 = clamp_to_region(base0, region0);

        TreeParams base1{};
        if (solution.is_valid(static_cast<size_t>(idx1))) {
            base1 = solution.get_params(static_cast<size_t>(idx1));
        } else {
            const float cx = 0.5f * (region1.min_x + region1.max_x);
            const float cy = 0.5f * (region1.min_y + region1.max_y);
            base1 = TreeParams{cx, cy, 0.0f};
        }
        base1 = clamp_to_region(base1, region1);

        std::vector<Index> candidates0;
        std::vector<Index> candidates1;
        std::vector<Index> union_candidates;
        {
            const auto& grid = solution.grid();
            const size_t needed = NEIGHBOR_DELTAS.size() * static_cast<size_t>(grid.capacity());
            union_candidates.reserve(needed * 2);
        }

        auto add_unique = [&](int idx) {
            if (idx < 0 || idx == idx0 || idx == idx1) return;
            for (int existing : union_candidates) {
                if (existing == idx) return;
            }
            union_candidates.push_back(static_cast<Index>(idx));
        };

        auto build_union_candidates = [&](const TreeParams& p0, const TreeParams& p1) {
            const FigureData d0 = build_figure_data(p0);
            const FigureData d1 = build_figure_data(p1);
            gather_candidates(d0.figure, solution, region_candidates0, scratch0, candidates0);
            gather_candidates(d1.figure, solution, region_candidates1, scratch1, candidates1);
            union_candidates.clear();
            for (int c : candidates0) add_unique(c);
            for (int c : candidates1) add_unique(c);
        };

        build_union_candidates(base0, base1);

        EvalFG start0{};
        EvalFG start1{};
        start0.f = std::numeric_limits<float>::infinity();
        start0.g = 0.0f;
        start0.cost = std::numeric_limits<float>::infinity();
        start1 = start0;
        if (config_.objective_type == ObjectiveType::Zero) {
            result.start_f = start0.f;
            result.start_g = start0.g;
        } else {
            result.start_f = start0.f + start1.f;
            result.start_g = std::max(start0.g, start1.g);
        }

        TreeParams best0 = base0;
        TreeParams best1 = base1;
        EvalFG best_fg0 = start0;
        EvalFG best_fg1 = start1;
        float best_cost = (config_.objective_type == ObjectiveType::Zero)
            ? start0.cost
            : (start0.cost + start1.cost);

        for (int i = 0; i < n_variations; ++i) {
            const float dx0 = pos_sigma * rng.uniform(-1.0f, 1.0f);
            const float dy0 = pos_sigma * rng.uniform(-1.0f, 1.0f);
            const float da0 = ang_sigma * rng.uniform(-1.0f, 1.0f);
            const float dx1 = pos_sigma * rng.uniform(-1.0f, 1.0f);
            const float dy1 = pos_sigma * rng.uniform(-1.0f, 1.0f);
            const float da1 = ang_sigma * rng.uniform(-1.0f, 1.0f);
            TreeParams cand0{
                base0.pos.x + dx0,
                base0.pos.y + dy0,
                base0.angle + da0
            };
            TreeParams cand1{
                base1.pos.x + dx1,
                base1.pos.y + dy1,
                base1.angle + da1
            };
            cand0 = clamp_to_region(cand0, region0);
            cand1 = clamp_to_region(cand1, region1);

            build_union_candidates(cand0, cand1);
            const EvalFG fg0 = evaluate_params_pair(cand0, eval, idx0, union_candidates, &cand1);
            const EvalFG fg1 = evaluate_params_pair(cand1, eval, idx1, union_candidates, &cand0);
            const float cost = (config_.objective_type == ObjectiveType::Zero)
                ? fg0.cost
                : (fg0.cost + fg1.cost);
            if (cost < best_cost) {
                best_cost = cost;
                best0 = cand0;
                best1 = cand1;
                best_fg0 = fg0;
                best_fg1 = fg1;
            }
        }

        result.params.set(0, best0);
        result.params.set(1, best1);
        if (config_.objective_type == ObjectiveType::Zero) {
            result.final_f = best_fg0.f;
            result.final_g = best_fg0.g;
        } else {
            result.final_f = best_fg0.f + best_fg1.f;
            result.final_g = std::max(best_fg0.g, best_fg1.g);
        }
        return result;
    }

    // Single-index fallback.
    const int target_index = indices[0];
    SearchRegion region = compute_search_region(
        solution, target_index, config_.constrain_to_cell, config_.prefer_current_cell, rng
    );
    const std::vector<Index>* region_candidates = region.has_candidates ? &region.candidates : nullptr;

    EvalScratch scratch{};
    {
        const auto& grid = solution.grid();
        const size_t needed = NEIGHBOR_DELTAS.size() * static_cast<size_t>(grid.capacity());
        scratch.candidates_storage.reserve(needed);
    }

    TreeParams base_params{};
    if (solution.is_valid(static_cast<size_t>(target_index))) {
        base_params = solution.get_params(static_cast<size_t>(target_index));
    } else {
        const float cx = 0.5f * (region.min_x + region.max_x);
        const float cy = 0.5f * (region.min_y + region.max_y);
        base_params = TreeParams{cx, cy, 0.0f};
    }
    base_params = clamp_to_region(base_params, region);

    EvalFG start_fg{};
    start_fg.f = std::numeric_limits<float>::infinity();
    start_fg.g = 0.0f;
    start_fg.cost = std::numeric_limits<float>::infinity();
    result.start_f = start_fg.f;
    result.start_g = start_fg.g;

    TreeParams best_params = base_params;
    EvalFG best_fg = start_fg;
    for (int i = 0; i < n_variations; ++i) {
        const float dx = pos_sigma * rng.uniform(-1.0f, 1.0f);
        const float dy = pos_sigma * rng.uniform(-1.0f, 1.0f);
        const float da = ang_sigma * rng.uniform(-1.0f, 1.0f);
        TreeParams cand{
            base_params.pos.x + dx,
            base_params.pos.y + dy,
            base_params.angle + da
        };
        cand = clamp_to_region(cand, region);
        const EvalFG cand_fg = evaluate_params(cand, eval, target_index, region_candidates, scratch, global_state);
        if (cand_fg.cost < best_fg.cost) {
            best_params = cand;
            best_fg = cand_fg;
        }
    }

    result.params.set(0, best_params);
    result.final_f = best_fg.f;
    result.final_g = best_fg.g;
    for (size_t i = 1; i < indices.size(); ++i) {
        result.params.set(i, best_params);
    }
    return result;
}

}  // namespace tree_packing
