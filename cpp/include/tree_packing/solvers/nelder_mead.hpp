#pragma once

#include "solver.hpp"
#include "../core/tree.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace tree_packing {

/**
 * Nelder-Mead solver for joint optimization of 1..N trees (practically 1-5).
 *
 * This implementation uses the incremental evaluation API on a temporary copy
 * of SolutionEval to score each simplex vertex. It is deterministic given the
 * RNG seed and the initial parameters.
 */
class NelderMeadSolver : public Solver {
public:
    enum class ObjectiveType {
        Distance,  // Use Problem objective
        Linf,      // Use max-abs hull coordinate across involved trees
        Zero       // Constraint satisfaction only (+ objective ceiling penalty)
    };

    struct Config {
        // Simplex step sizes per parameter type.
        float step_xy{0.25f};
        float step_ang{PI / 24.0f};
        // Randomize initial simplex around x0 (helps escape flat regions).
        bool randomize_simplex{true};
        float jitter_xy{0.5f};         // Additional xy jitter scale
        float jitter_ang{PI / 12.0f};  // Additional angle jitter scale

        // Nelder-Mead coefficients.
        float alpha{1.0f};  // reflection
        float gamma{2.0f};  // expansion
        float rho{0.5f};    // contraction
        float sigma{0.5f};  // shrink

        // Iteration budget and tolerances.
        int max_iters{80};
        float tol_f{1e-5f};
        float tol_x{1e-4f};

        // Penalized objective weight.
        float mu{1e6f};

        // Search region behavior.
        bool constrain_to_cell{true};
        bool prefer_current_cell{true};  // kept for API compatibility
        ObjectiveType objective_type{ObjectiveType::Distance};
    };

    NelderMeadSolver() = default;
    explicit NelderMeadSolver(const Config& config) : config_(config) {}

    [[nodiscard]] SolverResult solve(
        const SolutionEval& eval,
        std::span<const int> indices,
        RNG& rng,
        const GlobalState* global_state = nullptr
    ) const override;

    [[nodiscard]] SolverPtr clone() const override {
        auto ptr = std::make_unique<NelderMeadSolver>(config_);
        ptr->set_bounds(min_pos_, max_pos_);
        ptr->set_problem(problem_);
        return ptr;
    }

    [[nodiscard]] int max_group_size() const override { return 5; }

    [[nodiscard]] const Config& config() const { return config_; }
    Config& config() { return config_; }

private:
    Config config_;

    struct Regions {
        std::vector<SearchRegion> per_tree{};
    };

    [[nodiscard]] static float wrap_angle(float angle) {
        while (angle > PI) angle -= TWO_PI;
        while (angle < -PI) angle += TWO_PI;
        return angle;
    }

    [[nodiscard]] Regions compute_regions(
        const Solution& solution,
        std::span<const int> indices,
        RNG& rng
    ) const {
        Regions r{};
        r.per_tree.reserve(indices.size());
        for (int idx : indices) {
            r.per_tree.push_back(compute_search_region(
                solution, idx, config_.constrain_to_cell, config_.prefer_current_cell, rng
            ));
        }
        return r;
    }

    [[nodiscard]] TreeParamsSoA clamp_to_regions(
        std::span<const int> indices,
        const Regions& regions,
        const std::vector<float>& x
    ) const {
        const size_t n = indices.size();
        TreeParamsSoA params(n);
        for (size_t i = 0; i < n; ++i) {
            const size_t base = 3 * i;
            const auto& region = regions.per_tree[i];
            const float px = std::clamp(x[base + 0], region.min_x, region.max_x);
            const float py = std::clamp(x[base + 1], region.min_y, region.max_y);
            const float pa = wrap_angle(x[base + 2]);
            params.set(i, TreeParams{px, py, pa});
        }
        return params;
    }

    [[nodiscard]] std::vector<float> params_to_vec(
        std::span<const int> indices,
        const Solution& solution
    ) const {
        std::vector<float> x(3 * indices.size(), 0.0f);
        for (size_t i = 0; i < indices.size(); ++i) {
            const TreeParams p = solution.get_params(static_cast<size_t>(indices[i]));
            const size_t base = 3 * i;
            x[base + 0] = p.pos.x;
            x[base + 1] = p.pos.y;
            x[base + 2] = p.angle;
        }
        return x;
    }

    [[nodiscard]] float evaluate_vertex(
        const SolutionEval& base_eval,
        std::span<const int> indices,
        const Regions& regions,
        const std::vector<float>& x,
        float& out_f,
        float& out_g,
        const GlobalState* global_state
    ) const {
        const TreeParamsSoA params = clamp_to_regions(indices, regions, x);
        const auto& solution = base_eval.solution;
        const auto& grid = solution.grid();

        struct LocalData {
            Figure figure{};
            std::array<std::array<Vec2, 3>, TREE_NUM_TRIANGLES> normals{};
            std::array<AABB, TREE_NUM_TRIANGLES> tri_aabbs{};
            AABB aabb{};
            Vec2 center{};
        };

        std::vector<LocalData> locals;
        locals.resize(params.size());
        for (size_t i = 0; i < params.size(); ++i) {
            const TreeParams p = params.get(i);
            LocalData d{};
            d.figure = params_to_figure(p);
            get_tree_normals(p.angle, d.normals);
            compute_triangle_aabbs_and_figure_aabb(d.figure, d.tri_aabbs, d.aabb);
            d.center = d.figure[0].centroid();
            locals[i] = d;
        }

        float f = 0.0f;
        float g = 0.0f;

        if (config_.objective_type == ObjectiveType::Linf) {
            float max_abs = 0.0f;
            for (const auto& d : locals) {
                for (const auto& [ti, vi] : CONVEX_HULL_INDEXES) {
                    const auto& tri = d.figure[static_cast<size_t>(ti)];
                    const Vec2& v = (vi == 0) ? tri.v0 : (vi == 1) ? tri.v1 : tri.v2;
                    max_abs = std::max(max_abs, std::abs(v.x));
                    max_abs = std::max(max_abs, std::abs(v.y));
                }
            }
            f = max_abs;
        }

        // Evaluate constraints (and distance objective if requested) against fixed neighbors.
        for (size_t i = 0; i < locals.size(); ++i) {
            const auto& d = locals[i];
            auto [ci, cj] = grid.compute_ij(d.center);
            const size_t needed = NEIGHBOR_DELTAS.size() * static_cast<size_t>(grid.capacity());
            std::vector<Index> candidates(needed);
            const size_t n_candidates = grid.get_candidates_by_cell(ci, cj, candidates);
            candidates.resize(n_candidates);

            float total_score = 0.0f;
            int count = 0;
            float max_violation = 0.0f;

            for (int neighbor_idx : candidates) {
                if (neighbor_idx < 0 || neighbor_idx == indices[i]) continue;
                if (!solution.is_valid(static_cast<size_t>(neighbor_idx))) continue;

                const auto& neighbor_aabb = solution.aabbs()[static_cast<size_t>(neighbor_idx)];
                if (!d.aabb.intersects(neighbor_aabb)) {
                    if (config_.objective_type == ObjectiveType::Distance) {
                        total_score += std::log(2.0f);
                        ++count;
                    }
                    continue;
                }

                const auto& neighbor_fig = solution.figures()[static_cast<size_t>(neighbor_idx)];
                const auto& neighbor_normals = solution.normals()[static_cast<size_t>(neighbor_idx)];
                const auto& neighbor_tri_aabbs = solution.triangle_aabbs()[static_cast<size_t>(neighbor_idx)];

                float min_score = std::numeric_limits<float>::max();
                for (size_t ti = 0; ti < TREE_NUM_TRIANGLES; ++ti) {
                    if (!d.tri_aabbs[ti].intersects(neighbor_aabb)) continue;
                    for (size_t tj = 0; tj < TREE_NUM_TRIANGLES; ++tj) {
                        if (!d.tri_aabbs[ti].intersects(neighbor_tri_aabbs[tj])) continue;
                        const float score = triangles_intersection_score_from_normals(
                            d.figure[ti], neighbor_fig[tj], d.normals[ti], neighbor_normals[tj]
                        );
                        min_score = std::min(min_score, score);
                        max_violation = std::max(max_violation, score);
                    }
                }

                if (config_.objective_type == ObjectiveType::Distance) {
                    if (min_score == std::numeric_limits<float>::max()) {
                        min_score = -1.0f;
                    }
                    const float dist_score = std::log(1.0f + std::max(0.0f, -min_score));
                    total_score += dist_score;
                    ++count;
                }
            }

            if (config_.objective_type == ObjectiveType::Distance) {
                const float mean_score = count > 0 ? total_score / static_cast<float>(count) : 0.0f;
                const float tgt = std::max(1.0f, 1.0f);
                const float missing = std::max(0.0f, tgt - static_cast<float>(count));
                f += mean_score + std::log(2.0f) * missing;
            }

            g = std::max(g, max_violation);
        }

        // Constraints among updated trees.
        for (size_t i = 0; i < locals.size(); ++i) {
            for (size_t j = i + 1; j < locals.size(); ++j) {
                const auto& a = locals[i];
                const auto& b = locals[j];
                if (!a.aabb.intersects(b.aabb)) continue;
                for (size_t ti = 0; ti < TREE_NUM_TRIANGLES; ++ti) {
                    if (!a.tri_aabbs[ti].intersects(b.aabb)) continue;
                    for (size_t tj = 0; tj < TREE_NUM_TRIANGLES; ++tj) {
                        if (!a.tri_aabbs[ti].intersects(b.tri_aabbs[tj])) continue;
                        const float score = triangles_intersection_score_from_normals(
                            a.figure[ti], b.figure[tj], a.normals[ti], b.normals[tj]
                        );
                        g = std::max(g, score);
                    }
                }
            }
        }

        out_f = f;
        out_g = g;
        return f + compute_penalty(g, config_.mu);
    }
};

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

inline SolverResult NelderMeadSolver::solve(
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
    const size_t n_trees = indices.size();
    const size_t dim = 3 * n_trees;
    const size_t n_vertices = dim + 1;

    // Regions (cell-constrained when configured).
    const Regions regions = compute_regions(solution, indices, rng);

    // Starting point from current params (or region centers if invalid).
    std::vector<float> x0(dim, 0.0f);
    for (size_t i = 0; i < n_trees; ++i) {
        const int idx = indices[i];
        const size_t base = 3 * i;
        if (solution.is_valid(static_cast<size_t>(idx))) {
            const TreeParams p = solution.get_params(static_cast<size_t>(idx));
            x0[base + 0] = p.pos.x;
            x0[base + 1] = p.pos.y;
            x0[base + 2] = p.angle;
        } else {
            const auto& r = regions.per_tree[i];
            x0[base + 0] = 0.5f * (r.min_x + r.max_x);
            x0[base + 1] = 0.5f * (r.min_y + r.max_y);
            x0[base + 2] = 0.0f;
        }
    }

    // Initialize simplex around x0.
    std::vector<std::vector<float>> simplex(n_vertices, std::vector<float>(dim, 0.0f));
    simplex[0] = x0;
    for (size_t i = 0; i < dim; ++i) {
        simplex[i + 1] = x0;
        const bool is_angle = (i % 3) == 2;
        const float step = is_angle ? config_.step_ang : config_.step_xy;
        simplex[i + 1][i] += step;
    }

    // Optional randomized jitter to diversify the initial simplex.
    if (config_.randomize_simplex) {
        for (size_t v = 1; v < n_vertices; ++v) {
            for (size_t t = 0; t < n_trees; ++t) {
                const auto& region = regions.per_tree[t];
                const size_t base = 3 * t;
                simplex[v][base + 0] += rng.uniform(-config_.jitter_xy, config_.jitter_xy);
                simplex[v][base + 1] += rng.uniform(-config_.jitter_xy, config_.jitter_xy);
                simplex[v][base + 2] += rng.uniform(-config_.jitter_ang, config_.jitter_ang);
                // Keep jittered vertices within the allowed region.
                simplex[v][base + 0] = std::clamp(simplex[v][base + 0], region.min_x, region.max_x);
                simplex[v][base + 1] = std::clamp(simplex[v][base + 1], region.min_y, region.max_y);
                simplex[v][base + 2] = wrap_angle(simplex[v][base + 2]);
            }
        }
    }

    std::vector<float> values(n_vertices, std::numeric_limits<float>::max());
    std::vector<float> fvals(n_vertices, 0.0f);
    std::vector<float> gvals(n_vertices, 0.0f);
    for (size_t i = 0; i < n_vertices; ++i) {
        values[i] = evaluate_vertex(eval, indices, regions, simplex[i], fvals[i], gvals[i], global_state);
    }

    // Start metrics based on the first vertex.
    result.start_f = fvals[0];
    result.start_g = gvals[0];

    auto sort_simplex = [&]() {
        std::vector<size_t> order(n_vertices);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return values[a] < values[b];
        });

        auto simplex_copy = simplex;
        auto values_copy = values;
        auto f_copy = fvals;
        auto g_copy = gvals;
        for (size_t i = 0; i < n_vertices; ++i) {
            simplex[i] = simplex_copy[order[i]];
            values[i] = values_copy[order[i]];
            fvals[i] = f_copy[order[i]];
            gvals[i] = g_copy[order[i]];
        }
    };

    std::vector<float> centroid(dim, 0.0f);
    std::vector<float> xr(dim, 0.0f);
    std::vector<float> xe(dim, 0.0f);
    std::vector<float> xc(dim, 0.0f);

    for (int iter = 0; iter < std::max(1, config_.max_iters); ++iter) {
        sort_simplex();

        const float best = values[0];
        const float worst = values[n_vertices - 1];
        if (std::abs(worst - best) < config_.tol_f) {
            break;
        }

        // Compute centroid of all but worst.
        std::fill(centroid.begin(), centroid.end(), 0.0f);
        for (size_t i = 0; i < n_vertices - 1; ++i) {
            for (size_t d = 0; d < dim; ++d) {
                centroid[d] += simplex[i][d];
            }
        }
        const float inv = 1.0f / static_cast<float>(n_vertices - 1);
        for (float& v : centroid) v *= inv;

        // Reflection.
        const auto& xw = simplex[n_vertices - 1];
        for (size_t d = 0; d < dim; ++d) {
            xr[d] = centroid[d] + config_.alpha * (centroid[d] - xw[d]);
        }
        float fr = 0.0f, gr = 0.0f;
        const float vr = evaluate_vertex(eval, indices, regions, xr, fr, gr, global_state);

        if (vr < values[0]) {
            // Expansion.
            for (size_t d = 0; d < dim; ++d) {
                xe[d] = centroid[d] + config_.gamma * (xr[d] - centroid[d]);
            }
            float fe = 0.0f, ge = 0.0f;
            const float ve = evaluate_vertex(eval, indices, regions, xe, fe, ge, global_state);
            if (ve < vr) {
                simplex[n_vertices - 1] = xe;
                values[n_vertices - 1] = ve;
                fvals[n_vertices - 1] = fe;
                gvals[n_vertices - 1] = ge;
            } else {
                simplex[n_vertices - 1] = xr;
                values[n_vertices - 1] = vr;
                fvals[n_vertices - 1] = fr;
                gvals[n_vertices - 1] = gr;
            }
            continue;
        }

        if (vr < values[n_vertices - 2]) {
            // Accept reflection.
            simplex[n_vertices - 1] = xr;
            values[n_vertices - 1] = vr;
            fvals[n_vertices - 1] = fr;
            gvals[n_vertices - 1] = gr;
            continue;
        }

        // Contraction.
        const bool outside = (vr < values[n_vertices - 1]);
        for (size_t d = 0; d < dim; ++d) {
            const float dir = outside ? (xr[d] - centroid[d]) : (xw[d] - centroid[d]);
            xc[d] = centroid[d] + config_.rho * dir;
        }
        float fc = 0.0f, gc = 0.0f;
        const float vc = evaluate_vertex(eval, indices, regions, xc, fc, gc, global_state);
        if (vc < (outside ? vr : values[n_vertices - 1])) {
            simplex[n_vertices - 1] = xc;
            values[n_vertices - 1] = vc;
            fvals[n_vertices - 1] = fc;
            gvals[n_vertices - 1] = gc;
            continue;
        }

        // Shrink towards best.
        const auto& xb = simplex[0];
        float max_dx = 0.0f;
        for (size_t i = 1; i < n_vertices; ++i) {
            for (size_t d = 0; d < dim; ++d) {
                simplex[i][d] = xb[d] + config_.sigma * (simplex[i][d] - xb[d]);
                max_dx = std::max(max_dx, std::abs(simplex[i][d] - xb[d]));
            }
            values[i] = evaluate_vertex(eval, indices, regions, simplex[i], fvals[i], gvals[i], global_state);
        }
        if (max_dx < config_.tol_x) {
            break;
        }
    }

    sort_simplex();
    const TreeParamsSoA best_params = clamp_to_regions(indices, regions, simplex[0]);
    result.params = best_params;
    result.final_f = fvals[0];
    result.final_g = gvals[0];

    return result;
}

}  // namespace tree_packing
