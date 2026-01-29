/**
 * Detailed profiling of PSO solver computation breakdown.
 * Uses pre-computed normals and AABB culling (optimized path).
 */

#include "tree_packing/tree_packing.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>

using namespace tree_packing;
using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::micro>;

int main() {
    std::cout << "=== Detailed PSO Computation Profiling (normals + AABB culling) ===\n\n";

    // Setup
    constexpr int n_trees = 30;
    constexpr float side = 5.0f;
    constexpr uint64_t seed = 42;

    Problem problem = Problem::create_tree_packing_problem();
    Solution solution = Solution::init_random(n_trees, side, seed);
    SolutionEval eval = problem.eval(solution);

    const auto& grid = solution.grid();
    const auto& figures = solution.figures();
    const auto& all_normals = solution.normals();
    const auto& all_aabbs = solution.aabbs();
    const auto& all_tri_aabbs = solution.triangle_aabbs();

    std::cout << "Setup: " << n_trees << " trees, side=" << side << "\n";
    std::cout << "Grid cells: " << grid.grid_n() << " x " << grid.grid_n() << "\n\n";

    // Pick a random cell and get candidates
    RNG rng(seed);
    float center_x = rng.uniform(-side, side);
    float center_y = rng.uniform(-side, side);
    Vec2 center{center_x, center_y};
    auto [ci, cj] = grid.compute_ij(center);
    std::vector<Index> candidates = grid.get_candidates_by_cell(ci, cj);

    std::cout << "Test cell: (" << ci << ", " << cj << ")\n";
    std::cout << "Candidates in cell: " << candidates.size() << "\n\n";

    // Count valid candidates
    int valid_count = 0;
    for (int idx : candidates) {
        if (idx >= 0 && solution.is_valid(idx)) valid_count++;
    }
    std::cout << "Valid candidates: " << valid_count << "\n\n";

    // Profile individual operations
    constexpr int n_repeats = 1000;

    // 1. Profile params_to_figure + get_tree_normals + compute AABBs
    std::cout << "--- params_to_figure + normals + AABBs ---\n";
    TreeParams test_params{center, 0.5f};
    auto t0 = Clock::now();
    for (int i = 0; i < n_repeats; ++i) {
        Figure fig = params_to_figure(test_params);
        std::array<std::array<Vec2, 3>, TREE_NUM_TRIANGLES> normals;
        get_tree_normals(test_params.angle, normals);
        AABB fig_aabb;
        std::array<AABB, TREE_NUM_TRIANGLES> tri_aabbs;
        for (size_t t = 0; t < TREE_NUM_TRIANGLES; ++t) {
            const auto& tri = fig[t];
            tri_aabbs[t] = AABB();
            tri_aabbs[t].expand(tri.v0);
            tri_aabbs[t].expand(tri.v1);
            tri_aabbs[t].expand(tri.v2);
            fig_aabb.expand(tri.v0);
            fig_aabb.expand(tri.v1);
            fig_aabb.expand(tri.v2);
        }
        (void)fig;
        (void)normals;
        (void)fig_aabb;
        (void)tri_aabbs;
    }
    double params_to_fig_us = Duration(Clock::now() - t0).count() / n_repeats;
    std::cout << "  Time: " << std::fixed << std::setprecision(3) << params_to_fig_us << " us\n\n";

    // Prepare test figure and its data
    Figure test_fig = params_to_figure(test_params);
    std::array<std::array<Vec2, 3>, TREE_NUM_TRIANGLES> test_normals;
    get_tree_normals(test_params.angle, test_normals);
    AABB test_aabb;
    std::array<AABB, TREE_NUM_TRIANGLES> test_tri_aabbs;
    for (size_t i = 0; i < TREE_NUM_TRIANGLES; ++i) {
        const auto& tri = test_fig[i];
        test_tri_aabbs[i] = AABB();
        test_tri_aabbs[i].expand(tri.v0);
        test_tri_aabbs[i].expand(tri.v1);
        test_tri_aabbs[i].expand(tri.v2);
        test_aabb.expand(tri.v0);
        test_aabb.expand(tri.v1);
        test_aabb.expand(tri.v2);
    }

    // 2. Profile triangles_intersection_score_from_normals (single pair)
    std::cout << "--- triangles_intersection_score_from_normals (single pair) ---\n";
    Triangle tri1 = test_fig[0];
    Triangle tri2 = figures[0][0];
    const auto& n1 = test_normals[0];
    const auto& n2 = all_normals[0][0];
    t0 = Clock::now();
    for (int i = 0; i < n_repeats; ++i) {
        volatile float score = triangles_intersection_score_from_normals(tri1, tri2, n1, n2);
        (void)score;
    }
    double single_intersection_us = Duration(Clock::now() - t0).count() / n_repeats;
    std::cout << "  Time: " << single_intersection_us << " us\n\n";

    // 3. Profile full figure-to-figure intersection with AABB culling
    std::cout << "--- figure-to-figure intersection (with AABB culling) ---\n";
    const Figure& neighbor_fig = figures[0];
    const auto& neighbor_normals = all_normals[0];
    const auto& neighbor_aabb = all_aabbs[0];
    const auto& neighbor_tri_aabbs = all_tri_aabbs[0];
    int pairs_checked_fig = 0;
    t0 = Clock::now();
    for (int rep = 0; rep < n_repeats; ++rep) {
        float max_score = 0.0f;
        if (test_aabb.intersects(neighbor_aabb)) {
            for (size_t ti = 0; ti < TREE_NUM_TRIANGLES; ++ti) {
                if (!test_tri_aabbs[ti].intersects(neighbor_aabb)) continue;
                for (size_t tj = 0; tj < TREE_NUM_TRIANGLES; ++tj) {
                    if (!test_tri_aabbs[ti].intersects(neighbor_tri_aabbs[tj])) continue;
                    if (rep == 0) pairs_checked_fig++;
                    float score = triangles_intersection_score_from_normals(
                        test_fig[ti], neighbor_fig[tj],
                        test_normals[ti], neighbor_normals[tj]
                    );
                    max_score = std::max(max_score, score);
                }
            }
        }
        volatile float result = max_score;
        (void)result;
    }
    double fig_to_fig_us = Duration(Clock::now() - t0).count() / n_repeats;
    std::cout << "  Time: " << fig_to_fig_us << " us\n";
    std::cout << "  Triangle pairs checked: " << pairs_checked_fig << " / 25\n";
    if (pairs_checked_fig > 0) {
        std::cout << "  Per triangle pair: " << fig_to_fig_us / pairs_checked_fig << " us\n";
    }
    std::cout << "\n";

    // 4. Profile constraint computation with AABB culling
    std::cout << "--- compute_constraint (with AABB culling) ---\n";
    int total_pairs_constraint = 0;
    int figures_skipped_constraint = 0;
    t0 = Clock::now();
    for (int rep = 0; rep < n_repeats; ++rep) {
        float max_violation = 0.0f;
        for (int neighbor_idx : candidates) {
            if (neighbor_idx < 0 || !solution.is_valid(neighbor_idx)) continue;

            const auto& naabb = all_aabbs[neighbor_idx];
            // Figure-level AABB culling
            if (!test_aabb.intersects(naabb)) {
                if (rep == 0) figures_skipped_constraint++;
                continue;
            }

            const auto& nfig = figures[neighbor_idx];
            const auto& nnormals = all_normals[neighbor_idx];
            const auto& ntri_aabbs = all_tri_aabbs[neighbor_idx];

            for (size_t ti = 0; ti < TREE_NUM_TRIANGLES; ++ti) {
                if (!test_tri_aabbs[ti].intersects(naabb)) continue;
                for (size_t tj = 0; tj < TREE_NUM_TRIANGLES; ++tj) {
                    if (!test_tri_aabbs[ti].intersects(ntri_aabbs[tj])) continue;
                    if (rep == 0) total_pairs_constraint++;
                    float score = triangles_intersection_score_from_normals(
                        test_fig[ti], nfig[tj],
                        test_normals[ti], nnormals[tj]
                    );
                    max_violation = std::max(max_violation, score);
                }
            }
        }
        volatile float result = max_violation;
        (void)result;
    }
    double constraint_us = Duration(Clock::now() - t0).count() / n_repeats;
    std::cout << "  Time: " << constraint_us << " us\n";
    std::cout << "  Figures skipped (AABB): " << figures_skipped_constraint << " / " << valid_count << "\n";
    std::cout << "  Triangle pairs checked: " << total_pairs_constraint << " / " << valid_count * 25 << "\n";
    if (total_pairs_constraint > 0) {
        std::cout << "  Per triangle pair: " << constraint_us / total_pairs_constraint << " us\n";
    }
    std::cout << "\n";

    // 5. Profile objective computation with AABB culling
    std::cout << "--- compute_objective (with AABB culling) ---\n";
    int total_pairs_objective = 0;
    int figures_skipped_objective = 0;
    t0 = Clock::now();
    for (int rep = 0; rep < n_repeats; ++rep) {
        float total_score = 0.0f;
        int count = 0;
        for (int neighbor_idx : candidates) {
            if (neighbor_idx < 0 || !solution.is_valid(neighbor_idx)) continue;

            const auto& naabb = all_aabbs[neighbor_idx];
            // Figure-level AABB culling
            if (!test_aabb.intersects(naabb)) {
                if (rep == 0) figures_skipped_objective++;
                total_score += std::log(2.0f);  // Default distance score
                ++count;
                continue;
            }

            const auto& nfig = figures[neighbor_idx];
            const auto& nnormals = all_normals[neighbor_idx];
            const auto& ntri_aabbs = all_tri_aabbs[neighbor_idx];

            float min_score = std::numeric_limits<float>::max();
            for (size_t ti = 0; ti < TREE_NUM_TRIANGLES; ++ti) {
                if (!test_tri_aabbs[ti].intersects(naabb)) continue;
                for (size_t tj = 0; tj < TREE_NUM_TRIANGLES; ++tj) {
                    if (!test_tri_aabbs[ti].intersects(ntri_aabbs[tj])) continue;
                    if (rep == 0) total_pairs_objective++;
                    float score = triangles_intersection_score_from_normals(
                        test_fig[ti], nfig[tj],
                        test_normals[ti], nnormals[tj]
                    );
                    min_score = std::min(min_score, score);
                }
            }
            if (min_score == std::numeric_limits<float>::max()) {
                min_score = -1.0f;
            }
            float dist_score = std::log(1.0f + std::max(0.0f, -min_score));
            total_score += dist_score;
            ++count;
        }
        volatile float result = count > 0 ? total_score / count : 0.0f;
        (void)result;
    }
    double objective_us = Duration(Clock::now() - t0).count() / n_repeats;
    std::cout << "  Time: " << objective_us << " us\n";
    std::cout << "  Figures skipped (AABB): " << figures_skipped_objective << " / " << valid_count << "\n";
    std::cout << "  Triangle pairs checked: " << total_pairs_objective << " / " << valid_count * 25 << "\n";
    if (total_pairs_objective > 0) {
        std::cout << "  Per triangle pair: " << objective_us / total_pairs_objective << " us\n";
    }
    std::cout << "\n";

    // 5b. Profile L-infinity objective (no neighbor checks needed)
    std::cout << "--- compute_objective_linf (L-infinity) ---\n";
    t0 = Clock::now();
    for (int rep = 0; rep < n_repeats; ++rep) {
        float max_abs = 0.0f;
        for (size_t i = 0; i < TREE_NUM_TRIANGLES; ++i) {
            const auto& tri = test_fig[i];
            max_abs = std::max(max_abs, std::abs(tri.v0.x));
            max_abs = std::max(max_abs, std::abs(tri.v0.y));
            max_abs = std::max(max_abs, std::abs(tri.v1.x));
            max_abs = std::max(max_abs, std::abs(tri.v1.y));
            max_abs = std::max(max_abs, std::abs(tri.v2.x));
            max_abs = std::max(max_abs, std::abs(tri.v2.y));
        }
        volatile float result = max_abs;
        (void)result;
    }
    double objective_linf_us = Duration(Clock::now() - t0).count() / n_repeats;
    std::cout << "  Time: " << objective_linf_us << " us\n";
    std::cout << "  Speedup vs distance obj: " << std::setprecision(1) << (objective_us / objective_linf_us) << "x\n\n";

    // 6. Profile full fitness (objective + constraint)
    std::cout << "--- compute_fitness (obj + constraint) ---\n";
    double fitness_us = constraint_us + objective_us;
    std::cout << "  Estimated time (distance): " << fitness_us << " us\n";
    double fitness_linf_us = constraint_us + objective_linf_us;
    std::cout << "  Estimated time (L-inf):    " << fitness_linf_us << " us\n\n";

    // 7. Profile grid query
    std::cout << "--- grid.get_candidates_by_cell ---\n";
    t0 = Clock::now();
    for (int i = 0; i < n_repeats; ++i) {
        volatile auto cands = grid.get_candidates_by_cell(ci, cj);
        (void)cands;
    }
    double grid_query_us = Duration(Clock::now() - t0).count() / n_repeats;
    std::cout << "  Time: " << grid_query_us << " us\n\n";

    // Summary
    std::cout << std::string(60, '=') << "\n";
    std::cout << "SUMMARY - Single PSO evaluation breakdown\n";
    std::cout << std::string(60, '=') << "\n\n";

    std::cout << "Per-particle fitness evaluation:\n";
    std::cout << "  params + normals + AABBs:   " << std::setw(8) << params_to_fig_us << " us\n";
    std::cout << "  constraint (max violation): " << std::setw(8) << constraint_us << " us\n";
    std::cout << "  objective (distance score): " << std::setw(8) << objective_us << " us\n";
    std::cout << "  ----------------------------------------\n";
    double total_per_particle = params_to_fig_us + constraint_us + objective_us;
    std::cout << "  TOTAL per particle:         " << std::setw(8) << total_per_particle << " us\n\n";

    // PSO iteration estimate
    int n_particles = 16;
    int n_pso_iterations = 20;
    double pso_iteration_us = total_per_particle * n_particles;
    double full_pso_us = pso_iteration_us * n_pso_iterations;

    std::cout << "PSO with " << n_particles << " particles, " << n_pso_iterations << " iterations:\n";
    std::cout << "  Per PSO iteration:          " << std::setw(8) << pso_iteration_us / 1000.0 << " ms\n";
    std::cout << "  Full PSO solve:             " << std::setw(8) << full_pso_us / 1000.0 << " ms\n\n";

    // AABB culling stats
    int total_pairs_without_culling = valid_count * 25;
    int total_pairs_with_culling = (total_pairs_constraint + total_pairs_objective) / 2;  // Average
    double culling_ratio = 100.0 * (1.0 - static_cast<double>(total_pairs_with_culling) / total_pairs_without_culling);
    std::cout << "AABB culling effectiveness:\n";
    std::cout << "  Pairs without culling: " << total_pairs_without_culling << "\n";
    std::cout << "  Pairs with culling:    " << total_pairs_with_culling << " (avg)\n";
    std::cout << "  Reduction:             " << std::setprecision(1) << culling_ratio << "%\n\n";

    // Breakdown by component
    std::cout << "Time breakdown (% of fitness eval):\n";
    double total = params_to_fig_us + constraint_us + objective_us;
    std::cout << "  params + normals + AABBs: " << std::setw(5) << std::setprecision(1)
              << (params_to_fig_us / total * 100) << "%\n";
    std::cout << "  constraint:               " << std::setw(5) << (constraint_us / total * 100) << "%\n";
    std::cout << "  objective:                " << std::setw(5) << (objective_us / total * 100) << "%\n";

    return 0;
}
