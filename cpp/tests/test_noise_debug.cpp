#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "tree_packing/tree_packing.hpp"
#include "tree_packing/constraints/intersection_tree_filter.hpp"
#include "tree_packing/geometry/sat.hpp"
#include <iostream>
#include <iomanip>
#include <map>
#include <set>

using namespace tree_packing;
using Catch::Approx;

#ifndef INTERSECTION_DEBUG_LOGS
#define INTERSECTION_DEBUG_LOGS 0
#endif

#if INTERSECTION_DEBUG_LOGS
#define INTERSECTION_LOG(...) do { __VA_ARGS__; } while (0)
#else
#define INTERSECTION_LOG(...) do {} while (0)
#endif

// Helper to build a map of (i,j) -> violation from IntersectionMap
static std::map<std::pair<int,int>, float> build_violation_map_noise(const SolutionEval::IntersectionMap& imap) {
    std::map<std::pair<int,int>, float> result;
    for (size_t i = 0; i < imap.size(); ++i) {
        for (const auto& entry : imap[i]) {
            int j = entry.neighbor;
            if (j >= 0) {
                int a = std::min((int)i, j);
                int b = std::max((int)i, j);
                result[{a, b}] = entry.score;
            }
        }
    }
    return result;
}

TEST_CASE("Debug NoiseOptimizer zero noise", "[debug-noise]") {
    Problem problem = Problem::create_tree_packing_problem();
    Solution sol = Solution::init_random(15, 5.0f, 42);
    SolutionEval eval = problem.eval(sol);

    // Store original state
    std::vector<TreeParams> original_params;
    std::vector<bool> original_valid;
    for (size_t i = 0; i < sol.size(); ++i) {
        original_params.push_back(sol.get_params(i));
        original_valid.push_back(sol.is_valid(i));
    }

    auto original_violations = build_violation_map_noise(eval.intersection_map);
    double original_total_violation = eval.intersection_violation;

    NoiseOptimizer noise(0.0f, 3);  // Zero noise
    noise.set_problem(&problem);

    GlobalState global_state(42, eval);
    std::any state = noise.init_state(eval);

    float initial_score = problem.score(eval, global_state);

    INTERSECTION_LOG(std::cout << "\n=== INITIAL STATE ===" << std::endl);
    INTERSECTION_LOG(std::cout << "Score: " << initial_score << std::endl);
    INTERSECTION_LOG(std::cout << "Total intersection_violation: " << original_total_violation << std::endl);
    INTERSECTION_LOG(std::cout << "Number of intersecting pairs: " << original_violations.size() << std::endl);

    // Apply noise 20 times
    for (int i = 0; i < 20; ++i) {
        RNG rng(global_state.split_rng());
        noise.apply(eval, state, global_state, rng);
        global_state.next();
    }

    float final_score = problem.score(eval, global_state);
    auto final_violations = build_violation_map_noise(eval.intersection_map);
    double final_total_violation = eval.intersection_violation;

    INTERSECTION_LOG(std::cout << "\n=== FINAL STATE (after 20 zero-noise iterations) ===" << std::endl);
    INTERSECTION_LOG(std::cout << "Score: " << final_score << std::endl);
    INTERSECTION_LOG(std::cout << "Total intersection_violation: " << final_total_violation << std::endl);
    INTERSECTION_LOG(std::cout << "Number of intersecting pairs: " << final_violations.size() << std::endl);

    // Check if params changed
    bool params_changed = false;
    for (size_t i = 0; i < sol.size(); ++i) {
        if (original_valid[i] != eval.solution.is_valid(i)) {
            params_changed = true;
            INTERSECTION_LOG(std::cout << "Tree " << i << " validity changed!" << std::endl);
        } else if (eval.solution.is_valid(i)) {
            TreeParams p = eval.solution.get_params(i);
            if (std::abs(p.pos.x - original_params[i].pos.x) > 1e-6f ||
                std::abs(p.pos.y - original_params[i].pos.y) > 1e-6f ||
                std::abs(p.angle - original_params[i].angle) > 1e-6f) {
                params_changed = true;
                INTERSECTION_LOG(std::cout << "Tree " << i << " params changed!" << std::endl);
            }
        }
    }
    INTERSECTION_LOG(std::cout << "Params changed: " << (params_changed ? "YES" : "NO") << std::endl);

    // Find differing violations
    INTERSECTION_LOG(std::cout << "\n=== VIOLATION DIFFERENCES ===" << std::endl);
    std::set<std::pair<int,int>> all_pairs;
    for (const auto& [pair, _] : original_violations) all_pairs.insert(pair);
    for (const auto& [pair, _] : final_violations) all_pairs.insert(pair);

    IntersectionTreeFilter filter;

    int diff_count = 0;
    for (const auto& pair : all_pairs) {
        float orig_v = 0.0f, final_v = 0.0f;
        bool in_orig = original_violations.count(pair) > 0;
        bool in_final = final_violations.count(pair) > 0;

        if (in_orig) orig_v = original_violations.at(pair);
        if (in_final) final_v = final_violations.at(pair);

        if (std::abs(orig_v - final_v) > 1e-6f || in_orig != in_final) {
            diff_count++;
            int i = pair.first;
            int j = pair.second;

            INTERSECTION_LOG(std::cout << std::fixed << std::setprecision(8));
            INTERSECTION_LOG(std::cout << "\nPair (" << i << ", " << j << "):" << std::endl);
            INTERSECTION_LOG(std::cout << "  Original violation: " << (in_orig ? std::to_string(orig_v) : "NOT PRESENT") << std::endl);
            INTERSECTION_LOG(std::cout << "  Final violation: " << (in_final ? std::to_string(final_v) : "NOT PRESENT") << std::endl);

            // Print tree parameters
            const auto& pa = original_params[i];
            const auto& pb = original_params[j];
            INTERSECTION_LOG(std::cout << "  Tree " << i << ": pos=(" << pa.pos.x << ", " << pa.pos.y << "), angle=" << pa.angle << std::endl);
            INTERSECTION_LOG(std::cout << "  Tree " << j << ": pos=(" << pb.pos.x << ", " << pb.pos.y << "), angle=" << pb.angle << std::endl);

            // Check leaf indices for both orderings
            int leaf_ij = filter.leaf_index_for(eval.solution, i, j);
            int leaf_ji = filter.leaf_index_for(eval.solution, j, i);
            INTERSECTION_LOG(std::cout << "  Leaf (i,j)=" << leaf_ij << ", Leaf (j,i)=" << leaf_ji << std::endl);

            // Get predictions for both orderings
            std::array<uint8_t, 16> pred_ij, pred_ji;
            filter.leaf_pred_for(eval.solution, i, j, pred_ij);
            filter.leaf_pred_for(eval.solution, j, i, pred_ji);

            INTERSECTION_LOG(std::cout << "  Targets (i,j):");
            for (int t = 0; t < 16; ++t)
                INTERSECTION_LOG(if (pred_ij[t]) std::cout << " " << t);
            INTERSECTION_LOG(std::cout << std::endl);

            INTERSECTION_LOG(std::cout << "  Targets (j,i):");
            for (int t = 0; t < 16; ++t)
                INTERSECTION_LOG(if (pred_ji[t]) std::cout << " " << t);
            INTERSECTION_LOG(std::cout << std::endl);

            // Compute ground truth intersection
            Figure fig_i = params_to_figure(pa);
            Figure fig_j = params_to_figure(pb);
            float ground_truth = 0.0f;
            for (int ti = 0; ti < 5; ++ti) {
                for (int tj = 0; tj < 5; ++tj) {
                    float score = triangles_intersection_score(fig_i[ti], fig_j[tj]);
                    if (score > 0.0f) {
                        ground_truth += score;
                        INTERSECTION_LOG(std::cout << "  tri[" << ti << "] x tri[" << tj << "] = " << score << std::endl);
                    }
                }
            }
            INTERSECTION_LOG(std::cout << "  Ground truth sum: " << ground_truth << " (capped: " << std::min(ground_truth, 0.15f) << ")" << std::endl);
        }
    }

    INTERSECTION_LOG(std::cout << "\nTotal differing pairs: " << diff_count << std::endl);
    INTERSECTION_LOG(std::cout << "Score difference: " << (final_score - initial_score) << std::endl);

    REQUIRE(final_score == Approx(initial_score).margin(1e-5f));
}
