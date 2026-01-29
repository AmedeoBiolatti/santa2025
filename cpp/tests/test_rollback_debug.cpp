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

// Helper to build a map of (i,j) -> violation from IntersectionMap
std::map<std::pair<int,int>, float> build_violation_map(const SolutionEval::IntersectionMap& imap) {
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

// Compute full intersection score between two trees (checking all 25 triangle pairs)
float compute_full_intersection(const Solution& solution, size_t idx_a, size_t idx_b) {
    if (!solution.is_valid(idx_a) || !solution.is_valid(idx_b)) return 0.0f;

    const TreeParams pa = solution.get_params(idx_a);
    const TreeParams pb = solution.get_params(idx_b);

    Figure fig_a = params_to_figure(pa);
    Figure fig_b = params_to_figure(pb);

    float max_score = 0.0f;
    for (int ta = 0; ta < 5; ++ta) {
        for (int tb = 0; tb < 5; ++tb) {
            float score = triangles_intersection_score(fig_a[ta], fig_b[tb]);
            max_score = std::max(max_score, score);
        }
    }
    return max_score;
}

// Print detailed intersection analysis for a tree pair
void analyze_tree_pair(const Solution& solution, size_t idx_a, size_t idx_b, const char* label) {
    std::cout << "\n=== " << label << " Tree Pair (" << idx_a << ", " << idx_b << ") ===" << std::endl;

    // Check cache validity
    bool cache_valid = solution.validate_cache(1e-6f, false);
    std::cout << "Cache valid: " << (cache_valid ? "YES" : "NO") << std::endl;

    if (!solution.is_valid(idx_a) || !solution.is_valid(idx_b)) {
        std::cout << "One or both trees are INVALID" << std::endl;
        return;
    }

    const TreeParams pa = solution.get_params(idx_a);
    const TreeParams pb = solution.get_params(idx_b);

    std::cout << std::fixed << std::setprecision(8);
    std::cout << "Tree " << idx_a << ": pos=(" << pa.pos.x << ", " << pa.pos.y << "), angle=" << pa.angle << std::endl;
    std::cout << "Tree " << idx_b << ": pos=(" << pb.pos.x << ", " << pb.pos.y << "), angle=" << pb.angle << std::endl;

    // Compute features for tree filter
    IntersectionTreeFilter filter;
    std::array<float, 10> feat;
    if (filter.features_for(solution, idx_a, idx_b, feat)) {
        std::cout << "\nFeatures (a=" << idx_a << " -> b=" << idx_b << "):" << std::endl;
        std::cout << "  x=" << feat[0] << ", y=" << feat[1] << ", ang=" << feat[2] << std::endl;
        std::cout << "  r=" << feat[3] << ", cos=" << feat[4] << ", sin=" << feat[5] << std::endl;
        std::cout << "  aabb: min=(" << feat[6] << ", " << feat[8] << "), max=(" << feat[7] << ", " << feat[9] << ")" << std::endl;
    }

    // Get leaf index and predictions - test BOTH orderings
    std::array<uint8_t, 16> pred;
    int leaf_idx = -1;
    std::cout << "\n--- Order (a=" << idx_a << ", b=" << idx_b << ") ---" << std::endl;
    if (filter.leaf_pred_for(solution, idx_a, idx_b, pred, &leaf_idx)) {
        std::cout << "Leaf index: " << leaf_idx << std::endl;
        std::cout << "Leaf predictions (which targets to check):";
        for (int i = 0; i < 16; ++i) {
            if (pred[i]) std::cout << " " << i;
        }
        std::cout << std::endl;
    }

    std::array<float, 10> feat_ba;
    int leaf_idx_ba = -1;
    std::cout << "\n--- Order (a=" << idx_b << ", b=" << idx_a << ") [SWAPPED] ---" << std::endl;
    if (filter.features_for(solution, idx_b, idx_a, feat_ba)) {
        std::cout << "Features: x=" << feat_ba[0] << ", y=" << feat_ba[1] << ", ang=" << feat_ba[2] << std::endl;
    }
    std::array<uint8_t, 16> pred_ba;
    if (filter.leaf_pred_for(solution, idx_b, idx_a, pred_ba, &leaf_idx_ba)) {
        std::cout << "Leaf index: " << leaf_idx_ba << std::endl;
        std::cout << "Leaf predictions (which targets to check):";
        for (int i = 0; i < 16; ++i) {
            if (pred_ba[i]) std::cout << " " << i;
        }
        std::cout << std::endl;
    }

    // Check if swapped gives different pairs
    const auto* pairs_ab = filter.triangle_pairs_for(solution, idx_a, idx_b);
    const auto* pairs_ba = filter.triangle_pairs_for(solution, idx_b, idx_a);
    std::cout << "\nPairs count (a,b)=" << pairs_ab->size() << " vs (b,a)=" << pairs_ba->size() << std::endl;

    // Print triangle AABBs from cache vs freshly computed
    const auto& cached_tri_aabbs_a = solution.triangle_aabbs()[idx_a];
    const auto& cached_tri_aabbs_b = solution.triangle_aabbs()[idx_b];
    const auto& cached_aabb_a = solution.aabbs()[idx_a];
    const auto& cached_aabb_b = solution.aabbs()[idx_b];

    // Compute full intersection (all 25 pairs)
    Figure fig_a = params_to_figure(pa);
    Figure fig_b = params_to_figure(pb);

    // Compute fresh AABBs
    AABB fresh_aabb_a = compute_aabb(fig_a);
    AABB fresh_aabb_b = compute_aabb(fig_b);

    std::cout << "\nFigure AABBs (cached vs fresh):" << std::endl;
    std::cout << "  A cached: min=(" << cached_aabb_a.min.x << ", " << cached_aabb_a.min.y
              << "), max=(" << cached_aabb_a.max.x << ", " << cached_aabb_a.max.y << ")" << std::endl;
    std::cout << "  A fresh:  min=(" << fresh_aabb_a.min.x << ", " << fresh_aabb_a.min.y
              << "), max=(" << fresh_aabb_a.max.x << ", " << fresh_aabb_a.max.y << ")" << std::endl;
    std::cout << "  B cached: min=(" << cached_aabb_b.min.x << ", " << cached_aabb_b.min.y
              << "), max=(" << cached_aabb_b.max.x << ", " << cached_aabb_b.max.y << ")" << std::endl;
    std::cout << "  B fresh:  min=(" << fresh_aabb_b.min.x << ", " << fresh_aabb_b.min.y
              << "), max=(" << fresh_aabb_b.max.x << ", " << fresh_aabb_b.max.y << ")" << std::endl;

    // Check which triangle AABBs differ
    std::cout << "\nTriangle AABBs (showing mismatches):" << std::endl;
    for (int t = 0; t < 5; ++t) {
        AABB fresh_tri;
        fresh_tri.expand(fig_a[t]);
        bool match = (std::abs(cached_tri_aabbs_a[t].min.x - fresh_tri.min.x) < 1e-6f &&
                      std::abs(cached_tri_aabbs_a[t].min.y - fresh_tri.min.y) < 1e-6f &&
                      std::abs(cached_tri_aabbs_a[t].max.x - fresh_tri.max.x) < 1e-6f &&
                      std::abs(cached_tri_aabbs_a[t].max.y - fresh_tri.max.y) < 1e-6f);
        if (!match) {
            std::cout << "  MISMATCH tri_a[" << t << "]: cached=("
                      << cached_tri_aabbs_a[t].min.x << "," << cached_tri_aabbs_a[t].min.y << ")-("
                      << cached_tri_aabbs_a[t].max.x << "," << cached_tri_aabbs_a[t].max.y << ") fresh=("
                      << fresh_tri.min.x << "," << fresh_tri.min.y << ")-("
                      << fresh_tri.max.x << "," << fresh_tri.max.y << ")" << std::endl;
        }
    }
    for (int t = 0; t < 5; ++t) {
        AABB fresh_tri;
        fresh_tri.expand(fig_b[t]);
        bool match = (std::abs(cached_tri_aabbs_b[t].min.x - fresh_tri.min.x) < 1e-6f &&
                      std::abs(cached_tri_aabbs_b[t].min.y - fresh_tri.min.y) < 1e-6f &&
                      std::abs(cached_tri_aabbs_b[t].max.x - fresh_tri.max.x) < 1e-6f &&
                      std::abs(cached_tri_aabbs_b[t].max.y - fresh_tri.max.y) < 1e-6f);
        if (!match) {
            std::cout << "  MISMATCH tri_b[" << t << "]: cached=("
                      << cached_tri_aabbs_b[t].min.x << "," << cached_tri_aabbs_b[t].min.y << ")-("
                      << cached_tri_aabbs_b[t].max.x << "," << cached_tri_aabbs_b[t].max.y << ") fresh=("
                      << fresh_tri.min.x << "," << fresh_tri.min.y << ")-("
                      << fresh_tri.max.x << "," << fresh_tri.max.y << ")" << std::endl;
        }
    }

    std::cout << "\nAll triangle pair intersections:" << std::endl;
    float max_score = 0.0f;
    int max_ta = -1, max_tb = -1;
    for (int ta = 0; ta < 5; ++ta) {
        for (int tb = 0; tb < 5; ++tb) {
            float score = triangles_intersection_score(fig_a[ta], fig_b[tb]);
            if (score > 0.0f) {
                std::cout << "  tri_a[" << ta << "] x tri_b[" << tb << "]: " << score << std::endl;
            }
            if (score > max_score) {
                max_score = score;
                max_ta = ta;
                max_tb = tb;
            }
        }
    }
    std::cout << "Max intersection score: " << max_score;
    if (max_score > 0.0f) {
        std::cout << " (triangles " << max_ta << " x " << max_tb << ")";
    }
    std::cout << std::endl;

    // Check which triangle pairs the filter would return
    const auto* pairs = filter.triangle_pairs_for(solution, idx_a, idx_b);
    if (pairs) {
        std::cout << "\nFilter-selected pairs (" << pairs->size() << " pairs):" << std::endl;
        for (const auto& [ta, tb] : *pairs) {
            float score = triangles_intersection_score(fig_a[ta], fig_b[tb]);
            std::cout << "  tri_a[" << ta << "] x tri_b[" << tb << "]: " << score << std::endl;
        }
    }
}

TEST_CASE("Debug rollback violation differences", "[debug]") {
    Problem problem = Problem::create_tree_packing_problem();

    // Create a solution with 20 trees, spread to ensure some valid ones
    Solution sol = Solution::init_random(20, 4.0f, 42);
    SolutionEval eval = problem.eval(sol);

    // Store original params for comparison
    std::vector<TreeParams> original_params;
    std::vector<bool> original_valid;
    for (size_t i = 0; i < sol.size(); ++i) {
        original_params.push_back(sol.get_params(i));
        original_valid.push_back(sol.is_valid(i));
    }
    float original_score = problem.score(eval, GlobalState(42));

    // Store original violation map
    auto original_violations = build_violation_map(eval.intersection_map);
    float original_total_violation = eval.intersection_violation;

    std::cout << "\n=== ORIGINAL STATE ===" << std::endl;
    std::cout << "Score: " << original_score << std::endl;
    std::cout << "Total intersection_violation: " << original_total_violation << std::endl;
    std::cout << "Number of intersecting pairs: " << original_violations.size() << std::endl;

    // Analyze pair (4, 15) in original state
    analyze_tree_pair(eval.solution, 4, 15, "ORIGINAL");

    // Create optimizers
    RandomRuin ruin(5);
    RandomRecreate recreate(5);
    NoiseOptimizer noise(0.1f, 3);

    ruin.set_problem(&problem);
    recreate.set_problem(&problem);
    noise.set_problem(&problem);

    GlobalState global_state(42);
    RNG rng(123);

    // Mark checkpoint before any operations
    size_t checkpoint = global_state.mark_checkpoint();

    // Apply ruin
    std::any ruin_state = ruin.init_state(eval);
    ruin.apply(eval, ruin_state, global_state, rng);

    // Apply recreate
    std::any recreate_state = recreate.init_state(eval);
    recreate.apply(eval, recreate_state, global_state, rng);

    // Apply noise
    std::any noise_state = noise.init_state(eval);
    noise.apply(eval, noise_state, global_state, rng);

    // Now rollback to checkpoint
    global_state.rollback_to(problem, eval, checkpoint);

    // Get restored state
    auto restored_violations = build_violation_map(eval.intersection_map);
    float restored_total_violation = eval.intersection_violation;
    float restored_score = problem.score(eval, GlobalState(42));

    std::cout << "\n=== RESTORED STATE ===" << std::endl;
    std::cout << "Score: " << restored_score << std::endl;
    std::cout << "Total intersection_violation: " << restored_total_violation << std::endl;
    std::cout << "Number of intersecting pairs: " << restored_violations.size() << std::endl;

    // Analyze pair (4, 15) in restored state
    analyze_tree_pair(eval.solution, 4, 15, "RESTORED");

    // Also compute the "ground truth" full intersection
    std::cout << "\n=== GROUND TRUTH (full SAT check) ===" << std::endl;
    float full_intersection_4_15 = compute_full_intersection(eval.solution, 4, 15);
    std::cout << "Full intersection score for (4, 15): " << full_intersection_4_15 << std::endl;

    // Compare violations
    std::cout << "\n=== DIFFERENCES ===" << std::endl;

    std::set<std::pair<int,int>> all_pairs;
    for (const auto& [pair, _] : original_violations) all_pairs.insert(pair);
    for (const auto& [pair, _] : restored_violations) all_pairs.insert(pair);

    int diff_count = 0;
    for (const auto& pair : all_pairs) {
        float orig_v = 0.0f, rest_v = 0.0f;
        bool in_orig = original_violations.count(pair) > 0;
        bool in_rest = restored_violations.count(pair) > 0;

        if (in_orig) orig_v = original_violations.at(pair);
        if (in_rest) rest_v = restored_violations.at(pair);

        if (std::abs(orig_v - rest_v) > 1e-6f || in_orig != in_rest) {
            diff_count++;
            std::cout << "\nPair (" << pair.first << ", " << pair.second << "):" << std::endl;
            std::cout << "  Original violation: " << (in_orig ? std::to_string(orig_v) : "NOT PRESENT") << std::endl;
            std::cout << "  Restored violation: " << (in_rest ? std::to_string(rest_v) : "NOT PRESENT") << std::endl;
        }
    }

    std::cout << "\nTotal differing pairs: " << diff_count << std::endl;
    std::cout << "Score difference: " << (restored_score - original_score) << std::endl;
    std::cout << "Violation difference: " << (restored_total_violation - original_total_violation) << std::endl;

    REQUIRE(restored_score == Approx(original_score).margin(1e-6f));
}
