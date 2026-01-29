#include <catch2/catch_test_macros.hpp>
#include "tree_packing/tree_packing.hpp"
#include "tree_packing/constraints/intersection_tree_filter.hpp"
#include "tree_packing/geometry/sat.hpp"
#include <iostream>
#include <iomanip>
#include <set>

using namespace tree_packing;

// Compute exact intersection score (sum of all positive triangle pairs, capped at 0.15)
float compute_exact_intersection(const Figure& fig_a, const Figure& fig_b) {
    float total = 0.0f;
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            float score = triangles_intersection_score(fig_a[i], fig_b[j]);
            if (score > 0.0f) {
                total += score;
                if (total >= 0.15f) return 0.15f;
            }
        }
    }
    return total;
}

// Compute filtered intersection score using tree filter
float compute_filtered_intersection(
    const IntersectionTreeFilter& filter,
    const Solution& solution,
    size_t idx_a,
    size_t idx_b,
    const Figure& fig_a,
    const Figure& fig_b
) {
    const auto* pairs = filter.triangle_pairs_for(solution, idx_a, idx_b);
    if (!pairs) return compute_exact_intersection(fig_a, fig_b);

    float total = 0.0f;
    for (const auto& [i, j] : *pairs) {
        float score = triangles_intersection_score(fig_a[i], fig_b[j]);
        if (score > 0.0f) {
            total += score;
            if (total >= 0.15f) return 0.15f;
        }
    }
    return total;
}

TEST_CASE("Exhaustive filter vs exact comparison", "[filter-test]") {
    IntersectionTreeFilter filter;

    const int NUM_ITERATIONS = 100000;
    const float SIDE = 4.0f;
    const int NUM_TREES = 50;

    int mismatches = 0;
    int total_intersecting = 0;
    std::set<int> bad_leaves;

    std::cout << "\n=== Testing " << NUM_ITERATIONS << " random configurations ===" << std::endl;

    RNG rng(12345);

    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        // Generate random tree pair
        float x1 = rng.uniform(-SIDE/2, SIDE/2);
        float y1 = rng.uniform(-SIDE/2, SIDE/2);
        float a1 = rng.uniform(-PI, PI);

        float x2 = rng.uniform(-SIDE/2, SIDE/2);
        float y2 = rng.uniform(-SIDE/2, SIDE/2);
        float a2 = rng.uniform(-PI, PI);

        TreeParams p1{Vec2{x1, y1}, a1};
        TreeParams p2{Vec2{x2, y2}, a2};

        Figure fig1 = params_to_figure(p1);
        Figure fig2 = params_to_figure(p2);

        AABB aabb1 = compute_aabb(fig1);
        AABB aabb2 = compute_aabb(fig2);

        // AABB pre-filter
        if (!aabb1.intersects(aabb2)) continue;

        // Create minimal solution for filter lookup
        TreeParamsSoA params;
        params.push_back(p1);
        params.push_back(p2);
        Solution sol = Solution::init(params, 4, SIDE, 10);

        // Compute exact intersection
        float exact = compute_exact_intersection(fig1, fig2);
        if (exact <= 0.0f) continue;

        total_intersecting++;

        // Test both orderings
        float filtered_01 = compute_filtered_intersection(filter, sol, 0, 1, fig1, fig2);
        float filtered_10 = compute_filtered_intersection(filter, sol, 1, 0, fig2, fig1);

        bool mismatch_01 = std::abs(filtered_01 - exact) > 1e-6f;
        bool mismatch_10 = std::abs(filtered_10 - exact) > 1e-6f;

        if (mismatch_01 || mismatch_10) {
            mismatches++;

            int leaf_01 = filter.leaf_index_for(sol, 0, 1);
            int leaf_10 = filter.leaf_index_for(sol, 1, 0);

            if (mismatch_01) bad_leaves.insert(leaf_01);
            if (mismatch_10) bad_leaves.insert(leaf_10);

            if (mismatches <= 10) {  // Print first 10 mismatches
                std::cout << std::fixed << std::setprecision(6);
                std::cout << "\nMismatch #" << mismatches << " at iter " << iter << ":" << std::endl;
                std::cout << "  Tree 0: pos=(" << x1 << ", " << y1 << "), angle=" << a1 << std::endl;
                std::cout << "  Tree 1: pos=(" << x2 << ", " << y2 << "), angle=" << a2 << std::endl;
                std::cout << "  Exact: " << exact << std::endl;
                std::cout << "  Filtered (0,1): " << filtered_01 << " [leaf " << leaf_01 << "]" << std::endl;
                std::cout << "  Filtered (1,0): " << filtered_10 << " [leaf " << leaf_10 << "]" << std::endl;

                // Show which triangles actually intersect
                std::cout << "  Intersecting triangles:" << std::endl;
                for (int i = 0; i < 5; ++i) {
                    for (int j = 0; j < 5; ++j) {
                        float score = triangles_intersection_score(fig1[i], fig2[j]);
                        if (score > 0.0f) {
                            std::cout << "    tri[" << i << "] x tri[" << j << "] = " << score << std::endl;
                        }
                    }
                }

                // Show leaf predictions
                std::array<uint8_t, 16> pred_01, pred_10;
                filter.leaf_pred_for(sol, 0, 1, pred_01);
                filter.leaf_pred_for(sol, 1, 0, pred_10);

                std::cout << "  Targets (0,1):";
                for (int t = 0; t < 16; ++t) if (pred_01[t]) std::cout << " " << t;
                std::cout << std::endl;

                std::cout << "  Targets (1,0):";
                for (int t = 0; t < 16; ++t) if (pred_10[t]) std::cout << " " << t;
                std::cout << std::endl;
            }
        }

        if ((iter + 1) % 10000 == 0) {
            std::cout << "Progress: " << (iter + 1) << "/" << NUM_ITERATIONS
                      << " | Intersecting: " << total_intersecting
                      << " | Mismatches: " << mismatches << std::endl;
        }
    }

    std::cout << "\n=== SUMMARY ===" << std::endl;
    std::cout << "Total intersecting pairs tested: " << total_intersecting << std::endl;
    std::cout << "Total mismatches: " << mismatches << std::endl;
    std::cout << "Mismatch rate: " << (100.0 * mismatches / std::max(1, total_intersecting)) << "%" << std::endl;

    if (!bad_leaves.empty()) {
        std::cout << "Bad leaves (" << bad_leaves.size() << "): ";
        for (int leaf : bad_leaves) {
            std::cout << leaf << " ";
        }
        std::cout << std::endl;
    }

    REQUIRE(mismatches == 0);
}
