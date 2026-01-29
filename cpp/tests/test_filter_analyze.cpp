#include <catch2/catch_test_macros.hpp>
#include "tree_packing/tree_packing.hpp"
#include "tree_packing/constraints/intersection_tree_filter.hpp"
#include "tree_packing/geometry/sat.hpp"
#include <iostream>
#include <iomanip>
#include <map>
#include <set>

using namespace tree_packing;

// Target to triangle pairs mapping (from kTargetPairs)
// Target 0: (0,0), Target 1: (0,1), Target 2: (0,2), Target 3: (0,3),(0,4)
// Target 4: (1,0), Target 5: (1,1), Target 6: (1,2), Target 7: (1,3),(1,4)
// Target 8: (2,0), Target 9: (2,1), Target 10: (2,2), Target 11: (2,3),(2,4)
// Target 12: (3,0),(4,0), Target 13: (3,1),(4,1), Target 14: (3,2),(4,2), Target 15: (3,3),(3,4),(4,3),(4,4)

int get_target_for_pair(int tri_a, int tri_b) {
    // Group mapping: tri 0->g0, tri 1->g1, tri 2->g2, tri 3,4->g3
    int ga = (tri_a >= 3) ? 3 : tri_a;
    int gb = (tri_b >= 3) ? 3 : tri_b;
    return ga * 4 + gb;
}

TEST_CASE("Analyze bad leaves", "[analyze]") {
    IntersectionTreeFilter filter;

    const int NUM_ITERATIONS = 200000;
    const float SIDE = 4.0f;

    // Track which targets each leaf actually needs
    std::map<int, std::set<int>> leaf_needs_targets;
    std::map<int, std::set<int>> leaf_has_targets;

    RNG rng(12345);

    std::cout << "\n=== Analyzing " << NUM_ITERATIONS << " random configurations ===" << std::endl;

    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
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

        if (!aabb1.intersects(aabb2)) continue;

        // Check if any triangles intersect
        bool has_intersection = false;
        for (int i = 0; i < 5 && !has_intersection; ++i) {
            for (int j = 0; j < 5 && !has_intersection; ++j) {
                if (triangles_intersection_score(fig1[i], fig2[j]) > 0.0f) {
                    has_intersection = true;
                }
            }
        }
        if (!has_intersection) continue;

        TreeParamsSoA params;
        params.push_back(p1);
        params.push_back(p2);
        Solution sol = Solution::init(params, 4, SIDE, 10);

        // Test both orderings
        for (int order = 0; order < 2; ++order) {
            size_t idx_a = order == 0 ? 0 : 1;
            size_t idx_b = order == 0 ? 1 : 0;
            const Figure& fa = order == 0 ? fig1 : fig2;
            const Figure& fb = order == 0 ? fig2 : fig1;

            int leaf = filter.leaf_index_for(sol, idx_a, idx_b);
            if (leaf < 0) continue;

            // Record what targets this leaf has
            std::array<uint8_t, 16> pred;
            filter.leaf_pred_for(sol, idx_a, idx_b, pred);
            for (int t = 0; t < 16; ++t) {
                if (pred[t]) leaf_has_targets[leaf].insert(t);
            }

            // Record what targets this leaf actually needs
            for (int i = 0; i < 5; ++i) {
                for (int j = 0; j < 5; ++j) {
                    if (triangles_intersection_score(fa[i], fb[j]) > 0.0f) {
                        int target = get_target_for_pair(i, j);
                        leaf_needs_targets[leaf].insert(target);
                    }
                }
            }
        }

        if ((iter + 1) % 50000 == 0) {
            std::cout << "Progress: " << (iter + 1) << "/" << NUM_ITERATIONS << std::endl;
        }
    }

    std::cout << "\n=== LEAVES THAT NEED FIXES ===" << std::endl;

    std::vector<std::pair<int, std::set<int>>> fixes;

    for (const auto& [leaf, needs] : leaf_needs_targets) {
        const auto& has = leaf_has_targets[leaf];
        std::set<int> missing;
        for (int t : needs) {
            if (has.find(t) == has.end()) {
                missing.insert(t);
            }
        }

        if (!missing.empty()) {
            fixes.push_back({leaf, missing});
            std::cout << "\nLeaf " << leaf << ":" << std::endl;
            std::cout << "  Has targets:";
            for (int t : has) std::cout << " " << t;
            std::cout << std::endl;
            std::cout << "  Needs targets:";
            for (int t : needs) std::cout << " " << t;
            std::cout << std::endl;
            std::cout << "  MISSING:";
            for (int t : missing) std::cout << " " << t;
            std::cout << std::endl;

            // Output the fix needed
            std::cout << "  New prediction: ";
            for (int t = 0; t < 16; ++t) {
                bool should_have = (has.find(t) != has.end()) || (missing.find(t) != missing.end());
                std::cout << (should_have ? "1" : "0");
                if (t < 15) std::cout << ", ";
            }
            std::cout << std::endl;
        }
    }

    std::cout << "\n=== SUMMARY ===" << std::endl;
    std::cout << "Total leaves needing fixes: " << fixes.size() << std::endl;

    // Output in a format easy to copy-paste
    std::cout << "\n=== FIXES TO APPLY ===" << std::endl;
    for (const auto& [leaf, missing] : fixes) {
        const auto& has = leaf_has_targets[leaf];
        std::cout << "Leaf " << leaf << " (line " << (332 + leaf) << "): ";
        for (int t = 0; t < 16; ++t) {
            bool should_have = (has.find(t) != has.end()) || (missing.find(t) != missing.end());
            std::cout << (should_have ? "1" : "0");
            if (t < 15) std::cout << ", ";
        }
        std::cout << "," << std::endl;
    }

    REQUIRE(fixes.empty());
}
