#include <iostream>
#include <cmath>
#include "tree_packing/tree_packing.hpp"
#include "tree_packing/constraints/intersection.hpp"

using namespace tree_packing;

// Use relative tolerance for floating point comparison
// With many small values accumulating in different orders, expect ~0.01% relative difference
bool scores_match(float a, float b, float rel_tol = 1e-4f, float abs_tol = 1e-4f) {
    float diff = std::abs(a - b);
    float max_val = std::max(std::abs(a), std::abs(b));
    return diff <= abs_tol || diff <= max_val * rel_tol;
}

int main() {
    const int num_trees = 199;
    const float side = 1.0f;
    const uint64_t seed = 42;

    // Create a solution with random positions
    Solution solution = Solution::init_random(num_trees, side, seed);

    // Create intersection constraint
    IntersectionConstraint constraint;

    // Test 1: Full evaluation - compare both methods
    std::cout << "=== Test 1: Full Evaluation ===" << std::endl;

    SolutionEval::IntersectionMap map1, map2;
    int count1 = 0, count2 = 0;

    float score1 = constraint.eval(solution, map1, &count1);
    float score2 = constraint.eval_triangle_grid(solution, map2, &count2);

    std::cout << "Figure grid:   score=" << score1 << " count=" << count1 << std::endl;
    std::cout << "Triangle grid: score=" << score2 << " count=" << count2 << std::endl;

    bool pass1 = std::abs(score1 - score2) < 1e-4f && count1 == count2;
    std::cout << "Test 1: " << (pass1 ? "PASS" : "FAIL") << std::endl;

    // Test 2: Incremental update - modify a few trees
    std::cout << "\n=== Test 2: Incremental Update ===" << std::endl;

    // Modify a few trees
    std::vector<int> modified = {5, 10, 15};
    TreeParamsSoA new_params = solution.params();
    for (int idx : modified) {
        new_params.x[idx] += 0.1f;
        new_params.y[idx] -= 0.05f;
        new_params.angle[idx] += 0.2f;
    }

    // Update solution
    Solution updated = solution.update(new_params, modified);

    // Full eval on updated solution
    SolutionEval::IntersectionMap map3, map4;
    int count3 = 0, count4 = 0;

    float score3 = constraint.eval(updated, map3, &count3);
    float score4 = constraint.eval_triangle_grid(updated, map4, &count4);

    std::cout << "Figure grid:   score=" << score3 << " count=" << count3 << std::endl;
    std::cout << "Triangle grid: score=" << score4 << " count=" << count4 << std::endl;

    bool pass2 = std::abs(score3 - score4) < 1e-4f && count3 == count4;
    std::cout << "Test 2: " << (pass2 ? "PASS" : "FAIL") << std::endl;

    // Test 3: Incremental eval_update
    std::cout << "\n=== Test 3: Incremental eval_update ===" << std::endl;

    // Start with map1 state, update incrementally
    SolutionEval::IntersectionMap map5 = map1;
    SolutionEval::IntersectionMap map6 = map2;
    int count5 = count1, count6 = count2;

    float score5 = constraint.eval_update(updated, map5, modified, score1, count1, &count5);
    float score6 = constraint.eval_update_triangle_grid(updated, map6, modified, score2, count2, &count6);

    std::cout << "Figure grid incremental:   score=" << score5 << " count=" << count5 << std::endl;
    std::cout << "Triangle grid incremental: score=" << score6 << " count=" << count6 << std::endl;

    // Compare with full eval
    std::cout << "Figure grid full:          score=" << score3 << " count=" << count3 << std::endl;
    std::cout << "Triangle grid full:        score=" << score4 << " count=" << count4 << std::endl;

    bool pass3a = std::abs(score5 - score3) < 1e-4f;
    bool pass3b = std::abs(score6 - score4) < 1e-4f;
    bool pass3c = std::abs(score5 - score6) < 1e-4f;
    std::cout << "Test 3a (fig incr vs fig full):   " << (pass3a ? "PASS" : "FAIL") << std::endl;
    std::cout << "Test 3b (tri incr vs tri full):   " << (pass3b ? "PASS" : "FAIL") << std::endl;
    std::cout << "Test 3c (fig incr vs tri incr):   " << (pass3c ? "PASS" : "FAIL") << std::endl;

    // Summary
    std::cout << "\n=== Summary ===" << std::endl;
    bool all_pass = pass1 && pass2 && pass3a && pass3b && pass3c;
    std::cout << (all_pass ? "All tests PASSED" : "Some tests FAILED") << std::endl;

    return all_pass ? 0 : 1;
}
