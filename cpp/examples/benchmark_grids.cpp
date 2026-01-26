#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>
#include "tree_packing/tree_packing.hpp"
#include "tree_packing/constraints/intersection.hpp"

using namespace tree_packing;
using Clock = std::chrono::high_resolution_clock;

template<typename Func>
double time_ms(Func&& f, int iterations = 1) {
    auto start = Clock::now();
    for (int i = 0; i < iterations; ++i) {
        f();
    }
    auto end = Clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / iterations;
}

int main() {
    const int num_trees = 199;
    const float side = 1.0f;
    const uint64_t seed = 42;
    const int num_iterations = 100;

    std::cout << "=== Grid Comparison Benchmark ===" << std::endl;
    std::cout << "Trees: " << num_trees << ", Iterations: " << num_iterations << std::endl;
    std::cout << std::fixed << std::setprecision(3);

    // Create initial solution
    Solution solution = Solution::init_random(num_trees, side, seed);
    IntersectionConstraint constraint;

    // ============ Test 1: Full Evaluation ============
    std::cout << "\n--- Full Evaluation ---" << std::endl;

    SolutionEval::IntersectionMap map_fig, map_fig_hash;
    int count_fig = 0, count_fig_hash = 0;
    float score_fig = 0, score_fig_hash = 0;

    double time_fig_full = time_ms([&]() {
        score_fig = constraint.eval(solution, map_fig, &count_fig);
    }, num_iterations);

    double time_fig_hash_full = time_ms([&]() {
        score_fig_hash = constraint.eval_figure_hash(solution, map_fig_hash, &count_fig_hash);
    }, num_iterations);

    std::cout << "Figure grid: " << time_fig_full << " ms, score=" << score_fig << ", count=" << count_fig << std::endl;
    std::cout << "Figure hash: " << time_fig_hash_full << " ms, score=" << score_fig_hash << ", count=" << count_fig_hash << std::endl;
    std::cout << "Speedup: " << time_fig_full / time_fig_hash_full << "x" << std::endl;

    // ============ Test 2: Incremental Update (1 tree) ============
    std::cout << "\n--- Incremental Update (1 tree modified) ---" << std::endl;

    std::vector<int> modified_1 = {5};
    TreeParamsSoA new_params_1 = solution.params();
    new_params_1.x[5] += 0.1f;
    new_params_1.y[5] -= 0.05f;
    new_params_1.angle[5] += 0.2f;
    Solution updated_1 = solution.update(new_params_1, modified_1);

    // Reset maps
    constraint.eval(solution, map_fig, &count_fig);
    constraint.eval_figure_hash(solution, map_fig_hash, &count_fig_hash);
    SolutionEval::IntersectionMap map_fig_inc = map_fig;
    SolutionEval::IntersectionMap map_fig_hash_inc = map_fig_hash;

    double time_fig_inc1 = time_ms([&]() {
        map_fig_inc = map_fig;
        score_fig = constraint.eval_update(updated_1, map_fig_inc, modified_1, score_fig, count_fig, &count_fig);
    }, num_iterations);

    double time_fig_hash_inc1 = time_ms([&]() {
        map_fig_hash_inc = map_fig_hash;
        score_fig_hash = constraint.eval_update_figure_hash(updated_1, map_fig_hash_inc, modified_1, score_fig_hash, count_fig_hash, &count_fig_hash);
    }, num_iterations);

    std::cout << "Figure grid: " << time_fig_inc1 << " ms" << std::endl;
    std::cout << "Figure hash: " << time_fig_hash_inc1 << " ms" << std::endl;
    std::cout << "Speedup: " << time_fig_inc1 / time_fig_hash_inc1 << "x" << std::endl;

    // ============ Test 3: Incremental Update (5 trees) ============
    std::cout << "\n--- Incremental Update (5 trees modified) ---" << std::endl;

    std::vector<int> modified_5 = {5, 10, 15, 20, 25};
    TreeParamsSoA new_params_5 = solution.params();
    for (int idx : modified_5) {
        new_params_5.x[idx] += 0.1f;
        new_params_5.y[idx] -= 0.05f;
        new_params_5.angle[idx] += 0.2f;
    }
    Solution updated_5 = solution.update(new_params_5, modified_5);

    // Reset maps
    constraint.eval(solution, map_fig, &count_fig);
    constraint.eval_figure_hash(solution, map_fig_hash, &count_fig_hash);

    double time_fig_inc5 = time_ms([&]() {
        map_fig_inc = map_fig;
        score_fig = constraint.eval_update(updated_5, map_fig_inc, modified_5, score_fig, count_fig, &count_fig);
    }, num_iterations);

    double time_fig_hash_inc5 = time_ms([&]() {
        map_fig_hash_inc = map_fig_hash;
        score_fig_hash = constraint.eval_update_figure_hash(updated_5, map_fig_hash_inc, modified_5, score_fig_hash, count_fig_hash, &count_fig_hash);
    }, num_iterations);

    std::cout << "Figure grid: " << time_fig_inc5 << " ms" << std::endl;
    std::cout << "Figure hash: " << time_fig_hash_inc5 << " ms" << std::endl;
    std::cout << "Speedup: " << time_fig_inc5 / time_fig_hash_inc5 << "x" << std::endl;

    // ============ Test 4: Incremental Update (20 trees) ============
    std::cout << "\n--- Incremental Update (20 trees modified) ---" << std::endl;

    std::vector<int> modified_20;
    for (int i = 0; i < 20; ++i) modified_20.push_back(i * 5);
    TreeParamsSoA new_params_20 = solution.params();
    for (int idx : modified_20) {
        new_params_20.x[idx] += 0.1f;
        new_params_20.y[idx] -= 0.05f;
        new_params_20.angle[idx] += 0.2f;
    }
    Solution updated_20 = solution.update(new_params_20, modified_20);

    // Reset maps
    constraint.eval(solution, map_fig, &count_fig);
    constraint.eval_figure_hash(solution, map_fig_hash, &count_fig_hash);

    double time_fig_inc20 = time_ms([&]() {
        map_fig_inc = map_fig;
        score_fig = constraint.eval_update(updated_20, map_fig_inc, modified_20, score_fig, count_fig, &count_fig);
    }, num_iterations);

    double time_fig_hash_inc20 = time_ms([&]() {
        map_fig_hash_inc = map_fig_hash;
        score_fig_hash = constraint.eval_update_figure_hash(updated_20, map_fig_hash_inc, modified_20, score_fig_hash, count_fig_hash, &count_fig_hash);
    }, num_iterations);

    std::cout << "Figure grid: " << time_fig_inc20 << " ms" << std::endl;
    std::cout << "Figure hash: " << time_fig_hash_inc20 << " ms" << std::endl;
    std::cout << "Speedup: " << time_fig_inc20 / time_fig_hash_inc20 << "x" << std::endl;

    // ============ Summary ============
    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "| Operation              | Figure Grid | Figure Hash | Speedup |" << std::endl;
    std::cout << "|------------------------|-------------|-------------|---------|" << std::endl;
    std::cout << "| Full eval              | " << std::setw(9) << time_fig_full << " ms | "
              << std::setw(9) << time_fig_hash_full << " ms | " << std::setw(6) << time_fig_full/time_fig_hash_full << "x |" << std::endl;
    std::cout << "| Incremental (1 tree)   | " << std::setw(9) << time_fig_inc1 << " ms | "
              << std::setw(9) << time_fig_hash_inc1 << " ms | " << std::setw(6) << time_fig_inc1/time_fig_hash_inc1 << "x |" << std::endl;
    std::cout << "| Incremental (5 trees)  | " << std::setw(9) << time_fig_inc5 << " ms | "
              << std::setw(9) << time_fig_hash_inc5 << " ms | " << std::setw(6) << time_fig_inc5/time_fig_hash_inc5 << "x |" << std::endl;
    std::cout << "| Incremental (20 trees) | " << std::setw(9) << time_fig_inc20 << " ms | "
              << std::setw(9) << time_fig_hash_inc20 << " ms | " << std::setw(6) << time_fig_inc20/time_fig_hash_inc20 << "x |" << std::endl;

    return 0;
}
