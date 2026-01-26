#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include "tree_packing/tree_packing.hpp"

using namespace tree_packing;
using Catch::Approx;

namespace {
float objective_naive_from_params(const Solution& solution) {
    const auto& params = solution.params();
    const size_t n = params.size();
    if (n == 0) return 0.0f;

    float min_x = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float min_y = std::numeric_limits<float>::max();
    float max_y = std::numeric_limits<float>::lowest();
    int valid_count = 0;

    for (size_t i = 0; i < n; ++i) {
        if (!solution.is_valid(i)) continue;
        TreeParams p = params.get(i);
        for (size_t t = 0; t < TREE_NUM_TRIANGLES; ++t) {
            Triangle tri = transform_triangle(p.pos, p.angle, TREE_SHAPE[t]);
            min_x = std::min(min_x, std::min({tri.v0.x, tri.v1.x, tri.v2.x}));
            max_x = std::max(max_x, std::max({tri.v0.x, tri.v1.x, tri.v2.x}));
            min_y = std::min(min_y, std::min({tri.v0.y, tri.v1.y, tri.v2.y}));
            max_y = std::max(max_y, std::max({tri.v0.y, tri.v1.y, tri.v2.y}));
        }
        ++valid_count;
    }

    if (valid_count == 0) return 0.0f;
    float delta_x = max_x - min_x;
    float delta_y = max_y - min_y;
    float length = std::max(delta_x, delta_y);
    return (length * length) / static_cast<float>(n);
}

void require_eval_matches_full(const Problem& problem, const Solution& solution, const SolutionEval& eval) {
    SolutionEval full = problem.eval(solution);
    REQUIRE(eval.valid_count == full.valid_count);
    REQUIRE(eval.objective == Approx(full.objective).margin(1e-4f));
    REQUIRE(eval.intersection_violation == Approx(full.intersection_violation).margin(1e-4f));
    REQUIRE(eval.bounds_violation == Approx(full.bounds_violation).margin(1e-4f));
    REQUIRE(eval.min_x == Approx(full.min_x).margin(1e-4f));
    REQUIRE(eval.max_x == Approx(full.max_x).margin(1e-4f));
    REQUIRE(eval.min_y == Approx(full.min_y).margin(1e-4f));
    REQUIRE(eval.max_y == Approx(full.max_y).margin(1e-4f));
    REQUIRE(eval.total_violation() == Approx(full.total_violation()).margin(1e-4f));
}

void require_maps_equal(const SolutionEval::IntersectionMap& a,
                        const SolutionEval::IntersectionMap& b) {
    REQUIRE(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        const auto& lhs = a[i];
        const auto& rhs = b[i];
        if (!lhs && !rhs) {
            continue;
        }
        REQUIRE(static_cast<bool>(lhs) == static_cast<bool>(rhs));
        REQUIRE(lhs->size() == rhs->size());
        std::vector<std::pair<Index, float>> left;
        std::vector<std::pair<Index, float>> right;
        left.reserve(lhs->size());
        right.reserve(rhs->size());
        for (const auto& entry : *lhs) {
            left.emplace_back(entry.neighbor, entry.score);
        }
        for (const auto& entry : *rhs) {
            right.emplace_back(entry.neighbor, entry.score);
        }
        auto cmp = [](const std::pair<Index, float>& x, const std::pair<Index, float>& y) {
            if (x.first != y.first) return x.first < y.first;
            return x.second < y.second;
        };
        std::sort(left.begin(), left.end(), cmp);
        std::sort(right.begin(), right.end(), cmp);
        for (size_t j = 0; j < left.size(); ++j) {
            REQUIRE(left[j].first == right[j].first);
            REQUIRE(left[j].second == Approx(right[j].second).margin(1e-4f));
        }
    }
}

bool float_equal(float a, float b, float tol) {
    if (std::isnan(a) && std::isnan(b)) {
        return true;
    }
    return std::abs(a - b) <= tol;
}

void require_figures_match(const Solution& expected, const Solution& actual, float tol) {
    const auto& exp_figs = expected.figures();
    const auto& act_figs = actual.figures();
    REQUIRE(exp_figs.size() == act_figs.size());
    for (size_t i = 0; i < exp_figs.size(); ++i) {
        for (size_t t = 0; t < TREE_NUM_TRIANGLES; ++t) {
            const Triangle& e = exp_figs[i].triangles[t];
            const Triangle& a = act_figs[i].triangles[t];
            REQUIRE(float_equal(a.v0.x, e.v0.x, tol));
            REQUIRE(float_equal(a.v0.y, e.v0.y, tol));
            REQUIRE(float_equal(a.v1.x, e.v1.x, tol));
            REQUIRE(float_equal(a.v1.y, e.v1.y, tol));
            REQUIRE(float_equal(a.v2.x, e.v2.x, tol));
            REQUIRE(float_equal(a.v2.y, e.v2.y, tol));
        }
    }
}

void require_figures_unchanged(const Solution& before,
                               const Solution& after,
                               const std::vector<int>& changed,
                               float tol) {
    std::vector<char> is_changed(before.size(), false);
    for (int idx : changed) {
        if (idx >= 0 && static_cast<size_t>(idx) < is_changed.size()) {
            is_changed[static_cast<size_t>(idx)] = true;
        }
    }
    const auto& before_figs = before.figures();
    const auto& after_figs = after.figures();
    REQUIRE(before_figs.size() == after_figs.size());
    for (size_t i = 0; i < before_figs.size(); ++i) {
        if (is_changed[i]) {
            continue;
        }
        for (size_t t = 0; t < TREE_NUM_TRIANGLES; ++t) {
            const Triangle& b = before_figs[i].triangles[t];
            const Triangle& a = after_figs[i].triangles[t];
            REQUIRE(float_equal(a.v0.x, b.v0.x, tol));
            REQUIRE(float_equal(a.v0.y, b.v0.y, tol));
            REQUIRE(float_equal(a.v1.x, b.v1.x, tol));
            REQUIRE(float_equal(a.v1.y, b.v1.y, tol));
            REQUIRE(float_equal(a.v2.x, b.v2.x, tol));
            REQUIRE(float_equal(a.v2.y, b.v2.y, tol));
        }
    }
}

void require_figures_changed(const Solution& before,
                             const Solution& after,
                             const std::vector<int>& changed,
                             float tol) {
    const auto& before_figs = before.figures();
    const auto& after_figs = after.figures();
    for (int idx : changed) {
        if (idx < 0 || static_cast<size_t>(idx) >= before_figs.size()) {
            continue;
        }
        bool any_diff = false;
        for (size_t t = 0; t < TREE_NUM_TRIANGLES; ++t) {
            const Triangle& b = before_figs[static_cast<size_t>(idx)].triangles[t];
            const Triangle& a = after_figs[static_cast<size_t>(idx)].triangles[t];
            if ((std::isnan(a.v0.x) != std::isnan(b.v0.x)) ||
                (std::isnan(a.v0.y) != std::isnan(b.v0.y)) ||
                (std::isnan(a.v1.x) != std::isnan(b.v1.x)) ||
                (std::isnan(a.v1.y) != std::isnan(b.v1.y)) ||
                (std::isnan(a.v2.x) != std::isnan(b.v2.x)) ||
                (std::isnan(a.v2.y) != std::isnan(b.v2.y))) {
                any_diff = true;
                break;
            }
            if (std::abs(a.v0.x - b.v0.x) > tol || std::abs(a.v0.y - b.v0.y) > tol ||
                std::abs(a.v1.x - b.v1.x) > tol || std::abs(a.v1.y - b.v1.y) > tol ||
                std::abs(a.v2.x - b.v2.x) > tol || std::abs(a.v2.y - b.v2.y) > tol) {
                any_diff = true;
                break;
            }
        }
        CAPTURE(idx);
        REQUIRE(any_diff);
    }
}

void require_cache_valid(const Solution& solution) {
    REQUIRE(solution.validate_cache(1e-4f, true));
}
}  // namespace

TEST_CASE("Incremental eval matches full eval", "[incremental]") {
    Problem problem = Problem::create_tree_packing_problem();
    Solution sol = Solution::init_random(30, 8.0f, 42);
    SolutionEval prev = problem.eval(sol);

    Solution updated = sol;
    std::vector<int> indices = {2, 5, 7, 11};

    TreeParams p2 = updated.get_params(2);
    p2.pos.x += 0.25f;
    p2.pos.y -= 0.15f;
    p2.angle += 0.05f;
    updated.set_params(2, p2);

    updated.set_nan(5);

    TreeParams p7 = updated.get_params(7);
    p7.pos.x -= 0.4f;
    p7.pos.y += 0.3f;
    updated.set_params(7, p7);

    TreeParams p11 = updated.get_params(11);
    p11.angle -= 0.2f;
    updated.set_params(11, p11);

    SolutionEval inc = prev;
    std::vector<int> removed_indices = {5};
    problem.remove_and_eval(inc, removed_indices);

    std::vector<int> update_indices = {2, 7, 11};
    TreeParamsSoA new_params(update_indices.size());
    for (size_t i = 0; i < update_indices.size(); ++i) {
        new_params.set(i, updated.get_params(static_cast<size_t>(update_indices[i])));
    }
    problem.update_and_eval(inc, update_indices, new_params);
    require_eval_matches_full(problem, updated, inc);
}

TEST_CASE("Objective matches naive param iteration", "[incremental]") {
    Problem problem = Problem::create_tree_packing_problem();
    Solution sol = Solution::init_random(25, 9.0f, 123);

    float obj = problem.objective(sol);
    float naive = objective_naive_from_params(sol);
    REQUIRE(obj == Approx(naive).margin(1e-4f));

    // Mutate a few trees and compare again.
    sol.set_params(3, TreeParams(1.25f, -0.75f, 0.2f));
    sol.set_params(7, TreeParams(-2.5f, 1.5f, -0.4f));
    sol.set_nan(11);
    sol.recompute_cache();

    obj = problem.objective(sol);
    naive = objective_naive_from_params(sol);
    REQUIRE(obj == Approx(naive).margin(1e-4f));
}

TEST_CASE("Update and eval in-place", "[incremental]") {
    Problem problem = Problem::create_tree_packing_problem();
    Solution sol = Solution::init_random(12, 6.0f, 321);
    SolutionEval eval = problem.eval(sol);

    std::vector<int> indices = {1, 4, 9};
    std::vector<TreeParams> params = {
        TreeParams(1.5f, -2.0f, 0.3f),
        TreeParams(-1.0f, 2.2f, -0.6f),
        TreeParams(0.0f, 0.5f, 1.2f)
    };
    TreeParamsSoA new_params(indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        new_params.set(i, params[i]);
    }

    Solution before = sol;
    problem.update_and_eval(eval, indices, new_params);

    Solution updated = sol;
    for (size_t i = 0; i < indices.size(); ++i) {
        updated.set_params(static_cast<size_t>(indices[i]), params[i]);
    }
    require_cache_valid(eval.solution);
    require_figures_unchanged(before, eval.solution, indices, 1e-4f);
    require_figures_changed(before, eval.solution, indices, 1e-4f);
    require_figures_match(updated, eval.solution, 1e-4f);
    require_eval_matches_full(problem, updated, eval);
}

TEST_CASE("Incremental eval handles validity transitions", "[incremental]") {
    Problem problem = Problem::create_tree_packing_problem();
    Solution sol = Solution::init_random(20, 7.0f, 77);
    sol.set_nan(5);
    sol.set_nan(12);
    SolutionEval prev = problem.eval(sol);

    Solution updated = sol;

    updated.set_nan(2);
    updated.set_params(5, TreeParams(1.0f, -1.0f, 0.1f));
    TreeParams p7 = updated.get_params(7);
    p7.pos.x += 0.35f;
    updated.set_params(7, p7);
    updated.set_params(12, TreeParams(-2.0f, 1.5f, -0.2f));

    SolutionEval inc = prev;

    // Remove tree 2 (valid -> invalid)
    std::vector<int> removed_indices = {2};
    problem.remove_and_eval(inc, removed_indices);

    // Update tree 7 (valid -> valid)
    std::vector<int> update_indices = {7};
    TreeParamsSoA update_params(1);
    update_params.set(0, updated.get_params(7));
    problem.update_and_eval(inc, update_indices, update_params);

    // Insert trees 5 and 12 (invalid -> valid)
    std::vector<int> insert_indices = {5, 12};
    TreeParamsSoA insert_params(2);
    insert_params.set(0, updated.get_params(5));
    insert_params.set(1, updated.get_params(12));
    problem.insert_and_eval(inc, insert_indices, insert_params);

    require_eval_matches_full(problem, updated, inc);
}

TEST_CASE("Update and eval only updates valid trees", "[incremental]") {
    Problem problem = Problem::create_tree_packing_problem();
    Solution sol = Solution::init_random(20, 7.0f, 77);
    SolutionEval eval = problem.eval(sol);

    // All indices are valid trees
    std::vector<int> indices = {3, 7, 15};
    std::vector<TreeParams> params;
    params.reserve(indices.size());
    TreeParams p3 = sol.get_params(3);
    p3.pos.x += 0.5f;
    p3.pos.y -= 0.3f;
    params.emplace_back(p3);
    TreeParams p7 = sol.get_params(7);
    p7.pos.x += 0.35f;
    params.emplace_back(p7);
    TreeParams p15 = sol.get_params(15);
    p15.angle += 0.2f;
    params.emplace_back(p15);

    TreeParamsSoA new_params(indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        new_params.set(i, params[i]);
    }
    Solution before = sol;
    problem.update_and_eval(eval, indices, new_params);

    Solution updated = sol;
    updated.set_params(3, params[0]);
    updated.set_params(7, params[1]);
    updated.set_params(15, params[2]);
    require_cache_valid(eval.solution);
    require_figures_unchanged(before, eval.solution, indices, 1e-4f);
    require_figures_changed(before, eval.solution, indices, 1e-4f);
    require_figures_match(updated, eval.solution, 1e-4f);
    require_eval_matches_full(problem, updated, eval);
}

TEST_CASE("Insert and eval inserts invalid trees", "[incremental]") {
    Problem problem = Problem::create_tree_packing_problem();
    Solution sol = Solution::init_random(20, 7.0f, 77);
    sol.set_nan(5);
    sol.set_nan(12);
    SolutionEval eval = problem.eval(sol);

    // Insert trees 5 and 12 (invalid -> valid)
    std::vector<int> indices = {5, 12};
    std::vector<TreeParams> params;
    params.emplace_back(TreeParams(1.0f, -1.0f, 0.1f));
    params.emplace_back(TreeParams(-2.0f, 1.5f, -0.2f));

    TreeParamsSoA new_params(indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        new_params.set(i, params[i]);
    }
    Solution before = sol;
    problem.insert_and_eval(eval, indices, new_params);

    Solution updated = sol;
    updated.set_params(5, params[0]);
    updated.set_params(12, params[1]);
    require_cache_valid(eval.solution);
    require_figures_unchanged(before, eval.solution, indices, 1e-4f);
    require_figures_changed(before, eval.solution, indices, 1e-4f);
    require_figures_match(updated, eval.solution, 1e-4f);
    require_eval_matches_full(problem, updated, eval);
}

TEST_CASE("Remove and eval handles validity transitions", "[incremental]") {
    Problem problem = Problem::create_tree_packing_problem();
    Solution sol = Solution::init_random(20, 7.0f, 77);
    sol.set_nan(5);
    sol.set_nan(12);
    SolutionEval eval = problem.eval(sol);

    std::vector<int> indices = {2, 5};
    Solution before = sol;
    problem.remove_and_eval(eval, indices);

    Solution updated = sol;
    updated.set_nan(2);
    updated.set_nan(5);

    require_cache_valid(eval.solution);
    require_figures_unchanged(before, eval.solution, {2}, 1e-4f);
    require_figures_changed(before, eval.solution, {2}, 1e-4f);
    require_figures_match(updated, eval.solution, 1e-4f);
    require_eval_matches_full(problem, updated, eval);
}

TEST_CASE("Incremental eval handles bounds invalidation", "[incremental]") {
    Problem problem = Problem::create_tree_packing_problem();
    Solution sol = Solution::init_random(25, 9.0f, 1234);
    SolutionEval prev = problem.eval(sol);

    REQUIRE(prev.min_x_idx >= 0);
    int idx = prev.min_x_idx;
    Solution updated = sol;
    TreeParams p = updated.get_params(static_cast<size_t>(idx));
    p.pos.x += 10.0f;
    updated.set_params(static_cast<size_t>(idx), p);

    std::vector<int> indices = {idx};
    SolutionEval inc = prev;
    TreeParamsSoA new_params(indices.size());
    new_params.set(0, p);
    problem.update_and_eval(inc, indices, new_params);
    require_eval_matches_full(problem, updated, inc);
}

TEST_CASE("Update and eval handles bounds invalidation", "[incremental]") {
    Problem problem = Problem::create_tree_packing_problem();
    Solution sol = Solution::init_random(25, 9.0f, 1234);
    SolutionEval eval = problem.eval(sol);
    CAPTURE(eval.objective);

    REQUIRE(eval.min_x_idx >= 0);
    int idx = eval.min_x_idx;
    TreeParams p = sol.get_params(static_cast<size_t>(idx));
    p.pos.x += 10.0f;

    std::vector<int> indices = {idx};
    TreeParamsSoA new_params(indices.size());
    new_params.set(0, p);
    Solution before = sol;
    problem.update_and_eval(eval, indices, new_params);

    Solution updated = sol;
    updated.set_params(static_cast<size_t>(idx), p);
    require_cache_valid(eval.solution);
    require_figures_unchanged(before, eval.solution, indices, 1e-4f);
    require_figures_changed(before, eval.solution, indices, 1e-4f);
    require_figures_match(updated, eval.solution, 1e-4f);
    require_eval_matches_full(problem, updated, eval);
}

TEST_CASE("Incremental eval keeps intersection map consistent", "[incremental]") {
    Problem problem = Problem::create_tree_packing_problem();
    Solution sol = Solution::init_random(15, 6.0f, 222);
    SolutionEval prev = problem.eval(sol);

    Solution updated = sol;
    std::vector<int> indices = {1, 4, 9};
    updated.set_params(1, TreeParams(1.2f, 0.5f, 0.2f));
    updated.set_params(4, TreeParams(-1.1f, -0.3f, -0.4f));
    updated.set_params(9, TreeParams(0.4f, -1.5f, 0.9f));

    SolutionEval inc = prev;
    TreeParamsSoA new_params(indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        new_params.set(i, updated.get_params(static_cast<size_t>(indices[i])));
    }
    problem.update_and_eval(inc, indices, new_params);

    IntersectionConstraint constraint;
    SolutionEval::IntersectionMap map;
    float violation = constraint.eval(updated, map);
    REQUIRE(inc.intersection_violation == Approx(violation).margin(1e-4f));
    require_maps_equal(inc.intersection_map, map);
    require_eval_matches_full(problem, updated, inc);
}

TEST_CASE("Update and eval keeps intersection map consistent", "[incremental]") {
    Problem problem = Problem::create_tree_packing_problem();
    Solution sol = Solution::init_random(15, 6.0f, 222);
    SolutionEval eval = problem.eval(sol);

    std::vector<int> indices = {1, 4, 9};
    std::vector<TreeParams> params = {
        TreeParams(1.2f, 0.5f, 0.2f),
        TreeParams(-1.1f, -0.3f, -0.4f),
        TreeParams(0.4f, -1.5f, 0.9f)
    };
    TreeParamsSoA new_params(indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        new_params.set(i, params[i]);
    }
    Solution before = sol;
    problem.update_and_eval(eval, indices, new_params);

    Solution updated = sol;
    for (size_t i = 0; i < indices.size(); ++i) {
        updated.set_params(static_cast<size_t>(indices[i]), params[i]);
    }

    IntersectionConstraint constraint;
    SolutionEval::IntersectionMap map;
    float violation = constraint.eval(updated, map);
    REQUIRE(eval.intersection_violation == Approx(violation).margin(1e-4f));
    require_maps_equal(eval.intersection_map, map);
    require_cache_valid(eval.solution);
    require_figures_unchanged(before, eval.solution, indices, 1e-4f);
    require_figures_changed(before, eval.solution, indices, 1e-4f);
    require_figures_match(updated, eval.solution, 1e-4f);
    require_eval_matches_full(problem, updated, eval);
}
