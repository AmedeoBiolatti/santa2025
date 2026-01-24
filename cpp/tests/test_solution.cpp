#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "tree_packing/core/solution.hpp"
#include "tree_packing/core/tree.hpp"

using namespace tree_packing;
using Catch::Approx;

TEST_CASE("Solution construction", "[solution]") {
    SECTION("Default construction") {
        Solution sol(10);
        REQUIRE(sol.size() == 10);
    }

    SECTION("Random initialization") {
        Solution sol = Solution::init_random(20, 10.0f, 42);
        REQUIRE(sol.size() == 20);
        REQUIRE(sol.n_missing() == 0);

        // Check that figures and centers are computed
        REQUIRE(sol.figures().size() == 20);
        REQUIRE(sol.centers().size() == 20);
    }

    SECTION("Empty initialization") {
        Solution sol = Solution::init_empty(15);
        REQUIRE(sol.size() == 15);
        REQUIRE(sol.n_missing() == 15);
    }
}

TEST_CASE("Solution modification", "[solution]") {
    SECTION("Set params") {
        Solution sol = Solution::init_random(5, 10.0f, 42);
        auto before_normals = sol.normals()[2];

        TreeParams new_params(1.0f, 2.0f, 0.5f);
        sol.set_params(2, new_params);

        TreeParams retrieved = sol.get_params(2);
        REQUIRE(retrieved.pos.x == Approx(1.0f));
        REQUIRE(retrieved.pos.y == Approx(2.0f));
        REQUIRE(retrieved.angle == Approx(0.5f));

        std::array<std::array<Vec2, 3>, TREE_NUM_TRIANGLES> expected_normals{};
        get_tree_normals(new_params.angle, expected_normals);
        const auto& after_normals = sol.normals()[2];
        bool changed = false;
        for (size_t i = 0; i < TREE_NUM_TRIANGLES; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                if (before_normals[i][j].x != after_normals[i][j].x ||
                    before_normals[i][j].y != after_normals[i][j].y) {
                    changed = true;
                }
                REQUIRE(after_normals[i][j].x == Approx(expected_normals[i][j].x).margin(1e-6f));
                REQUIRE(after_normals[i][j].y == Approx(expected_normals[i][j].y).margin(1e-6f));
            }
        }
        REQUIRE(changed);
    }

    SECTION("Set NaN") {
        Solution sol = Solution::init_random(5, 10.0f, 42);
        REQUIRE(sol.n_missing() == 0);

        sol.set_nan(2);
        REQUIRE(sol.n_missing() == 1);
        REQUIRE_FALSE(sol.is_valid(2));
    }

    SECTION("Update with indices") {
        Solution sol = Solution::init_random(5, 10.0f, 42);
        TreeParamsSoA new_params = sol.params();
        new_params.set(1, TreeParams(5.0f, 5.0f, 1.0f));
        new_params.set(3, TreeParams(-5.0f, -5.0f, -1.0f));

        std::vector<int> indices = {1, 3};
        Solution updated = sol.update(new_params, indices);

        REQUIRE(updated.get_params(1).pos.x == Approx(5.0f));
        REQUIRE(updated.get_params(3).pos.x == Approx(-5.0f));
    }
}


TEST_CASE("SolutionEval", "[solution]") {
    SECTION("Total violation") {
        SolutionEval eval;
        eval.intersection_violation = 1.5f;
        eval.bounds_violation = 0.5f;

        REQUIRE(eval.total_violation() == Approx(2.0f));
    }
}

TEST_CASE("Solution cache validation", "[solution]") {
    Solution sol = Solution::init_random(10, 10.0f, 42);
    REQUIRE(sol.validate_cache(1e-4f, true));

    TreeParams new_params(1.0f, 2.0f, 0.5f);
    sol.set_params(3, new_params);
    REQUIRE(sol.validate_cache(1e-4f, true));

    sol.set_nan(6);
    REQUIRE(sol.validate_cache(1e-4f, true));

    TreeParamsSoA updated_params = sol.params();
    updated_params.set(1, TreeParams(5.0f, 5.0f, 1.0f));
    std::vector<int> indices = {1};
    Solution updated = sol.update(updated_params, indices);
    REQUIRE(updated.validate_cache(1e-4f, true));
}
