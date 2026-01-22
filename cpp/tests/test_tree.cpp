#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "tree_packing/core/tree.hpp"

using namespace tree_packing;
using Catch::Approx;

TEST_CASE("Tree transformations", "[tree]") {
    SECTION("rotate_point") {
        Vec2 p(1.0f, 0.0f);
        Vec2 rotated = rotate_point(p, PI / 2.0f);
        REQUIRE(rotated.x == Approx(0.0f).margin(1e-6f));
        REQUIRE(rotated.y == Approx(1.0f));
    }

    SECTION("transform_point") {
        Vec2 offset(1.0f, 1.0f);
        Vec2 p(1.0f, 0.0f);
        Vec2 result = transform_point(offset, 0.0f, p);
        REQUIRE(result.x == 2.0f);
        REQUIRE(result.y == 1.0f);
    }

    SECTION("params_to_figure") {
        TreeParams params(0.0f, 0.0f, 0.0f);
        Figure fig = params_to_figure(params);

        // Check that figure has correct number of triangles
        REQUIRE(fig.size() == TREE_NUM_TRIANGLES);

        // First triangle tip should be at (0, 0.8) for zero params
        REQUIRE(fig.triangles[0].v0.x == Approx(0.0f));
        REQUIRE(fig.triangles[0].v0.y == Approx(0.8f));
    }

    SECTION("get_tree_center") {
        TreeParams params(0.0f, 0.0f, 0.0f);
        Vec2 center = get_tree_center(params);

        // Center should be offset by CENTER_Y in y direction for angle=0
        REQUIRE(center.x == Approx(0.0f).margin(1e-6f));
        REQUIRE(center.y == Approx(CENTER_Y));
    }
}

TEST_CASE("Batch tree operations", "[tree]") {
    SECTION("params_to_figures") {
        TreeParamsSoA params(3);
        params.set(0, TreeParams(0.0f, 0.0f, 0.0f));
        params.set(1, TreeParams(1.0f, 0.0f, PI / 2.0f));
        params.set(2, TreeParams(0.0f, 1.0f, PI));

        std::vector<Figure> figures;
        params_to_figures(params, figures);

        REQUIRE(figures.size() == 3);
        REQUIRE_FALSE(figures[0].is_nan());
        REQUIRE_FALSE(figures[1].is_nan());
        REQUIRE_FALSE(figures[2].is_nan());
    }

    SECTION("get_tree_centers") {
        TreeParamsSoA params(2);
        params.set(0, TreeParams(0.0f, 0.0f, 0.0f));
        params.set(1, TreeParams(1.0f, 1.0f, PI));

        std::vector<Vec2> centers;
        get_tree_centers(params, centers);

        REQUIRE(centers.size() == 2);
        REQUIRE(centers[0].y == Approx(CENTER_Y));
    }
}
