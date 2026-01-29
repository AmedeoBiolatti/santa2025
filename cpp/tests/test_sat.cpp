#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <algorithm>
#include <cstddef>
#include "tree_packing/geometry/sat.hpp"
#include "tree_packing/geometry/utilities.hpp"
#include "tree_packing/core/tree.hpp"

using namespace tree_packing;
using Catch::Approx;

TEST_CASE("SAT edge normals", "[sat]") {
    SECTION("compute_edge_normals") {
        Triangle tri(Vec2(0.0f, 0.0f), Vec2(1.0f, 0.0f), Vec2(0.5f, 1.0f));
        auto normals = compute_edge_normals(tri);

        REQUIRE(normals.size() == 3);

        // First edge (0,0) to (1,0) should have normal perpendicular to (1,0)
        // Perpendicular of (1,0) is (0,1) or (0,-1)
        REQUIRE(std::abs(normals[0].x) == Approx(0.0f).margin(1e-6f));
        REQUIRE(std::abs(normals[0].y) == Approx(1.0f));
    }
}

TEST_CASE("SAT projection", "[sat]") {
    SECTION("project_triangle") {
        Triangle tri(Vec2(0.0f, 0.0f), Vec2(2.0f, 0.0f), Vec2(1.0f, 2.0f));
        Vec2 axis(1.0f, 0.0f);  // X-axis

        auto proj = project_triangle(tri, axis);
        REQUIRE(proj.min == 0.0f);
        REQUIRE(proj.max == 2.0f);

        Vec2 axis_y(0.0f, 1.0f);  // Y-axis
        auto proj_y = project_triangle(tri, axis_y);
        REQUIRE(proj_y.min == 0.0f);
        REQUIRE(proj_y.max == 2.0f);
    }
}

TEST_CASE("SAT intersection", "[sat]") {
    SECTION("Overlapping triangles") {
        Triangle t1(Vec2(0.0f, 0.0f), Vec2(2.0f, 0.0f), Vec2(1.0f, 2.0f));
        Triangle t2(Vec2(0.5f, 0.5f), Vec2(2.5f, 0.5f), Vec2(1.5f, 2.5f));

        REQUIRE(triangles_intersect(t1, t2));

        float score = triangles_intersection_score(t1, t2);
        REQUIRE(score > 0.0f);
    }

    SECTION("Separated triangles") {
        Triangle t1(Vec2(0.0f, 0.0f), Vec2(1.0f, 0.0f), Vec2(0.5f, 1.0f));
        Triangle t2(Vec2(5.0f, 0.0f), Vec2(6.0f, 0.0f), Vec2(5.5f, 1.0f));

        REQUIRE_FALSE(triangles_intersect(t1, t2));

        float score = triangles_intersection_score(t1, t2);
        REQUIRE(score <= 0.0f);
    }

    SECTION("Same triangle") {
        Triangle t(Vec2(0.0f, 0.0f), Vec2(1.0f, 0.0f), Vec2(0.5f, 1.0f));

        REQUIRE(triangles_intersect(t, t));

        float score = triangles_intersection_score(t, t);
        REQUIRE(score > 0.0f);
    }

}

TEST_CASE("Figure intersection", "[sat]") {
    SECTION("Overlapping figures") {
        // Create two simple figures
        Figure f1, f2;
        for (int i = 0; i < 5; ++i) {
            f1.triangles[i] = Triangle(
                Vec2(0.0f + i * 0.1f, 0.0f),
                Vec2(0.5f + i * 0.1f, 0.0f),
                Vec2(0.25f + i * 0.1f, 0.5f)
            );
            f2.triangles[i] = Triangle(
                Vec2(0.2f + i * 0.1f, 0.2f),
                Vec2(0.7f + i * 0.1f, 0.2f),
                Vec2(0.45f + i * 0.1f, 0.7f)
            );
        }

        float score = figure_intersection_score(f1, f2);
        REQUIRE(score > 0.0f);
    }

    SECTION("Separated figures") {
        Figure f1, f2;
        for (int i = 0; i < 5; ++i) {
            f1.triangles[i] = Triangle(
                Vec2(0.0f, 0.0f),
                Vec2(0.5f, 0.0f),
                Vec2(0.25f, 0.5f)
            );
            f2.triangles[i] = Triangle(
                Vec2(10.0f, 10.0f),
                Vec2(10.5f, 10.0f),
                Vec2(10.25f, 10.5f)
            );
        }

        float score = figure_intersection_score(f1, f2);
        REQUIRE(score == 0.0f);
    }
}

TEST_CASE("Intersection matrix", "[sat]") {
    SECTION("Small matrix") {
        std::vector<Figure> figures(3);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 5; ++j) {
                figures[i].triangles[j] = Triangle(
                    Vec2(i * 0.1f, 0.0f),
                    Vec2(i * 0.1f + 0.5f, 0.0f),
                    Vec2(i * 0.1f + 0.25f, 0.5f)
                );
            }
        }

        std::vector<float> matrix;
        compute_intersection_matrix(figures, matrix);

        REQUIRE(matrix.size() == 9);

        // Diagonal should be zero
        REQUIRE(matrix[0] == 0.0f);
        REQUIRE(matrix[4] == 0.0f);
        REQUIRE(matrix[8] == 0.0f);

        // Symmetric
        REQUIRE(matrix[1] == matrix[3]);
        REQUIRE(matrix[2] == matrix[6]);
        REQUIRE(matrix[5] == matrix[7]);
    }
}

TEST_CASE("Triangle pair intersection scores utility", "[sat]") {
    SECTION("Far apart trees produce all zeros") {
        TreeParams a(Vec2(0.0f, 0.0f), 0.0f);
        TreeParams b(Vec2(100.0f, 100.0f), 0.0f);

        const auto scores = triangle_pair_intersection_scores(a, b);
        for (size_t i = 0; i < TREE_NUM_TRIANGLES; ++i) {
            for (size_t j = 0; j < TREE_NUM_TRIANGLES; ++j) {
                REQUIRE(scores[i][j] == 0.0f);
            }
        }
    }

    SECTION("Sum of positive pair scores matches figure score") {
        TreeParams a(Vec2(0.0f, 0.0f), 0.0f);
        TreeParams b(Vec2(0.05f, 0.02f), 0.1f);

        const auto scores = triangle_pair_intersection_scores(a, b);
        float sum_pos = 0.0f;
        for (size_t i = 0; i < TREE_NUM_TRIANGLES; ++i) {
            for (size_t j = 0; j < TREE_NUM_TRIANGLES; ++j) {
                sum_pos += std::max(0.0f, scores[i][j]);
            }
        }

        const Figure fa = params_to_figure(a);
        const Figure fb = params_to_figure(b);
        const float figure_score = figure_intersection_score(fa, fb);
        REQUIRE(sum_pos == Approx(figure_score).margin(1e-5f));
    }
}
