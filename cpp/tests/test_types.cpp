#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "tree_packing/core/types.hpp"

using namespace tree_packing;
using Catch::Approx;

TEST_CASE("Vec2 operations", "[types]") {
    SECTION("Construction") {
        Vec2 v1;
        REQUIRE(v1.x == 0.0f);
        REQUIRE(v1.y == 0.0f);

        Vec2 v2(1.0f, 2.0f);
        REQUIRE(v2.x == 1.0f);
        REQUIRE(v2.y == 2.0f);
    }

    SECTION("Arithmetic") {
        Vec2 a(1.0f, 2.0f);
        Vec2 b(3.0f, 4.0f);

        Vec2 sum = a + b;
        REQUIRE(sum.x == 4.0f);
        REQUIRE(sum.y == 6.0f);

        Vec2 diff = b - a;
        REQUIRE(diff.x == 2.0f);
        REQUIRE(diff.y == 2.0f);

        Vec2 scaled = a * 2.0f;
        REQUIRE(scaled.x == 2.0f);
        REQUIRE(scaled.y == 4.0f);
    }

    SECTION("Dot product") {
        Vec2 a(1.0f, 0.0f);
        Vec2 b(0.0f, 1.0f);
        REQUIRE(a.dot(b) == 0.0f);

        Vec2 c(1.0f, 2.0f);
        Vec2 d(3.0f, 4.0f);
        REQUIRE(c.dot(d) == 11.0f);
    }

    SECTION("Length") {
        Vec2 v(3.0f, 4.0f);
        REQUIRE(v.length() == Approx(5.0f));
        REQUIRE(v.length_squared() == 25.0f);
    }

    SECTION("Normalized") {
        Vec2 v(3.0f, 4.0f);
        Vec2 n = v.normalized();
        REQUIRE(n.x == Approx(0.6f));
        REQUIRE(n.y == Approx(0.8f));
    }

    SECTION("Perpendicular") {
        Vec2 v(1.0f, 0.0f);
        Vec2 p = v.perpendicular();
        REQUIRE(p.x == 0.0f);
        REQUIRE(p.y == 1.0f);
    }

    SECTION("NaN checks") {
        Vec2 v(1.0f, 2.0f);
        REQUIRE_FALSE(v.is_nan());

        Vec2 nan_v = Vec2::nan();
        REQUIRE(nan_v.is_nan());
    }
}

TEST_CASE("Mat2 operations", "[types]") {
    SECTION("Identity") {
        Mat2 m;
        Vec2 v(1.0f, 2.0f);
        Vec2 result = m * v;
        REQUIRE(result.x == v.x);
        REQUIRE(result.y == v.y);
    }

    SECTION("Rotation") {
        Mat2 rot = Mat2::rotation(PI / 2.0f);  // 90 degrees
        Vec2 v(1.0f, 0.0f);
        Vec2 result = rot * v;
        REQUIRE(result.x == Approx(0.0f).margin(1e-6f));
        REQUIRE(result.y == Approx(1.0f));
    }
}

TEST_CASE("Triangle operations", "[types]") {
    SECTION("Construction") {
        Triangle tri(Vec2(0.0f, 0.0f), Vec2(1.0f, 0.0f), Vec2(0.5f, 1.0f));
        REQUIRE(tri.v0.x == 0.0f);
        REQUIRE(tri.v1.x == 1.0f);
        REQUIRE(tri.v2.y == 1.0f);
    }

    SECTION("Centroid") {
        Triangle tri(Vec2(0.0f, 0.0f), Vec2(3.0f, 0.0f), Vec2(0.0f, 3.0f));
        Vec2 c = tri.centroid();
        REQUIRE(c.x == Approx(1.0f));
        REQUIRE(c.y == Approx(1.0f));
    }
}

TEST_CASE("TreeParams operations", "[types]") {
    SECTION("Construction") {
        TreeParams p(1.0f, 2.0f, 0.5f);
        REQUIRE(p.pos.x == 1.0f);
        REQUIRE(p.pos.y == 2.0f);
        REQUIRE(p.angle == 0.5f);
    }

    SECTION("NaN") {
        TreeParams p;
        REQUIRE_FALSE(p.is_nan());

        p.set_nan();
        REQUIRE(p.is_nan());
    }
}

TEST_CASE("TreeParamsSoA operations", "[types]") {
    SECTION("Construction") {
        TreeParamsSoA params(10);
        REQUIRE(params.size() == 10);
    }

    SECTION("Get/Set") {
        TreeParamsSoA params(5);
        params.set(2, TreeParams(1.0f, 2.0f, 0.5f));

        TreeParams p = params.get(2);
        REQUIRE(p.pos.x == 1.0f);
        REQUIRE(p.pos.y == 2.0f);
        REQUIRE(p.angle == 0.5f);
    }

    SECTION("NaN count") {
        TreeParamsSoA params(5);
        params.set_nan(1);
        params.set_nan(3);
        REQUIRE(params.count_nan() == 2);
    }
}
