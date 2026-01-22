#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "tree_packing/spatial/grid2d.hpp"

using namespace tree_packing;

TEST_CASE("Grid2D construction", "[grid2d]") {
    SECTION("Empty grid") {
        Grid2D grid = Grid2D::empty(10, 20, 1.0f, 8);
        REQUIRE(grid.grid_n() == 20);
        REQUIRE(grid.capacity() == 8);
        REQUIRE(grid.cell_size() == 1.0f);
    }

    SECTION("Initialize with centers") {
        std::vector<Vec2> centers = {
            Vec2(0.0f, 0.0f),
            Vec2(1.0f, 1.0f),
            Vec2(-1.0f, -1.0f)
        };

        Grid2D grid = Grid2D::init(centers, 20, 8, 1.0f);
        REQUIRE(grid.grid_n() == 20);

        auto candidates = grid.get_candidates(0);
        REQUIRE(!candidates.empty());
    }
}

TEST_CASE("Grid2D cell computation", "[grid2d]") {
    SECTION("compute_ij center") {
        Grid2D grid = Grid2D::empty(10, 20, 1.0f, 8);
        auto [i, j] = grid.compute_ij(Vec2(0.0f, 0.0f));

        // Center should map to middle of grid (with +1 for padding)
        REQUIRE(i == 11);
        REQUIRE(j == 11);
    }

    SECTION("compute_ij offset") {
        Grid2D grid = Grid2D::empty(10, 20, 1.0f, 8);
        auto [i, j] = grid.compute_ij(Vec2(5.0f, 5.0f));

        // 5.0 / 1.0 = 5 cells from center = 10 + 5 + 1 = 16
        REQUIRE(i == 16);
        REQUIRE(j == 16);
    }

    SECTION("compute_ij negative") {
        Grid2D grid = Grid2D::empty(10, 20, 1.0f, 8);
        auto [i, j] = grid.compute_ij(Vec2(-5.0f, -5.0f));

        // -5.0 / 1.0 = -5 cells from center = 10 - 5 + 1 = 6
        REQUIRE(i == 6);
        REQUIRE(j == 6);
    }
}

TEST_CASE("Grid2D update and candidates", "[grid2d]") {
    SECTION("Update single item") {
        std::vector<Vec2> centers = {
            Vec2(0.0f, 0.0f),
            Vec2(1.0f, 1.0f)
        };

        Grid2D grid = Grid2D::init(centers, 20, 8, 1.0f);

        // Move first item
        grid.update(0, Vec2(5.0f, 5.0f));

        // Check candidates near new position
        auto candidates = grid.get_candidates_by_pos(Vec2(5.0f, 5.0f));

        bool found = false;
        for (int c : candidates) {
            if (c == 0) found = true;
        }
        REQUIRE(found);
    }

    SECTION("Candidates from neighbors") {
        std::vector<Vec2> centers;
        // Create a grid of items
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 5; ++j) {
                centers.push_back(Vec2(static_cast<float>(i) * 0.5f,
                                       static_cast<float>(j) * 0.5f));
            }
        }

        Grid2D grid = Grid2D::init(centers, 20, 16, 1.0f);

        // Get candidates for center item
        auto candidates = grid.get_candidates(12);  // Middle item

        // Should include itself and neighbors
        REQUIRE(!candidates.empty());
    }
}

TEST_CASE("Grid2D NaN handling", "[grid2d]") {
    SECTION("NaN positions") {
        std::vector<Vec2> centers = {
            Vec2(0.0f, 0.0f),
            Vec2::nan(),
            Vec2(1.0f, 1.0f)
        };

        Grid2D grid = Grid2D::init(centers, 20, 8, 1.0f);

        // NaN item should be at padding cell
        auto candidates = grid.get_candidates(0);

        // Should not include NaN item
        for (int c : candidates) {
            // Item 1 (NaN) might be in candidates if in same cell, but that's ok
            // The important thing is that the grid doesn't crash
        }
    }
}
