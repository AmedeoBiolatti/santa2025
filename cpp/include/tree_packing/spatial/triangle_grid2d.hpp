#pragma once

#include "../core/types.hpp"
#include "../core/tree.hpp"
#include "grid2d.hpp"  // For NEIGHBOR_DELTAS
#include <array>
#include <vector>
#include <cstdint>

namespace tree_packing {

// Packed triangle identifier: figure index + triangle index (0-4)
struct TriangleId {
    int16_t figure_idx{-1};
    int8_t triangle_idx{-1};
    int8_t padding{0};

    bool operator==(const TriangleId& other) const {
        return figure_idx == other.figure_idx && triangle_idx == other.triangle_idx;
    }

    bool is_valid() const { return figure_idx >= 0 && triangle_idx >= 0; }
};

// 2D spatial grid for triangle-level queries
// Each triangle is stored in exactly ONE cell (based on AABB center)
// Queries check 9 neighboring cells - no deduplication needed
class TriangleGrid2D {
public:
    static constexpr int DEFAULT_CAPACITY = 32;  // Fewer items per cell with smaller cells
    static constexpr float DEFAULT_CELL_SIZE = 0.25f;  // Smaller cells, check 5x5 neighbors

    TriangleGrid2D() = default;

    // Create empty grid for given number of figures
    static TriangleGrid2D empty(
        size_t num_figures,
        int n = 12,
        float cell_size = DEFAULT_CELL_SIZE,
        int capacity = DEFAULT_CAPACITY,
        float center = 0.0f
    );

    // Initialize grid with triangle AABBs
    static TriangleGrid2D init(
        const std::vector<std::array<AABB, TREE_NUM_TRIANGLES>>& triangle_aabbs,
        const std::vector<char>& valid,
        int n = 12,
        float cell_size = DEFAULT_CELL_SIZE,
        int capacity = DEFAULT_CAPACITY,
        float center = 0.0f
    );

    // Add all triangles of a figure to the grid
    void add_figure(int figure_idx, const std::array<AABB, TREE_NUM_TRIANGLES>& tri_aabbs);

    // Remove all triangles of a figure from the grid
    void remove_figure(int figure_idx);

    // Update a figure (remove old, add new)
    void update_figure(int figure_idx, const std::array<AABB, TREE_NUM_TRIANGLES>& tri_aabbs);

    // Get candidate triangles from 9 neighboring cells around query position
    // Returns count of valid entries written to candidates
    size_t get_candidates_by_cell(int i, int j, std::vector<TriangleId>& candidates) const;

    // Get candidates by AABB center position
    size_t get_candidates(const AABB& query_aabb, std::vector<TriangleId>& candidates) const;

    // Get candidate triangles for a specific figure's triangle
    size_t get_candidates_for_triangle(
        int figure_idx,
        int triangle_idx,
        const AABB& tri_aabb,
        std::vector<TriangleId>& candidates
    ) const;

    // Compute cell indices for a position
    std::pair<int, int> compute_ij(const Vec2& pos) const;

    // Grid parameters
    [[nodiscard]] int grid_n() const { return n_; }
    [[nodiscard]] int grid_N() const { return N_; }
    [[nodiscard]] int capacity() const { return capacity_; }
    [[nodiscard]] float cell_size() const { return cell_size_; }
    [[nodiscard]] float center() const { return center_; }
    [[nodiscard]] size_t num_figures() const { return tri_cells_.size(); }

private:
    int n_{12};           // Number of cells per dimension
    int N_{14};           // N = n + 2 (with padding)
    int capacity_{DEFAULT_CAPACITY};
    float cell_size_{DEFAULT_CELL_SIZE};
    float center_{0.0f};

    // Flat cell storage: ij2k_[i * N * capacity + j * capacity + slot] = TriangleId
    std::vector<TriangleId> ij2k_;

    // Count per cell: ij2n_[i * N + j] = number of triangles in cell
    std::vector<int> ij2n_;

    // Reverse mapping: tri_cells_[figure_idx][triangle_idx] = (i, j) cell coords
    // Stores the single cell each triangle is in (for O(1) removal)
    std::vector<std::array<std::pair<int8_t, int8_t>, TREE_NUM_TRIANGLES>> tri_cells_;

    [[nodiscard]] int cell_index(int i, int j) const {
        return i * N_ * capacity_ + j * capacity_;
    }

    [[nodiscard]] int count_index(int i, int j) const {
        return i * N_ + j;
    }

    // Add a single triangle to the grid (at its AABB center)
    void add_triangle(int figure_idx, int triangle_idx, const AABB& tri_aabb);

    // Remove a single triangle from the grid
    void remove_triangle(int figure_idx, int triangle_idx);

    void add_item_to_cell(const TriangleId& tid, int i, int j);
    void remove_item_from_cell(const TriangleId& tid, int i, int j);
};

}  // namespace tree_packing