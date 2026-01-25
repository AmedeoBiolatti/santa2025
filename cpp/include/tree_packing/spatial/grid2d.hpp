#pragma once

#include "../core/types.hpp"
#include "../core/tree.hpp"
#include <array>
#include <vector>

namespace tree_packing {

// 9-neighborhood deltas for candidate search
constexpr std::array<std::pair<int, int>, 9> NEIGHBOR_DELTAS = {{
    {0, -1}, {-1, -1}, {+1, -1},
    {0, 0},  {-1, 0},  {+1, 0},
    {0, +1}, {-1, +1}, {+1, +1}
}};

// 2D spatial grid for efficient neighbor queries
// Uses padding to avoid boundary checks
class Grid2D {
public:
    Grid2D() = default;

    // Create empty grid
    static Grid2D empty(size_t num_items, int n = 20, float size = 1.04f, int capacity = 16, float center = 0.0f);

    // Initialize grid with centers
    static Grid2D init(const std::vector<Vec2>& centers, int n = 16, int capacity = 8, float size = 1.04f, float center = 0.0f);

    // Update position of item k
    void update(int k, const Vec2& new_center);
    void insert(int k, const Vec2& new_center);
    void remove(int k);

    // Get candidate indices near item k (from its cell's neighborhood)
    // Returns indices, with -1 for empty slots
    std::vector<Index> get_candidates(int k) const;
    void get_candidates(int k, std::vector<Index>& out) const;

    // Get candidate indices near a position
    std::vector<Index> get_candidates_by_pos(const Vec2& pos) const;
    void get_candidates_by_pos(const Vec2& pos, std::vector<Index>& out) const;

    // Get candidate indices near a cell
    std::vector<Index> get_candidates_by_cell(int i, int j) const;
    void get_candidates_by_cell(int i, int j, std::vector<Index>& out) const;

    // Get item indices in a cell
    std::vector<Index> get_items_in_cell(int i, int j) const;
    void get_items_in_cell(int i, int j, std::vector<Index>& out) const;

    // Get count of items in a cell
    [[nodiscard]] int cell_count(int i, int j) const;

    // Get all non-empty cells (for efficient iteration)
    [[nodiscard]] const std::vector<std::pair<int, int>>& non_empty_cells() const { return non_empty_cells_; }

    // Compute cell indices for a position
    std::pair<int, int> compute_ij(const Vec2& pos) const;

    // Get cell indices for an item
    std::pair<int, int> get_item_cell(int k) const;

    // Compute bounds for a cell index (including padding cells)
    AABB cell_bounds(int i, int j) const;

    // Compute bounds for a cell index expanded by CENTER_R
    AABB cell_bounds_expanded(int i, int j) const;

    std::pair<float, int> get_min_x(const std::vector<AABB>& aabbs) const;
    std::pair<float, int> get_max_x(const std::vector<AABB>& aabbs) const;
    std::pair<float, int> get_min_y(const std::vector<AABB>& aabbs) const;
    std::pair<float, int> get_max_y(const std::vector<AABB>& aabbs) const;

    // Grid parameters
    [[nodiscard]] int grid_n() const { return n_; }
    [[nodiscard]] int grid_N() const { return N_; }
    [[nodiscard]] int capacity() const { return capacity_; }
    [[nodiscard]] float cell_size() const { return size_; }
private:
    int n_{20};          // Number of cells per dimension
    int N_{22};          // N = n + 2 (with padding)
    int capacity_{8};    // Max items per cell
    float size_{1.04f};  // Cell size
    float center_{0.0f}; // Grid center
    //
    std::vector<int> i2n_;
    std::vector<int> j2n_;
    int min_i_{22}, max_i_{0}, min_j_{22}, max_j_{0};
    // ij2k[i * N * capacity + j * capacity + slot] = item index (-1 if empty)
    std::vector<int> ij2k_;
    // ij2n[i * N + j] = number of items in cell (i, j)
    std::vector<int> ij2n_;
    // k2ij[k * 2 + 0] = i, k2ij[k * 2 + 1] = j
    std::vector<int> k2ij_;
    // Precomputed cell bounds (including padding cells)
    std::vector<AABB> cell_bounds_;
    // Precomputed cell bounds expanded by CENTER_R
    std::vector<AABB> cell_bounds_expanded_;
    // List of non-empty cells for efficient iteration
    std::vector<std::pair<int, int>> non_empty_cells_;
    // Map from cell (i*N+j) to index in non_empty_cells_ (-1 if empty)
    std::vector<int> cell_to_non_empty_idx_;

    [[nodiscard]] int cell_index(int i, int j) const {
        return i * N_ * capacity_ + j * capacity_;
    }

    [[nodiscard]] int count_index(int i, int j) const {
        return i * N_ + j;
    }

    [[nodiscard]] int item_ij_index(int k) const {
        return k * 2;
    }

    void add_item_to_cell(int k, int i, int j);
    void remove_item_from_cell(int k, int i, int j);
};

}  // namespace tree_packing
