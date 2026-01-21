#include "tree_packing/spatial/grid2d.hpp"
#include <algorithm>
#include <cmath>

namespace tree_packing {

Grid2D Grid2D::empty(size_t num_items, int n, float size, int capacity, float center) {
    Grid2D grid;
    grid.n_ = n;
    grid.N_ = n + 2;  // Padding
    grid.capacity_ = capacity;
    grid.size_ = size;
    grid.center_ = center;

    // Initialize arrays
    grid.ij2k_.resize(grid.N_ * grid.N_ * capacity, -1);
    grid.ij2n_.resize(grid.N_ * grid.N_, 0);
    grid.k2ij_.resize(num_items * 2, 0);

    return grid;
}

Grid2D Grid2D::init(const std::vector<Vec2>& centers, int n, int capacity, float size, float center) {
    Grid2D grid = empty(centers.size(), n, size, capacity, center);

    // Add each item to its cell
    for (size_t k = 0; k < centers.size(); ++k) {
        if (centers[k].is_nan()) continue;

        auto [i, j] = grid.compute_ij(centers[k]);
        grid.add_item_to_cell(static_cast<int>(k), i, j);
        grid.k2ij_[k * 2 + 0] = i;
        grid.k2ij_[k * 2 + 1] = j;
    }

    return grid;
}

std::pair<int, int> Grid2D::compute_ij(const Vec2& pos) const {
    if (pos.is_nan()) {
        return {0, 0};  // Invalid position goes to padding cell
    }

    int n_half = n_ / 2;
    float px = pos.x - center_;
    float py = pos.y - center_;

    int i = static_cast<int>(std::floor(px / size_));
    int j = static_cast<int>(std::floor(py / size_));

    i = std::clamp(i, -n_half, n_half - 1) + n_half;
    j = std::clamp(j, -n_half, n_half - 1) + n_half;

    // Add 1 for padding
    return {i + 1, j + 1};
}

void Grid2D::add_item_to_cell(int k, int i, int j) {
    int base = cell_index(i, j);

    // Find first empty slot
    for (int s = 0; s < capacity_; ++s) {
        if (ij2k_[base + s] < 0) {
            ij2k_[base + s] = k;
            ij2n_[count_index(i, j)]++;
            return;
        }
    }
    // Cell is full - this shouldn't happen if capacity is set correctly
}

void Grid2D::remove_item_from_cell(int k, int i, int j) {
    int base = cell_index(i, j);

    for (int s = 0; s < capacity_; ++s) {
        if (ij2k_[base + s] == k) {
            ij2k_[base + s] = -1;
            ij2n_[count_index(i, j)]--;
            return;
        }
    }
}

void Grid2D::update(int k, const Vec2& new_center) {
    int old_i = k2ij_[k * 2 + 0];
    int old_j = k2ij_[k * 2 + 1];

    auto [new_i, new_j] = compute_ij(new_center);

    // Only update if cell changed
    if (old_i != new_i || old_j != new_j) {
        remove_item_from_cell(k, old_i, old_j);
        add_item_to_cell(k, new_i, new_j);
        k2ij_[k * 2 + 0] = new_i;
        k2ij_[k * 2 + 1] = new_j;
    }
}

std::vector<Index> Grid2D::get_candidates(int k) const {
    std::vector<Index> candidates;
    get_candidates(k, candidates);
    return candidates;
}

void Grid2D::get_candidates(int k, std::vector<Index>& out) const {
    int i = k2ij_[k * 2 + 0];
    int j = k2ij_[k * 2 + 1];
    get_candidates_by_cell(i, j, out);
}

std::vector<Index> Grid2D::get_candidates_by_pos(const Vec2& pos) const {
    std::vector<Index> candidates;
    get_candidates_by_pos(pos, candidates);
    return candidates;
}

void Grid2D::get_candidates_by_pos(const Vec2& pos, std::vector<Index>& out) const {
    auto [i, j] = compute_ij(pos);
    get_candidates_by_cell(i, j, out);
}

std::vector<Index> Grid2D::get_candidates_by_cell(int i, int j) const {
    std::vector<Index> candidates;
    get_candidates_by_cell(i, j, candidates);
    return candidates;
}

void Grid2D::get_candidates_by_cell(int i, int j, std::vector<Index>& out) const {
    size_t needed = NEIGHBOR_DELTAS.size() * static_cast<size_t>(capacity_);
    if (out.size() != needed) {
        out.resize(needed);
    }
    std::fill(out.begin(), out.end(), static_cast<Index>(-1));
    size_t out_idx = 0;
    // Check all 9 neighboring cells
    for (const auto& [di, dj] : NEIGHBOR_DELTAS) {
        int ni = i + di;
        int nj = j + dj;

        // Skip if out of bounds (shouldn't happen due to padding)
        if (ni < 0 || ni >= N_ || nj < 0 || nj >= N_) continue;

        int base = cell_index(ni, nj);
        for (int s = 0; s < capacity_; ++s) {
            int item = ij2k_[base + s];
            out[out_idx++] = static_cast<Index>(item);
        }
    }
}

}  // namespace tree_packing
