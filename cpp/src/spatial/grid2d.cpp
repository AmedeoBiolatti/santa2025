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
    grid.cell_bounds_.resize(grid.N_ * grid.N_);
    grid.cell_bounds_expanded_.resize(grid.N_ * grid.N_);
    grid.cell_to_non_empty_idx_.resize(grid.N_ * grid.N_, -1);
    grid.non_empty_cells_.reserve(grid.N_ * grid.N_ / 4);  // Expect ~25% occupancy
    for (int i = 0; i < grid.N_; ++i) {
        for (int j = 0; j < grid.N_; ++j) {
            int n_half = grid.n_ / 2;
            int gi = i - 1 - n_half;
            int gj = j - 1 - n_half;
            float min_x = grid.center_ + static_cast<float>(gi) * grid.size_;
            float min_y = grid.center_ + static_cast<float>(gj) * grid.size_;
            Vec2 min{min_x, min_y};
            Vec2 max{min_x + grid.size_, min_y + grid.size_};
            AABB bounds{min, max};
            grid.cell_bounds_[i * grid.N_ + j] = bounds;
            bounds.min.x -= CENTER_R;
            bounds.min.y -= CENTER_R;
            bounds.max.x += CENTER_R;
            bounds.max.y += CENTER_R;
            grid.cell_bounds_expanded_[i * grid.N_ + j] = bounds;
        }
    }

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

std::pair<int, int> Grid2D::get_item_cell(int k) const {
    return {k2ij_[k * 2 + 0], k2ij_[k * 2 + 1]};
}

AABB Grid2D::cell_bounds(int i, int j) const {
#ifndef NDEBUG
    if (i < 0 || i >= N_ || j < 0 || j >= N_) {
        return AABB{};
    }
#endif
    return cell_bounds_[i * N_ + j];
}

AABB Grid2D::cell_bounds_expanded(int i, int j) const {
#ifndef NDEBUG
    if (i < 0 || i >= N_ || j < 0 || j >= N_) {
        return AABB{};
    }
#endif
    return cell_bounds_expanded_[i * N_ + j];
}

void Grid2D::add_item_to_cell(int k, int i, int j) {
    int base = cell_index(i, j);
    int cnt_idx = count_index(i, j);

    // Find first empty slot
    for (int s = 0; s < capacity_; ++s) {
        if (ij2k_[base + s] < 0) {
            ij2k_[base + s] = k;
            int old_count = ij2n_[cnt_idx]++;
            // Track non-empty cells: add when count goes 0->1
            if (old_count == 0) {
                cell_to_non_empty_idx_[cnt_idx] = static_cast<int>(non_empty_cells_.size());
                non_empty_cells_.emplace_back(i, j);
            }
            return;
        }
    }
    // Cell is full - this shouldn't happen if capacity is set correctly
}

void Grid2D::remove_item_from_cell(int k, int i, int j) {
    int base = cell_index(i, j);
    int cnt_idx = count_index(i, j);

    for (int s = 0; s < capacity_; ++s) {
        if (ij2k_[base + s] == k) {
            ij2k_[base + s] = -1;
            int new_count = --ij2n_[cnt_idx];
            // Track non-empty cells: remove when count goes 1->0
            if (new_count == 0) {
                int idx = cell_to_non_empty_idx_[cnt_idx];
                if (idx >= 0 && idx < static_cast<int>(non_empty_cells_.size())) {
                    // Swap with last and pop
                    auto& last = non_empty_cells_.back();
                    int last_cnt_idx = last.first * N_ + last.second;
                    non_empty_cells_[idx] = last;
                    cell_to_non_empty_idx_[last_cnt_idx] = idx;
                    non_empty_cells_.pop_back();
                }
                cell_to_non_empty_idx_[cnt_idx] = -1;
            }
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

void Grid2D::insert(int k, const Vec2& new_center) {
    auto [new_i, new_j] = compute_ij(new_center);
    add_item_to_cell(k, new_i, new_j);
    k2ij_[k * 2 + 0] = new_i;
    k2ij_[k * 2 + 1] = new_j;
}

void Grid2D::remove(int k) {
    int old_i = k2ij_[k * 2 + 0];
    int old_j = k2ij_[k * 2 + 1];

    // Only update if cell changed
    remove_item_from_cell(k, old_i, old_j);
    k2ij_[k * 2 + 0] = -1;
    k2ij_[k * 2 + 1] = -1;
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

std::vector<Index> Grid2D::get_items_in_cell(int i, int j) const {
    std::vector<Index> items;
    get_items_in_cell(i, j, items);
    return items;
}

void Grid2D::get_items_in_cell(int i, int j, std::vector<Index>& out) const {
    size_t needed = static_cast<size_t>(capacity_);
    if (out.size() != needed) {
        out.resize(needed);
    }
#ifndef NDEBUG
    if (i < 0 || i >= N_ || j < 0 || j >= N_) {
        std::fill(out.begin(), out.end(), static_cast<Index>(-1));
        return;
    }
#endif
    // Copy all slots directly - ij2k_ already has -1 for empty slots
    int base = cell_index(i, j);
    for (int s = 0; s < capacity_; ++s) {
        out[static_cast<size_t>(s)] = static_cast<Index>(ij2k_[base + s]);
    }
}

int Grid2D::cell_count(int i, int j) const {
#ifndef NDEBUG
    if (i < 0 || i >= N_ || j < 0 || j >= N_) {
        return 0;
    }
#endif
    return ij2n_[count_index(i, j)];
}

}  // namespace tree_packing
