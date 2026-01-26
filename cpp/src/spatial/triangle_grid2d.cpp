#include "tree_packing/spatial/triangle_grid2d.hpp"
#include <algorithm>
#include <cmath>

namespace tree_packing {

TriangleGrid2D TriangleGrid2D::empty(
    size_t num_figures,
    int n,
    float cell_size,
    int capacity,
    float center
) {
    TriangleGrid2D grid;
    grid.n_ = n;
    grid.N_ = n + 4;  // Padding for 5x5 boundary safety (need 2 on each side)
    grid.capacity_ = capacity;
    grid.cell_size_ = cell_size;
    grid.center_ = center;

    // Initialize flat cell storage
    size_t total_slots = static_cast<size_t>(grid.N_ * grid.N_ * capacity);
    grid.ij2k_.resize(total_slots);  // Default-constructed TriangleIds are invalid

    // Initialize counts
    grid.ij2n_.resize(static_cast<size_t>(grid.N_ * grid.N_), 0);

    // Initialize reverse mapping with invalid cell coords (-1, -1)
    grid.tri_cells_.resize(num_figures);
    for (auto& fig_tris : grid.tri_cells_) {
        for (auto& cell : fig_tris) {
            cell = {-1, -1};
        }
    }

    return grid;
}

TriangleGrid2D TriangleGrid2D::init(
    const std::vector<std::array<AABB, TREE_NUM_TRIANGLES>>& triangle_aabbs,
    const std::vector<char>& valid,
    int n,
    float cell_size,
    int capacity,
    float center
) {
    TriangleGrid2D grid = empty(triangle_aabbs.size(), n, cell_size, capacity, center);

    for (size_t fig = 0; fig < triangle_aabbs.size(); ++fig) {
        if (!valid[fig]) continue;
        grid.add_figure(static_cast<int>(fig), triangle_aabbs[fig]);
    }

    return grid;
}

std::pair<int, int> TriangleGrid2D::compute_ij(const Vec2& pos) const {
    if (pos.is_nan()) {
        return {0, 0};  // Invalid position goes to padding cell
    }

    int n_half = n_ / 2;
    float px = pos.x - center_;
    float py = pos.y - center_;

    int i = static_cast<int>(std::floor(px / cell_size_));
    int j = static_cast<int>(std::floor(py / cell_size_));

    i = std::clamp(i, -n_half, n_half - 1) + n_half;
    j = std::clamp(j, -n_half, n_half - 1) + n_half;

    // Add 2 for padding (5x5 neighborhood needs 2 cells on each side)
    return {i + 2, j + 2};
}

void TriangleGrid2D::add_item_to_cell(const TriangleId& tid, int i, int j) {
    int base = cell_index(i, j);
    int cnt_idx = count_index(i, j);
    int old_count = ij2n_[cnt_idx];

    // Cell is full - shouldn't happen if capacity is set correctly
    if (old_count >= capacity_) return;

    // Append at end
    ij2k_[base + old_count] = tid;
    ij2n_[cnt_idx]++;
}

void TriangleGrid2D::remove_item_from_cell(const TriangleId& tid, int i, int j) {
    int base = cell_index(i, j);
    int cnt_idx = count_index(i, j);
    int count = ij2n_[cnt_idx];

    for (int s = 0; s < count; ++s) {
        if (ij2k_[base + s] == tid) {
            // Swap with last to keep compact
            ij2k_[base + s] = ij2k_[base + count - 1];
            ij2k_[base + count - 1] = TriangleId{};  // Clear last slot
            ij2n_[cnt_idx]--;
            return;
        }
    }
}

void TriangleGrid2D::add_triangle(int figure_idx, int triangle_idx, const AABB& tri_aabb) {
    if (tri_aabb.min.x != tri_aabb.min.x) return;  // NaN check

    // Compute center of AABB
    Vec2 center{
        (tri_aabb.min.x + tri_aabb.max.x) * 0.5f,
        (tri_aabb.min.y + tri_aabb.max.y) * 0.5f
    };

    auto [i, j] = compute_ij(center);

    TriangleId tid{static_cast<int16_t>(figure_idx), static_cast<int8_t>(triangle_idx), 0};

    // Add to cell
    add_item_to_cell(tid, i, j);

    // Store reverse mapping
    if (static_cast<size_t>(figure_idx) >= tri_cells_.size()) {
        tri_cells_.resize(static_cast<size_t>(figure_idx) + 1);
        for (auto& cell : tri_cells_.back()) {
            cell = {-1, -1};
        }
    }
    tri_cells_[figure_idx][triangle_idx] = {static_cast<int8_t>(i), static_cast<int8_t>(j)};
}

void TriangleGrid2D::remove_triangle(int figure_idx, int triangle_idx) {
    if (static_cast<size_t>(figure_idx) >= tri_cells_.size()) return;

    auto [i, j] = tri_cells_[figure_idx][triangle_idx];
    if (i < 0 || j < 0) return;  // Not in grid

    TriangleId tid{static_cast<int16_t>(figure_idx), static_cast<int8_t>(triangle_idx), 0};
    remove_item_from_cell(tid, i, j);

    // Clear reverse mapping
    tri_cells_[figure_idx][triangle_idx] = {-1, -1};
}

void TriangleGrid2D::add_figure(int figure_idx, const std::array<AABB, TREE_NUM_TRIANGLES>& tri_aabbs) {
    for (size_t t = 0; t < TREE_NUM_TRIANGLES; ++t) {
        add_triangle(figure_idx, static_cast<int>(t), tri_aabbs[t]);
    }
}

void TriangleGrid2D::remove_figure(int figure_idx) {
    for (size_t t = 0; t < TREE_NUM_TRIANGLES; ++t) {
        remove_triangle(figure_idx, static_cast<int>(t));
    }
}

void TriangleGrid2D::update_figure(int figure_idx, const std::array<AABB, TREE_NUM_TRIANGLES>& tri_aabbs) {
    remove_figure(figure_idx);
    add_figure(figure_idx, tri_aabbs);
}

size_t TriangleGrid2D::get_candidates_by_cell(int i, int j, std::vector<TriangleId>& candidates) const {
    size_t out_idx = 0;
    TriangleId* out_ptr = candidates.data();

    // Check all 25 neighboring cells (5x5) for finer grid
    for (const auto& [di, dj] : NEIGHBOR_DELTAS_5x5) {
        int ni = i + di;
        int nj = j + dj;
        // Bounds check for extended neighborhood
        if (ni < 0 || ni >= N_ || nj < 0 || nj >= N_) continue;
        int cnt_idx = count_index(ni, nj);
        int count = ij2n_[cnt_idx];
        if (count == 0) continue;

        int base = cell_index(ni, nj);
        for (int s = 0; s < count; ++s) {
            out_ptr[out_idx++] = ij2k_[base + s];
        }
    }

    return out_idx;
}

size_t TriangleGrid2D::get_candidates(const AABB& query_aabb, std::vector<TriangleId>& candidates) const {
    // Compute center of query AABB
    Vec2 center{
        (query_aabb.min.x + query_aabb.max.x) * 0.5f,
        (query_aabb.min.y + query_aabb.max.y) * 0.5f
    };

    auto [i, j] = compute_ij(center);
    return get_candidates_by_cell(i, j, candidates);
}

size_t TriangleGrid2D::get_candidates_for_triangle(
    int figure_idx,
    int triangle_idx,
    const AABB& tri_aabb,
    std::vector<TriangleId>& candidates
) const {
    size_t count = get_candidates(tri_aabb, candidates);

    // Filter out same figure (in-place)
    size_t write_idx = 0;
    for (size_t read_idx = 0; read_idx < count; ++read_idx) {
        if (candidates[read_idx].figure_idx != figure_idx) {
            candidates[write_idx++] = candidates[read_idx];
        }
    }

    (void)triangle_idx;  // Not used for filtering

    return write_idx;
}

}  // namespace tree_packing
