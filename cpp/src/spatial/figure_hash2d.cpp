#include "tree_packing/spatial/figure_hash2d.hpp"
#include <algorithm>
#include <cmath>

namespace tree_packing {

FigureHash2D FigureHash2D::empty(size_t num_figures, float cell_size) {
    FigureHash2D hash;
    hash.cell_size_ = cell_size;

    // Initialize reverse mapping with invalid cell coords
    hash.fig_cells_.resize(num_figures, {-32768, -32768});

    // Reserve expected number of buckets
    hash.buckets_.reserve(num_figures / 4 + 1);

    return hash;
}

FigureHash2D FigureHash2D::init(
    const std::vector<Vec2>& centers,
    const std::vector<char>& valid,
    float cell_size
) {
    FigureHash2D hash = empty(centers.size(), cell_size);

    for (size_t fig = 0; fig < centers.size(); ++fig) {
        if (!valid[fig]) continue;
        hash.insert(static_cast<int>(fig), centers[fig]);
    }

    return hash;
}

std::pair<int, int> FigureHash2D::compute_ij(const Vec2& pos) const {
    if (pos.is_nan()) {
        return {0, 0};
    }

    int i = static_cast<int>(std::floor(pos.x / cell_size_));
    int j = static_cast<int>(std::floor(pos.y / cell_size_));

    return {i, j};
}

void FigureHash2D::insert(int figure_idx, const Vec2& center) {
    if (center.is_nan()) return;

    auto [i, j] = compute_ij(center);

    // Add to bucket (creates bucket if doesn't exist)
    FigureCellKey key(i, j);
    auto& bucket = buckets_[key];
    if (bucket.empty()) {
        bucket.reserve(16);
    }
    bucket.push_back(static_cast<Index>(figure_idx));

    // Store reverse mapping
    if (static_cast<size_t>(figure_idx) >= fig_cells_.size()) {
        fig_cells_.resize(static_cast<size_t>(figure_idx) + 1, {-32768, -32768});
    }
    fig_cells_[figure_idx] = {static_cast<int16_t>(i), static_cast<int16_t>(j)};
}

void FigureHash2D::remove(int figure_idx) {
    if (static_cast<size_t>(figure_idx) >= fig_cells_.size()) return;

    auto [i, j] = fig_cells_[figure_idx];
    if (i == -32768 && j == -32768) return;  // Not in hash

    FigureCellKey key(i, j);
    auto it = buckets_.find(key);
    if (it == buckets_.end()) return;

    // Remove from bucket (swap with last for O(1))
    auto& bucket = it->second;
    Index idx = static_cast<Index>(figure_idx);
    for (size_t s = 0; s < bucket.size(); ++s) {
        if (bucket[s] == idx) {
            bucket[s] = bucket.back();
            bucket.pop_back();
            break;
        }
    }

    // Clear reverse mapping
    fig_cells_[figure_idx] = {-32768, -32768};
}

void FigureHash2D::update(int figure_idx, const Vec2& center) {
    remove(figure_idx);
    insert(figure_idx, center);
}

size_t FigureHash2D::get_candidates_by_cell(int i, int j, std::vector<Index>& candidates) const {
    size_t out_idx = 0;

    // Check all 9 neighboring cells (3x3)
    for (const auto& [di, dj] : NEIGHBOR_DELTAS) {
        int ni = i + di;
        int nj = j + dj;

        FigureCellKey key(ni, nj);
        auto it = buckets_.find(key);
        if (it == buckets_.end()) continue;

        const auto& bucket = it->second;
        for (Index idx : bucket) {
            candidates[out_idx++] = idx;
        }
    }

    return out_idx;
}

size_t FigureHash2D::get_candidates(const Vec2& pos, std::vector<Index>& candidates) const {
    auto [i, j] = compute_ij(pos);
    return get_candidates_by_cell(i, j, candidates);
}

}  // namespace tree_packing
