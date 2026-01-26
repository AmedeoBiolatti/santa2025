#include "tree_packing/spatial/triangle_hash2d.hpp"
#include <algorithm>
#include <cmath>

namespace tree_packing {

TriangleHash2D TriangleHash2D::empty(size_t num_figures, float cell_size) {
    TriangleHash2D hash;
    hash.cell_size_ = cell_size;

    // Initialize reverse mapping with invalid cell coords
    hash.tri_cells_.resize(num_figures);
    for (auto& fig_tris : hash.tri_cells_) {
        for (auto& cell : fig_tris) {
            cell = {-32768, -32768};  // Invalid marker
        }
    }

    // Reserve expected number of buckets (rough estimate)
    hash.buckets_.reserve(num_figures * 2);

    return hash;
}

TriangleHash2D TriangleHash2D::init(
    const std::vector<std::array<AABB, TREE_NUM_TRIANGLES>>& triangle_aabbs,
    const std::vector<char>& valid,
    float cell_size
) {
    TriangleHash2D hash = empty(triangle_aabbs.size(), cell_size);

    for (size_t fig = 0; fig < triangle_aabbs.size(); ++fig) {
        if (!valid[fig]) continue;
        hash.add_figure(static_cast<int>(fig), triangle_aabbs[fig]);
    }

    return hash;
}

std::pair<int, int> TriangleHash2D::compute_ij(const Vec2& pos) const {
    if (pos.is_nan()) {
        return {0, 0};
    }

    int i = static_cast<int>(std::floor(pos.x / cell_size_));
    int j = static_cast<int>(std::floor(pos.y / cell_size_));

    return {i, j};
}

void TriangleHash2D::add_triangle(int figure_idx, int triangle_idx, const AABB& tri_aabb) {
    if (tri_aabb.min.x != tri_aabb.min.x) return;  // NaN check

    // Compute center of AABB
    Vec2 center{
        (tri_aabb.min.x + tri_aabb.max.x) * 0.5f,
        (tri_aabb.min.y + tri_aabb.max.y) * 0.5f
    };

    auto [i, j] = compute_ij(center);

    TriangleHashId tid{static_cast<int16_t>(figure_idx), static_cast<int8_t>(triangle_idx), 0};

    // Add to bucket (creates bucket if doesn't exist)
    CellKey key(i, j);
    auto& bucket = buckets_[key];
    if (bucket.empty()) {
        bucket.reserve(DEFAULT_BUCKET_RESERVE);
    }
    bucket.push_back(tid);

    // Store reverse mapping
    if (static_cast<size_t>(figure_idx) >= tri_cells_.size()) {
        tri_cells_.resize(static_cast<size_t>(figure_idx) + 1);
        for (auto& cell : tri_cells_.back()) {
            cell = {-32768, -32768};
        }
    }
    tri_cells_[figure_idx][triangle_idx] = {static_cast<int16_t>(i), static_cast<int16_t>(j)};
}

void TriangleHash2D::remove_triangle(int figure_idx, int triangle_idx) {
    if (static_cast<size_t>(figure_idx) >= tri_cells_.size()) return;

    auto [i, j] = tri_cells_[figure_idx][triangle_idx];
    if (i == -32768 && j == -32768) return;  // Not in hash

    CellKey key(i, j);
    auto it = buckets_.find(key);
    if (it == buckets_.end()) return;

    TriangleHashId tid{static_cast<int16_t>(figure_idx), static_cast<int8_t>(triangle_idx), 0};

    // Remove from bucket (swap with last for O(1))
    auto& bucket = it->second;
    for (size_t s = 0; s < bucket.size(); ++s) {
        if (bucket[s] == tid) {
            bucket[s] = bucket.back();
            bucket.pop_back();
            break;
        }
    }

    // Clear reverse mapping
    tri_cells_[figure_idx][triangle_idx] = {-32768, -32768};
}

void TriangleHash2D::add_figure(int figure_idx, const std::array<AABB, TREE_NUM_TRIANGLES>& tri_aabbs) {
    for (size_t t = 0; t < TREE_NUM_TRIANGLES; ++t) {
        add_triangle(figure_idx, static_cast<int>(t), tri_aabbs[t]);
    }
}

void TriangleHash2D::remove_figure(int figure_idx) {
    for (size_t t = 0; t < TREE_NUM_TRIANGLES; ++t) {
        remove_triangle(figure_idx, static_cast<int>(t));
    }
}

void TriangleHash2D::update_figure(int figure_idx, const std::array<AABB, TREE_NUM_TRIANGLES>& tri_aabbs) {
    remove_figure(figure_idx);
    add_figure(figure_idx, tri_aabbs);
}

size_t TriangleHash2D::get_candidates_by_cell(int i, int j, std::vector<TriangleHashId>& candidates) const {
    size_t out_idx = 0;

    // Check all 9 neighboring cells (3x3)
    for (const auto& [di, dj] : NEIGHBOR_DELTAS) {
        int ni = i + di;
        int nj = j + dj;

        CellKey key(ni, nj);
        auto it = buckets_.find(key);
        if (it == buckets_.end()) continue;

        const auto& bucket = it->second;
        for (const auto& tid : bucket) {
            candidates[out_idx++] = tid;
        }
    }

    return out_idx;
}

size_t TriangleHash2D::get_candidates(const AABB& query_aabb, std::vector<TriangleHashId>& candidates) const {
    // Compute center of query AABB
    Vec2 center{
        (query_aabb.min.x + query_aabb.max.x) * 0.5f,
        (query_aabb.min.y + query_aabb.max.y) * 0.5f
    };

    auto [i, j] = compute_ij(center);
    return get_candidates_by_cell(i, j, candidates);
}

size_t TriangleHash2D::get_candidates_for_triangle(
    int figure_idx,
    int triangle_idx,
    const AABB& tri_aabb,
    std::vector<TriangleHashId>& candidates
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
