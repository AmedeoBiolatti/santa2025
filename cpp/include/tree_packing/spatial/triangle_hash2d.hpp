#pragma once

#include "../core/types.hpp"
#include "../core/tree.hpp"
#include "grid2d.hpp"  // For NEIGHBOR_DELTAS
#include <array>
#include <vector>
#include <cstdint>
#include <unordered_map>

namespace tree_packing {

// Packed triangle identifier: figure index + triangle index (0-4)
struct TriangleHashId {
    int16_t figure_idx{-1};
    int8_t triangle_idx{-1};
    int8_t padding{0};

    bool operator==(const TriangleHashId& other) const {
        return figure_idx == other.figure_idx && triangle_idx == other.triangle_idx;
    }

    bool is_valid() const { return figure_idx >= 0 && triangle_idx >= 0; }
};

// Cell key for hashing (i, j coordinates packed into single int)
struct CellKey {
    int32_t key;

    CellKey() : key(0) {}
    CellKey(int i, int j) : key((i << 16) | (j & 0xFFFF)) {}

    bool operator==(const CellKey& other) const { return key == other.key; }

    int i() const { return key >> 16; }
    int j() const { return static_cast<int16_t>(key & 0xFFFF); }
};

struct CellKeyHash {
    size_t operator()(const CellKey& k) const {
        // Simple hash mixing
        return std::hash<int32_t>{}(k.key);
    }
};

// 2D spatial hash for triangle-level queries
// Each triangle is stored in exactly ONE bucket (based on AABB center)
// Only non-empty buckets exist - efficient for sparse distributions
class TriangleHash2D {
public:
    static constexpr float DEFAULT_CELL_SIZE = 0.5f;  // Larger cells OK with hash lookup
    static constexpr size_t DEFAULT_BUCKET_RESERVE = 16;

    TriangleHash2D() = default;

    // Create empty hash grid for given number of figures
    static TriangleHash2D empty(size_t num_figures, float cell_size = DEFAULT_CELL_SIZE);

    // Initialize hash grid with triangle AABBs
    static TriangleHash2D init(
        const std::vector<std::array<AABB, TREE_NUM_TRIANGLES>>& triangle_aabbs,
        const std::vector<char>& valid,
        float cell_size = DEFAULT_CELL_SIZE
    );

    // Add all triangles of a figure to the hash
    void add_figure(int figure_idx, const std::array<AABB, TREE_NUM_TRIANGLES>& tri_aabbs);

    // Remove all triangles of a figure from the hash
    void remove_figure(int figure_idx);

    // Update a figure (remove old, add new)
    void update_figure(int figure_idx, const std::array<AABB, TREE_NUM_TRIANGLES>& tri_aabbs);

    // Get candidate triangles from neighboring buckets around query cell
    // Returns count of valid entries written to candidates
    size_t get_candidates_by_cell(int i, int j, std::vector<TriangleHashId>& candidates) const;

    // Get candidates by AABB center position
    size_t get_candidates(const AABB& query_aabb, std::vector<TriangleHashId>& candidates) const;

    // Get candidate triangles for a specific figure's triangle
    size_t get_candidates_for_triangle(
        int figure_idx,
        int triangle_idx,
        const AABB& tri_aabb,
        std::vector<TriangleHashId>& candidates
    ) const;

    // Compute cell indices for a position
    std::pair<int, int> compute_ij(const Vec2& pos) const;

    // Grid parameters
    [[nodiscard]] float cell_size() const { return cell_size_; }
    [[nodiscard]] size_t num_figures() const { return tri_cells_.size(); }
    [[nodiscard]] size_t num_buckets() const { return buckets_.size(); }

private:
    float cell_size_{DEFAULT_CELL_SIZE};

    // Hash map: cell key -> list of triangles in that cell
    std::unordered_map<CellKey, std::vector<TriangleHashId>, CellKeyHash> buckets_;

    // Reverse mapping: tri_cells_[figure_idx][triangle_idx] = (i, j) cell coords
    // Stores the single cell each triangle is in (for O(1) removal)
    std::vector<std::array<std::pair<int16_t, int16_t>, TREE_NUM_TRIANGLES>> tri_cells_;

    // Add a single triangle to the hash (at its AABB center)
    void add_triangle(int figure_idx, int triangle_idx, const AABB& tri_aabb);

    // Remove a single triangle from the hash
    void remove_triangle(int figure_idx, int triangle_idx);
};

}  // namespace tree_packing
