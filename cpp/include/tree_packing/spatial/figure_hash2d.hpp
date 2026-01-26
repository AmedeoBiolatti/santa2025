#pragma once

#include "../core/types.hpp"
#include "grid2d.hpp"  // For NEIGHBOR_DELTAS
#include <vector>
#include <cstdint>
#include <unordered_map>

namespace tree_packing {

// Cell key for hashing (i, j coordinates packed into single int)
struct FigureCellKey {
    int32_t key;

    FigureCellKey() : key(0) {}
    FigureCellKey(int i, int j) : key((i << 16) | (j & 0xFFFF)) {}

    bool operator==(const FigureCellKey& other) const { return key == other.key; }

    int i() const { return key >> 16; }
    int j() const { return static_cast<int16_t>(key & 0xFFFF); }
};

struct FigureCellKeyHash {
    size_t operator()(const FigureCellKey& k) const {
        return std::hash<int32_t>{}(k.key);
    }
};

// 2D spatial hash for figure-level queries
// Each figure is stored in exactly ONE bucket (based on center position)
// Only non-empty buckets exist - efficient for sparse distributions
class FigureHash2D {
public:
    static constexpr float DEFAULT_CELL_SIZE = 1.0f;  // Same as THR for figure grid

    FigureHash2D() = default;

    // Create empty hash for given number of figures
    static FigureHash2D empty(size_t num_figures, float cell_size = DEFAULT_CELL_SIZE);

    // Initialize hash with figure centers
    static FigureHash2D init(
        const std::vector<Vec2>& centers,
        const std::vector<char>& valid,
        float cell_size = DEFAULT_CELL_SIZE
    );

    // Add a figure to the hash
    void insert(int figure_idx, const Vec2& center);

    // Remove a figure from the hash
    void remove(int figure_idx);

    // Update a figure (remove old, add new)
    void update(int figure_idx, const Vec2& center);

    // Get candidate figures from neighboring buckets around query cell
    // Returns count of valid entries written to candidates
    size_t get_candidates_by_cell(int i, int j, std::vector<Index>& candidates) const;

    // Get candidates by position
    size_t get_candidates(const Vec2& pos, std::vector<Index>& candidates) const;

    // Get the cell for a figure (for incremental updates)
    std::pair<int, int> get_item_cell(int figure_idx) const {
        if (static_cast<size_t>(figure_idx) >= fig_cells_.size()) {
            return {-32768, -32768};
        }
        return fig_cells_[figure_idx];
    }

    // Compute cell indices for a position
    std::pair<int, int> compute_ij(const Vec2& pos) const;

    // Hash parameters
    [[nodiscard]] float cell_size() const { return cell_size_; }
    [[nodiscard]] size_t num_figures() const { return fig_cells_.size(); }
    [[nodiscard]] size_t num_buckets() const { return buckets_.size(); }
    [[nodiscard]] int capacity() const { return static_cast<int>(fig_cells_.size()); }

private:
    float cell_size_{DEFAULT_CELL_SIZE};

    // Hash map: cell key -> list of figure indices in that cell
    std::unordered_map<FigureCellKey, std::vector<Index>, FigureCellKeyHash> buckets_;

    // Reverse mapping: fig_cells_[figure_idx] = (i, j) cell coords
    std::vector<std::pair<int16_t, int16_t>> fig_cells_;
};

}  // namespace tree_packing
