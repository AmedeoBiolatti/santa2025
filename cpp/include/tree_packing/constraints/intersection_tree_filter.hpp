#pragma once

#include "../core/solution.hpp"
#include "../core/tree.hpp"
#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace tree_packing {

class IntersectionTreeFilter {
public:
    // Default constructor loads from default data file path
    IntersectionTreeFilter();

    // Constructor with explicit data file path
    explicit IntersectionTreeFilter(const std::string& data_file_path);

    IntersectionTreeFilter(int depth, int num_features, int num_targets);

    // Load model from a text file
    bool load_from_file(const std::string& path);

    void set_model(
        int depth,
        int num_features,
        int num_targets,
        std::vector<float> w,
        std::vector<float> b,
        std::vector<uint8_t> leaf_pred
    );

    [[nodiscard]] bool ready() const;

    [[nodiscard]] const std::vector<std::pair<int, int>>* triangle_pairs_for(
        const Solution& solution,
        size_t idx_a,
        size_t idx_b
    ) const;

    // Debug helpers: retrieve leaf index / predictions for a pair
    [[nodiscard]] int leaf_index_for(
        const Solution& solution,
        size_t idx_a,
        size_t idx_b
    ) const;

    [[nodiscard]] bool leaf_pred_for(
        const Solution& solution,
        size_t idx_a,
        size_t idx_b,
        std::array<uint8_t, 16>& out_pred,
        int* out_leaf_idx = nullptr
    ) const;

    // Debug helper: retrieve feature vector for a pair
    [[nodiscard]] bool features_for(
        const Solution& solution,
        size_t idx_a,
        size_t idx_b,
        std::array<float, 10>& out_feat
    ) const;

private:
    int depth_{0};
    int num_features_{0};
    int num_targets_{0};
    std::vector<float> w_;
    std::vector<float> b_;
    std::vector<uint8_t> leaf_pred_;

    // Precomputed pairs for each leaf (avoids runtime generation)
    std::vector<std::vector<std::pair<int, int>>> leaf_pairs_;
    // All 25 triangle pairs for "all ones" leaves
    std::vector<std::pair<int, int>> all_pairs_;
    // Flag per leaf: true if leaf predicts all pairs
    std::vector<bool> leaf_is_all_ones_;
    // Cached ready state
    bool ready_{false};

    void precompute_leaf_pairs();

    [[nodiscard]] bool compute_leaf_index(
        const Solution& solution,
        size_t idx_a,
        size_t idx_b,
        int* out_leaf_idx
    ) const;

    // Fast path: compute leaf index without validation (caller must ensure valid input)
    [[nodiscard]] int compute_leaf_index_fast(
        const TreeParams& a,
        const TreeParams& b
    ) const;
};

}  // namespace tree_packing
