#include "tree_packing/constraints/intersection_tree_filter.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace tree_packing {
namespace {

constexpr int kGroupCount = 4;
constexpr int kTargetCount = kGroupCount * kGroupCount;
constexpr int kMaxPairsPerTarget = 4;
constexpr int kTargetPairCounts[kTargetCount] = {
    1, 1, 1, 2, 1, 1, 1, 2,
    1, 1, 1, 2, 2, 2, 2, 4,
};
constexpr int kTargetPairs[kTargetCount][kMaxPairsPerTarget][2] = {
    { {0, 0}, {-1, -1}, {-1, -1}, {-1, -1} },
    { {0, 1}, {-1, -1}, {-1, -1}, {-1, -1} },
    { {0, 2}, {-1, -1}, {-1, -1}, {-1, -1} },
    { {0, 3}, {0, 4}, {-1, -1}, {-1, -1} },
    { {1, 0}, {-1, -1}, {-1, -1}, {-1, -1} },
    { {1, 1}, {-1, -1}, {-1, -1}, {-1, -1} },
    { {1, 2}, {-1, -1}, {-1, -1}, {-1, -1} },
    { {1, 3}, {1, 4}, {-1, -1}, {-1, -1} },
    { {2, 0}, {-1, -1}, {-1, -1}, {-1, -1} },
    { {2, 1}, {-1, -1}, {-1, -1}, {-1, -1} },
    { {2, 2}, {-1, -1}, {-1, -1}, {-1, -1} },
    { {2, 3}, {2, 4}, {-1, -1}, {-1, -1} },
    { {3, 0}, {4, 0}, {-1, -1}, {-1, -1} },
    { {3, 1}, {4, 1}, {-1, -1}, {-1, -1} },
    { {3, 2}, {4, 2}, {-1, -1}, {-1, -1} },
    { {3, 3}, {3, 4}, {4, 3}, {4, 4} },
};

// Default path for the data file (relative to working directory)
const char* kDefaultDataPath = "data/intersection_tree_filter.txt";

// Search paths for the data file
const char* kSearchPaths[] = {
    "data/intersection_tree_filter.txt",
    "../data/intersection_tree_filter.txt",
    "../../data/intersection_tree_filter.txt",
    "../../../data/intersection_tree_filter.txt",
    "cpp/data/intersection_tree_filter.txt",
    "../cpp/data/intersection_tree_filter.txt",
};

// Fast angle normalization using fmod (avoids loops)
inline float normalize_angle_0_2pi_fast(float angle) {
    angle = std::fmod(angle, TWO_PI);
    if (angle < 0.0f) angle += TWO_PI;
    return angle;
}

inline float normalize_angle_0_2pi(float angle) {
    while (angle < 0.0f) angle += TWO_PI;
    while (angle >= TWO_PI) angle -= TWO_PI;
    return angle;
}

std::string find_data_file() {
    for (const auto* path : kSearchPaths) {
        std::ifstream f(path);
        if (f.good()) {
            return path;
        }
    }
    return "";
}

}  // namespace

IntersectionTreeFilter::IntersectionTreeFilter() {
    std::string path = find_data_file();
    if (!path.empty()) {
        load_from_file(path);
    }
}

IntersectionTreeFilter::IntersectionTreeFilter(const std::string& data_file_path) {
    if (!load_from_file(data_file_path)) {
        throw std::runtime_error("Failed to load intersection tree filter from: " + data_file_path);
    }
}

IntersectionTreeFilter::IntersectionTreeFilter(int depth, int num_features, int num_targets)
    : depth_(depth), num_features_(num_features), num_targets_(num_targets) {}

bool IntersectionTreeFilter::load_from_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return false;
    }

    std::string line;
    int depth = 0, num_features = 0, num_targets = 0;

    // Skip comments and empty lines until we find the header
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        if (iss >> depth >> num_features >> num_targets) {
            break;
        }
    }

    if (depth <= 0 || num_features <= 0 || num_targets <= 0) {
        return false;
    }

    const int num_internal = (1 << depth) - 1;
    const int num_leaves = (1 << depth);
    const int w_size = num_internal * num_features;
    const int b_size = num_internal;
    const int pred_size = num_leaves * num_targets;

    std::vector<float> w, b;
    std::vector<uint8_t> leaf_pred;
    w.reserve(w_size);
    b.reserve(b_size);
    leaf_pred.reserve(pred_size);

    // Read all values, skipping comments
    float val;
    int ival;
    while (file >> std::ws) {
        if (file.peek() == '#') {
            std::getline(file, line);
            continue;
        }

        // Read W values
        if (static_cast<int>(w.size()) < w_size) {
            if (file >> val) {
                w.push_back(val);
            }
        }
        // Read B values
        else if (static_cast<int>(b.size()) < b_size) {
            if (file >> val) {
                b.push_back(val);
            }
        }
        // Read leaf prediction values
        else if (static_cast<int>(leaf_pred.size()) < pred_size) {
            if (file >> ival) {
                leaf_pred.push_back(static_cast<uint8_t>(ival));
            }
        } else {
            break;
        }
    }

    // Validate sizes
    if (static_cast<int>(w.size()) != w_size ||
        static_cast<int>(b.size()) != b_size ||
        static_cast<int>(leaf_pred.size()) != pred_size) {
        return false;
    }

    // Set the model
    depth_ = depth;
    num_features_ = num_features;
    num_targets_ = num_targets;
    w_ = std::move(w);
    b_ = std::move(b);
    leaf_pred_ = std::move(leaf_pred);

    // Precompute pairs for fast lookup
    precompute_leaf_pairs();

    return true;
}

void IntersectionTreeFilter::set_model(
    int depth,
    int num_features,
    int num_targets,
    std::vector<float> w,
    std::vector<float> b,
    std::vector<uint8_t> leaf_pred
) {
    depth_ = depth;
    num_features_ = num_features;
    num_targets_ = num_targets;
    w_ = std::move(w);
    b_ = std::move(b);
    leaf_pred_ = std::move(leaf_pred);

    // Precompute pairs for fast lookup
    precompute_leaf_pairs();
}

void IntersectionTreeFilter::precompute_leaf_pairs() {
    // Build the "all pairs" vector (all 25 triangle pairs)
    all_pairs_.clear();
    for (int t = 0; t < kTargetCount; ++t) {
        const int count = kTargetPairCounts[t];
        for (int k = 0; k < count; ++k) {
            all_pairs_.emplace_back(kTargetPairs[t][k][0], kTargetPairs[t][k][1]);
        }
    }

    const int num_leaves = (depth_ <= 0) ? 1 : (1 << depth_);

    // Precompute pairs for each leaf
    leaf_pairs_.resize(num_leaves);
    leaf_is_all_ones_.resize(num_leaves);

    for (int leaf = 0; leaf < num_leaves; ++leaf) {
        const int base = leaf * num_targets_;
        auto& pairs = leaf_pairs_[leaf];
        pairs.clear();

        bool all_ones = true;
        for (int t = 0; t < kTargetCount; ++t) {
            if (leaf_pred_[base + t]) {
                const int count = kTargetPairCounts[t];
                for (int k = 0; k < count; ++k) {
                    pairs.emplace_back(kTargetPairs[t][k][0], kTargetPairs[t][k][1]);
                }
            } else {
                all_ones = false;
            }
        }
        leaf_is_all_ones_[leaf] = all_ones;
    }

    // Cache ready state
    const int num_internal = (depth_ <= 0) ? 0 : (1 << depth_) - 1;
    ready_ = (num_features_ > 0 && num_targets_ > 0 &&
              static_cast<int>(w_.size()) == num_internal * num_features_ &&
              static_cast<int>(b_.size()) == num_internal &&
              static_cast<int>(leaf_pred_.size()) == num_leaves * num_targets_);
}

bool IntersectionTreeFilter::ready() const {
    return ready_;
}

bool IntersectionTreeFilter::compute_leaf_index(
    const Solution& solution,
    size_t idx_a,
    size_t idx_b,
    int* out_leaf_idx
) const {
    if (!out_leaf_idx) {
        return false;
    }
    if (idx_a >= solution.size() || idx_b >= solution.size()) {
        return false;
    }
    if (!solution.is_valid(idx_a) || !solution.is_valid(idx_b)) {
        return false;
    }
    if (!ready() || num_targets_ != kTargetCount) {
        return false;
    }

    const TreeParams a = solution.get_params(idx_a);
    const TreeParams b = solution.get_params(idx_b);

    Vec2 rel = rotate_point(b.pos - a.pos, -a.angle);
    float ang = normalize_angle_0_2pi(b.angle - a.angle);

    if (rel.x < 0.0f) {
        rel.x = -rel.x;
        ang = normalize_angle_0_2pi(PI - ang);
    }

    const float x = rel.x;
    const float y = rel.y;
    const float r = std::sqrt(x * x + y * y);
    const float c = std::cos(ang);
    const float s = std::sin(ang);

    // Feature vector: [x, y, r, cos(a), sin(a), x*cos, y*cos, x*sin, y*sin, r*cos]
    float features[10] = {x, y, r, c, s, x * c, y * c, x * s, y * s, r * c};

    const int num_internal = (1 << depth_) - 1;
    int node = 0;

    while (node < num_internal) {
        const float* row = &w_[node * num_features_];
        float sum = b_[node];
        for (int f = 0; f < num_features_; ++f) {
            sum += row[f] * features[f];
        }
        node = (sum <= 0.0f) ? (2 * node + 1) : (2 * node + 2);
    }

    *out_leaf_idx = node - num_internal;
    return true;
}

int IntersectionTreeFilter::leaf_index_for(
    const Solution& solution,
    size_t idx_a,
    size_t idx_b
) const {
    int leaf_idx = -1;
    if (!compute_leaf_index(solution, idx_a, idx_b, &leaf_idx)) {
        return -1;
    }
    return leaf_idx;
}

bool IntersectionTreeFilter::leaf_pred_for(
    const Solution& solution,
    size_t idx_a,
    size_t idx_b,
    std::array<uint8_t, 16>& out_pred,
    int* out_leaf_idx
) const {
    int leaf_idx = -1;
    if (!compute_leaf_index(solution, idx_a, idx_b, &leaf_idx)) {
        return false;
    }

    if (out_leaf_idx) {
        *out_leaf_idx = leaf_idx;
    }

    const int num_leaves = (1 << depth_);
    if (leaf_idx < 0 || leaf_idx >= num_leaves) {
        return false;
    }

    const int base = leaf_idx * num_targets_;
    for (int t = 0; t < 16; ++t) {
        out_pred[t] = (t < num_targets_) ? leaf_pred_[base + t] : 1;
    }
    return true;
}

bool IntersectionTreeFilter::features_for(
    const Solution& solution,
    size_t idx_a,
    size_t idx_b,
    std::array<float, 10>& out_feat
) const {
    if (idx_a >= solution.size() || idx_b >= solution.size()) {
        return false;
    }
    if (!solution.is_valid(idx_a) || !solution.is_valid(idx_b)) {
        return false;
    }

    const TreeParams a = solution.get_params(idx_a);
    const TreeParams b = solution.get_params(idx_b);

    Vec2 rel = rotate_point(b.pos - a.pos, -a.angle);
    float ang = normalize_angle_0_2pi(b.angle - a.angle);

    if (rel.x < 0.0f) {
        rel.x = -rel.x;
        ang = normalize_angle_0_2pi(PI - ang);
    }

    const float x = rel.x;
    const float y = rel.y;
    const float r = std::sqrt(x * x + y * y);
    const float c = std::cos(ang);
    const float s = std::sin(ang);

    out_feat = {x, y, r, c, s, x * c, y * c, x * s, y * s, r * c};
    return true;
}

int IntersectionTreeFilter::compute_leaf_index_fast(
    const TreeParams& a,
    const TreeParams& b
) const {
    // Compute relative position and angle
    const float dx = b.pos.x - a.pos.x;
    const float dy = b.pos.y - a.pos.y;
    const float cos_a = std::cos(-a.angle);
    const float sin_a = std::sin(-a.angle);

    float x = dx * cos_a - dy * sin_a;
    const float y = dx * sin_a + dy * cos_a;

    float ang = b.angle - a.angle;

    // Normalize: ensure x >= 0
    if (x < 0.0f) {
        x = -x;
        ang = PI - ang;
    }

    // Fast angle normalization
    ang = normalize_angle_0_2pi_fast(ang);

    const float r = std::sqrt(x * x + y * y);
    const float c = std::cos(ang);
    const float s = std::sin(ang);

    // Feature vector (unrolled for speed)
    const float f0 = x;
    const float f1 = y;
    const float f2 = r;
    const float f3 = c;
    const float f4 = s;
    const float f5 = x * c;
    const float f6 = y * c;
    const float f7 = x * s;
    const float f8 = y * s;
    const float f9 = r * c;

    // Tree traversal with unrolled dot product
    const int num_internal = (1 << depth_) - 1;
    int node = 0;

    while (node < num_internal) {
        const float* w = &w_[node * 10];  // num_features_ == 10
        const float sum = b_[node] +
            w[0] * f0 + w[1] * f1 + w[2] * f2 + w[3] * f3 + w[4] * f4 +
            w[5] * f5 + w[6] * f6 + w[7] * f7 + w[8] * f8 + w[9] * f9;
        node = (sum <= 0.0f) ? (2 * node + 1) : (2 * node + 2);
    }

    return node - num_internal;
}

const std::vector<std::pair<int, int>>* IntersectionTreeFilter::triangle_pairs_for(
    const Solution& solution,
    size_t idx_a,
    size_t idx_b
) const {
    // Fast path checks
    if (!ready_ || idx_a >= solution.size() || idx_b >= solution.size()) {
        return nullptr;
    }
    if (!solution.is_valid(idx_a) || !solution.is_valid(idx_b)) {
        return nullptr;
    }

    // Compute leaf index using fast path
    const TreeParams& a = solution.get_params(idx_a);
    const TreeParams& b = solution.get_params(idx_b);
    const int leaf_idx = compute_leaf_index_fast(a, b);

    // Return precomputed pairs (no runtime generation needed)
    return &leaf_pairs_[leaf_idx];
}

}  // namespace tree_packing
