/*
 * Intersection Filtering using Lipschitz Bounds and Decision Forests (Octree Method)
 *
 * BUILD: cd cpp && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
 * RUN:   ./build/intersection_filtering
 *
 * PROBLEM:
 *   For a tree at (x, y, angle) relative to a tree at origin, determine which of the
 *   25 triangle pairs (5x5) can possibly intersect. We use the signed intersection
 *   score g(x,y,angle) where g > 0 means intersection.
 *
 * APPROACH:
 *   1. Build Octree with adaptive refinement in (x, y, angle) space
 *   2. Use Lipschitz bounds to determine sign of g for each cell
 *   3. Train decision forest by traversing Octree + Forest together
 *   4. Prune OUTSIDE subtrees, use INSIDE bounds directly
 *
 * DOMAIN:
 *   x, y in [-2.0116, 2.0116] (grid neighborhood)
 *   angle in [0, 2*pi]
 */

#include "tree_packing/geometry/intersection_score.hpp"
#include "tree_packing/core/types.hpp"
#include "tree_packing/core/tree.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <queue>
#include <random>
#include <vector>

using namespace tree_packing;

// ============================================================================
// BASIC TYPES
// ============================================================================

struct Interval {
    float lo, hi;
    float center() const { return 0.5f * (lo + hi); }
    float width() const { return hi - lo; }
};

// Compute the range of cos(angle) over an angle interval [a_lo, a_hi]
// cos has max at 0, 2π, 4π, ... and min at π, 3π, ...
Interval cos_range(float a_lo, float a_hi) {
    float c_lo = std::cos(a_lo);
    float c_hi = std::cos(a_hi);
    float min_val = std::min(c_lo, c_hi);
    float max_val = std::max(c_lo, c_hi);

    // Check if interval contains a maximum (cos = 1 at 2πk)
    float first_max = std::ceil(a_lo / (2.0f * PI)) * 2.0f * PI;
    if (first_max <= a_hi) max_val = 1.0f;

    // Check if interval contains a minimum (cos = -1 at π + 2πk)
    float first_min = std::ceil((a_lo - PI) / (2.0f * PI)) * 2.0f * PI + PI;
    if (first_min <= a_hi) min_val = -1.0f;

    return {min_val, max_val};
}

// Compute the range of sin(angle) over an angle interval [a_lo, a_hi]
// sin has max at π/2, 5π/2, ... and min at 3π/2, 7π/2, ...
Interval sin_range(float a_lo, float a_hi) {
    float s_lo = std::sin(a_lo);
    float s_hi = std::sin(a_hi);
    float min_val = std::min(s_lo, s_hi);
    float max_val = std::max(s_lo, s_hi);

    // Check if interval contains a maximum (sin = 1 at π/2 + 2πk)
    float half_pi = PI / 2.0f;
    float first_max = std::ceil((a_lo - half_pi) / (2.0f * PI)) * 2.0f * PI + half_pi;
    if (first_max <= a_hi) max_val = 1.0f;

    // Check if interval contains a minimum (sin = -1 at 3π/2 + 2πk)
    float three_half_pi = 3.0f * PI / 2.0f;
    float first_min = std::ceil((a_lo - three_half_pi) / (2.0f * PI)) * 2.0f * PI + three_half_pi;
    if (first_min <= a_hi) min_val = -1.0f;

    return {min_val, max_val};
}

// Compute the range of r = sqrt(x^2 + y^2) over a box [x_lo, x_hi] × [y_lo, y_hi]
Interval r_range(float x_lo, float x_hi, float y_lo, float y_hi) {
    // r_max is always at one of the 4 corners (farthest from origin)
    float r00 = std::sqrt(x_lo * x_lo + y_lo * y_lo);
    float r01 = std::sqrt(x_lo * x_lo + y_hi * y_hi);
    float r10 = std::sqrt(x_hi * x_hi + y_lo * y_lo);
    float r11 = std::sqrt(x_hi * x_hi + y_hi * y_hi);
    float r_max = std::max({r00, r01, r10, r11});

    // r_min depends on whether the box contains or straddles the origin
    float r_min;
    bool x_contains_zero = (x_lo <= 0 && x_hi >= 0);
    bool y_contains_zero = (y_lo <= 0 && y_hi >= 0);

    if (x_contains_zero && y_contains_zero) {
        // Origin is inside the box
        r_min = 0.0f;
    } else if (x_contains_zero) {
        // Box straddles x-axis, closest point is on x-axis
        r_min = std::min(std::abs(y_lo), std::abs(y_hi));
    } else if (y_contains_zero) {
        // Box straddles y-axis, closest point is on y-axis
        r_min = std::min(std::abs(x_lo), std::abs(x_hi));
    } else {
        // Box is in one quadrant, closest point is a corner
        r_min = std::min({r00, r01, r10, r11});
    }

    return {r_min, r_max};
}

struct Box3D {
    Interval x, y, angle;
    float volume() const { return x.width() * y.width() * angle.width(); }

    // Derived feature ranges
    Interval cos_angle() const { return cos_range(angle.lo, angle.hi); }
    Interval sin_angle() const { return sin_range(angle.lo, angle.hi); }
    Interval radius() const { return r_range(x.lo, x.hi, y.lo, y.hi); }
};

struct ScoreBounds {
    std::array<float, NUM_SCORE_TARGETS> lower;
    std::array<float, NUM_SCORE_TARGETS> upper;
};

struct SamplePoint {
    float x, y, angle;
    std::array<float, NUM_SCORE_TARGETS> scores;
};

struct LipschitzConstants {
    std::array<float, NUM_SCORE_TARGETS> L_x;
    std::array<float, NUM_SCORE_TARGETS> L_y;
    std::array<float, NUM_SCORE_TARGETS> L_angle;
};

// ============================================================================
// LIPSCHITZ CONSTANTS
// ============================================================================

std::array<float, TREE_NUM_TRIANGLES> compute_triangle_max_vertex_dist() {
    std::array<float, TREE_NUM_TRIANGLES> result;
    for (size_t i = 0; i < TREE_NUM_TRIANGLES; ++i) {
        const auto& tri = TREE_SHAPE[i];
        float d0 = tri.v0.length();
        float d1 = tri.v1.length();
        float d2 = tri.v2.length();
        result[i] = std::max({d0, d1, d2});
    }
    return result;
}

LipschitzConstants compute_lipschitz_constants_analytical(const Box3D& domain) {
    LipschitzConstants L;
    float max_abs_x = std::max(std::abs(domain.x.lo), std::abs(domain.x.hi));
    float max_abs_y = std::max(std::abs(domain.y.lo), std::abs(domain.y.hi));
    float max_pos = std::sqrt(max_abs_x * max_abs_x + max_abs_y * max_abs_y);

    static const auto tri_dists = compute_triangle_max_vertex_dist();

    for (size_t i = 0; i < TREE_NUM_TRIANGLES; ++i) {
        for (size_t j = 0; j < TREE_NUM_TRIANGLES; ++j) {
            size_t t = i * TREE_NUM_TRIANGLES + j;
            float r_A = tri_dists[i];
            float r_B = tri_dists[j];
            float bound_sum = 2.0f * (r_A + r_B + max_pos);
            L.L_x[t] = bound_sum;
            L.L_y[t] = bound_sum;
            L.L_angle[t] = (r_A + max_pos) * bound_sum;
        }
    }
    return L;
}

// ============================================================================
// BOUND COMPUTATION
// ============================================================================

ScoreBounds compute_bounds_from_samples(const Box3D& box, const LipschitzConstants& L,
                                         const std::vector<SamplePoint>& samples) {
    ScoreBounds bounds;
    bounds.upper.fill(std::numeric_limits<float>::max());
    bounds.lower.fill(std::numeric_limits<float>::lowest());

    for (const auto& s : samples) {
        float max_dx = std::max(std::abs(s.x - box.x.lo), std::abs(s.x - box.x.hi));
        float max_dy = std::max(std::abs(s.y - box.y.lo), std::abs(s.y - box.y.hi));
        float max_da = std::max(std::abs(s.angle - box.angle.lo), std::abs(s.angle - box.angle.hi));

        for (size_t t = 0; t < NUM_SCORE_TARGETS; ++t) {
            float margin = L.L_x[t] * max_dx + L.L_y[t] * max_dy + L.L_angle[t] * max_da;
            bounds.upper[t] = std::min(bounds.upper[t], s.scores[t] + margin);
            bounds.lower[t] = std::max(bounds.lower[t], s.scores[t] - margin);
        }
    }
    return bounds;
}

ScoreBounds compute_bounds_opposing_corners(const Box3D& box, const LipschitzConstants& L,
                                             const std::vector<SamplePoint>& samples) {
    ScoreBounds bounds;
    bounds.upper.fill(std::numeric_limits<float>::max());
    bounds.lower.fill(std::numeric_limits<float>::lowest());

    if (samples.size() < 9) return bounds;
    size_t corner_start = samples.size() - 9;
    const SamplePoint* corners = &samples[corner_start];

    for (size_t t = 0; t < NUM_SCORE_TARGETS; ++t) {
        float W = L.L_x[t] * box.x.width() + L.L_y[t] * box.y.width() + L.L_angle[t] * box.angle.width();
        for (int i = 0; i < 4; ++i) {
            float g1 = corners[i].scores[t];
            float g2 = corners[7 - i].scores[t];
            float sum = g1 + g2;
            bounds.upper[t] = std::min(bounds.upper[t], (sum + W) * 0.5f);
            bounds.lower[t] = std::max(bounds.lower[t], (sum - W) * 0.5f);
        }
    }
    return bounds;
}

struct DiagonalPoint {
    float x, y, angle;
    std::array<float, NUM_SCORE_TARGETS> scores;
};

std::pair<float, float> compute_segment_bounds_recursive(
    size_t t, const DiagonalPoint& p1, const DiagonalPoint& p2,
    float W_segment, int depth, int& num_evals) {
    float g1 = p1.scores[t], g2 = p2.scores[t];
    float sum = g1 + g2;
    float upper = (sum + W_segment) * 0.5f;
    float lower = (sum - W_segment) * 0.5f;

    if ((lower > 0) || (upper < 0) || depth <= 0) return {upper, lower};

    DiagonalPoint mid;
    mid.x = (p1.x + p2.x) * 0.5f;
    mid.y = (p1.y + p2.y) * 0.5f;
    mid.angle = (p1.angle + p2.angle) * 0.5f;
    compute_intersection_scores(mid.x, mid.y, mid.angle, mid.scores);
    num_evals++;

    float W_half = W_segment * 0.5f;
    auto [upper1, lower1] = compute_segment_bounds_recursive(t, p1, mid, W_half, depth - 1, num_evals);
    auto [upper2, lower2] = compute_segment_bounds_recursive(t, mid, p2, W_half, depth - 1, num_evals);
    return {std::min(upper, std::max(upper1, upper2)), std::max(lower, std::min(lower1, lower2))};
}

ScoreBounds compute_bounds_diagonal_recursive(const Box3D& box, const LipschitzConstants& L,
                                               const std::vector<SamplePoint>& samples,
                                               int max_depth, int& total_evals,
                                               const ScoreBounds* existing_bounds = nullptr) {
    ScoreBounds bounds;
    bounds.upper.fill(std::numeric_limits<float>::max());
    bounds.lower.fill(std::numeric_limits<float>::lowest());

    if (samples.size() < 9) return bounds;
    size_t corner_start = samples.size() - 9;
    const SamplePoint* corners = &samples[corner_start];

    std::array<DiagonalPoint, 8> diag_corners;
    for (int i = 0; i < 8; ++i) {
        diag_corners[i].x = corners[i].x;
        diag_corners[i].y = corners[i].y;
        diag_corners[i].angle = corners[i].angle;
        diag_corners[i].scores = corners[i].scores;
    }

    total_evals = 0;
    for (size_t t = 0; t < NUM_SCORE_TARGETS; ++t) {
        if (existing_bounds && ((existing_bounds->lower[t] > 0) || (existing_bounds->upper[t] < 0))) {
            bounds.upper[t] = existing_bounds->upper[t];
            bounds.lower[t] = existing_bounds->lower[t];
            continue;
        }
        float W = L.L_x[t] * box.x.width() + L.L_y[t] * box.y.width() + L.L_angle[t] * box.angle.width();
        for (int i = 0; i < 4; ++i) {
            auto [upper_i, lower_i] = compute_segment_bounds_recursive(
                t, diag_corners[i], diag_corners[7 - i], W, max_depth, total_evals);
            bounds.upper[t] = std::min(bounds.upper[t], upper_i);
            bounds.lower[t] = std::max(bounds.lower[t], lower_i);
            if ((bounds.lower[t] > 0) || (bounds.upper[t] < 0)) break;
        }
    }
    return bounds;
}

void combine_bounds(ScoreBounds& dst, const ScoreBounds& src) {
    for (size_t t = 0; t < NUM_SCORE_TARGETS; ++t) {
        dst.upper[t] = std::min(dst.upper[t], src.upper[t]);
        dst.lower[t] = std::max(dst.lower[t], src.lower[t]);
    }
}

// ============================================================================
// SAMPLING
// ============================================================================

std::vector<SamplePoint> generate_samples(const Box3D& box, int num_random_samples, unsigned seed = 123) {
    std::vector<SamplePoint> samples;
    samples.reserve(num_random_samples + 9);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist_x(box.x.lo, box.x.hi);
    std::uniform_real_distribution<float> dist_y(box.y.lo, box.y.hi);
    std::uniform_real_distribution<float> dist_a(box.angle.lo, box.angle.hi);

    for (int i = 0; i < num_random_samples; ++i) {
        SamplePoint s;
        s.x = dist_x(rng); s.y = dist_y(rng); s.angle = dist_a(rng);
        compute_intersection_scores(s.x, s.y, s.angle, s.scores);
        samples.push_back(s);
    }

    // 8 corners
    for (int ix = 0; ix < 2; ++ix) {
        for (int iy = 0; iy < 2; ++iy) {
            for (int ia = 0; ia < 2; ++ia) {
                SamplePoint s;
                s.x = (ix == 0) ? box.x.lo : box.x.hi;
                s.y = (iy == 0) ? box.y.lo : box.y.hi;
                s.angle = (ia == 0) ? box.angle.lo : box.angle.hi;
                compute_intersection_scores(s.x, s.y, s.angle, s.scores);
                samples.push_back(s);
            }
        }
    }

    // Center
    SamplePoint center;
    center.x = box.x.center(); center.y = box.y.center(); center.angle = box.angle.center();
    compute_intersection_scores(center.x, center.y, center.angle, center.scores);
    samples.push_back(center);

    return samples;
}

// ============================================================================
// CELL (used for Octree nodes and evaluation)
// ============================================================================

struct Cell {
    Box3D box;
    LipschitzConstants L;
    ScoreBounds bounds;
    std::vector<SamplePoint> samples;
    std::array<bool, NUM_SCORE_TARGETS> solved;
    int num_uncertain;

    float volume() const { return box.volume(); }

    void update_solved() {
        num_uncertain = 0;
        for (size_t t = 0; t < NUM_SCORE_TARGETS; ++t) {
            solved[t] = (bounds.lower[t] > 0) || (bounds.upper[t] < 0);
            if (!solved[t]) num_uncertain++;
        }
    }
};

uint64_t compute_cell_signature(const Cell& cell) {
    uint64_t sig = 0;
    for (size_t t = 0; t < NUM_SCORE_TARGETS; ++t) {
        int status = 0;
        if (cell.bounds.upper[t] < 0) status = 1;
        else if (cell.bounds.lower[t] > 0) status = 2;
        sig |= (static_cast<uint64_t>(status) << (2 * t));
    }
    return sig;
}

// ============================================================================
// OCTREE
// ============================================================================

// Forward declarations
class DecisionForest;
bool bounds_are_valid(const ScoreBounds& bounds);
uint64_t bounds_to_signature(const ScoreBounds& bounds);
enum class CellLeafRelation;
CellLeafRelation classify_cell_leaf(const DecisionForest& forest, int tree_idx,
                                     int target_leaf_id, const Box3D& box);

struct OctreeNode {
    Box3D box;
    LipschitzConstants L;
    ScoreBounds bounds;
    std::vector<SamplePoint> samples;
    std::array<bool, NUM_SCORE_TARGETS> solved;
    int num_uncertain;
    uint64_t signature;
    int parent_idx;
    int children_start;  // -1 if leaf
    int depth;
    bool deleted;

    bool is_leaf() const { return children_start < 0 && !deleted; }
    float volume() const { return box.volume(); }

    void compute_signature() {
        signature = 0;
        num_uncertain = 0;
        for (size_t t = 0; t < NUM_SCORE_TARGETS; ++t) {
            int status = 0;
            if (bounds.upper[t] < 0) status = 1;
            else if (bounds.lower[t] > 0) status = 2;
            else num_uncertain++;
            solved[t] = (status != 0);
            signature |= (static_cast<uint64_t>(status) << (2 * t));
        }
    }
};

class Octree {
public:
    std::vector<OctreeNode> nodes;
    Box3D domain;
    int samples_per_cell;

    Octree(const Box3D& dom, int samples = 50) : domain(dom), samples_per_cell(samples) {
        OctreeNode root;
        root.box = domain;
        root.L = compute_lipschitz_constants_analytical(domain);
        root.samples = generate_samples(domain, samples, 0);
        root.bounds = compute_bounds_from_samples(domain, root.L, root.samples);
        ScoreBounds corner_bounds = compute_bounds_opposing_corners(domain, root.L, root.samples);
        combine_bounds(root.bounds, corner_bounds);
        root.compute_signature();
        root.parent_idx = -1;
        root.children_start = -1;
        root.depth = 0;
        root.deleted = false;
        nodes.push_back(root);
    }

    void split_node(int node_idx) {
        if (!nodes[node_idx].is_leaf()) return;

        int children_start = static_cast<int>(nodes.size());
        nodes[node_idx].children_start = children_start;

        // Cache values from parent before push_back (to avoid reference invalidation)
        Box3D parent_box = nodes[node_idx].box;
        int parent_depth = nodes[node_idx].depth;

        float mid_x = parent_box.x.center();
        float mid_y = parent_box.y.center();
        float mid_a = parent_box.angle.center();
        unsigned seed = static_cast<unsigned>(node_idx * 8);

        for (int ix = 0; ix < 2; ++ix) {
            for (int iy = 0; iy < 2; ++iy) {
                for (int ia = 0; ia < 2; ++ia) {
                    OctreeNode child;
                    child.box.x.lo = (ix == 0) ? parent_box.x.lo : mid_x;
                    child.box.x.hi = (ix == 0) ? mid_x : parent_box.x.hi;
                    child.box.y.lo = (iy == 0) ? parent_box.y.lo : mid_y;
                    child.box.y.hi = (iy == 0) ? mid_y : parent_box.y.hi;
                    child.box.angle.lo = (ia == 0) ? parent_box.angle.lo : mid_a;
                    child.box.angle.hi = (ia == 0) ? mid_a : parent_box.angle.hi;

                    child.L = compute_lipschitz_constants_analytical(child.box);
                    child.samples = generate_samples(child.box, samples_per_cell, seed++);
                    child.bounds = compute_bounds_from_samples(child.box, child.L, child.samples);
                    ScoreBounds corner_bounds = compute_bounds_opposing_corners(child.box, child.L, child.samples);
                    combine_bounds(child.bounds, corner_bounds);
                    child.compute_signature();
                    child.parent_idx = node_idx;
                    child.children_start = -1;
                    child.depth = parent_depth + 1;
                    child.deleted = false;
                    nodes.push_back(child);
                }
            }
        }
    }

    void refine(int target_leaves, int max_depth = 10) {
        // Reserve space to avoid reallocation issues
        nodes.reserve(target_leaves * 2);

        auto cmp = [this](int a, int b) {
            float score_a = nodes[a].volume() * nodes[a].num_uncertain;
            float score_b = nodes[b].volume() * nodes[b].num_uncertain;
            return score_a < score_b;
        };
        std::priority_queue<int, std::vector<int>, decltype(cmp)> queue(cmp);

        if (nodes[0].num_uncertain > 0) queue.push(0);
        int num_leaves = 1;

        while (!queue.empty() && num_leaves < target_leaves) {
            int idx = queue.top();
            queue.pop();

            // Check BEFORE split (don't keep reference across push_back)
            if (!nodes[idx].is_leaf() || nodes[idx].num_uncertain == 0 || nodes[idx].depth >= max_depth) continue;

            split_node(idx);
            num_leaves += 7;

            // Access children_start AFTER split (node reference would be invalid)
            int children_start = nodes[idx].children_start;
            for (int c = 0; c < 8; ++c) {
                int child_idx = children_start + c;
                if (nodes[child_idx].num_uncertain > 0 && nodes[child_idx].depth < max_depth) {
                    queue.push(child_idx);
                }
            }
        }
    }

    void apply_diagonal_refinement(int max_depth = 4) {
        for (auto& node : nodes) {
            if (!node.is_leaf() || node.num_uncertain == 0) continue;
            int total_evals = 0;
            ScoreBounds diag_bounds = compute_bounds_diagonal_recursive(
                node.box, node.L, node.samples, max_depth, total_evals, &node.bounds);
            combine_bounds(node.bounds, diag_bounds);
            node.compute_signature();
        }
    }

    // Compute how much feature bounds degrade when merging children into parent
    // Returns max ratio of (parent_width / avg_child_width) for derived features
    float compute_feature_degradation(int parent_idx) const {
        const OctreeNode& parent = nodes[parent_idx];
        if (parent.is_leaf()) return 1.0f;

        // Compute parent's derived feature ranges
        Interval parent_cos = parent.box.cos_angle();
        Interval parent_sin = parent.box.sin_angle();
        Interval parent_r = parent.box.radius();

        // Compute average child ranges
        float avg_cos_width = 0, avg_sin_width = 0, avg_r_width = 0;
        int valid_children = 0;
        for (int c = 0; c < 8; ++c) {
            const OctreeNode& child = nodes[parent.children_start + c];
            if (child.deleted || !child.is_leaf()) continue;
            avg_cos_width += child.box.cos_angle().width();
            avg_sin_width += child.box.sin_angle().width();
            avg_r_width += child.box.radius().width();
            valid_children++;
        }

        if (valid_children == 0) return 1.0f;
        avg_cos_width /= valid_children;
        avg_sin_width /= valid_children;
        avg_r_width /= valid_children;

        // Compute degradation ratios (how much wider parent is vs avg child)
        float cos_ratio = (avg_cos_width > 1e-6f) ? parent_cos.width() / avg_cos_width : 1.0f;
        float sin_ratio = (avg_sin_width > 1e-6f) ? parent_sin.width() / avg_sin_width : 1.0f;
        float r_ratio = (avg_r_width > 1e-6f) ? parent_r.width() / avg_r_width : 1.0f;

        return std::max({cos_ratio, sin_ratio, r_ratio});
    }

    // Merge siblings with same signature, optionally limiting feature degradation
    // max_degradation_ratio: skip merge if feature bounds would get this much wider
    void merge_siblings(float max_degradation_ratio = std::numeric_limits<float>::infinity()) {
        int max_depth = 0;
        for (const auto& node : nodes) max_depth = std::max(max_depth, node.depth);

        int merges_done = 0, merges_skipped = 0;

        for (int d = max_depth - 1; d >= 0; --d) {
            for (size_t i = 0; i < nodes.size(); ++i) {
                OctreeNode& node = nodes[i];
                if (node.depth != d || node.is_leaf()) continue;

                bool all_same = true;
                uint64_t first_sig = nodes[node.children_start].signature;
                for (int c = 0; c < 8; ++c) {
                    int child_idx = node.children_start + c;
                    if (!nodes[child_idx].is_leaf() || nodes[child_idx].signature != first_sig) {
                        all_same = false;
                        break;
                    }
                }

                if (!all_same) continue;

                // Check feature degradation
                float degradation = compute_feature_degradation(static_cast<int>(i));
                if (degradation > max_degradation_ratio) {
                    merges_skipped++;
                    continue;  // Skip this merge to preserve feature accuracy
                }

                // Proceed with merge
                ScoreBounds merged_bounds;
                merged_bounds.upper.fill(std::numeric_limits<float>::lowest());
                merged_bounds.lower.fill(std::numeric_limits<float>::max());

                for (int c = 0; c < 8; ++c) {
                    const OctreeNode& child = nodes[node.children_start + c];
                    for (size_t t = 0; t < NUM_SCORE_TARGETS; ++t) {
                        merged_bounds.upper[t] = std::max(merged_bounds.upper[t], child.bounds.upper[t]);
                        merged_bounds.lower[t] = std::min(merged_bounds.lower[t], child.bounds.lower[t]);
                    }
                    nodes[node.children_start + c].deleted = true;
                }

                node.signature = first_sig;
                node.bounds = merged_bounds;
                node.num_uncertain = 0;
                for (size_t t = 0; t < NUM_SCORE_TARGETS; ++t) {
                    node.solved[t] = (merged_bounds.upper[t] < 0) || (merged_bounds.lower[t] > 0);
                    if (!node.solved[t]) node.num_uncertain++;
                }
                node.children_start = -1;
                merges_done++;
            }
        }

        if (merges_skipped > 0) {
            std::printf("  Merges: %d done, %d skipped (degradation > %.1fx)\n",
                       merges_done, merges_skipped, max_degradation_ratio);
        }
    }

    std::vector<Cell> to_cells() const {
        std::vector<Cell> cells;
        for (const auto& node : nodes) {
            if (!node.is_leaf() || node.deleted) continue;
            Cell c;
            c.box = node.box;
            c.L = node.L;
            c.bounds = node.bounds;
            c.samples = node.samples;
            c.solved = node.solved;
            c.num_uncertain = node.num_uncertain;
            cells.push_back(c);
        }
        return cells;
    }

    void print_stats() const {
        int num_leaves = 0, num_internal = 0, num_deleted = 0;
        int total_uncertain = 0;
        float total_volume = 0;
        int max_depth = 0;

        for (const auto& node : nodes) {
            if (node.deleted) { num_deleted++; continue; }
            if (node.is_leaf()) {
                num_leaves++;
                total_uncertain += node.num_uncertain;
                total_volume += node.box.volume();
                max_depth = std::max(max_depth, node.depth);
            } else {
                num_internal++;
            }
        }

        std::printf("Octree: %d leaves, %d internal, %d deleted, max_depth=%d\n",
                   num_leaves, num_internal, num_deleted, max_depth);
        std::printf("  Volume: %.4f, Uncertain: %d / %d\n",
                   total_volume, total_uncertain, num_leaves * static_cast<int>(NUM_SCORE_TARGETS));
    }
};

// ============================================================================
// DECISION FOREST (5 features: x, y, cos(angle), sin(angle), r)
// ============================================================================

// Split function: coef_x*x + coef_y*y + coef_cos*cos(a) + coef_sin*sin(a) + coef_r*r
// Compute min/max of split function over a box
Interval compute_split_range(float coef_x, float coef_y, float coef_cos, float coef_sin, float coef_r,
                              const Box3D& box) {
    // Linear terms: x, y have simple interval arithmetic
    float x_contrib_lo = (coef_x >= 0) ? coef_x * box.x.lo : coef_x * box.x.hi;
    float x_contrib_hi = (coef_x >= 0) ? coef_x * box.x.hi : coef_x * box.x.lo;
    float y_contrib_lo = (coef_y >= 0) ? coef_y * box.y.lo : coef_y * box.y.hi;
    float y_contrib_hi = (coef_y >= 0) ? coef_y * box.y.hi : coef_y * box.y.lo;

    // Derived features: use precomputed ranges
    Interval cos_iv = box.cos_angle();
    Interval sin_iv = box.sin_angle();
    Interval r_iv = box.radius();

    float cos_contrib_lo = (coef_cos >= 0) ? coef_cos * cos_iv.lo : coef_cos * cos_iv.hi;
    float cos_contrib_hi = (coef_cos >= 0) ? coef_cos * cos_iv.hi : coef_cos * cos_iv.lo;
    float sin_contrib_lo = (coef_sin >= 0) ? coef_sin * sin_iv.lo : coef_sin * sin_iv.hi;
    float sin_contrib_hi = (coef_sin >= 0) ? coef_sin * sin_iv.hi : coef_sin * sin_iv.lo;
    float r_contrib_lo = (coef_r >= 0) ? coef_r * r_iv.lo : coef_r * r_iv.hi;
    float r_contrib_hi = (coef_r >= 0) ? coef_r * r_iv.hi : coef_r * r_iv.lo;

    return {
        x_contrib_lo + y_contrib_lo + cos_contrib_lo + sin_contrib_lo + r_contrib_lo,
        x_contrib_hi + y_contrib_hi + cos_contrib_hi + sin_contrib_hi + r_contrib_hi
    };
}

class DecisionForest {
public:
    std::vector<bool> is_leaf;
    std::vector<float> coef_x, coef_y, coef_cos, coef_sin, coef_r, threshold;
    std::vector<int> left_child, right_child;
    std::vector<uint64_t> signature;
    std::vector<int> leaf_id;
    std::vector<int> tree_offsets, tree_num_leaves;
    Box3D domain;
    int num_trees, num_nodes;

    DecisionForest() : num_trees(0), num_nodes(0) {}

    void initialize_random(const Box3D& dom, int n_trees, int max_depth, unsigned seed = 42) {
        domain = dom;
        num_trees = n_trees;
        num_nodes = 0;
        is_leaf.clear(); coef_x.clear(); coef_y.clear();
        coef_cos.clear(); coef_sin.clear(); coef_r.clear();
        threshold.clear(); left_child.clear(); right_child.clear();
        signature.clear(); leaf_id.clear();
        tree_offsets.clear(); tree_num_leaves.clear();
        tree_offsets.reserve(n_trees + 1);
        tree_num_leaves.reserve(n_trees);

        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> coef_dist(-1.0f, 1.0f);

        for (int t = 0; t < n_trees; ++t) {
            tree_offsets.push_back(num_nodes);
            int leaves_before = static_cast<int>(std::count(is_leaf.begin(), is_leaf.end(), true));
            rng.seed(seed + t * 1000);
            int leaf_counter = 0;
            build_random_node(0, dom, max_depth, rng, coef_dist, leaf_counter);
            int leaves_after = static_cast<int>(std::count(is_leaf.begin(), is_leaf.end(), true));
            tree_num_leaves.push_back(leaves_after - leaves_before);
        }
        tree_offsets.push_back(num_nodes);
    }

    // Evaluate split function at a point
    float eval_split(int idx, float x, float y, float angle) const {
        float c = std::cos(angle), s = std::sin(angle);
        float r = std::sqrt(x * x + y * y);
        return coef_x[idx] * x + coef_y[idx] * y +
               coef_cos[idx] * c + coef_sin[idx] * s + coef_r[idx] * r;
    }

    int get_leaf_index(int tree_idx, float x, float y, float angle) const {
        int idx = tree_offsets[tree_idx];
        while (!is_leaf[idx]) {
            idx = (eval_split(idx, x, y, angle) < threshold[idx]) ? left_child[idx] : right_child[idx];
        }
        return leaf_id[idx];
    }

    uint64_t get_signature(int tree_idx, float x, float y, float angle) const {
        int idx = tree_offsets[tree_idx];
        while (!is_leaf[idx]) {
            idx = (eval_split(idx, x, y, angle) < threshold[idx]) ? left_child[idx] : right_child[idx];
        }
        return signature[idx];
    }

    // Get split function range over a box
    Interval get_split_range(int idx, const Box3D& box) const {
        return compute_split_range(coef_x[idx], coef_y[idx], coef_cos[idx], coef_sin[idx], coef_r[idx], box);
    }

    void set_node_signature(int node_idx, uint64_t sig) { signature[node_idx] = sig; }

    std::vector<int> get_tree_leaf_indices(int tree_idx) const {
        std::vector<int> leaves;
        int start = tree_offsets[tree_idx], end = tree_offsets[tree_idx + 1];
        for (int i = start; i < end; ++i) {
            if (is_leaf[i]) leaves.push_back(i);
        }
        return leaves;
    }

    int total_leaves() const {
        int total = 0;
        for (int n : tree_num_leaves) total += n;
        return total;
    }

private:
    int alloc_node() {
        int idx = num_nodes++;
        is_leaf.push_back(true);
        coef_x.push_back(0); coef_y.push_back(0);
        coef_cos.push_back(0); coef_sin.push_back(0); coef_r.push_back(0);
        threshold.push_back(0);
        left_child.push_back(-1); right_child.push_back(-1);
        signature.push_back(0); leaf_id.push_back(-1);
        return idx;
    }

    int build_random_node(int depth, const Box3D& box, int max_depth,
                          std::mt19937& rng, std::uniform_real_distribution<float>& coef_dist,
                          int& leaf_counter) {
        int idx = alloc_node();
        if (depth >= max_depth) {
            is_leaf[idx] = true;
            leaf_id[idx] = leaf_counter++;
            return idx;
        }

        is_leaf[idx] = false;

        // Random coefficients for all 5 features
        coef_x[idx] = coef_dist(rng);
        coef_y[idx] = coef_dist(rng);
        coef_cos[idx] = coef_dist(rng);
        coef_sin[idx] = coef_dist(rng);
        coef_r[idx] = coef_dist(rng) * 0.5f;  // Scale down r since it has different magnitude

        // Normalize
        float norm = std::sqrt(coef_x[idx]*coef_x[idx] + coef_y[idx]*coef_y[idx] +
                              coef_cos[idx]*coef_cos[idx] + coef_sin[idx]*coef_sin[idx] +
                              coef_r[idx]*coef_r[idx]);
        if (norm > 1e-8f) {
            coef_x[idx] /= norm; coef_y[idx] /= norm;
            coef_cos[idx] /= norm; coef_sin[idx] /= norm; coef_r[idx] /= norm;
        }

        // Use split range to pick threshold in the middle
        Interval split_range = get_split_range(idx, box);
        threshold[idx] = 0.5f * (split_range.lo + split_range.hi);

        left_child[idx] = build_random_node(depth + 1, box, max_depth, rng, coef_dist, leaf_counter);
        right_child[idx] = build_random_node(depth + 1, box, max_depth, rng, coef_dist, leaf_counter);
        return idx;
    }
};

// ============================================================================
// OPTIMIZED DECISION FOREST (axis-aligned splits, greedy TN maximization)
// Features: y, cos(angle), r  (removed x and sin - they contribute <1% TN gain)
// ============================================================================

// Feature extraction for a cell (3 features: y, cos, r)
struct CellFeatures {
    float y, cos_a, r;

    static CellFeatures from_box(const Box3D& box) {
        CellFeatures f;
        f.y = box.y.center();
        float a = box.angle.center();
        f.cos_a = std::cos(a);
        float x = box.x.center();
        f.r = std::sqrt(x * x + f.y * f.y);
        return f;
    }

    float get(int feature_idx) const {
        switch (feature_idx) {
            case 0: return y;
            case 1: return cos_a;
            case 2: return r;
            default: return 0;
        }
    }
};

// Get feature interval for a box
Interval get_feature_interval(const Box3D& box, int feature_idx) {
    switch (feature_idx) {
        case 0: return box.y;
        case 1: return box.cos_angle();
        case 2: return box.radius();
        default: return {0, 0};
    }
}

class OptimizedForest {
public:
    // Split structure: axis-aligned or oblique
    std::vector<bool> is_leaf;
    std::vector<bool> is_oblique;    // true = oblique split, false = axis-aligned
    std::vector<int> split_feature;  // for axis-aligned: 0=y, 1=cos, 2=r
    std::vector<float> threshold;
    // Oblique split coefficients: coef_y*y + coef_cos*cos + coef_r*r < threshold
    std::vector<float> coef_y, coef_cos, coef_r;
    std::vector<int> left_child, right_child;
    std::vector<uint64_t> signature;
    std::vector<int> tree_offsets;
    int num_trees, num_nodes;

    // Feature importance tracking (3 features: y, cos, r)
    std::array<int, 3> feature_split_count{};      // How many times each feature was used
    std::array<int, 3> feature_tn_gain{};          // Total TN gain from splits on each feature
    std::array<int, 3> feature_cells_affected{};   // Total cells affected by splits on each feature
    int oblique_split_count{0};  // Track oblique splits added during refinement

    static constexpr int NUM_FEATURES = 3;
    static constexpr const char* FEATURE_NAMES[3] = {"y", "cos", "r"};

    OptimizedForest() : num_trees(0), num_nodes(0) {
        feature_split_count.fill(0);
        feature_tn_gain.fill(0);
        feature_cells_affected.fill(0);
    }

    // Train forest on cells using greedy TN maximization
    void train(const std::vector<Cell>& cells, int n_trees, int max_depth, int num_threshold_candidates = 20) {
        num_trees = n_trees;
        num_nodes = 0;
        is_leaf.clear(); is_oblique.clear(); split_feature.clear(); threshold.clear();
        coef_y.clear(); coef_cos.clear(); coef_r.clear();
        left_child.clear(); right_child.clear(); signature.clear();
        tree_offsets.clear();
        tree_offsets.reserve(n_trees + 1);

        // Reset feature importance
        feature_split_count.fill(0);
        feature_tn_gain.fill(0);
        feature_cells_affected.fill(0);
        oblique_split_count = 0;

        // Precompute cell features and signatures
        std::vector<CellFeatures> features(cells.size());
        std::vector<uint64_t> cell_sigs(cells.size());
        for (size_t i = 0; i < cells.size(); ++i) {
            features[i] = CellFeatures::from_box(cells[i].box);
            cell_sigs[i] = compute_cell_signature(cells[i]);
        }

        // Build trees with different random subsets for diversity
        std::mt19937 rng(42);
        for (int t = 0; t < n_trees; ++t) {
            tree_offsets.push_back(num_nodes);

            // Use all cells but shuffle threshold candidates for diversity
            std::vector<int> cell_indices(cells.size());
            std::iota(cell_indices.begin(), cell_indices.end(), 0);

            build_optimized_node(cells, features, cell_sigs, cell_indices,
                                0, max_depth, num_threshold_candidates, rng);
        }
        tree_offsets.push_back(num_nodes);
    }

    // Evaluate oblique split at a point: coef_y*y + coef_cos*cos + coef_r*r
    float eval_oblique_split(int idx, const CellFeatures& f) const {
        return coef_y[idx] * f.y + coef_cos[idx] * f.cos_a + coef_r[idx] * f.r;
    }

    // Get signature for a point
    uint64_t get_signature(int tree_idx, float x, float y, float angle) const {
        CellFeatures f;
        f.y = y;
        f.cos_a = std::cos(angle);
        f.r = std::sqrt(x * x + y * y);

        int idx = tree_offsets[tree_idx];
        while (!is_leaf[idx]) {
            float val;
            if (is_oblique[idx]) {
                val = eval_oblique_split(idx, f);
            } else {
                val = f.get(split_feature[idx]);
            }
            idx = (val < threshold[idx]) ? left_child[idx] : right_child[idx];
        }
        return signature[idx];
    }

    void print_tree_structure(int tree_idx, int max_nodes = 15) const {
        std::printf("Tree %d structure:\n", tree_idx);
        int start = tree_offsets[tree_idx];
        int end = tree_offsets[tree_idx + 1];
        int printed = 0;
        for (int i = start; i < end && printed < max_nodes; ++i, ++printed) {
            if (is_leaf[i]) {
                int tn_count = 0;
                for (size_t t = 0; t < NUM_SCORE_TARGETS; ++t) {
                    if (((signature[i] >> (2*t)) & 0x3) == 1) tn_count++;
                }
                std::printf("  [%d] LEAF sig=%lu (TN=%d targets)\n", i - start, signature[i], tn_count);
            } else if (is_oblique[i]) {
                std::printf("  [%d] OBLIQUE %.2f*y + %.2f*cos + %.2f*r < %.3f -> L:%d R:%d\n",
                           i - start, coef_y[i], coef_cos[i], coef_r[i], threshold[i],
                           left_child[i] - start, right_child[i] - start);
            } else {
                std::printf("  [%d] SPLIT %s < %.3f -> L:%d R:%d\n",
                           i - start, FEATURE_NAMES[split_feature[i]], threshold[i],
                           left_child[i] - start, right_child[i] - start);
            }
        }
        if (end - start > max_nodes) std::printf("  ... (%d more nodes)\n", end - start - max_nodes);
    }

private:
    int alloc_node() {
        int idx = num_nodes++;
        is_leaf.push_back(true);
        is_oblique.push_back(false);
        split_feature.push_back(-1);
        threshold.push_back(0);
        coef_y.push_back(0);
        coef_cos.push_back(0);
        coef_r.push_back(0);
        left_child.push_back(-1);
        right_child.push_back(-1);
        signature.push_back(0);
        return idx;
    }

    // Compute combined signature for a set of cells (conservative: only mark negative if ALL cells agree)
    uint64_t compute_combined_signature(const std::vector<uint64_t>& cell_sigs,
                                        const std::vector<int>& indices) const {
        if (indices.empty()) return 0;

        uint64_t combined = 0;
        for (size_t t = 0; t < NUM_SCORE_TARGETS; ++t) {
            bool all_negative = true;  // All cells have upper < 0
            bool all_positive = true;  // All cells have lower > 0

            for (int idx : indices) {
                int status = (cell_sigs[idx] >> (2 * t)) & 0x3;
                if (status != 1) all_negative = false;  // 1 = definitely negative
                if (status != 2) all_positive = false;  // 2 = definitely positive
            }

            int combined_status = 0;
            if (all_negative) combined_status = 1;
            else if (all_positive) combined_status = 2;
            combined |= (static_cast<uint64_t>(combined_status) << (2 * t));
        }
        return combined;
    }

    // Count TN (true negatives) in a signature
    int count_tn(uint64_t sig, const std::vector<uint64_t>& cell_sigs,
                 const std::vector<int>& indices) const {
        int tn = 0;
        for (size_t t = 0; t < NUM_SCORE_TARGETS; ++t) {
            int predicted = (sig >> (2 * t)) & 0x3;
            if (predicted == 1) {  // Predicting negative
                // Count cells that are actually negative
                for (int idx : indices) {
                    int actual = (cell_sigs[idx] >> (2 * t)) & 0x3;
                    if (actual == 1) tn++;  // Actual negative, predicted negative
                }
            }
        }
        return tn;
    }

    // Evaluate a split: returns (left_indices, right_indices, total_tn_gain)
    std::tuple<std::vector<int>, std::vector<int>, int> evaluate_split(
        const std::vector<CellFeatures>& features,
        const std::vector<uint64_t>& cell_sigs,
        const std::vector<int>& indices,
        int feature_idx, float thresh,
        int current_tn) const {

        std::vector<int> left_indices, right_indices;
        for (int idx : indices) {
            if (features[idx].get(feature_idx) < thresh) {
                left_indices.push_back(idx);
            } else {
                right_indices.push_back(idx);
            }
        }

        if (left_indices.empty() || right_indices.empty()) {
            return {left_indices, right_indices, -1};  // Invalid split
        }

        uint64_t left_sig = compute_combined_signature(cell_sigs, left_indices);
        uint64_t right_sig = compute_combined_signature(cell_sigs, right_indices);

        int left_tn = count_tn(left_sig, cell_sigs, left_indices);
        int right_tn = count_tn(right_sig, cell_sigs, right_indices);
        int new_tn = left_tn + right_tn;

        return {left_indices, right_indices, new_tn - current_tn};
    }

    int build_optimized_node(const std::vector<Cell>& cells,
                             const std::vector<CellFeatures>& features,
                             const std::vector<uint64_t>& cell_sigs,
                             const std::vector<int>& indices,
                             int depth, int max_depth, int num_candidates,
                             std::mt19937& rng) {
        int idx = alloc_node();

        // Compute current signature
        uint64_t current_sig = compute_combined_signature(cell_sigs, indices);
        int current_tn = count_tn(current_sig, cell_sigs, indices);

        // Stop if max depth reached or too few cells
        if (depth >= max_depth || indices.size() < 4) {
            is_leaf[idx] = true;
            signature[idx] = current_sig;
            return idx;
        }

        // Find best split (axis-aligned OR oblique)
        int best_feature = -1;
        float best_threshold = 0;
        int best_gain = 0;
        std::vector<int> best_left, best_right;
        bool best_is_oblique = false;
        float best_cy = 0, best_cc = 0, best_cr = 0;

        // Try axis-aligned splits
        for (int f = 0; f < NUM_FEATURES; ++f) {
            std::vector<float> values;
            values.reserve(indices.size());
            for (int i : indices) {
                values.push_back(features[i].get(f));
            }
            std::sort(values.begin(), values.end());

            for (int q = 1; q < num_candidates; ++q) {
                size_t pos = (values.size() * q) / num_candidates;
                if (pos == 0 || pos >= values.size()) continue;
                float thresh = 0.5f * (values[pos - 1] + values[pos]);

                auto [left, right, gain] = evaluate_split(features, cell_sigs, indices, f, thresh, current_tn);
                if (gain > best_gain) {
                    best_gain = gain;
                    best_feature = f;
                    best_threshold = thresh;
                    best_left = std::move(left);
                    best_right = std::move(right);
                    best_is_oblique = false;
                }
            }
        }

        // Oblique splits disabled for this run (axis-aligned only)

        // If no improvement, make leaf
        if (best_gain <= 0) {
            is_leaf[idx] = true;
            signature[idx] = current_sig;
            return idx;
        }

        // Track feature importance (only for axis-aligned splits)
        if (!best_is_oblique) {
            feature_split_count[best_feature]++;
            feature_tn_gain[best_feature] += best_gain;
            feature_cells_affected[best_feature] += static_cast<int>(indices.size());
        } else {
            oblique_split_count++;
        }

        // Make split
        is_leaf[idx] = false;
        is_oblique[idx] = best_is_oblique;
        if (best_is_oblique) {
            coef_y[idx] = best_cy;
            coef_cos[idx] = best_cc;
            coef_r[idx] = best_cr;
        } else {
            split_feature[idx] = best_feature;
        }
        threshold[idx] = best_threshold;
        left_child[idx] = build_optimized_node(cells, features, cell_sigs, best_left,
                                               depth + 1, max_depth, num_candidates, rng);
        right_child[idx] = build_optimized_node(cells, features, cell_sigs, best_right,
                                               depth + 1, max_depth, num_candidates, rng);
        return idx;
    }

public:
    void print_feature_importance() const {
        // Compute total for normalization
        int total_splits = 0, total_tn_gain = 0, total_cells = 0;
        for (int i = 0; i < NUM_FEATURES; ++i) {
            total_splits += feature_split_count[i];
            total_tn_gain += feature_tn_gain[i];
            total_cells += feature_cells_affected[i];
        }

        std::printf("\nFeature Importance:\n");
        std::printf("%-8s %8s %8s %10s %10s %10s\n",
                   "Feature", "Splits", "Split%", "TN_Gain", "TN_Gain%", "Cells");
        std::printf("%s\n", std::string(58, '-').c_str());

        // Sort by TN gain (descending)
        std::array<int, NUM_FEATURES> order = {0, 1, 2};
        std::sort(order.begin(), order.end(), [this](int a, int b) {
            return feature_tn_gain[a] > feature_tn_gain[b];
        });

        for (int i : order) {
            float split_pct = (total_splits > 0) ? 100.0f * feature_split_count[i] / total_splits : 0;
            float tn_pct = (total_tn_gain > 0) ? 100.0f * feature_tn_gain[i] / total_tn_gain : 0;
            std::printf("%-8s %8d %7.1f%% %10d %9.1f%% %10d\n",
                       FEATURE_NAMES[i], feature_split_count[i], split_pct,
                       feature_tn_gain[i], tn_pct, feature_cells_affected[i]);
        }

        // Recommendation
        std::printf("\nRecommendation: ");
        std::vector<const char*> removable;
        for (int i : order) {
            if (feature_split_count[i] == 0) {
                removable.push_back(FEATURE_NAMES[i]);
            } else {
                float tn_pct = (total_tn_gain > 0) ? 100.0f * feature_tn_gain[i] / total_tn_gain : 0;
                if (tn_pct < 1.0f) {
                    removable.push_back(FEATURE_NAMES[i]);
                }
            }
        }
        if (removable.empty()) {
            std::printf("All features contribute significantly.\n");
        } else {
            std::printf("Consider removing: ");
            for (size_t i = 0; i < removable.size(); ++i) {
                if (i > 0) std::printf(", ");
                std::printf("%s", removable[i]);
            }
            std::printf(" (unused or <1%% TN gain)\n");
        }
    }

    // =========================================================================
    // POST-PROCESSING REFINEMENT
    // =========================================================================

    // Refinement 1: Fine-tune thresholds to maximize TN
    int refine_thresholds(const std::vector<Cell>& cells, int num_steps = 10) {
        std::vector<CellFeatures> features(cells.size());
        std::vector<uint64_t> cell_sigs(cells.size());
        for (size_t i = 0; i < cells.size(); ++i) {
            features[i] = CellFeatures::from_box(cells[i].box);
            cell_sigs[i] = compute_cell_signature(cells[i]);
        }

        int total_improvement = 0;

        for (int t = 0; t < num_trees; ++t) {
            int start = tree_offsets[t], end = tree_offsets[t + 1];
            for (int idx = start; idx < end; ++idx) {
                if (is_leaf[idx]) continue;

                int feat = split_feature[idx];
                float orig_thresh = threshold[idx];

                // Find cells that reach this node
                std::vector<int> node_cells;
                for (size_t c = 0; c < cells.size(); ++c) {
                    if (reaches_node(t, idx, features[c])) {
                        node_cells.push_back(static_cast<int>(c));
                    }
                }
                if (node_cells.size() < 4) continue;

                // Collect feature values at this node
                std::vector<float> values;
                for (int c : node_cells) values.push_back(features[c].get(feat));
                std::sort(values.begin(), values.end());

                // Try different thresholds
                int best_tn = compute_subtree_tn(idx, node_cells, features, cell_sigs);
                float best_thresh = orig_thresh;

                for (int s = 1; s < num_steps; ++s) {
                    size_t pos = (values.size() * s) / num_steps;
                    if (pos == 0 || pos >= values.size()) continue;
                    float new_thresh = 0.5f * (values[pos - 1] + values[pos]);

                    threshold[idx] = new_thresh;
                    int new_tn = compute_subtree_tn(idx, node_cells, features, cell_sigs);

                    if (new_tn > best_tn) {
                        best_tn = new_tn;
                        best_thresh = new_thresh;
                    }
                }

                if (best_thresh != orig_thresh) {
                    threshold[idx] = best_thresh;
                    total_improvement += best_tn - compute_subtree_tn(idx, node_cells, features, cell_sigs);
                } else {
                    threshold[idx] = orig_thresh;
                }
            }
        }
        return total_improvement;
    }

    // Refinement 2: Ensemble-aware signature relaxation
    // For each cell, if the ensemble already correctly predicts it, individual leaves can be more aggressive
    int refine_with_ensemble(const std::vector<Cell>& cells) {
        std::vector<CellFeatures> features(cells.size());
        std::vector<uint64_t> cell_sigs(cells.size());
        for (size_t i = 0; i < cells.size(); ++i) {
            features[i] = CellFeatures::from_box(cells[i].box);
            cell_sigs[i] = compute_cell_signature(cells[i]);
        }

        int improvements = 0;

        // For each leaf, check if we can safely predict more targets as negative
        for (int t = 0; t < num_trees; ++t) {
            int start = tree_offsets[t], end = tree_offsets[t + 1];
            for (int idx = start; idx < end; ++idx) {
                if (!is_leaf[idx]) continue;

                // Find cells in this leaf
                std::vector<int> leaf_cells;
                for (size_t c = 0; c < cells.size(); ++c) {
                    if (get_leaf_node(t, features[c]) == idx) {
                        leaf_cells.push_back(static_cast<int>(c));
                    }
                }
                if (leaf_cells.empty()) continue;

                // For each target, check if we can predict negative
                uint64_t new_sig = signature[idx];
                for (size_t target = 0; target < NUM_SCORE_TARGETS; ++target) {
                    int current_status = (signature[idx] >> (2 * target)) & 0x3;
                    if (current_status == 1) continue;  // Already predicting negative

                    // Check if ALL cells in this leaf are actually negative for this target
                    bool all_negative = true;
                    for (int c : leaf_cells) {
                        int cell_status = (cell_sigs[c] >> (2 * target)) & 0x3;
                        if (cell_status != 1) {  // Not definitely negative
                            all_negative = false;
                            break;
                        }
                    }

                    if (all_negative) {
                        // We can safely predict negative for this target
                        new_sig = (new_sig & ~(3ULL << (2 * target))) | (1ULL << (2 * target));
                        improvements++;
                    }
                }
                signature[idx] = new_sig;
            }
        }
        return improvements;
    }

    // Refinement 3: Re-split leaves with uncertain targets (SAFE version)
    // Only creates splits that maintain FN=0 guarantee
    int refine_split_leaves(const std::vector<Cell>& cells, int max_extra_depth = 2) {
        (void)max_extra_depth;  // Reserved for future use

        std::vector<CellFeatures> features(cells.size());
        std::vector<uint64_t> cell_sigs(cells.size());
        for (size_t i = 0; i < cells.size(); ++i) {
            features[i] = CellFeatures::from_box(cells[i].box);
            cell_sigs[i] = compute_cell_signature(cells[i]);
        }

        int total_new_tn = 0;

        // Collect leaves that could benefit from splitting
        std::vector<std::pair<int, int>> leaves_to_split;  // (tree_idx, node_idx)
        for (int t = 0; t < num_trees; ++t) {
            int start = tree_offsets[t], end = tree_offsets[t + 1];
            for (int idx = start; idx < end; ++idx) {
                if (!is_leaf[idx]) continue;

                // Count uncertain targets in this leaf
                int uncertain = 0;
                for (size_t target = 0; target < NUM_SCORE_TARGETS; ++target) {
                    if (((signature[idx] >> (2 * target)) & 0x3) == 0) uncertain++;
                }
                if (uncertain > 0) {
                    leaves_to_split.push_back({t, idx});
                }
            }
        }

        for (auto [tree_idx, leaf_idx] : leaves_to_split) {
            // Find cells in this leaf
            std::vector<int> leaf_cells;
            for (size_t c = 0; c < cells.size(); ++c) {
                if (get_leaf_node(tree_idx, features[c]) == leaf_idx) {
                    leaf_cells.push_back(static_cast<int>(c));
                }
            }
            if (leaf_cells.size() < 4) continue;

            // Try to find a good split that maintains safety
            int current_tn = count_tn(signature[leaf_idx], cell_sigs, leaf_cells);
            int best_gain = 0;
            int best_feature = -1;
            float best_threshold = 0;
            std::vector<int> best_left, best_right;
            uint64_t best_left_sig = 0, best_right_sig = 0;

            for (int f = 0; f < NUM_FEATURES; ++f) {
                std::vector<float> values;
                for (int c : leaf_cells) values.push_back(features[c].get(f));
                std::sort(values.begin(), values.end());

                for (int q = 1; q < 20; ++q) {
                    size_t pos = (values.size() * q) / 20;
                    if (pos == 0 || pos >= values.size()) continue;
                    float thresh = 0.5f * (values[pos - 1] + values[pos]);

                    // Split cells
                    std::vector<int> left_cells, right_cells;
                    for (int c : leaf_cells) {
                        if (features[c].get(f) < thresh) left_cells.push_back(c);
                        else right_cells.push_back(c);
                    }
                    if (left_cells.empty() || right_cells.empty()) continue;

                    // Compute conservative signatures
                    uint64_t left_sig = compute_combined_signature(cell_sigs, left_cells);
                    uint64_t right_sig = compute_combined_signature(cell_sigs, right_cells);

                    // SAFETY CHECK: new signatures must not predict negative for any target
                    // unless ALL cells in that partition are DEFINITELY negative (status=1)
                    // This is conservative: uncertain cells (status=0) could be positive!
                    bool safe = true;
                    for (size_t t = 0; t < NUM_SCORE_TARGETS && safe; ++t) {
                        int left_status = (left_sig >> (2 * t)) & 0x3;
                        int right_status = (right_sig >> (2 * t)) & 0x3;

                        // If predicting negative, verify ALL cells are definitely negative
                        if (left_status == 1) {
                            for (int c : left_cells) {
                                int cell_status = (cell_sigs[c] >> (2 * t)) & 0x3;
                                if (cell_status != 1) { safe = false; break; }  // Not definitely negative!
                            }
                        }
                        if (safe && right_status == 1) {
                            for (int c : right_cells) {
                                int cell_status = (cell_sigs[c] >> (2 * t)) & 0x3;
                                if (cell_status != 1) { safe = false; break; }  // Not definitely negative!
                            }
                        }
                    }

                    if (!safe) continue;

                    // Compute TN gain
                    int left_tn = count_tn(left_sig, cell_sigs, left_cells);
                    int right_tn = count_tn(right_sig, cell_sigs, right_cells);
                    int gain = left_tn + right_tn - current_tn;

                    if (gain > best_gain) {
                        best_gain = gain;
                        best_feature = f;
                        best_threshold = thresh;
                        best_left = std::move(left_cells);
                        best_right = std::move(right_cells);
                        best_left_sig = left_sig;
                        best_right_sig = right_sig;
                    }
                }
            }

            if (best_gain > 0) {
                // Convert leaf to split node
                is_leaf[leaf_idx] = false;
                split_feature[leaf_idx] = best_feature;
                threshold[leaf_idx] = best_threshold;

                // Create new leaf nodes
                int left_idx = alloc_node();
                int right_idx = alloc_node();
                left_child[leaf_idx] = left_idx;
                right_child[leaf_idx] = right_idx;

                is_leaf[left_idx] = true;
                is_leaf[right_idx] = true;
                signature[left_idx] = best_left_sig;
                signature[right_idx] = best_right_sig;

                total_new_tn += best_gain;

                // Track feature importance
                feature_split_count[best_feature]++;
                feature_tn_gain[best_feature] += best_gain;
                feature_cells_affected[best_feature] += static_cast<int>(leaf_cells.size());
            }
        }
        return total_new_tn;
    }

    // Safe refinement: Strengthen leaf signatures by checking if we can mark more targets as negative
    // Only modifies signatures, never changes tree structure
    int refine_strengthen_signatures(const std::vector<Cell>& cells, bool verbose = false) {
        std::vector<CellFeatures> features(cells.size());
        std::vector<uint64_t> cell_sigs(cells.size());
        for (size_t i = 0; i < cells.size(); ++i) {
            features[i] = CellFeatures::from_box(cells[i].box);
            cell_sigs[i] = compute_cell_signature(cells[i]);
        }

        int total_improvements = 0;

        for (int t = 0; t < num_trees; ++t) {
            int start = tree_offsets[t], end = tree_offsets[t + 1];
            for (int idx = start; idx < end; ++idx) {
                if (!is_leaf[idx]) continue;

                // Find cells in this leaf
                std::vector<int> leaf_cells;
                for (size_t c = 0; c < cells.size(); ++c) {
                    if (get_leaf_node(t, features[c]) == idx) {
                        leaf_cells.push_back(static_cast<int>(c));
                    }
                }
                if (leaf_cells.empty()) continue;

                // Recompute signature from cells actually in this leaf
                uint64_t new_sig = compute_combined_signature(cell_sigs, leaf_cells);

                // Count improvements (targets we can now predict vs before)
                for (size_t target = 0; target < NUM_SCORE_TARGETS; ++target) {
                    int old_status = (signature[idx] >> (2 * target)) & 0x3;
                    int new_status = (new_sig >> (2 * target)) & 0x3;
                    // Only count if we're strengthening (going from uncertain to certain)
                    if (old_status == 0 && new_status != 0) total_improvements++;
                }

                signature[idx] = new_sig;
            }
        }
        return total_improvements;
    }

    // Compute oblique split range over a cell's feature intervals
    // Split: coef_y*y + coef_cos*cos + coef_r*r
    Interval get_oblique_split_range(float cy, float cc, float cr, const Box3D& box) const {
        Interval y_iv = box.y;
        Interval cos_iv = box.cos_angle();
        Interval r_iv = box.radius();

        // Interval arithmetic for linear combination
        float lo = 0, hi = 0;

        // y contribution
        if (cy >= 0) { lo += cy * y_iv.lo; hi += cy * y_iv.hi; }
        else { lo += cy * y_iv.hi; hi += cy * y_iv.lo; }

        // cos contribution
        if (cc >= 0) { lo += cc * cos_iv.lo; hi += cc * cos_iv.hi; }
        else { lo += cc * cos_iv.hi; hi += cc * cos_iv.lo; }

        // r contribution
        if (cr >= 0) { lo += cr * r_iv.lo; hi += cr * r_iv.hi; }
        else { lo += cr * r_iv.hi; hi += cr * r_iv.lo; }

        return {lo, hi};
    }

    // Evaluate oblique split for a cell using INTERVAL ARITHMETIC (safe)
    // Returns: -1 = all left, +1 = all right, 0 = straddles
    int classify_cell_oblique(float cy, float cc, float cr, float thresh, const Box3D& box) const {
        Interval split_iv = get_oblique_split_range(cy, cc, cr, box);
        if (split_iv.hi < thresh) return -1;  // All points go left
        if (split_iv.lo >= thresh) return +1; // All points go right
        return 0;  // Straddles - uncertain
    }

    // Compute TN for an oblique split using interval-based cell routing (SAFE)
    int compute_oblique_split_tn_safe(
        float cy, float cc, float cr, float thresh,
        const std::vector<Cell>& cells,
        const std::vector<int>& cell_indices,
        const std::vector<uint64_t>& cell_sigs,
        uint64_t left_sig, uint64_t right_sig
    ) const {
        int tn = 0;
        for (int idx : cell_indices) {
            int side = classify_cell_oblique(cy, cc, cr, thresh, cells[idx].box);
            uint64_t sig = (side < 0) ? left_sig : (side > 0) ? right_sig : 0;
            // For straddling cells (side == 0), use uncertain signature (0 = no predictions)
            for (size_t t = 0; t < NUM_SCORE_TARGETS; ++t) {
                int predicted = (sig >> (2 * t)) & 0x3;
                int actual = (cell_sigs[idx] >> (2 * t)) & 0x3;
                if (predicted == 1 && actual == 1) tn++;  // Correct negative prediction
            }
        }
        return tn;
    }

    // Oblique fine-tuning: try to replace axis-aligned splits with oblique ones
    // Uses interval arithmetic for SAFE cell routing
    int refine_oblique_splits(const std::vector<Cell>& cells, int num_angles = 12, bool verbose = false) {
        std::vector<CellFeatures> features(cells.size());
        std::vector<uint64_t> cell_sigs(cells.size());
        for (size_t i = 0; i < cells.size(); ++i) {
            features[i] = CellFeatures::from_box(cells[i].box);
            cell_sigs[i] = compute_cell_signature(cells[i]);
        }

        int total_improvement = 0;
        int nodes_upgraded = 0;

        for (int t = 0; t < num_trees; ++t) {
            int start = tree_offsets[t], end = tree_offsets[t + 1];
            for (int idx = start; idx < end; ++idx) {
                if (is_leaf[idx]) continue;

                // Find cells that reach this node (using current tree structure)
                std::vector<int> node_cells;
                for (size_t c = 0; c < cells.size(); ++c) {
                    if (reaches_node(t, idx, features[c])) {
                        node_cells.push_back(static_cast<int>(c));
                    }
                }
                if (node_cells.size() < 8) continue;

                // Compute current TN with existing split (using center-based routing for baseline)
                int current_tn = compute_subtree_tn(idx, node_cells, features, cell_sigs);

                // Store original split configuration
                bool orig_oblique = is_oblique[idx];
                int orig_feature = split_feature[idx];
                float orig_thresh = threshold[idx];
                float orig_cy = coef_y[idx], orig_cc = coef_cos[idx], orig_cr = coef_r[idx];

                // Best found so far
                int best_tn = current_tn;
                float best_cy = 0, best_cc = 0, best_cr = 0, best_thresh = 0;
                bool found_better = false;

                // Try oblique splits: combinations of 2 features
                for (int pair = 0; pair < 3; ++pair) {
                    for (int ai = 0; ai < num_angles; ++ai) {
                        float angle = static_cast<float>(ai) * M_PI / num_angles;
                        float c1 = std::cos(angle);
                        float c2 = std::sin(angle);

                        float cy = 0, cc = 0, cr = 0;
                        switch (pair) {
                            case 0: cy = c1; cc = c2; break;
                            case 1: cy = c1; cr = c2; break;
                            case 2: cc = c1; cr = c2; break;
                        }

                        // Collect cell split intervals to find good threshold candidates
                        std::vector<float> boundaries;
                        boundaries.reserve(node_cells.size() * 2);
                        for (int c : node_cells) {
                            Interval iv = get_oblique_split_range(cy, cc, cr, cells[c].box);
                            boundaries.push_back(iv.lo);
                            boundaries.push_back(iv.hi);
                        }
                        std::sort(boundaries.begin(), boundaries.end());

                        // Sample up to 30 threshold candidates at quantile positions
                        constexpr int MAX_THRESH_CANDIDATES = 30;
                        int step = std::max(1, static_cast<int>(boundaries.size()) / MAX_THRESH_CANDIDATES);
                        for (size_t bi = step; bi < boundaries.size(); bi += step) {
                            if (boundaries[bi] - boundaries[bi-1] < 1e-6f) continue;
                            float thresh = 0.5f * (boundaries[bi-1] + boundaries[bi]);

                            // Partition cells using interval arithmetic (with early exit on straddlers)
                            std::vector<int> left_cells, right_cells;
                            left_cells.reserve(node_cells.size());
                            right_cells.reserve(node_cells.size());
                            bool has_straddlers = false;
                            for (int c : node_cells) {
                                int side = classify_cell_oblique(cy, cc, cr, thresh, cells[c].box);
                                if (side < 0) left_cells.push_back(c);
                                else if (side > 0) right_cells.push_back(c);
                                else { has_straddlers = true; break; }  // Early exit on straddler
                            }

                            // Skip if any cells straddle (unsafe) or empty partition
                            if (has_straddlers || left_cells.empty() || right_cells.empty()) continue;

                            // Compute conservative signatures for left and right
                            uint64_t left_sig = compute_combined_signature(cell_sigs, left_cells);
                            uint64_t right_sig = compute_combined_signature(cell_sigs, right_cells);

                            // Compute TN
                            int left_tn = count_tn(left_sig, cell_sigs, left_cells);
                            int right_tn = count_tn(right_sig, cell_sigs, right_cells);
                            int new_tn = left_tn + right_tn;

                            if (new_tn > best_tn) {
                                best_tn = new_tn;
                                best_cy = cy;
                                best_cc = cc;
                                best_cr = cr;
                                best_thresh = thresh;
                                found_better = true;
                            }
                        }
                    }
                }

                // Also try 3-way oblique (all features)
                for (int ri = 0; ri < num_angles; ++ri) {
                    float phi = static_cast<float>(ri) * 2.0f * M_PI / num_angles;
                    float theta = static_cast<float>(ri % 6 + 1) * M_PI / 12.0f;
                    float cy = std::sin(theta) * std::cos(phi);
                    float cc = std::sin(theta) * std::sin(phi);
                    float cr = std::cos(theta);

                    std::vector<float> boundaries;
                    boundaries.reserve(node_cells.size() * 2);
                    for (int c : node_cells) {
                        Interval iv = get_oblique_split_range(cy, cc, cr, cells[c].box);
                        boundaries.push_back(iv.lo);
                        boundaries.push_back(iv.hi);
                    }
                    std::sort(boundaries.begin(), boundaries.end());

                    // Sample up to 30 threshold candidates
                    int step3 = std::max(1, static_cast<int>(boundaries.size()) / 30);
                    for (size_t bi = step3; bi < boundaries.size(); bi += step3) {
                        if (boundaries[bi] - boundaries[bi-1] < 1e-6f) continue;
                        float thresh = 0.5f * (boundaries[bi-1] + boundaries[bi]);

                        std::vector<int> left_cells, right_cells;
                        left_cells.reserve(node_cells.size());
                        right_cells.reserve(node_cells.size());
                        bool has_straddlers = false;
                        for (int c : node_cells) {
                            int side = classify_cell_oblique(cy, cc, cr, thresh, cells[c].box);
                            if (side < 0) left_cells.push_back(c);
                            else if (side > 0) right_cells.push_back(c);
                            else { has_straddlers = true; break; }  // Early exit
                        }

                        if (has_straddlers || left_cells.empty() || right_cells.empty()) continue;

                        uint64_t left_sig = compute_combined_signature(cell_sigs, left_cells);
                        uint64_t right_sig = compute_combined_signature(cell_sigs, right_cells);
                        int new_tn = count_tn(left_sig, cell_sigs, left_cells) + count_tn(right_sig, cell_sigs, right_cells);

                        if (new_tn > best_tn) {
                            best_tn = new_tn;
                            best_cy = cy;
                            best_cc = cc;
                            best_cr = cr;
                            best_thresh = thresh;
                            found_better = true;
                        }
                    }
                }

                // Apply best configuration if found
                if (found_better) {
                    is_oblique[idx] = true;
                    coef_y[idx] = best_cy;
                    coef_cos[idx] = best_cc;
                    coef_r[idx] = best_cr;
                    threshold[idx] = best_thresh;
                    total_improvement += best_tn - current_tn;
                    if (!orig_oblique) {
                        nodes_upgraded++;
                        oblique_split_count++;
                    }
                } else {
                    // Restore original
                    is_oblique[idx] = orig_oblique;
                    split_feature[idx] = orig_feature;
                    threshold[idx] = orig_thresh;
                    coef_y[idx] = orig_cy;
                    coef_cos[idx] = orig_cc;
                    coef_r[idx] = orig_cr;
                }
            }
        }

        if (verbose) {
            std::printf("  Oblique refinement: %d nodes upgraded, +%d TN\n", nodes_upgraded, total_improvement);
        }
        return total_improvement;
    }

    // Run all refinements
    void refine_all(const std::vector<Cell>& cells, bool verbose = true) {
        if (verbose) std::printf("\n--- Post-processing refinement ---\n");

        // Pass 1: Strengthen signatures by recomputing from actual cell assignments
        int improvements = refine_strengthen_signatures(cells, verbose);
        if (verbose) std::printf("  Signature strengthening: +%d target predictions\n", improvements);

        // Oblique refinement disabled (provides no benefit with axis-aligned training)
        (void)oblique_split_count;
    }

private:
    bool reaches_node(int tree_idx, int target_node, const CellFeatures& f) const {
        int idx = tree_offsets[tree_idx];
        while (idx != target_node && !is_leaf[idx]) {
            float val;
            if (is_oblique[idx]) {
                val = coef_y[idx] * f.y + coef_cos[idx] * f.cos_a + coef_r[idx] * f.r;
            } else {
                val = f.get(split_feature[idx]);
            }
            idx = (val < threshold[idx]) ? left_child[idx] : right_child[idx];
        }
        return idx == target_node;
    }

    int get_leaf_node(int tree_idx, const CellFeatures& f) const {
        int idx = tree_offsets[tree_idx];
        while (!is_leaf[idx]) {
            float val;
            if (is_oblique[idx]) {
                val = coef_y[idx] * f.y + coef_cos[idx] * f.cos_a + coef_r[idx] * f.r;
            } else {
                val = f.get(split_feature[idx]);
            }
            idx = (val < threshold[idx]) ? left_child[idx] : right_child[idx];
        }
        return idx;
    }

    int compute_subtree_tn(int node_idx, const std::vector<int>& cells,
                           const std::vector<CellFeatures>& features,
                           const std::vector<uint64_t>& cell_sigs) const {
        if (is_leaf[node_idx]) {
            return count_tn(signature[node_idx], cell_sigs, cells);
        }

        std::vector<int> left_cells, right_cells;
        float thresh = threshold[node_idx];
        for (int c : cells) {
            float val;
            if (is_oblique[node_idx]) {
                val = coef_y[node_idx] * features[c].y + coef_cos[node_idx] * features[c].cos_a + coef_r[node_idx] * features[c].r;
            } else {
                val = features[c].get(split_feature[node_idx]);
            }
            if (val < thresh) left_cells.push_back(c);
            else right_cells.push_back(c);
        }

        return compute_subtree_tn(left_child[node_idx], left_cells, features, cell_sigs) +
               compute_subtree_tn(right_child[node_idx], right_cells, features, cell_sigs);
    }
};

constexpr const char* OptimizedForest::FEATURE_NAMES[3];

// ============================================================================
// OCTREE DYNAMIC DEPTH TRAVERSAL
// ============================================================================

enum class CellLeafRelation { INSIDE, BORDER, OUTSIDE };

// Interval relation with threshold
enum class IntervalThresholdRelation {
    ALL_LEFT,    // interval.hi < threshold (all values go left)
    ALL_RIGHT,   // interval.lo >= threshold (all values go right)
    SPLIT        // interval straddles threshold
};

IntervalThresholdRelation classify_interval_threshold(const Interval& iv, float threshold) {
    if (iv.hi < threshold) return IntervalThresholdRelation::ALL_LEFT;
    if (iv.lo >= threshold) return IntervalThresholdRelation::ALL_RIGHT;
    return IntervalThresholdRelation::SPLIT;
}

// Recursively classify cell-leaf relation using interval arithmetic
// Returns: does the cell overlap with the given leaf, and if so, is it fully inside?
CellLeafRelation classify_cell_leaf_recursive(const DecisionForest& forest, int node_idx,
                                               int target_leaf_id, const Box3D& box) {
    if (forest.is_leaf[node_idx]) {
        // At a leaf: check if it's the target
        return (forest.leaf_id[node_idx] == target_leaf_id)
            ? CellLeafRelation::INSIDE : CellLeafRelation::OUTSIDE;
    }

    // Internal node: compute split range and classify
    Interval split_range = forest.get_split_range(node_idx, box);
    auto rel = classify_interval_threshold(split_range, forest.threshold[node_idx]);

    if (rel == IntervalThresholdRelation::ALL_LEFT) {
        // Entire cell goes left
        return classify_cell_leaf_recursive(forest, forest.left_child[node_idx], target_leaf_id, box);
    } else if (rel == IntervalThresholdRelation::ALL_RIGHT) {
        // Entire cell goes right
        return classify_cell_leaf_recursive(forest, forest.right_child[node_idx], target_leaf_id, box);
    } else {
        // Cell is split: check both children
        auto left_rel = classify_cell_leaf_recursive(forest, forest.left_child[node_idx], target_leaf_id, box);
        auto right_rel = classify_cell_leaf_recursive(forest, forest.right_child[node_idx], target_leaf_id, box);

        // Combine results
        if (left_rel == CellLeafRelation::OUTSIDE && right_rel == CellLeafRelation::OUTSIDE) {
            return CellLeafRelation::OUTSIDE;
        } else if (left_rel == CellLeafRelation::INSIDE && right_rel == CellLeafRelation::INSIDE) {
            return CellLeafRelation::INSIDE;  // Both sides fully in target (shouldn't happen but handle it)
        } else {
            return CellLeafRelation::BORDER;  // Partial overlap
        }
    }
}

CellLeafRelation classify_cell_leaf(const DecisionForest& forest, int tree_idx,
                                     int target_leaf_id, const Box3D& box) {
    int root_idx = forest.tree_offsets[tree_idx];
    return classify_cell_leaf_recursive(forest, root_idx, target_leaf_id, box);
}

bool bounds_are_valid(const ScoreBounds& bounds) {
    for (size_t t = 0; t < NUM_SCORE_TARGETS; ++t) {
        if (bounds.lower[t] > bounds.upper[t]) return false;
    }
    return true;
}

uint64_t bounds_to_signature(const ScoreBounds& bounds) {
    if (!bounds_are_valid(bounds)) return 0;
    uint64_t sig = 0;
    for (size_t t = 0; t < NUM_SCORE_TARGETS; ++t) {
        int status = 0;
        if (bounds.upper[t] < 0) status = 1;
        else if (bounds.lower[t] > 0) status = 2;
        sig |= (static_cast<uint64_t>(status) << (2 * t));
    }
    return sig;
}

ScoreBounds compute_leaf_bounds_dynamic_helper(const Octree& tree, const DecisionForest& forest,
                                                int tree_idx, int target_leaf_id, int node_idx) {
    const OctreeNode& node = tree.nodes[node_idx];
    CellLeafRelation rel = classify_cell_leaf(forest, tree_idx, target_leaf_id, node.box);

    if (rel == CellLeafRelation::OUTSIDE) {
        ScoreBounds empty;
        empty.upper.fill(std::numeric_limits<float>::lowest());
        empty.lower.fill(std::numeric_limits<float>::max());
        return empty;
    }

    if (node.is_leaf()) return node.bounds;

    // Internal node: recurse to children
    ScoreBounds bounds;
    bounds.upper.fill(std::numeric_limits<float>::lowest());
    bounds.lower.fill(std::numeric_limits<float>::max());

    for (int c = 0; c < 8; ++c) {
        int child_idx = node.children_start + c;
        if (tree.nodes[child_idx].deleted) continue;
        ScoreBounds child_bounds = compute_leaf_bounds_dynamic_helper(tree, forest, tree_idx, target_leaf_id, child_idx);
        for (size_t t = 0; t < NUM_SCORE_TARGETS; ++t) {
            bounds.upper[t] = std::max(bounds.upper[t], child_bounds.upper[t]);
            bounds.lower[t] = std::min(bounds.lower[t], child_bounds.lower[t]);
        }
    }
    return bounds;
}

void train_forest_with_octree(const Octree& tree, DecisionForest& forest) {
    // Build leaf_id -> node_idx lookup
    std::vector<std::vector<int>> leaf_to_node(forest.num_trees);
    for (int tree_idx = 0; tree_idx < forest.num_trees; ++tree_idx) {
        leaf_to_node[tree_idx].resize(forest.tree_num_leaves[tree_idx], -1);
        int start = forest.tree_offsets[tree_idx], end = forest.tree_offsets[tree_idx + 1];
        for (int i = start; i < end; ++i) {
            if (forest.is_leaf[i]) leaf_to_node[tree_idx][forest.leaf_id[i]] = i;
        }
    }

    for (int tree_idx = 0; tree_idx < forest.num_trees; ++tree_idx) {
        int num_leaves = forest.tree_num_leaves[tree_idx];
        for (int lid = 0; lid < num_leaves; ++lid) {
            ScoreBounds bounds = compute_leaf_bounds_dynamic_helper(tree, forest, tree_idx, lid, 0);
            uint64_t sig = bounds_to_signature(bounds);
            int node_idx = leaf_to_node[tree_idx][lid];
            if (node_idx >= 0) forest.set_node_signature(node_idx, sig);
        }
    }
}

// ============================================================================
// EVALUATION HELPERS
// ============================================================================

void precompute_cell_signatures(const std::vector<Cell>& cells, std::vector<uint64_t>& out) {
    out.resize(cells.size());
    for (size_t c = 0; c < cells.size(); ++c) {
        out[c] = compute_cell_signature(cells[c]);
    }
}

void get_signatures_batch(const DecisionForest& forest, const std::vector<Cell>& cells,
                          std::vector<uint64_t>& out) {
    out.resize(cells.size());
    for (size_t c = 0; c < cells.size(); ++c) {
        const Cell& cell = cells[c];
        float x = cell.box.x.center(), y = cell.box.y.center(), a = cell.box.angle.center();

        std::array<std::array<int, 3>, NUM_SCORE_TARGETS> votes{};
        for (int t = 0; t < forest.num_trees; ++t) {
            uint64_t sig = forest.get_signature(t, x, y, a);
            for (size_t i = 0; i < NUM_SCORE_TARGETS; ++i) {
                votes[i][(sig >> (2 * i)) & 0x3]++;
            }
        }

        uint64_t result = 0;
        for (size_t i = 0; i < NUM_SCORE_TARGETS; ++i) {
            int best = 0, best_count = votes[i][0];
            for (int s = 1; s < 3; ++s) {
                if (votes[i][s] > best_count) { best_count = votes[i][s]; best = s; }
            }
            result |= (static_cast<uint64_t>(best) << (2 * i));
        }
        out[c] = result;
    }
}

void evaluate_forest_fast(const std::vector<uint64_t>& forest_sigs,
                          const std::vector<uint64_t>& cell_sigs,
                          int& tp, int& fp, int& tn, int& fn) {
    tp = fp = tn = fn = 0;
    for (size_t c = 0; c < cell_sigs.size(); ++c) {
        uint64_t fs = forest_sigs[c], cs = cell_sigs[c];
        for (size_t t = 0; t < NUM_SCORE_TARGETS; ++t) {
            int forest_status = (fs >> (2 * t)) & 0x3;
            int cell_status = (cs >> (2 * t)) & 0x3;
            bool actual_positive = (cell_status == 2);
            bool actual_negative = (cell_status == 1);
            bool predict_negative = (forest_status == 1);

            if (actual_positive) { if (predict_negative) fn++; else tp++; }
            else if (actual_negative) { if (predict_negative) tn++; else fp++; }
            else { if (predict_negative) fn++; else tp++; }
        }
    }
}

// Get signatures for optimized forest (with voting)
void get_signatures_batch_optimized(const OptimizedForest& forest, const std::vector<Cell>& cells,
                                     std::vector<uint64_t>& out) {
    out.resize(cells.size());
    for (size_t c = 0; c < cells.size(); ++c) {
        const Cell& cell = cells[c];
        float x = cell.box.x.center(), y = cell.box.y.center(), a = cell.box.angle.center();

        std::array<std::array<int, 3>, NUM_SCORE_TARGETS> votes{};
        for (int t = 0; t < forest.num_trees; ++t) {
            uint64_t sig = forest.get_signature(t, x, y, a);
            for (size_t i = 0; i < NUM_SCORE_TARGETS; ++i) {
                votes[i][(sig >> (2 * i)) & 0x3]++;
            }
        }

        uint64_t result = 0;
        for (size_t i = 0; i < NUM_SCORE_TARGETS; ++i) {
            int best = 0, best_count = votes[i][0];
            for (int s = 1; s < 3; ++s) {
                if (votes[i][s] > best_count) { best_count = votes[i][s]; best = s; }
            }
            result |= (static_cast<uint64_t>(best) << (2 * i));
        }
        out[c] = result;
    }
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    Box3D full_domain = {{-2.0116f, 2.0116f}, {-2.0116f, 2.0116f}, {0.0f, 2.0f * PI}};

    using Clock = std::chrono::high_resolution_clock;
    using Ms = std::chrono::milliseconds;

    std::printf("========== Octree-Based Intersection Filtering ==========\n");
    std::printf("Domain: x,y in [-2.01, 2.01], angle in [0, 2pi]\n");
    std::printf("Targets: 25 triangle pairs (5x5)\n\n");

    // Build Octree
    std::printf("--- Building Octree ---\n");
    auto t0 = Clock::now();
    Octree octree(full_domain, 30);
    octree.refine(500000, 12);  // 500k cells, max depth 12
    auto t1 = Clock::now();
    std::printf("Build time: %ld ms\n", std::chrono::duration_cast<Ms>(t1 - t0).count());
    octree.print_stats();

    // Diagonal refinement
    t0 = Clock::now();
    octree.apply_diagonal_refinement(4);
    t1 = Clock::now();
    std::printf("Diagonal refinement: %ld ms\n", std::chrono::duration_cast<Ms>(t1 - t0).count());
    octree.print_stats();

    // Test merging with feature degradation limit (2x = don't merge if features get 2x wider)
    std::printf("\n--- Merging with feature degradation limit ---\n");
    octree.merge_siblings(2.0f);  // Only merge if degradation <= 2x
    octree.print_stats();

    // Get cells for evaluation
    auto cells = octree.to_cells();
    int n_cells = static_cast<int>(cells.size());

    // Show example cell feature ranges
    std::printf("\n--- Example Cell Feature Ranges (5 features: x, y, cos, sin, r) ---\n");
    std::printf("%6s %20s %12s %12s %12s\n",
               "Cell", "angle_range", "cos_range", "sin_range", "r_range");
    std::printf("%s\n", std::string(72, '-').c_str());

    // Show 10 cells with different angle ranges
    std::vector<int> example_indices;
    for (int i = 0; i < n_cells && example_indices.size() < 10; i += n_cells / 10) {
        example_indices.push_back(i);
    }

    for (int idx : example_indices) {
        const Cell& c = cells[idx];
        Interval cos_iv = c.box.cos_angle();
        Interval sin_iv = c.box.sin_angle();
        Interval r_iv = c.box.radius();
        std::printf("%6d [%5.2f,%5.2f] [%5.2f,%5.2f] [%5.2f,%5.2f] [%5.2f,%5.2f]\n",
                   idx, c.box.angle.lo, c.box.angle.hi,
                   cos_iv.lo, cos_iv.hi, sin_iv.lo, sin_iv.hi,
                   r_iv.lo, r_iv.hi);
    }

    // Statistics on feature ranges
    float total_cos_width = 0, total_sin_width = 0, total_r_width = 0;
    for (const Cell& c : cells) {
        total_cos_width += c.box.cos_angle().width();
        total_sin_width += c.box.sin_angle().width();
        total_r_width += c.box.radius().width();
    }
    Interval full_r = full_domain.radius();
    std::printf("\nFeature range statistics:\n");
    std::printf("  Avg cos(angle) width: %.4f (max 2.0)\n", total_cos_width / n_cells);
    std::printf("  Avg sin(angle) width: %.4f (max 2.0)\n", total_sin_width / n_cells);
    std::printf("  Avg r width:          %.4f (max %.2f)\n", total_r_width / n_cells, full_r.hi - full_r.lo);

    // Precompute cell signatures
    std::vector<uint64_t> cell_sigs, forest_sigs;
    precompute_cell_signatures(cells, cell_sigs);
    forest_sigs.reserve(n_cells);

    // =========================================================================
    // TEST 1: OPTIMIZED FOREST (greedy TN maximization)
    // =========================================================================
    std::printf("\n========== OPTIMIZED FOREST (greedy axis-aligned splits) ==========\n");
    std::printf("Training optimized forest on %d cells...\n", n_cells);

    t0 = Clock::now();
    OptimizedForest opt_forest;
    opt_forest.train(cells, 8, 8, 30);  // 8 trees, depth 8, 30 threshold candidates
    t1 = Clock::now();
    std::printf("Training time: %ld ms\n", std::chrono::duration_cast<Ms>(t1 - t0).count());

    // Show tree structure
    opt_forest.print_tree_structure(0, 10);

    // Evaluate
    t0 = Clock::now();
    get_signatures_batch_optimized(opt_forest, cells, forest_sigs);
    int opt_tp, opt_fp, opt_tn, opt_fn;
    evaluate_forest_fast(forest_sigs, cell_sigs, opt_tp, opt_fp, opt_tn, opt_fn);
    t1 = Clock::now();

    double opt_fpr = (opt_fn + opt_tp > 0) ? 100.0 * opt_fn / (opt_fn + opt_tp) : 0.0;
    double opt_fnr = (opt_fp + opt_tn > 0) ? 100.0 * opt_fp / (opt_fp + opt_tn) : 0.0;

    std::printf("\nOptimized Forest Results (BEFORE refinement):\n");
    std::printf("  TP: %d, FP: %d, TN: %d, FN: %d\n", opt_tp, opt_fp, opt_tn, opt_fn);
    std::printf("  FPR: %.4f%% (must be 0%%)\n", opt_fpr);
    std::printf("  FNR: %.2f%% (lower = better filtering)\n", opt_fnr);

    // Store pre-refinement values
    int pre_tn = opt_tn, pre_fp = opt_fp;
    double pre_fnr = opt_fnr;

    // =========================================================================
    // POST-PROCESSING REFINEMENT
    // =========================================================================
    t0 = Clock::now();
    opt_forest.refine_all(cells, true);
    t1 = Clock::now();
    std::printf("Refinement time: %ld ms\n", std::chrono::duration_cast<Ms>(t1 - t0).count());

    // Re-evaluate after refinement
    get_signatures_batch_optimized(opt_forest, cells, forest_sigs);
    evaluate_forest_fast(forest_sigs, cell_sigs, opt_tp, opt_fp, opt_tn, opt_fn);

    opt_fpr = (opt_fn + opt_tp > 0) ? 100.0 * opt_fn / (opt_fn + opt_tp) : 0.0;
    opt_fnr = (opt_fp + opt_tn > 0) ? 100.0 * opt_fp / (opt_fp + opt_tn) : 0.0;

    std::printf("\nOptimized Forest Results (AFTER refinement):\n");
    std::printf("  TP: %d, FP: %d, TN: %d, FN: %d\n", opt_tp, opt_fp, opt_tn, opt_fn);
    std::printf("  FPR: %.4f%% (must be 0%%)\n", opt_fpr);
    std::printf("  FNR: %.2f%% (lower = better filtering)\n", opt_fnr);

    std::printf("\nRefinement improvement:\n");
    std::printf("  TN: %d -> %d (+%d, %.1f%% more filtering)\n",
               pre_tn, opt_tn, opt_tn - pre_tn, 100.0 * (opt_tn - pre_tn) / std::max(1, pre_tn));
    std::printf("  FNR: %.2f%% -> %.2f%% (%.2f pp improvement)\n",
               pre_fnr, opt_fnr, pre_fnr - opt_fnr);

    // Feature importance
    opt_forest.print_feature_importance();

    // =========================================================================
    // TEST 2: RANDOM FOREST (for comparison)
    // =========================================================================
    std::printf("\n========== RANDOM FOREST (baseline comparison) ==========\n");
    // Definitions (g(x) > 0 = intersection = positive):
    //   FPR = FN / (FN + TP) = P(predict neg | actual pos) = missed intersections (MUST BE 0%)
    //   FNR = FP / (FP + TN) = P(predict pos | actual neg) = wasted checks (lower is better)
    std::printf("Testing 100 random forests (4 trees, depth 3)...\n");

    std::vector<double> fprs, fnrs;
    long long total_tp = 0, total_fp = 0, total_tn = 0, total_fn = 0;
    long long train_us = 0, eval_us = 0;

    t0 = Clock::now();
    for (int i = 0; i < 100; ++i) {
        DecisionForest forest;
        forest.initialize_random(full_domain, 4, 3, 1000 + i * 17);

        auto tt0 = Clock::now();
        train_forest_with_octree(octree, forest);
        auto tt1 = Clock::now();
        train_us += std::chrono::duration_cast<std::chrono::microseconds>(tt1 - tt0).count();

        auto te0 = Clock::now();
        get_signatures_batch(forest, cells, forest_sigs);
        int tp, fp, tn, fn;
        evaluate_forest_fast(forest_sigs, cell_sigs, tp, fp, tn, fn);
        auto te1 = Clock::now();
        eval_us += std::chrono::duration_cast<std::chrono::microseconds>(te1 - te0).count();

        double fpr = (fn + tp > 0) ? 100.0 * fn / (fn + tp) : 0.0;
        double fnr = (fp + tn > 0) ? 100.0 * fp / (fp + tn) : 0.0;
        fprs.push_back(fpr);
        fnrs.push_back(fnr);
        total_tp += tp; total_fp += fp; total_tn += tn; total_fn += fn;
    }
    t1 = Clock::now();
    auto total_ms = std::chrono::duration_cast<Ms>(t1 - t0).count();

    // Compute aggregate stats
    double avg_fpr = 0, avg_fnr = 0;
    for (size_t i = 0; i < fprs.size(); ++i) { avg_fpr += fprs[i]; avg_fnr += fnrs[i]; }
    avg_fpr /= fprs.size(); avg_fnr /= fnrs.size();

    std::printf("\nRandom Forest Results (avg over 100):\n");
    std::printf("  Total time: %ld ms (%.1f ms/forest)\n", total_ms, total_ms / 100.0);
    std::printf("  TP: %lld, FP: %lld, TN: %lld, FN: %lld\n", total_tp, total_fp, total_tn, total_fn);
    std::printf("  FPR: %.4f%% avg\n", avg_fpr);
    std::printf("  FNR: %.2f%% avg\n", avg_fnr);

    int safe = 0;
    for (double f : fprs) if (f == 0.0) safe++;
    std::printf("  Safe forests (FPR=0): %d / 100\n", safe);

    // =========================================================================
    // TEST 3: AABB FILTERING (baseline)
    // =========================================================================
    std::printf("\n========== AABB FILTERING (baseline) ==========\n");

    // Compute AABB for reference tree at origin (angle=0)
    Figure ref_figure = params_to_figure(TreeParams{Vec2{0, 0}, 0.0f});
    AABB ref_aabb = compute_aabb(ref_figure);
    std::array<AABB, TREE_NUM_TRIANGLES> ref_tri_aabbs;
    for (size_t i = 0; i < TREE_NUM_TRIANGLES; ++i) {
        ref_tri_aabbs[i] = AABB{};
        ref_tri_aabbs[i].expand(ref_figure[i].v0);
        ref_tri_aabbs[i].expand(ref_figure[i].v1);
        ref_tri_aabbs[i].expand(ref_figure[i].v2);
    }

    // Test AABB filtering on all cells (using cell centers)
    int64_t aabb_overall_tp = 0, aabb_overall_fp = 0, aabb_overall_tn = 0, aabb_overall_fn = 0;
    int64_t aabb_pair_tp = 0, aabb_pair_fp = 0, aabb_pair_tn = 0, aabb_pair_fn = 0;

    t0 = Clock::now();
    for (int i = 0; i < n_cells; ++i) {
        float x = cells[i].box.x.center();
        float y = cells[i].box.y.center();
        float angle = cells[i].box.angle.center();

        // Compute transformed figure and AABBs
        Figure fig = params_to_figure(TreeParams{Vec2{x, y}, angle});
        AABB fig_aabb = compute_aabb(fig);
        std::array<AABB, TREE_NUM_TRIANGLES> fig_tri_aabbs;
        for (size_t ti = 0; ti < TREE_NUM_TRIANGLES; ++ti) {
            fig_tri_aabbs[ti] = AABB{};
            fig_tri_aabbs[ti].expand(fig[ti].v0);
            fig_tri_aabbs[ti].expand(fig[ti].v1);
            fig_tri_aabbs[ti].expand(fig[ti].v2);
        }

        // Get actual intersection status from cell signature
        uint64_t sig = cell_sigs[i];

        // Overall AABB filter
        bool overall_overlap = ref_aabb.intersects(fig_aabb);
        bool any_actual_intersection = false;
        for (size_t t = 0; t < NUM_SCORE_TARGETS; ++t) {
            int status = (sig >> (2 * t)) & 0x3;
            if (status == 2) { any_actual_intersection = true; break; }  // status=2 means positive
        }

        if (overall_overlap) {
            if (any_actual_intersection) aabb_overall_tp++;
            else aabb_overall_fp++;
        } else {
            if (any_actual_intersection) aabb_overall_fn++;
            else aabb_overall_tn++;
        }

        // Per-triangle-pair AABB filter
        for (size_t ti = 0; ti < TREE_NUM_TRIANGLES; ++ti) {
            for (size_t tj = 0; tj < TREE_NUM_TRIANGLES; ++tj) {
                size_t pair_idx = ti * TREE_NUM_TRIANGLES + tj;
                bool pair_overlap = ref_tri_aabbs[ti].intersects(fig_tri_aabbs[tj]);
                int status = (sig >> (2 * pair_idx)) & 0x3;
                bool actual_intersects = (status == 2);  // status=2 means definitely positive

                if (pair_overlap) {
                    if (actual_intersects) aabb_pair_tp++;
                    else aabb_pair_fp++;
                } else {
                    if (actual_intersects) aabb_pair_fn++;
                    else aabb_pair_tn++;
                }
            }
        }
    }
    t1 = Clock::now();
    auto aabb_ms = std::chrono::duration_cast<Ms>(t1 - t0).count();

    // Overall AABB stats
    double aabb_overall_fpr = (aabb_overall_fn + aabb_overall_tp > 0) ?
        100.0 * aabb_overall_fn / (aabb_overall_fn + aabb_overall_tp) : 0.0;
    double aabb_overall_fnr = (aabb_overall_fp + aabb_overall_tn > 0) ?
        100.0 * aabb_overall_fp / (aabb_overall_fp + aabb_overall_tn) : 0.0;

    // Per-pair AABB stats
    double aabb_pair_fpr = (aabb_pair_fn + aabb_pair_tp > 0) ?
        100.0 * aabb_pair_fn / (aabb_pair_fn + aabb_pair_tp) : 0.0;
    double aabb_pair_fnr = (aabb_pair_fp + aabb_pair_tn > 0) ?
        100.0 * aabb_pair_fp / (aabb_pair_fp + aabb_pair_tn) : 0.0;

    std::printf("Evaluation time: %ld ms (%d cells)\n", aabb_ms, n_cells);

    std::printf("\nOverall Figure AABB Filter:\n");
    std::printf("  TP: %ld, FP: %ld, TN: %ld, FN: %ld\n",
               (long)aabb_overall_tp, (long)aabb_overall_fp, (long)aabb_overall_tn, (long)aabb_overall_fn);
    std::printf("  FPR: %.4f%% (must be 0%%)\n", aabb_overall_fpr);
    std::printf("  FNR: %.2f%% (lower = better filtering)\n", aabb_overall_fnr);

    std::printf("\nPer-Triangle-Pair AABB Filter:\n");
    std::printf("  TP: %ld, FP: %ld, TN: %ld, FN: %ld\n",
               (long)aabb_pair_tp, (long)aabb_pair_fp, (long)aabb_pair_tn, (long)aabb_pair_fn);
    std::printf("  FPR: %.4f%% (must be 0%%)\n", aabb_pair_fpr);
    std::printf("  FNR: %.2f%% (lower = better filtering)\n", aabb_pair_fnr);

    // Estimate cost: AABB check = ~10 ops (4 comparisons), intersection = ~150 ops
    int64_t aabb_pair_checks_needed = aabb_pair_tp + aabb_pair_fp;
    double aabb_pair_check_fraction = 100.0 * aabb_pair_checks_needed / (n_cells * 25);
    std::printf("\n  Pairs needing full check: %.1f%%\n", aabb_pair_check_fraction);

    // =========================================================================
    // TEST 4: CASCADED FILTERING
    // =========================================================================
    std::printf("\n========== CASCADED FILTERING ==========\n");

    // Cascade 1: Figure AABB -> Per-pair AABB
    int64_t cascade1_fig_pass = 0;  // Tree-pairs where figure AABBs overlap
    int64_t cascade1_pair_checks = 0;  // Per-pair AABB checks done
    int64_t cascade1_intersect_checks = 0;  // Full intersection checks needed
    int64_t cascade1_tp = 0, cascade1_fn = 0;

    // Cascade 2: Decision Forest -> Per-pair AABB
    int64_t cascade2_forest_pass = 0;  // Pairs not filtered by forest
    int64_t cascade2_aabb_checks = 0;  // Per-pair AABB checks done
    int64_t cascade2_intersect_checks = 0;  // Full intersection checks needed
    int64_t cascade2_tp = 0, cascade2_fn = 0;

    for (int i = 0; i < n_cells; ++i) {
        float x = cells[i].box.x.center();
        float y = cells[i].box.y.center();
        float angle = cells[i].box.angle.center();

        // Compute transformed figure and AABBs
        Figure fig = params_to_figure(TreeParams{Vec2{x, y}, angle});
        AABB fig_aabb = compute_aabb(fig);
        std::array<AABB, TREE_NUM_TRIANGLES> fig_tri_aabbs;
        for (size_t ti = 0; ti < TREE_NUM_TRIANGLES; ++ti) {
            fig_tri_aabbs[ti] = AABB{};
            fig_tri_aabbs[ti].expand(fig[ti].v0);
            fig_tri_aabbs[ti].expand(fig[ti].v1);
            fig_tri_aabbs[ti].expand(fig[ti].v2);
        }

        // Get cell signature and forest signature
        uint64_t cell_sig = cell_sigs[i];
        uint64_t forest_sig = forest_sigs[i];

        // CASCADE 1: Figure AABB -> Per-pair AABB
        bool fig_overlap = ref_aabb.intersects(fig_aabb);
        if (fig_overlap) {
            cascade1_fig_pass++;
            // Do per-pair AABB checks
            for (size_t ti = 0; ti < TREE_NUM_TRIANGLES; ++ti) {
                for (size_t tj = 0; tj < TREE_NUM_TRIANGLES; ++tj) {
                    cascade1_pair_checks++;
                    size_t pair_idx = ti * TREE_NUM_TRIANGLES + tj;
                    bool pair_overlap = ref_tri_aabbs[ti].intersects(fig_tri_aabbs[tj]);
                    if (pair_overlap) {
                        cascade1_intersect_checks++;
                        int status = (cell_sig >> (2 * pair_idx)) & 0x3;
                        if (status == 2) cascade1_tp++;
                    } else {
                        // Filtered by per-pair AABB - check if we missed any
                        int status = (cell_sig >> (2 * pair_idx)) & 0x3;
                        if (status == 2) cascade1_fn++;
                    }
                }
            }
        } else {
            // Filtered by figure AABB - check if we missed any
            for (size_t t = 0; t < NUM_SCORE_TARGETS; ++t) {
                int status = (cell_sig >> (2 * t)) & 0x3;
                if (status == 2) cascade1_fn++;
            }
        }

        // CASCADE 2: Decision Forest -> Per-pair AABB
        for (size_t ti = 0; ti < TREE_NUM_TRIANGLES; ++ti) {
            for (size_t tj = 0; tj < TREE_NUM_TRIANGLES; ++tj) {
                size_t pair_idx = ti * TREE_NUM_TRIANGLES + tj;
                int forest_status = (forest_sig >> (2 * pair_idx)) & 0x3;
                int actual_status = (cell_sig >> (2 * pair_idx)) & 0x3;

                if (forest_status != 1) {  // Forest didn't filter (not predicting negative)
                    cascade2_forest_pass++;
                    // Do per-pair AABB check
                    cascade2_aabb_checks++;
                    bool pair_overlap = ref_tri_aabbs[ti].intersects(fig_tri_aabbs[tj]);
                    if (pair_overlap) {
                        cascade2_intersect_checks++;
                        if (actual_status == 2) cascade2_tp++;
                    } else {
                        // Filtered by AABB
                        if (actual_status == 2) cascade2_fn++;
                    }
                } else {
                    // Filtered by forest - verify no FN
                    if (actual_status == 2) cascade2_fn++;
                }
            }
        }
    }

    double cascade1_fpr = (cascade1_fn + cascade1_tp > 0) ? 100.0 * cascade1_fn / (cascade1_fn + cascade1_tp) : 0.0;
    double cascade2_fpr = (cascade2_fn + cascade2_tp > 0) ? 100.0 * cascade2_fn / (cascade2_fn + cascade2_tp) : 0.0;

    std::printf("\nCascade 1: Figure AABB -> Per-pair AABB\n");
    std::printf("  Tree-pairs passing figure AABB: %ld / %d (%.1f%%)\n",
               (long)cascade1_fig_pass, n_cells, 100.0 * cascade1_fig_pass / n_cells);
    std::printf("  Per-pair AABB checks: %ld\n", (long)cascade1_pair_checks);
    std::printf("  Full intersection checks: %ld (%.1f%% of all pairs)\n",
               (long)cascade1_intersect_checks, 100.0 * cascade1_intersect_checks / (n_cells * 25));
    std::printf("  FPR: %.4f%%\n", cascade1_fpr);

    std::printf("\nCascade 2: Decision Forest -> Per-pair AABB\n");
    std::printf("  Pairs passing forest filter: %ld (%.1f%%)\n",
               (long)cascade2_forest_pass, 100.0 * cascade2_forest_pass / (n_cells * 25));
    std::printf("  Per-pair AABB checks: %ld\n", (long)cascade2_aabb_checks);
    std::printf("  Full intersection checks: %ld (%.1f%% of all pairs)\n",
               (long)cascade2_intersect_checks, 100.0 * cascade2_intersect_checks / (n_cells * 25));
    std::printf("  FPR: %.4f%%\n", cascade2_fpr);

    // =========================================================================
    // COMPARISON SUMMARY
    // =========================================================================
    std::printf("\n========== COMPARISON SUMMARY ==========\n");
    std::printf("%-22s %10s %10s %12s %12s\n", "Metric", "Optimized", "AABB-Pair", "AABB-Figure", "Random");
    std::printf("%s\n", std::string(68, '-').c_str());
    std::printf("%-22s %9.4f%% %9.4f%% %11.4f%% %11.4f%%\n", "FPR (must be 0%)",
               opt_fpr, aabb_pair_fpr, aabb_overall_fpr, avg_fpr);
    std::printf("%-22s %9.2f%% %9.2f%% %11.2f%% %11.2f%%\n", "FNR (lower=better)",
               opt_fnr, aabb_pair_fnr, aabb_overall_fnr, avg_fnr);
    std::printf("%-22s %10ld %10ld %12ld %12ld\n", "TN (filtered)",
               (long)opt_tn, (long)aabb_pair_tn, (long)aabb_overall_tn, (long)(total_tn / 100));

    // Cost analysis
    std::printf("\n--- Cost Analysis (per tree-pair, avg ops) ---\n");
    double opt_check_frac = (double)(opt_tp + opt_fp) / (n_cells * 25);
    double aabb_check_frac = (double)aabb_pair_checks_needed / (n_cells * 25);
    double forest_cost = 8 * 8;  // 8 trees, ~8 avg depth
    double aabb_fig_cost = 4;    // 1 figure AABB check
    double aabb_pair_cost = 4;   // per pair AABB check
    double intersect_cost = 150; // Full SAT intersection

    // Cascade 1 costs
    double c1_fig_pass_frac = (double)cascade1_fig_pass / n_cells;
    double c1_intersect_frac = (double)cascade1_intersect_checks / (n_cells * 25);
    double c1_filter_cost = aabb_fig_cost + c1_fig_pass_frac * 25 * aabb_pair_cost;
    double c1_effective = c1_filter_cost + c1_intersect_frac * 25 * intersect_cost;

    // Cascade 2 costs
    double c2_forest_pass_frac = (double)cascade2_forest_pass / (n_cells * 25);
    double c2_intersect_frac = (double)cascade2_intersect_checks / (n_cells * 25);
    double c2_filter_cost = forest_cost + c2_forest_pass_frac * 25 * aabb_pair_cost;
    double c2_effective = c2_filter_cost + c2_intersect_frac * 25 * intersect_cost;

    std::printf("%-28s %8s %10s %8s\n", "Method", "Filter", "Effective", "Speedup");
    std::printf("%s\n", std::string(58, '-').c_str());
    double baseline = 25 * intersect_cost;
    std::printf("%-28s %8.0f %10.0f %7.1fx\n", "No filter", 0.0, baseline, 1.0);
    std::printf("%-28s %8.0f %10.0f %7.1fx\n", "AABB per-pair only",
               25 * aabb_pair_cost, 25 * aabb_pair_cost + aabb_check_frac * 25 * intersect_cost,
               baseline / (25 * aabb_pair_cost + aabb_check_frac * 25 * intersect_cost));
    std::printf("%-28s %8.0f %10.0f %7.1fx\n", "Decision Forest only",
               forest_cost, forest_cost + opt_check_frac * 25 * intersect_cost,
               baseline / (forest_cost + opt_check_frac * 25 * intersect_cost));
    std::printf("%-28s %8.0f %10.0f %7.1fx\n", "FigAABB -> PairAABB",
               c1_filter_cost, c1_effective, baseline / c1_effective);
    std::printf("%-28s %8.0f %10.0f %7.1fx\n", "Forest -> PairAABB",
               c2_filter_cost, c2_effective, baseline / c2_effective);

    return 0;
}
