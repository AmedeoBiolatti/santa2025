#include "tree_packing/constraints/intersection.hpp"
#include "tree_packing/geometry/sat.hpp"
#include "tree_packing/core/tree.hpp"
#include <algorithm>
#include <iostream>


namespace tree_packing {
namespace {

SolutionEval::IntersectionList& ensure_unique_row(
    SolutionEval::IntersectionMap& map,
    size_t idx
) {
    if (idx >= map.size()) {
        map.resize(idx + 1);
    }
    auto& row_ptr = map[idx];
    if (!row_ptr) {
        row_ptr = std::make_shared<SolutionEval::IntersectionList>();
    }
    return *row_ptr;
}

void erase_entry_with_back(
    SolutionEval::IntersectionMap& map,
    size_t row_idx,
    size_t entry_idx
) {
    auto& row = ensure_unique_row(map, row_idx);
    size_t last = row.size() - 1;
    if (entry_idx != last) {
        auto moved = row[last];
        row[entry_idx] = moved;
        auto& neighbor_row = ensure_unique_row(map, static_cast<size_t>(moved.neighbor));
        neighbor_row[static_cast<size_t>(moved.back_index)].back_index = static_cast<int>(entry_idx);
    }
    row.pop_back();
}

void add_pair(
    SolutionEval::IntersectionMap& map,
    size_t a_idx,
    size_t b_idx,
    float score
) {
    auto& row_a = ensure_unique_row(map, a_idx);
    auto& row_b = ensure_unique_row(map, b_idx);
    int a_pos = static_cast<int>(row_a.size());
    int b_pos = static_cast<int>(row_b.size());
    row_a.push_back({static_cast<Index>(b_idx), score, b_pos});
    row_b.push_back({static_cast<Index>(a_idx), score, a_pos});
}

}  // namespace
float IntersectionConstraint::compute_pair_score_from_normals_per_triangle(
    const Figure& f0,
    const Figure& f1,
    const std::array<std::array<Vec2, 3>, TREE_NUM_TRIANGLES>& n0,
    const std::array<std::array<Vec2, 3>, TREE_NUM_TRIANGLES>& n1,
    const std::array<AABB, TREE_NUM_TRIANGLES>& taabb0,
    const std::array<AABB, TREE_NUM_TRIANGLES>& taabb1,
    const AABB& aabb0,
    const AABB& aabb1,
    const Vec2& c0,
    const Vec2& c1
) const {
    const Triangle* t0 = f0.triangles.data();
    const Triangle* t1 = f1.triangles.data();
    const auto* n0p = n0.data();
    const auto* n1p = n1.data();
    const AABB* a0 = taabb0.data();
    const AABB* a1 = taabb1.data();

    if (aabb0.max.x < aabb1.min.x || aabb1.max.x < aabb0.min.x ||
        aabb0.max.y < aabb1.min.y || aabb1.max.y < aabb0.min.y) {
        return 0.0f;
    }

    Vec2 diff = c0 - c1;
    float dist2 = diff.length_squared();
    if (dist2 >= THR2) {
        return 0.0f;
    }

    float total = 0.0f;
    int i_list[TREE_NUM_TRIANGLES];
    int j_list[TREE_NUM_TRIANGLES];
    int i_count = 0;
    int j_count = 0;
    for (size_t i = 0; i < TREE_NUM_TRIANGLES; ++i) {
        if (a0[i].intersects(aabb1)) {
            i_list[i_count++] = static_cast<int>(i);
        }
    }
    for (size_t j = 0; j < TREE_NUM_TRIANGLES; ++j) {
        if (a1[j].intersects(aabb0)) {
            j_list[j_count++] = static_cast<int>(j);
        }
    }
    int pair_i[TREE_NUM_TRIANGLES * TREE_NUM_TRIANGLES];
    int pair_j[TREE_NUM_TRIANGLES * TREE_NUM_TRIANGLES];
    int pair_count = 0;
    for (int ii = 0; ii < i_count; ++ii) {
        int i = i_list[ii];
        for (int jj = 0; jj < j_count; ++jj) {
            int j = j_list[jj];
            if (!a0[i].intersects(a1[j])) {
                continue;
            }
            pair_i[pair_count] = i;
            pair_j[pair_count] = j;
            ++pair_count;
        }
    }
    for (int p = 0; p < pair_count; ++p) {
        int i = pair_i[p];
        int j = pair_j[p];
        float score = triangles_intersection_score_from_normals(
            t0[i], t1[j], n0p[i], n1p[j], EPSILON
        );
        if (score > 0.0f) {
            total += score;
        }
    }

    return total;
}

float IntersectionConstraint::eval(
    const Solution& solution,
    SolutionEval::IntersectionMap& map,
    int* out_count
) const {
    const auto& figures = solution.figures();
    const auto& normals = solution.normals();
    const auto& aabbs = solution.aabbs();
    const auto& tri_aabbs = solution.triangle_aabbs();
    const auto& centers = solution.centers();
    const auto& grid = solution.grid();
    size_t n = figures.size();

    map.resize(n);
    int reserve_size = static_cast<int>(NEIGHBOR_DELTAS.size()) * grid.capacity();
    for (auto& row : map) {
        if (!row) {
            row = std::make_shared<SolutionEval::IntersectionList>();
        } else {
            row->clear();
        }
        row->reserve(reserve_size);
    }

    float total_violation = 0.0f;
    int count = 0;

    for (size_t i = 0; i < n; ++i) {
        if (!solution.is_valid(i)) continue;

        int i_idx = static_cast<int>(i);
        auto [ci, cj] = grid.get_item_cell(i_idx);

        size_t n_candidates = grid.get_candidates_by_cell(ci, cj, candidates_);
        size_t t = 0;
        for (Index c : candidates_) {
            if ((t++) >= n_candidates) break;
            if (c < 0) continue;
            int c_idx = static_cast<int>(c);
            if (static_cast<size_t>(c_idx) <= i) continue;
            if (!solution.is_valid(static_cast<size_t>(c_idx))) continue;

            float score = compute_pair_score_from_normals_per_triangle(
                figures[i],
                figures[c_idx],
                normals[i],
                normals[c_idx],
                tri_aabbs[i],
                tri_aabbs[c_idx],
                aabbs[i],
                aabbs[c_idx],
                centers[i],
                centers[c_idx]
            );
            if (score <= 0.0f) continue;
            add_pair(map, i, static_cast<size_t>(c_idx), score);
            total_violation += 2.0f * score;  // Count both directions
            count += 2;
        }

    }

    if (out_count) {
        *out_count = count;
    }
    return total_violation;
}

float IntersectionConstraint::eval_update(
    const Solution& solution,
    SolutionEval::IntersectionMap& map,
    const std::vector<int>& modified_indices,
    float prev_total,
    int prev_count,
    int* out_count
) const {
    const auto& figures = solution.figures();
    const auto& normals = solution.normals();
    const auto& aabbs = solution.aabbs();
    const auto& tri_aabbs = solution.triangle_aabbs();
    const auto& centers = solution.centers();
    const auto& new_grid = solution.grid();
    size_t n = solution.figures().size();

    float total = prev_total;
    int count = prev_count;
    modified_.clear();

    // Remove old values for modified indices
    int reserve_size = static_cast<int>(NEIGHBOR_DELTAS.size()) * new_grid.capacity();
    for (int idx : modified_indices) {
        if (idx < 0 || static_cast<size_t>(idx) >= n) continue;
        modified_.insert(idx);

        auto& row = ensure_unique_row(map, static_cast<size_t>(idx));
        size_t row_size = row.size();
        for (const auto& entry : row) {
            total -= 2.0f * entry.score;
            erase_entry_with_back(map, static_cast<size_t>(entry.neighbor), static_cast<size_t>(entry.back_index));
        }
        count -= static_cast<int>(row_size) * 2;
        row.clear();
        row.reserve(reserve_size);
    }

    // Compute new values for modified indices using new grid
    for (int idx : modified_indices) {
        if (idx < 0 || static_cast<size_t>(idx) >= n) continue;
        if (!solution.is_valid(static_cast<size_t>(idx))) continue;

        auto [ci, cj] = new_grid.get_item_cell(idx);

        size_t n_candidates = new_grid.get_candidates_by_cell(ci, cj, candidates_);
        size_t i = 0;
        for (Index c : candidates_) {
            if ((i++) >= n_candidates) break;
            if (c < 0) continue;
            int c_idx = static_cast<int>(c);
            if (c_idx == idx) continue;
            if (c_idx > idx && modified_.contains(c_idx)) continue;

            float score = compute_pair_score_from_normals_per_triangle(
                figures[idx],
                figures[c_idx],
                normals[idx],
                normals[c_idx],
                tri_aabbs[idx],
                tri_aabbs[c_idx],
                aabbs[idx],
                aabbs[c_idx],
                centers[idx],
                centers[c_idx]
            );
            if (score <= 0.0f) continue;
            add_pair(map, static_cast<size_t>(idx), static_cast<size_t>(c_idx), score);
            total += 2.0f * score;
            count += 2;
        }
    }

    if (count < 0) {
        count = 0;
    }
    if (count == 0) {
        total = 0.0f;
    }
    if (out_count) {
        *out_count = count;
    }
    return total;
}

float IntersectionConstraint::eval_remove(
    const Solution& solution,
    SolutionEval::IntersectionMap& map,
    const std::vector<int>& removed_indices,
    float prev_total,
    int prev_count,
    int* out_count
) const {
    size_t n = solution.figures().size();
    if (map.size() != n) {
        map.resize(n);
    }

    float total = prev_total;
    int count = prev_count;
    int reserve_size = static_cast<int>(NEIGHBOR_DELTAS.size()) * solution.grid().capacity();

    for (int idx : removed_indices) {
        if (idx < 0 || static_cast<size_t>(idx) >= n) continue;

        auto& row = ensure_unique_row(map, static_cast<size_t>(idx));
        size_t row_size = row.size();
        for (const auto& entry : row) {
            total -= 2.0f * entry.score;
            erase_entry_with_back(map, static_cast<size_t>(entry.neighbor), static_cast<size_t>(entry.back_index));
        }
        count -= static_cast<int>(row_size) * 2;
        row.clear();
        row.reserve(reserve_size);
    }

    if (count < 0) {
        count = 0;
    }
    if (count == 0) {
        total = 0.0f;
    }
    if (out_count) {
        *out_count = count;
    }
    return total;
}

}  // namespace tree_packing
