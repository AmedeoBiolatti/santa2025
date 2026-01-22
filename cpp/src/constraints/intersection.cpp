#include "tree_packing/constraints/intersection.hpp"
#include "tree_packing/geometry/sat.hpp"
#include "tree_packing/core/tree.hpp"
#include <algorithm>

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

namespace tree_packing {
namespace {
struct IntersectionScratch {
    std::vector<Index> candidates;
    std::vector<int> seen;
    std::vector<char> is_modified;
};

IntersectionScratch& scratch() {
    static thread_local IntersectionScratch s;
    return s;
}

SolutionEval::IntersectionList& ensure_unique_row(
    SolutionEval::IntersectionMap& map,
    size_t idx
) {
    auto& row_ptr = map[idx];
    if (!row_ptr) {
        row_ptr = std::make_shared<SolutionEval::IntersectionList>();
    } else if (!row_ptr.unique()) {
        row_ptr = std::make_shared<SolutionEval::IntersectionList>(*row_ptr);
    }
    return *row_ptr;
}

bool erase_pair(std::vector<std::pair<Index, float>>& list, int key, float& score_out) {
    Index target = static_cast<Index>(key);
    for (size_t i = 0; i < list.size(); ++i) {
        if (list[i].first == target) {
            score_out = list[i].second;
            if (i + 1 != list.size()) {
                list[i] = list.back();
            }
            list.pop_back();
            return true;
        }
    }
    return false;
}
}  // namespace

float IntersectionConstraint::compute_pair_score(
    const Figure& f0,
    const Figure& f1,
    const Vec2& c0,
    const Vec2& c1
) const {
    // Quick distance check using bounding spheres
    Vec2 diff = c0 - c1;
    float dist2 = diff.length_squared();

    // If centers are too far apart, skip detailed check
    if (dist2 >= THR2) {
        return 0.0f;
    }

    return figure_intersection_score(f0, f1, EPSILON, false, true);
}

float IntersectionConstraint::compute_pair_score_from_normals(
    const Figure& f0,
    const Figure& f1,
    const std::array<std::array<Vec2, 3>, TREE_NUM_TRIANGLES>& n0,
    const std::array<std::array<Vec2, 3>, TREE_NUM_TRIANGLES>& n1,
    const Vec2& c0,
    const Vec2& c1
) const {
    Vec2 diff = c0 - c1;
    float dist2 = diff.length_squared();

    if (dist2 >= THR2) {
        return 0.0f;
    }

    return figure_intersection_score_from_normals(f0, f1, n0, n1, EPSILON, false, true);
}

float IntersectionConstraint::eval(
    const Solution& solution,
    SolutionEval::IntersectionMap& map,
    int* out_count
) const {
    const auto& figures = solution.figures();
    const auto& normals = solution.normals();
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

    auto& work = scratch();
    auto& candidates = work.candidates;
    auto& seen = work.seen;
    candidates.reserve(static_cast<size_t>(reserve_size));
    if (seen.size() != n) {
        seen.assign(n, -1);
    } else {
        std::fill(seen.begin(), seen.end(), -1);
    }
    int stamp = 0;

    for (size_t i = 0; i < n; ++i) {
        if (!solution.is_valid(i)) continue;

        grid.get_candidates(static_cast<int>(i), candidates);
        ++stamp;
        int i_idx = static_cast<int>(i);
        for (Index c : candidates) {
            if (c < 0) continue;
            int c_idx = static_cast<int>(c);
            if (static_cast<size_t>(c_idx) <= i) continue;
            if (!solution.is_valid(static_cast<size_t>(c_idx))) continue;
            if (seen[c_idx] == stamp) continue;
            seen[c_idx] = stamp;

            float score = compute_pair_score_from_normals(
                figures[i],
                figures[c_idx],
                normals[i],
                normals[c_idx],
                centers[i],
                centers[c_idx]
            );
            if (score <= 0.0f) continue;
            map[i]->emplace_back(static_cast<Index>(c_idx), score);
            map[c_idx]->emplace_back(static_cast<Index>(i_idx), score);
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
    const auto& centers = solution.centers();
    const auto& new_grid = solution.grid();
    size_t n = figures.size();

    float total = prev_total;
    int count = prev_count;
    int reserve_size = static_cast<int>(NEIGHBOR_DELTAS.size()) * new_grid.capacity();
    auto& work = scratch();
    auto& is_modified = work.is_modified;
    if (is_modified.size() != n) {
        is_modified.assign(n, 0);
    } else {
        std::fill(is_modified.begin(), is_modified.end(), 0);
    }
    for (int idx : modified_indices) {
        if (idx < 0 || static_cast<size_t>(idx) >= n) continue;
        is_modified[static_cast<size_t>(idx)] = 1;
    }

    // Remove old values for modified indices
    for (int idx : modified_indices) {
        if (idx < 0 || static_cast<size_t>(idx) >= n) continue;

        auto& row = ensure_unique_row(map, static_cast<size_t>(idx));
        size_t row_size = row.size();
        for (const auto& [c, score] : row) {
            total -= 2.0f * score;
            float removed = 0.0f;
            auto& neighbor = ensure_unique_row(map, static_cast<size_t>(c));
            erase_pair(neighbor, idx, removed);
        }
        count -= static_cast<int>(row_size) * 2;
        row.clear();
        row.reserve(reserve_size);
    }

    // Compute new values for modified indices using new grid
    auto& candidates = work.candidates;
    auto& seen = work.seen;
    candidates.reserve(static_cast<size_t>(reserve_size));
    if (seen.size() != n) {
        seen.assign(n, -1);
    } else {
        std::fill(seen.begin(), seen.end(), -1);
    }
    int stamp = 0;
    for (int idx : modified_indices) {
        if (idx < 0 || static_cast<size_t>(idx) >= n) continue;
        if (!solution.is_valid(static_cast<size_t>(idx))) continue;

        new_grid.get_candidates(idx, candidates);
        ++stamp;
        for (Index c : candidates) {
            if (c < 0) continue;
            int c_idx = static_cast<int>(c);
            if (c_idx == idx) continue;
            if (!solution.is_valid(static_cast<size_t>(c_idx))) continue;
            if (is_modified[static_cast<size_t>(c_idx)] && c_idx < idx) continue;
            if (seen[c_idx] == stamp) continue;
            seen[c_idx] = stamp;

            float score = compute_pair_score_from_normals(
                figures[idx],
                figures[c_idx],
                normals[idx],
                normals[c_idx],
                centers[idx],
                centers[c_idx]
            );
            if (score <= 0.0f) continue;
            ensure_unique_row(map, static_cast<size_t>(idx)).emplace_back(static_cast<Index>(c_idx), score);
            ensure_unique_row(map, static_cast<size_t>(c_idx)).emplace_back(static_cast<Index>(idx), score);
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

}  // namespace tree_packing
