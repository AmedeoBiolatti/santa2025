#include "tree_packing/constraints/intersection.hpp"
#include "tree_packing/geometry/sat.hpp"
#include "tree_packing/core/tree.hpp"

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

namespace tree_packing {

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

float IntersectionConstraint::eval(
    const Solution& solution,
    std::vector<std::unordered_map<int, float>>& map
) const {
    const auto& figures = solution.figures();
    const auto& centers = solution.centers();
    const auto& grid = solution.grid();
    size_t n = figures.size();

    map.clear();
    map.resize(n);
    int reserve_size = static_cast<int>(NEIGHBOR_DELTAS.size()) * grid.capacity();
    for (auto& row : map) {
        row.reserve(reserve_size);
    }

    float total_violation = 0.0f;

    std::vector<int> candidates;
    candidates.reserve(static_cast<size_t>(reserve_size));

    for (size_t i = 0; i < n; ++i) {
        if (!solution.is_valid(i)) continue;

        grid.get_candidates(static_cast<int>(i), candidates);
        for (int c : candidates) {
            if (c < 0 || static_cast<size_t>(c) <= i) continue;
            if (!solution.is_valid(static_cast<size_t>(c))) continue;

            float score = compute_pair_score(figures[i], figures[c], centers[i], centers[c]);
            if (score <= 0.0f) continue;
            map[i][c] = score;
            map[c][static_cast<int>(i)] = score;
            total_violation += 2.0f * score;  // Count both directions
        }
    }

    return total_violation;
}

float IntersectionConstraint::eval_update(
    const Solution& solution,
    const Solution& prev_solution,
    std::vector<std::unordered_map<int, float>>& map,
    const std::vector<int>& modified_indices,
    float prev_total
) const {
    const auto& figures = solution.figures();
    const auto& centers = solution.centers();
    const auto& new_grid = solution.grid();
    size_t n = figures.size();

    float total = prev_total;
    int reserve_size = static_cast<int>(NEIGHBOR_DELTAS.size()) * new_grid.capacity();
    std::vector<char> is_modified(n, 0);
    for (int idx : modified_indices) {
        if (idx < 0 || static_cast<size_t>(idx) >= n) continue;
        is_modified[static_cast<size_t>(idx)] = 1;
    }

    // Remove old values for modified indices
    for (int idx : modified_indices) {
        if (idx < 0 || static_cast<size_t>(idx) >= n) continue;

        auto& row = map[static_cast<size_t>(idx)];
        for (const auto& [c, score] : row) {
            total -= 2.0f * score;
            map[static_cast<size_t>(c)].erase(idx);
        }
        row.clear();
        row.reserve(reserve_size);
    }

    // Compute new values for modified indices using new grid
    std::vector<int> candidates;
    candidates.reserve(static_cast<size_t>(reserve_size));
    for (int idx : modified_indices) {
        if (idx < 0 || static_cast<size_t>(idx) >= n) continue;
        if (!solution.is_valid(static_cast<size_t>(idx))) continue;

        new_grid.get_candidates(idx, candidates);
        for (int c : candidates) {
            if (c < 0 || c == idx) continue;
            if (!solution.is_valid(static_cast<size_t>(c))) continue;
            if (is_modified[static_cast<size_t>(c)] && c < idx) continue;

            float score = compute_pair_score(
                figures[idx], figures[c],
                centers[idx], centers[c]
            );
            if (score <= 0.0f) continue;
            map[static_cast<size_t>(idx)][c] = score;
            map[static_cast<size_t>(c)][idx] = score;
            total += 2.0f * score;
        }
    }

    return total;
}

}  // namespace tree_packing
