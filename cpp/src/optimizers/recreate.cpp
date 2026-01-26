#include "tree_packing/optimizers/recreate.hpp"
#include "tree_packing/core/tree.hpp"
#include "tree_packing/spatial/grid2d.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

namespace tree_packing {

RandomRecreate::RandomRecreate(int max_recreate, float box_size, float delta, bool verbose)
    : max_recreate_(max_recreate)
    , box_size_(box_size)
    , delta_(delta)
    , verbose_(verbose)
{
    if (max_recreate_ < 1) max_recreate_ = 1;
}

std::any RandomRecreate::init_state(const SolutionEval& solution) {
    (void)solution;
    RandomRecreateState state;
    state.iteration = 0;
    state.indices.reserve(8);
    state.new_params.reserve(8);
    return state;
}

void RandomRecreate::apply(
    SolutionEval& solution,
    std::any& state,
    GlobalState& global_state,
    RNG& rng
) {
    float max_abs = solution.solution.max_max_abs();

    auto* rec_state = std::any_cast<RandomRecreateState>(&state);
    if (!rec_state) {
        state = init_state(solution);
        rec_state = std::any_cast<RandomRecreateState>(&state);
    }

    // Get removed indices directly from Solution (no copy)
    const auto& removed = solution.solution.removed_indices();
    if (removed.empty()) {
        rec_state->iteration++;
        return;
    }

    // If no valid trees, use default box
    if (max_abs == 0.0f) {
        max_abs = box_size_ / 2.0f;
    }

    float minval = -max_abs - delta_;
    float maxval = max_abs + delta_;

    // Select indices to recreate (up to max_recreate)
    int n_recreate = std::min(max_recreate_, static_cast<int>(removed.size()));

    // Copy only the indices we need (not the whole vector)
    auto& indices = rec_state->indices;
    indices.assign(removed.begin(), removed.begin() + n_recreate);

    // Create new params
    auto& new_params = rec_state->new_params;
    new_params.resize(static_cast<size_t>(n_recreate));
    for (int i = 0; i < n_recreate; ++i) {
        TreeParams p{
            rng.uniform(minval, maxval),
            rng.uniform(minval, maxval),
            rng.uniform(-PI, PI)
        };
        new_params.set(static_cast<size_t>(i), p);
    }

    // Push insertions to update stack for rollback support
    auto& stack = global_state.update_stack();
    for (int idx : indices) {
        stack.push_insert(idx);
    }

    // Evaluate the new solution
    problem_->insert_and_eval(solution, indices, new_params);

    // Update state
    rec_state->iteration++;

    if (verbose_) {
        std::cout << "[RandomRecreate] iter=" << rec_state->iteration
                  << " removed=" << solution.solution.removed_indices().size()
                  << " recreated=" << n_recreate << "\n";
    }
}

OptimizerPtr RandomRecreate::clone() const {
    return std::make_unique<RandomRecreate>(*this);
}

GridCellRecreate::GridCellRecreate(
    int max_recreate,
    int cell_min,
    int cell_max,
    int neighbor_min,
    int neighbor_max,
    bool verbose
)
    : max_recreate_(max_recreate)
    , cell_min_(cell_min)
    , cell_max_(cell_max)
    , neighbor_min_(neighbor_min)
    , neighbor_max_(neighbor_max)
    , verbose_(verbose)
{
    if (max_recreate_ < 1) max_recreate_ = 1;
    if (cell_min_ < 0) cell_min_ = 0;
    if (cell_max_ < cell_min_) cell_max_ = cell_min_;
}

std::any GridCellRecreate::init_state(const SolutionEval& solution) {
    GridCellRecreateState state;
    state.iteration = 0;
    state.indices.reserve(8);
    state.new_params.reserve(8);
    int N = solution.solution.grid().grid_N();
    state.candidate_cells.reserve(static_cast<size_t>(N * N));
    return state;
}

void GridCellRecreate::apply(
    SolutionEval& solution,
    std::any& state,
    GlobalState& global_state,
    RNG& rng
) {
    auto* rec_state = std::any_cast<GridCellRecreateState>(&state);
    if (!rec_state) {
        state = init_state(solution);
        rec_state = std::any_cast<GridCellRecreateState>(&state);
    }

    // Get removed indices directly from Solution (no copy)
    const auto& removed = solution.solution.removed_indices();
    if (removed.empty()) {
        rec_state->iteration++;
        return;
    }

    const Grid2D& grid = solution.solution.grid();
    int n = grid.grid_n();

    auto& candidate_cells = rec_state->candidate_cells;
    candidate_cells.clear();
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= n; ++j) {
            int count = grid.cell_count(i, j);
            if (count < cell_min_ || count > cell_max_) {
                continue;
            }
            int neighbor_count = 0;
            for (const auto& [di, dj] : NEIGHBOR_DELTAS) {
                neighbor_count += grid.cell_count(i + di, j + dj);
            }
            if (neighbor_count < neighbor_min_) {
                continue;
            }
            if (neighbor_max_ >= 0 && neighbor_count > neighbor_max_) {
                continue;
            }
            candidate_cells.emplace_back(i, j);
        }
    }

    if (candidate_cells.empty()) {
        rec_state->iteration++;
        return;
    }

    int n_recreate = std::min(max_recreate_, static_cast<int>(removed.size()));
    // Copy only the indices we need (not the whole vector)
    auto& indices = rec_state->indices;
    indices.assign(removed.begin(), removed.begin() + n_recreate);

    auto& new_params = rec_state->new_params;
    new_params.resize(static_cast<size_t>(n_recreate));
    for (int k = 0; k < n_recreate; ++k) {
        int choice = rng.randint(0, static_cast<int>(candidate_cells.size() - 1));
        auto [ci, cj] = candidate_cells[static_cast<size_t>(choice)];
        AABB bounds = grid.cell_bounds(ci, cj);
        TreeParams p{
            rng.uniform(bounds.min.x, bounds.max.x),
            rng.uniform(bounds.min.y, bounds.max.y),
            rng.uniform(-PI, PI)
        };
        new_params.set(static_cast<size_t>(k), p);
    }

    // Push insertions to update stack for rollback support
    auto& stack = global_state.update_stack();
    for (int idx : indices) {
        stack.push_insert(idx);
    }

    problem_->insert_and_eval(solution, indices, new_params);

    rec_state->iteration++;

    if (verbose_) {
        std::cout << "[GridCellRecreate] iter=" << rec_state->iteration
                  << " removed=" << solution.solution.removed_indices().size()
                  << " recreated=" << n_recreate
                  << " cells=" << candidate_cells.size() << "\n";
    }
}

OptimizerPtr GridCellRecreate::clone() const {
    return std::make_unique<GridCellRecreate>(*this);
}

}  // namespace tree_packing
