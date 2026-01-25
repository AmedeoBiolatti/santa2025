#include "tree_packing/optimizers/ruin.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

namespace tree_packing {

// RandomRuin implementation
RandomRuin::RandomRuin(int n_remove, bool verbose)
    : n_remove_(n_remove), verbose_(verbose) {
    if (n_remove_ < 1) n_remove_ = 1;
}

std::any RandomRuin::init_state(const SolutionEval& solution) {
    (void)solution;
    RuinState state;
    state.indices.reserve(8);
    return state;
}

void RandomRuin::apply(
    SolutionEval& solution,
    std::any& state,
    GlobalState& global_state,
    RNG& rng
) {
    const auto& params = solution.solution.params();
    size_t n = params.size();

    // Select random indices to remove
    auto* ruin_state = std::any_cast<RuinState>(&state);
    if (!ruin_state) {
        state = init_state(solution);
        ruin_state = std::any_cast<RuinState>(&state);
    }
    auto& indices = ruin_state->indices;
    indices.clear();
    indices.reserve(static_cast<size_t>(n_remove_));
    if (n_remove_ >= static_cast<int>(n)) {
        indices.resize(n);
        for (size_t i = 0; i < n; ++i) {
            indices[i] = static_cast<int>(i);
        }
    } else {
        while (static_cast<int>(indices.size()) < n_remove_) {
            int idx = rng.randint(0, static_cast<int>(n) - 1);
            bool exists = false;
            for (int v : indices) {
                if (v == idx) {
                    exists = true;
                    break;
                }
            }
            if (!exists) {
                indices.push_back(idx);
            }
        }
    }

    // Push removals to update stack for rollback support
    auto& stack = global_state.update_stack();
    for (int idx : indices) {
        if (idx >= 0 && static_cast<size_t>(idx) < n && solution.solution.is_valid(static_cast<size_t>(idx))) {
            TreeParams prev = solution.solution.get_params(static_cast<size_t>(idx));
            stack.push_remove(idx, prev);
        }
    }

    problem_->remove_and_eval(solution, indices);

    if (verbose_) {
        std::cout << "[RandomRuin] removed=" << indices.size() << "\n";
    }
}

OptimizerPtr RandomRuin::clone() const {
    return std::make_unique<RandomRuin>(*this);
}

// CellRuin implementation
CellRuin::CellRuin(int n_remove, bool verbose)
    : n_remove_(n_remove), verbose_(verbose) {
    if (n_remove_ < 1) n_remove_ = 1;
}

std::any CellRuin::init_state(const SolutionEval& solution) {
    RuinState state;
    state.indices.reserve(8);
    state.distances.reserve(solution.solution.size());
    // For CellRuin specific scratch
    int N = solution.solution.grid().grid_N();
    state.eligible_cells.reserve(static_cast<size_t>(N * N));
    state.cell_items.resize(static_cast<size_t>(solution.solution.grid().capacity()));
    return state;
}

void CellRuin::apply(
    SolutionEval& solution,
    std::any& state,
    GlobalState& global_state,
    RNG& rng
) {
    const auto& params = solution.solution.params();
    const auto& grid = solution.solution.grid();
    int N = grid.grid_N();

    auto* ruin_state = std::any_cast<RuinState>(&state);
    if (!ruin_state) {
        state = init_state(solution);
        ruin_state = std::any_cast<RuinState>(&state);
    }
    auto& indices = ruin_state->indices;
    indices.clear();

    auto shuffle_indices = [&](std::vector<int>& vals) {
        for (int i = static_cast<int>(vals.size()) - 1; i > 0; --i) {
            int j = rng.randint(0, i);
            std::swap(vals[static_cast<size_t>(i)], vals[static_cast<size_t>(j)]);
        }
    };

    auto& eligible_cells = ruin_state->eligible_cells;
    eligible_cells.clear();
    // Use non_empty_cells() for O(occupied) instead of O(N^2) iteration
    for (const auto& [i, j] : grid.non_empty_cells()) {
        // Skip padding cells
        if (i < 1 || i >= N - 1 || j < 1 || j >= N - 1) continue;
        if (grid.cell_count(i, j) >= n_remove_) {
            eligible_cells.emplace_back(i, j);
        }
    }

    if (eligible_cells.empty()) {
        // Fallback: remove random valid indices.
        for (size_t i = 0; i < params.size(); ++i) {
            if (solution.solution.is_valid(i)) {
                indices.push_back(static_cast<int>(i));
            }
        }
        if (indices.empty()) {
            return;
        }
        shuffle_indices(indices);
        int count = std::min(n_remove_, static_cast<int>(indices.size()));
        indices.resize(static_cast<size_t>(count));
    } else {
        auto [ci, cj] = eligible_cells[rng.randint(0, static_cast<int>(eligible_cells.size()) - 1)];
        auto& cell_items = ruin_state->cell_items;
        grid.get_items_in_cell(ci, cj, cell_items);
        for (Index idx : cell_items) {
            if (idx < 0) continue;
            indices.push_back(static_cast<int>(idx));
        }
        if (static_cast<int>(indices.size()) < n_remove_) {
            indices.clear();
            for (size_t i = 0; i < params.size(); ++i) {
                if (solution.solution.is_valid(i)) {
                    indices.push_back(static_cast<int>(i));
                }
            }
        }
        if (indices.empty()) {
            return;
        }
        shuffle_indices(indices);
        indices.resize(static_cast<size_t>(n_remove_));
    }

    // Push removals to update stack for rollback support
    auto& stack = global_state.update_stack();
    for (int idx : indices) {
        if (idx >= 0 && static_cast<size_t>(idx) < params.size() && solution.solution.is_valid(static_cast<size_t>(idx))) {
            TreeParams prev = solution.solution.get_params(static_cast<size_t>(idx));
            stack.push_remove(idx, prev);
        }
    }

    problem_->remove_and_eval(solution, indices);

    if (verbose_) {
        std::cout << "[CellRuin] removed=" << indices.size() << "\n";
    }
}

OptimizerPtr CellRuin::clone() const {
    return std::make_unique<CellRuin>(*this);
}

}  // namespace tree_packing
