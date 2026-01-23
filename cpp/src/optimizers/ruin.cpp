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
    RuinState state;
    state.indices.reserve(static_cast<size_t>(n_remove_));
    return state;
}

void RandomRuin::apply(
    SolutionEval& solution,
    std::any& state,
    GlobalState& global_state,
    RNG& rng
) {
    (void)global_state;
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

    // Create new solution with updated params
    ruin_state->prev_params.resize(indices.size());
    for (size_t k = 0; k < indices.size(); ++k) {
        int idx = indices[k];
        if (idx < 0 || static_cast<size_t>(idx) >= n) {
            ruin_state->prev_params.set_nan(k);
            continue;
        }
        if (solution.solution.is_valid(static_cast<size_t>(idx))) {
            ruin_state->prev_params.set(k, solution.solution.get_params(static_cast<size_t>(idx)));
        } else {
            ruin_state->prev_params.set_nan(k);
        }
    }
    problem_->remove_and_eval(solution, indices);

    if (verbose_) {
        std::cout << "[RandomRuin] removed=" << indices.size() << "\n";
    }
}

void RandomRuin::rollback(SolutionEval& solution, std::any& state) {
    auto* ruin_state = std::any_cast<RuinState>(&state);
    if (!ruin_state) {
        return;
    }
    const auto& indices = ruin_state->indices;
    if (indices.empty()) {
        return;
    }
    std::vector<int> valid_indices;
    std::vector<int> invalid_indices;
    TreeParamsSoA valid_params;
    for (size_t k = 0; k < indices.size(); ++k) {
        int idx = indices[k];
        if (ruin_state->prev_params.is_nan(k)) {
            invalid_indices.push_back(idx);
        } else {
            size_t out_idx = valid_indices.size();
            valid_indices.push_back(idx);
            valid_params.resize(valid_indices.size());
            valid_params.set(out_idx, ruin_state->prev_params.get(k));
        }
    }
    if (!invalid_indices.empty()) {
        problem_->remove_and_eval(solution, invalid_indices);
    }
    if (!valid_indices.empty()) {
        problem_->update_and_eval(solution, valid_indices, valid_params);
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
    state.indices.reserve(static_cast<size_t>(n_remove_));
    state.distances.reserve(solution.solution.size());
    return state;
}

void CellRuin::apply(
    SolutionEval& solution,
    std::any& state,
    GlobalState& global_state,
    RNG& rng
) {
    (void)global_state;
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

    std::vector<std::pair<int, int>> eligible_cells;
    eligible_cells.reserve(static_cast<size_t>(N * N));
    for (int i = 1; i < N - 1; ++i) {
        for (int j = 1; j < N - 1; ++j) {
            if (grid.cell_count(i, j) >= n_remove_) {
                eligible_cells.emplace_back(i, j);
            }
        }
    }

    if (eligible_cells.empty()) {
        // Fallback: remove random valid indices.
        for (size_t i = 0; i < params.size(); ++i) {
            if (!params.is_nan(i)) {
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
        std::vector<Index> cell_items;
        grid.get_items_in_cell(ci, cj, cell_items);
        for (Index idx : cell_items) {
            if (idx < 0) continue;
            indices.push_back(static_cast<int>(idx));
        }
        if (static_cast<int>(indices.size()) < n_remove_) {
            indices.clear();
            for (size_t i = 0; i < params.size(); ++i) {
                if (!params.is_nan(i)) {
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

    // Create new solution with updated params
    ruin_state->prev_params.resize(indices.size());
    for (size_t k = 0; k < indices.size(); ++k) {
        int idx = indices[k];
        if (idx < 0 || static_cast<size_t>(idx) >= params.size()) {
            ruin_state->prev_params.set_nan(k);
            continue;
        }
        if (solution.solution.is_valid(static_cast<size_t>(idx))) {
            ruin_state->prev_params.set(k, solution.solution.get_params(static_cast<size_t>(idx)));
        } else {
            ruin_state->prev_params.set_nan(k);
        }
    }
    problem_->remove_and_eval(solution, indices);

    if (verbose_) {
        std::cout << "[CellRuin] removed=" << indices.size() << "\n";
    }
}

void CellRuin::rollback(SolutionEval& solution, std::any& state) {
    auto* ruin_state = std::any_cast<RuinState>(&state);
    if (!ruin_state) {
        return;
    }
    const auto& indices = ruin_state->indices;
    if (indices.empty()) {
        return;
    }
    std::vector<int> valid_indices;
    std::vector<int> invalid_indices;
    TreeParamsSoA valid_params;
    for (size_t k = 0; k < indices.size(); ++k) {
        int idx = indices[k];
        if (ruin_state->prev_params.is_nan(k)) {
            invalid_indices.push_back(idx);
        } else {
            size_t out_idx = valid_indices.size();
            valid_indices.push_back(idx);
            valid_params.resize(valid_indices.size());
            valid_params.set(out_idx, ruin_state->prev_params.get(k));
        }
    }
    if (!invalid_indices.empty()) {
        problem_->remove_and_eval(solution, invalid_indices);
    }
    if (!valid_indices.empty()) {
        problem_->update_and_eval(solution, valid_indices, valid_params);
    }
}

OptimizerPtr CellRuin::clone() const {
    return std::make_unique<CellRuin>(*this);
}

}  // namespace tree_packing
