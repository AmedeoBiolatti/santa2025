#include "tree_packing/optimizers/recreate.hpp"
#include "tree_packing/core/tree.hpp"
#include "tree_packing/spatial/grid2d.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
namespace {
    struct OpenMPInit {
        OpenMPInit() {
#ifdef OMP_NUM_THREADS_DEFAULT
            omp_set_num_threads(OMP_NUM_THREADS_DEFAULT);
#endif
        }
    };
    static OpenMPInit omp_init;
}
#endif

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

// ============================================================================
// SolverRecreate implementation
// ============================================================================

SolverRecreate::SolverRecreate(
    SolverPtr solver,
    int max_recreate,
    int num_samples,
    bool verbose
)
    : solver_(std::move(solver))
    , max_recreate_(max_recreate)
    , num_samples_(num_samples)
    , verbose_(verbose)
{
    if (max_recreate_ < 1) max_recreate_ = 1;
    if (num_samples_ < 1) num_samples_ = 1;
}

std::any SolverRecreate::init_state(const SolutionEval& solution) {
    (void)solution;
    SolverRecreateState state;
    state.iteration = 0;
    state.indices.reserve(8);
    state.new_params.reserve(8);
    state.sample_params.reserve(static_cast<size_t>(num_samples_));
    return state;
}

void SolverRecreate::apply(
    SolutionEval& solution,
    std::any& state,
    GlobalState& global_state,
    RNG& rng
) {
    auto* rec_state = std::any_cast<SolverRecreateState>(&state);
    if (!rec_state) {
        state = init_state(solution);
        rec_state = std::any_cast<SolverRecreateState>(&state);
    }

    // Get removed indices directly from Solution
    const auto& removed = solution.solution.removed_indices();
    if (removed.empty()) {
        rec_state->iteration++;
        return;
    }

    // Determine search bounds based on current solution
    float max_abs = solution.solution.max_max_abs();
    float delta = 0.5f;
    float search_min, search_max;

    if (max_abs > 0.0f && std::isfinite(max_abs)) {
        search_min = -max_abs - delta;
        search_max = max_abs + delta;
    } else {
        // No valid trees, use default bounds
        search_min = solver_->min_pos();
        search_max = solver_->max_pos();
    }

    // Clamp to solver bounds
    search_min = std::max(search_min, solver_->min_pos());
    search_max = std::min(search_max, solver_->max_pos());

    // Select indices to recreate (up to max_recreate)
    int n_recreate = std::min(max_recreate_, static_cast<int>(removed.size()));

    auto& indices = rec_state->indices;
    indices.assign(removed.begin(), removed.begin() + n_recreate);

    auto& new_params = rec_state->new_params;
    new_params.resize(static_cast<size_t>(n_recreate));

    // Process each tree to recreate
    for (int k = 0; k < n_recreate; ++k) {
        int target_idx = indices[static_cast<size_t>(k)];

        // Multi-start: run solver from multiple random initial positions
        // and pick the best result
        float best_score = std::numeric_limits<float>::max();
        TreeParams best_params{};

        for (int sample = 0; sample < num_samples_; ++sample) {
            // Create a sub-RNG for this sample
            RNG sample_rng = rng.split();

            // Run solver for this index
            auto result = solver_->solve_single(solution, target_idx, sample_rng, &global_state);

            // Compute score: f + mu * max(0, g)
            float mu = global_state.mu();
            float score = result.final_f + mu * std::max(0.0f, result.final_g);

            if (score < best_score) {
                best_score = score;
                best_params = result.params.get(0);
            }
        }

        new_params.set(static_cast<size_t>(k), best_params);

        if (verbose_) {
            std::cout << "[SolverRecreate] tree " << target_idx
                      << " -> pos=(" << best_params.pos.x << ", " << best_params.pos.y << ")"
                      << " ang=" << best_params.angle
                      << " score=" << best_score << "\n";
        }
    }

    // Push insertions to update stack for rollback support
    auto& stack = global_state.update_stack();
    for (int idx : indices) {
        stack.push_insert(idx);
    }

    // Evaluate the new solution
    problem_->insert_and_eval(solution, indices, new_params);

    rec_state->iteration++;

    if (verbose_) {
        std::cout << "[SolverRecreate] iter=" << rec_state->iteration
                  << " removed=" << solution.solution.removed_indices().size()
                  << " recreated=" << n_recreate << "\n";
    }
}

OptimizerPtr SolverRecreate::clone() const {
    return std::make_unique<SolverRecreate>(
        solver_->clone(),
        max_recreate_,
        num_samples_,
        verbose_
    );
}

// ============================================================================
// SolverOptimize implementation
// ============================================================================

SolverOptimize::SolverOptimize(
    SolverPtr solver,
    int max_optimize,
    int group_size,
    bool same_cell_pairs,
    int num_samples,
    bool verbose
)
    : solver_(std::move(solver))
    , max_optimize_(max_optimize)
    , group_size_(group_size)
    , same_cell_pairs_(same_cell_pairs)
    , num_samples_(num_samples)
    , verbose_(verbose)
{
    if (max_optimize_ < 1) max_optimize_ = 1;
    if (group_size_ < 1) group_size_ = 1;
    if (group_size_ > 2) group_size_ = 2;
    if (num_samples_ < 1) num_samples_ = 1;
}

std::any SolverOptimize::init_state(const SolutionEval& solution) {
    SolverOptimizeState state;
    state.iteration = 0;
    state.indices.reserve(8);
    state.valid_indices.reserve(solution.solution.size());
    state.old_params.reserve(8);
    state.new_params.reserve(8);
    return state;
}

void SolverOptimize::apply(
    SolutionEval& solution,
    std::any& state,
    GlobalState& global_state,
    RNG& rng
) {

    auto* opt_state = std::any_cast<SolverOptimizeState>(&state);
    if (!opt_state) {
        state = init_state(solution);
        opt_state = std::any_cast<SolverOptimizeState>(&state);
    }

    const auto& sol = solution.solution;
    size_t n_trees = sol.size();

    if (solver_ && group_size_ > solver_->max_group_size()) {
        throw std::runtime_error(
            "SolverOptimize group_size exceeds solver max_group_size"
        );
    }

    // Build list of valid tree indices
    auto& valid_indices = opt_state->valid_indices;
    valid_indices.clear();
    for (size_t i = 0; i < n_trees; ++i) {
        if (sol.is_valid(i)) {
            valid_indices.push_back(static_cast<int>(i));
        }
    }

    if (valid_indices.empty()) {
        opt_state->iteration++;
        return;
    }

    // Select random indices to optimize, grouped by group_size_
    const int group_size = group_size_;
    const int max_groups_by_valid = static_cast<int>(valid_indices.size()) / group_size;
    const int n_groups = std::min(max_optimize_, max_groups_by_valid);
    const int n_updates = n_groups * group_size;
    if (n_groups <= 0) {
        opt_state->iteration++;
        return;
    }

    auto& indices = opt_state->indices;
    indices.clear();
    indices.reserve(static_cast<size_t>(n_updates));

    // Optional same-cell pairing when optimizing in pairs.
    if (group_size == 2 && same_cell_pairs_) {
        auto& used_flags = opt_state->used_flags;
        used_flags.assign(n_trees, static_cast<char>(0));

        auto& candidate_cells = opt_state->candidate_cells;
        auto& cell_items = opt_state->cell_items;
        auto& cell_valid = opt_state->cell_valid;
        candidate_cells.clear();

        const auto& grid = sol.grid();
        const size_t needed = NEIGHBOR_DELTAS.size() * static_cast<size_t>(grid.capacity());
        cell_items.reserve(needed);
        cell_valid.reserve(needed);

        // Build list of cells that contain at least two valid trees.
        for (const auto& [ci, cj] : grid.non_empty_cells()) {
            cell_items = grid.get_items_in_cell(ci, cj);
            int count_valid = 0;
            for (int idx : cell_items) {
                if (idx < 0) continue;
                if (sol.is_valid(static_cast<size_t>(idx))) {
                    ++count_valid;
                    if (count_valid >= 2) {
                        candidate_cells.emplace_back(ci, cj);
                        break;
                    }
                }
            }
        }

        // Greedily sample pairs from candidate cells without replacement.
        for (int g = 0; g < n_groups; ++g) {
            bool selected = false;
            const int max_attempts = std::max(4, static_cast<int>(candidate_cells.size()) * 2);
            for (int attempt = 0; attempt < max_attempts && !candidate_cells.empty(); ++attempt) {
                const int cell_choice = rng.randint(0, static_cast<int>(candidate_cells.size()) - 1);
                const auto [ci, cj] = candidate_cells[static_cast<size_t>(cell_choice)];

                cell_items = grid.get_items_in_cell(ci, cj);
                cell_valid.clear();
                for (int idx : cell_items) {
                    if (idx < 0) continue;
                    if (!sol.is_valid(static_cast<size_t>(idx))) continue;
                    if (used_flags[static_cast<size_t>(idx)]) continue;
                    cell_valid.push_back(idx);
                }

                if (cell_valid.size() < 2) {
                    continue;
                }

                const int i0 = rng.randint(0, static_cast<int>(cell_valid.size()) - 1);
                int i1 = rng.randint(0, static_cast<int>(cell_valid.size()) - 2);
                if (i1 >= i0) i1 += 1;

                const int idx0 = cell_valid[static_cast<size_t>(i0)];
                const int idx1 = cell_valid[static_cast<size_t>(i1)];
                indices.push_back(idx0);
                indices.push_back(idx1);
                used_flags[static_cast<size_t>(idx0)] = static_cast<char>(1);
                used_flags[static_cast<size_t>(idx1)] = static_cast<char>(1);
                selected = true;
            }

            if (!selected) {
                break;
            }
        }

        // If we could not fill all requested pairs, fall back to random selection
        // for the remainder (still without replacement).
        const int remaining_updates = n_updates - static_cast<int>(indices.size());
        if (remaining_updates > 0) {
            valid_indices.clear();
            for (size_t i = 0; i < n_trees; ++i) {
                if (!sol.is_valid(i)) continue;
                if (used_flags[i]) continue;
                valid_indices.push_back(static_cast<int>(i));
            }
            const int take = std::min(remaining_updates, static_cast<int>(valid_indices.size()));
            for (int i = 0; i < take; ++i) {
                int remaining = static_cast<int>(valid_indices.size()) - i;
                int choice = rng.randint(0, remaining - 1);
                std::swap(valid_indices[static_cast<size_t>(i)],
                          valid_indices[static_cast<size_t>(i + choice)]);
                indices.push_back(valid_indices[static_cast<size_t>(i)]);
            }
        }
    } else {
        if (group_size == 1) {
            // Match NoiseOptimizer selection: rng.choice without replacement.
            auto& selected = opt_state->cell_valid;
            selected.clear();
            rng.choice(static_cast<int>(valid_indices.size()), n_updates, selected);
            for (int s : selected) {
                indices.push_back(valid_indices[static_cast<size_t>(s)]);
            }
        } else {
            // Fisher-Yates partial shuffle to select random indices without replacement.
            for (int i = 0; i < n_updates; ++i) {
                int remaining = static_cast<int>(valid_indices.size()) - i;
                int choice = rng.randint(0, remaining - 1);
                std::swap(valid_indices[static_cast<size_t>(i)],
                          valid_indices[static_cast<size_t>(i + choice)]);
                indices.push_back(valid_indices[static_cast<size_t>(i)]);
            }
        }
    }

    const int actual_updates = static_cast<int>(indices.size());
    const int actual_groups = actual_updates / group_size;
    if (actual_groups <= 0) {
        opt_state->iteration++;
        return;
    }

    // Save old params for potential rollback comparison
    auto& old_params = opt_state->old_params;
    old_params.resize(static_cast<size_t>(actual_updates));
    for (int k = 0; k < actual_updates; ++k) {
        old_params.set(static_cast<size_t>(k), sol.get_params(static_cast<size_t>(indices[static_cast<size_t>(k)])));
    }

    auto& new_params = opt_state->new_params;
    new_params.resize(static_cast<size_t>(actual_updates));

    const bool split_rngs = solver_->wants_rng_split();
    std::vector<RNG> tree_rngs;
    if (split_rngs) {
        // Pre-split RNGs for each group (must be done sequentially for reproducibility)
        tree_rngs.reserve(static_cast<size_t>(actual_groups));
        for (int k = 0; k < actual_groups; ++k) {
            tree_rngs.push_back(rng.split());
        }
    }

    // Cache mu for use in parallel region
    const float mu = global_state.mu();

    // Process each tree to optimize (parallelized)
    #ifdef _OPENMP
    if (split_rngs) {
        #pragma omp parallel
        {
            // Each thread gets its own solver clone
            auto thread_solver = solver_->clone();

            #pragma omp for schedule(dynamic)
            for (int g = 0; g < actual_groups; ++g) {
                const size_t base = static_cast<size_t>(g * group_size);
                const int idx0 = indices[base];
                const int idx1 = (group_size == 2) ? indices[base + 1] : -1;
                const TreeParams current0 = sol.get_params(static_cast<size_t>(idx0));
                const TreeParams current1 = (group_size == 2)
                    ? sol.get_params(static_cast<size_t>(idx1))
                    : TreeParams{};

                // Multi-start: run solver from multiple positions and pick the best
                float best_score = std::numeric_limits<float>::max();
                TreeParams best0 = current0;
                TreeParams best1 = current1;

                // Use the pre-split RNG for this group
                RNG local_rng = tree_rngs[static_cast<size_t>(g)];

                for (int sample = 0; sample < num_samples_; ++sample) {
                    RNG sample_rng = local_rng.split();

                    SolverResult result;
                    if (group_size == 2) {
                        std::array<int, 2> pair{idx0, idx1};
                        result = thread_solver->solve(solution, pair, sample_rng, &global_state);
                    } else {
                        result = thread_solver->solve_single(solution, idx0, sample_rng, &global_state);
                    }

                    // Compute score: f + mu * max(0, g)
                    float score = result.final_f + mu * std::max(0.0f, result.final_g);

                    if (score < best_score) {
                        best_score = score;
                        best0 = result.params.get(0);
                        if (group_size == 2) {
                            best1 = result.params.get(1);
                        }
                    }
                }

                new_params.set(base, best0);
                if (group_size == 2) {
                    new_params.set(base + 1, best1);
                }
            }
        }
    } else
    #endif
    // Non-OpenMP fallback (or RNG not split)
    for (int g = 0; g < actual_groups; ++g) {
        const size_t base = static_cast<size_t>(g * group_size);
        const int idx0 = indices[base];
        const int idx1 = (group_size == 2) ? indices[base + 1] : -1;
        const TreeParams current0 = sol.get_params(static_cast<size_t>(idx0));
        const TreeParams current1 = (group_size == 2)
            ? sol.get_params(static_cast<size_t>(idx1))
            : TreeParams{};

        // Multi-start: run solver from multiple positions and pick the best
        float best_score = std::numeric_limits<float>::max();
        float init_score = best_score;
        TreeParams best0 = current0;
        TreeParams best1 = current1;

        for (int sample = 0; sample < num_samples_; ++sample) {
            SolverResult result;
            if (split_rngs) {
                RNG sample_rng = tree_rngs[static_cast<size_t>(g)].split();
                if (group_size == 2) {
                    std::array<int, 2> pair{idx0, idx1};
                    result = solver_->solve(solution, pair, sample_rng, &global_state);
                } else {
                    result = solver_->solve_single(solution, idx0, sample_rng, &global_state);
                }
            } else {
                if (group_size == 2) {
                    std::array<int, 2> pair{idx0, idx1};
                    result = solver_->solve(solution, pair, rng, &global_state);
                } else {
                    result = solver_->solve_single(solution, idx0, rng, &global_state);
                }
            }

            // Compute score: f + mu * max(0, g)
            float score = result.final_f + mu * std::max(0.0f, result.final_g);

            if (score < best_score) {
                if (init_score > 1e12) init_score = score;
                best_score = score;
                best0 = result.params.get(0);
                if (group_size == 2) {
                    best1 = result.params.get(1);
                }
            }
        }

        new_params.set(base, best0);
        if (group_size == 2) {
            new_params.set(base + 1, best1);
        }

        if (verbose_) {
            std::cout << "[SolverOptimize] group " << g
                      << " idx0=" << idx0
                      << " (" << current0.pos.x << ", " << current0.pos.y << ")"
                      << " -> (" << best0.pos.x << ", " << best0.pos.y << ")";
            if (group_size == 2) {
                std::cout << " | idx1=" << idx1
                          << " (" << current1.pos.x << ", " << current1.pos.y << ")"
                          << " -> (" << best1.pos.x << ", " << best1.pos.y << ")";
            }
            std::cout << " score: " << init_score << "->" << best_score << "\n";
        }
    }
    // Push updates to the stack for rollback support
    auto& stack = global_state.update_stack();
    for (int k = 0; k < actual_updates; ++k) {
        int idx = indices[static_cast<size_t>(k)];
        stack.push_update(idx, old_params.get(static_cast<size_t>(k)));
    }

    // Apply the update
    problem_->update_and_eval(solution, indices, new_params);

    opt_state->iteration++;

    if (verbose_) {
        std::cout << "[SolverOptimize] iter=" << opt_state->iteration
                  << " optimized_groups=" << actual_groups
                  << " group_size=" << group_size << "\n";
    }
}

OptimizerPtr SolverOptimize::clone() const {
    return std::make_unique<SolverOptimize>(
        solver_->clone(),
        max_optimize_,
        group_size_,
        same_cell_pairs_,
        num_samples_,
        verbose_
    );
}

}  // namespace tree_packing
