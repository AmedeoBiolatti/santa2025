#pragma once

#include "optimizer.hpp"
#include "../solvers/solver.hpp"
#include <vector>

namespace tree_packing {

// Random recreate: places removed trees at random positions
class RandomRecreate : public Optimizer {
public:
    explicit RandomRecreate(
        int max_recreate = 1,
        float box_size = 5.0f,
        float delta = 0.35f,
        bool verbose = false
    );

    std::any init_state(const SolutionEval& solution) override;

    void apply(
        SolutionEval& solution,
        std::any& state,
        GlobalState& global_state,
        RNG& rng
    ) override;

    [[nodiscard]] OptimizerPtr clone() const override;

private:
    int max_recreate_;
    float box_size_;
    float delta_;
    bool verbose_;
};

// GridCellRecreate: recreates trees in low-occupancy grid cells with neighbors
class GridCellRecreate : public Optimizer {
public:
    explicit GridCellRecreate(
        int max_recreate = 1,
        int cell_min = 0,
        int cell_max = 4,
        int neighbor_min = 1,
        int neighbor_max = -1,
        bool verbose = false
    );

    std::any init_state(const SolutionEval& solution) override;

    void apply(
        SolutionEval& solution,
        std::any& state,
        GlobalState& global_state,
        RNG& rng
    ) override;

    [[nodiscard]] OptimizerPtr clone() const override;

private:
    int max_recreate_;
    int cell_min_;
    int cell_max_;
    int neighbor_min_;
    int neighbor_max_;
    bool verbose_;
};

// State for RandomRecreate (simplified - rollback now uses UpdateStack)
struct RandomRecreateState {
    int iteration{0};
    std::vector<int> indices;
    TreeParamsSoA new_params;
};

// State for GridCellRecreate (simplified - rollback now uses UpdateStack)
struct GridCellRecreateState {
    int iteration{0};
    std::vector<int> indices;
    std::vector<std::pair<int, int>> candidate_cells;
    TreeParamsSoA new_params;
};

// SolverRecreate: uses a Solver to find optimal placements for removed trees
class SolverRecreate : public Optimizer {
public:
    /**
     * Create a recreate optimizer that uses a solver for placement optimization.
     *
     * @param solver The solver to use (ownership transferred)
     * @param max_recreate Maximum number of trees to recreate per step
     * @param num_samples Number of random initial positions to try per tree
     * @param verbose Print debug info
     */
    explicit SolverRecreate(
        SolverPtr solver,
        int max_recreate = 1,
        int num_samples = 16,
        bool verbose = false
    );

    void set_problem(Problem* problem) override {
        Optimizer::set_problem(problem);
        if (solver_) {
            solver_->set_problem(problem);
        }
    }

    std::any init_state(const SolutionEval& solution) override;

    void apply(
        SolutionEval& solution,
        std::any& state,
        GlobalState& global_state,
        RNG& rng
    ) override;

    [[nodiscard]] OptimizerPtr clone() const override;

    // Access the underlying solver
    [[nodiscard]] const Solver& solver() const { return *solver_; }
    [[nodiscard]] Solver& solver() { return *solver_; }

private:
    SolverPtr solver_;
    int max_recreate_;
    int num_samples_;
    bool verbose_;
};

// State for SolverRecreate
struct SolverRecreateState {
    int iteration{0};
    std::vector<int> indices;
    TreeParamsSoA new_params;
    TreeParamsSoA sample_params;  // For multi-start sampling
};

// SolverOptimize: selects random valid trees and uses a Solver to find better placements
// Unlike SolverRecreate, this doesn't require a ruin step first - it directly
// optimizes existing tree placements.
class SolverOptimize : public Optimizer {
public:
    /**
     * Create an optimizer that uses a solver to improve existing placements.
     *
     * @param solver The solver to use (ownership transferred)
     * @param max_optimize Maximum number of trees to optimize per step
     * @param group_size Number of trees per solver call (1 or 2)
     * @param same_cell_pairs If true and group_size==2, select pairs from the same grid cell
     * @param num_samples Number of solver runs per tree (multi-start)
     * @param verbose Print debug info
     */
    explicit SolverOptimize(
        SolverPtr solver,
        int max_optimize = 1,
        int group_size = 1,
        bool same_cell_pairs = false,
        int num_samples = 8,
        bool verbose = false
    );

    void set_problem(Problem* problem) override {
        Optimizer::set_problem(problem);
        if (solver_) {
            solver_->set_problem(problem);
        }
    }

    std::any init_state(const SolutionEval& solution) override;

    void apply(
        SolutionEval& solution,
        std::any& state,
        GlobalState& global_state,
        RNG& rng
    ) override;

    [[nodiscard]] OptimizerPtr clone() const override;

    // Access the underlying solver
    [[nodiscard]] const Solver& solver() const { return *solver_; }
    [[nodiscard]] Solver& solver() { return *solver_; }

private:
    SolverPtr solver_;
    int max_optimize_;
    int group_size_;
    bool same_cell_pairs_;
    int num_samples_;
    bool verbose_;
};

// State for SolverOptimize
struct SolverOptimizeState {
    int iteration{0};
    std::vector<int> indices;
    std::vector<int> valid_indices;  // Scratch buffer for valid tree indices
    std::vector<std::pair<int, int>> candidate_cells; // Cells with >=2 eligible trees
    std::vector<Index> cell_items;   // Scratch buffer for cell contents
    std::vector<int> cell_valid;     // Scratch buffer for eligible trees in a cell
    std::vector<char> used_flags;    // Tracks trees already selected this step
    TreeParamsSoA old_params;        // For rollback
    TreeParamsSoA new_params;
};

}  // namespace tree_packing
