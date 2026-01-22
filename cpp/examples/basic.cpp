#include <iostream>
#include <iomanip>
#include "tree_packing/tree_packing.hpp"

using namespace tree_packing;

int main() {
    std::cout << "Tree Packing Optimization Example\n";
    std::cout << "==================================\n\n";

    // Create problem
    Problem problem = Problem::create_tree_packing_problem();
    std::cout << "Created problem with bounds [" << problem.min_pos()
              << ", " << problem.max_pos() << "]\n\n";

    // Create initial solution
    const int num_trees = 67;
    const float side = 10.0f;
    const uint64_t seed = 42;

    Solution initial = Solution::init_random(num_trees, side, seed);
    SolutionEval eval = problem.eval(initial);

    std::cout << "Initial solution:\n";
    std::cout << "  Trees: " << num_trees << "\n";
    std::cout << "  Objective: " << eval.objective << "\n";
    std::cout << "  Intersection violation: " << eval.intersection_violation << "\n";
    std::cout << "  Bounds violation: " << eval.bounds_violation << "\n\n";

    // Create optimizers
    std::vector<OptimizerPtr> ruin_ops;
    std::vector<OptimizerPtr> recreate_ops;

    ruin_ops.push_back(std::make_unique<RandomRuin>(1));
    ruin_ops.push_back(std::make_unique<SpatialRuin>(2));
    recreate_ops.push_back(std::make_unique<RandomRecreate>(1));
    recreate_ops.push_back(std::make_unique<RandomRecreate>(2));

    auto alns = std::make_unique<ALNS>(
        std::move(ruin_ops),
        std::move(recreate_ops),
        0.01f,  // reaction_factor
        1.0f,   // reward_improve
        0.0f,   // reward_no_improve
        1e-3f   // min_weight
    );

    SimulatedAnnealing sa(
        std::move(alns),
        1000.0f,    // initial_temp
        1e-6f,      // min_temp
        CoolingSchedule::Exponential,
        0.9995f,    // cooling_rate
        -1,         // patience (no reheating)
        false       // verbose
    );

    sa.set_problem(&problem);

    // Initialize state
    std::any state = sa.init_state(eval);
    GlobalState global_state(seed, eval);
    // Note: Default mu=1e6 is very high. For better SA exploration, consider lower values.
    // global_state.set_mu(1e4f);

    // Run optimization
    const int num_iterations = 5000;
    std::cout << "Running " << num_iterations << " iterations...\n";
    std::cout << std::fixed << std::setprecision(6);

    for (int i = 0; i < num_iterations; ++i) {
        RNG rng(global_state.split_rng());
        sa.apply(eval, state, global_state, rng);
        global_state.maybe_update_best(problem, eval);
        global_state.next();

        // Print progress every 500 iterations
        if ((i + 1) % 500 == 0) {
            std::cout << "  Iteration " << std::setw(5) << (i + 1)
                      << " | Best: " << global_state.best_feasible_score()
                      << " | Obj: " << eval.objective
                      << " | Viol: " << eval.total_violation() << "\n";
        }
    }

    std::cout << "\nOptimization complete!\n";
    std::cout << "  Best overall score: " << global_state.best_score() << "\n";
    std::cout << "  Best feasible score: " << global_state.best_feasible_score() << "\n";
    std::cout << "  Total iterations: " << global_state.iteration() << "\n";

    return 0;
}
