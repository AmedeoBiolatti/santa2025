/**
 * Example demonstrating the Solver interface with SolverRecreate optimizer.
 *
 * This example shows how to:
 * 1. Create a ParticleSwarmSolver
 * 2. Wrap it in a SolverRecreate optimizer
 * 3. Use it in an optimization loop with ruin-and-recreate
 */

#include "tree_packing/tree_packing.hpp"
#include <iostream>
#include <iomanip>

using namespace tree_packing;

int main() {
    std::cout << "=== Solver Example ===\n\n";

    // Create problem
    Problem problem = Problem::create_tree_packing_problem();

    // Initialize solution with some trees
    constexpr int num_trees = 30;
    constexpr float side = 6.0f;
    constexpr uint64_t seed = 42;

    Solution initial = Solution::init_random(num_trees, side, seed);
    SolutionEval eval = problem.eval(initial);

    std::cout << "Initial solution:\n"
              << "  Trees: " << num_trees << "\n"
              << "  Missing: " << eval.n_missing() << "\n"
              << "  Objective: " << eval.objective << "\n"
              << "  Intersection violation: " << eval.intersection_violation << "\n"
              << "  Bounds violation: " << eval.bounds_violation << "\n\n";

    // Create PSO solver with custom config
    ParticleSwarmSolver::Config pso_config;
    pso_config.n_particles = 32;
    pso_config.n_iterations = 30;
    pso_config.w = 0.7f;
    pso_config.c1 = 1.5f;
    pso_config.c2 = 1.5f;
    pso_config.mu0 = 1.0f;
    pso_config.mu_max = 1e4f;

    auto pso_solver = std::make_unique<ParticleSwarmSolver>(pso_config);
    pso_solver->set_bounds(-8.0f, 8.0f);

    // Create SolverRecreate optimizer
    auto solver_recreate = std::make_unique<SolverRecreate>(
        std::move(pso_solver),
        /*max_recreate=*/2,
        /*num_samples=*/8,
        /*verbose=*/false
    );
    solver_recreate->set_problem(&problem);

    // Create ruin operator
    auto random_ruin = std::make_unique<RandomRuin>(
        /*max_ruin=*/3,
        /*verbose=*/false
    );
    random_ruin->set_problem(&problem);

    // Create global state
    GlobalState global_state(seed, eval);
    global_state.set_mu(1e4f);

    // Initialize optimizer states
    std::any ruin_state = random_ruin->init_state(eval);
    std::any recreate_state = solver_recreate->init_state(eval);

    std::cout << "Running optimization with PSO-based recreate...\n\n";

    // Run optimization loop
    constexpr int n_iterations = 50;
    for (int iter = 0; iter < n_iterations; ++iter) {
        RNG rng(global_state.split_rng());

        // Ruin: remove some trees
        random_ruin->apply(eval, ruin_state, global_state, rng);

        // Recreate: use PSO solver to find good placements
        RNG recreate_rng = rng.split();
        solver_recreate->apply(eval, recreate_state, global_state, recreate_rng);

        // Update best solution
        global_state.maybe_update_best(problem, eval);
        global_state.next();

        // Print progress
        if ((iter + 1) % 10 == 0) {
            float score = problem.score(eval, global_state);
            std::cout << "Iter " << std::setw(3) << (iter + 1) << ": "
                      << "obj=" << std::fixed << std::setprecision(4) << eval.objective
                      << " int_viol=" << eval.intersection_violation
                      << " score=" << score
                      << " best=" << global_state.best_score()
                      << "\n";
        }
    }

    std::cout << "\nFinal results:\n"
              << "  Best score: " << global_state.best_score() << "\n"
              << "  Best feasible score: " << global_state.best_feasible_score() << "\n"
              << "  Final objective: " << eval.objective << "\n"
              << "  Final violation: " << eval.total_violation() << "\n";

    // Compare with random recreate
    std::cout << "\n=== Comparison with RandomRecreate ===\n\n";

    // Reset solution
    Solution initial2 = Solution::init_random(num_trees, side, seed);
    SolutionEval eval2 = problem.eval(initial2);
    GlobalState global_state2(seed, eval2);
    global_state2.set_mu(1e4f);

    auto random_recreate = std::make_unique<RandomRecreate>(
        /*max_recreate=*/2,
        /*box_size=*/side,
        /*delta=*/0.5f,
        /*verbose=*/false
    );
    random_recreate->set_problem(&problem);

    auto random_ruin2 = std::make_unique<RandomRuin>(3, false);
    random_ruin2->set_problem(&problem);

    std::any ruin_state2 = random_ruin2->init_state(eval2);
    std::any recreate_state2 = random_recreate->init_state(eval2);

    std::cout << "Running optimization with Random recreate...\n\n";

    for (int iter = 0; iter < n_iterations; ++iter) {
        RNG rng(global_state2.split_rng());

        random_ruin2->apply(eval2, ruin_state2, global_state2, rng);

        RNG recreate_rng = rng.split();
        random_recreate->apply(eval2, recreate_state2, global_state2, recreate_rng);

        global_state2.maybe_update_best(problem, eval2);
        global_state2.next();

        if ((iter + 1) % 10 == 0) {
            float score = problem.score(eval2, global_state2);
            std::cout << "Iter " << std::setw(3) << (iter + 1) << ": "
                      << "obj=" << std::fixed << std::setprecision(4) << eval2.objective
                      << " int_viol=" << eval2.intersection_violation
                      << " score=" << score
                      << " best=" << global_state2.best_score()
                      << "\n";
        }
    }

    std::cout << "\nRandom recreate results:\n"
              << "  Best score: " << global_state2.best_score() << "\n"
              << "  Best feasible score: " << global_state2.best_feasible_score() << "\n";

    std::cout << "\n=== Summary (Ruin+Recreate) ===\n"
              << "PSO-based recreate best: " << global_state.best_score() << "\n"
              << "Random recreate best:    " << global_state2.best_score() << "\n";

    // =========================================================================
    // Test SolverOptimize (direct optimization without ruin)
    // =========================================================================
    std::cout << "\n=== SolverOptimize (direct optimization) ===\n\n";

    // Reset solution
    Solution initial3 = Solution::init_random(num_trees, side, seed);
    SolutionEval eval3 = problem.eval(initial3);
    GlobalState global_state3(seed, eval3);
    global_state3.set_mu(1e4f);

    std::cout << "Initial solution:\n"
              << "  Objective: " << eval3.objective << "\n"
              << "  Intersection violation: " << eval3.intersection_violation << "\n\n";

    // Create PSO solver for SolverOptimize
    ParticleSwarmSolver::Config pso_config3;
    pso_config3.n_particles = 24;
    pso_config3.n_iterations = 20;

    auto pso_solver3 = std::make_unique<ParticleSwarmSolver>(pso_config3);
    pso_solver3->set_bounds(-8.0f, 8.0f);

    // Create SolverOptimize - directly optimizes random trees without ruin
    auto solver_optimize = std::make_unique<SolverOptimize>(
        std::move(pso_solver3),
        /*max_optimize=*/3,   // Optimize 3 trees per iteration
        /*num_samples=*/4,
        /*verbose=*/false
    );
    solver_optimize->set_problem(&problem);

    std::any optimize_state = solver_optimize->init_state(eval3);

    std::cout << "Running SolverOptimize (no ruin step needed)...\n\n";

    for (int iter = 0; iter < n_iterations; ++iter) {
        RNG rng(global_state3.split_rng());

        // Directly optimize random trees - no ruin needed!
        solver_optimize->apply(eval3, optimize_state, global_state3, rng);

        global_state3.maybe_update_best(problem, eval3);
        global_state3.next();

        if ((iter + 1) % 10 == 0) {
            float score = problem.score(eval3, global_state3);
            std::cout << "Iter " << std::setw(3) << (iter + 1) << ": "
                      << "obj=" << std::fixed << std::setprecision(4) << eval3.objective
                      << " int_viol=" << eval3.intersection_violation
                      << " score=" << score
                      << " best=" << global_state3.best_score()
                      << "\n";
        }
    }

    std::cout << "\nSolverOptimize results:\n"
              << "  Best score: " << global_state3.best_score() << "\n"
              << "  Final objective: " << eval3.objective << "\n"
              << "  Final violation: " << eval3.total_violation() << "\n";

    std::cout << "\n=== Final Summary ===\n"
              << "Ruin + PSO Recreate best:  " << global_state.best_score() << "\n"
              << "Ruin + Random Recreate:    " << global_state2.best_score() << "\n"
              << "SolverOptimize (no ruin):  " << global_state3.best_score() << "\n";

    return 0;
}
