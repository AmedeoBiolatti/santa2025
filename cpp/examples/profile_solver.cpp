/**
 * Profiling script for SolverOptimize and ParticleSwarmSolver.
 *
 * Measures timing for different configurations and compares
 * cell-constrained vs global search performance.
 */

#include "tree_packing/tree_packing.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>

using namespace tree_packing;
using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::milli>;

struct ProfileResult {
    double setup_time_ms{0};
    double solver_creation_time_ms{0};
    double total_optimization_time_ms{0};
    double avg_iteration_time_ms{0};
    double min_iteration_time_ms{0};
    double max_iteration_time_ms{0};
    double iterations_per_second{0};
    int total_solver_calls{0};
    double solver_calls_per_second{0};
    float initial_objective{0};
    float initial_violation{0};
    float final_objective{0};
    float final_violation{0};
    float best_score{0};
};

ProfileResult profile_solver_optimize(
    int n_trees,
    float side,
    int n_iterations,
    int n_particles,
    int pso_iterations,
    int max_optimize,
    int num_samples,
    bool constrain_to_cell,
    uint64_t seed
) {
    ProfileResult result;

    // Setup
    auto t0 = Clock::now();
    Problem problem = Problem::create_tree_packing_problem();
    Solution solution = Solution::init_random(n_trees, side, seed);
    SolutionEval eval = problem.eval(solution);
    result.setup_time_ms = Duration(Clock::now() - t0).count();
    result.initial_objective = eval.objective;
    result.initial_violation = eval.intersection_violation;

    // Create PSO config
    ParticleSwarmSolver::Config config;
    config.n_particles = n_particles;
    config.n_iterations = pso_iterations;
    config.constrain_to_cell = constrain_to_cell;
    config.mu_max = 1e5f;

    // Create solver and optimizer
    t0 = Clock::now();
    auto pso = std::make_unique<ParticleSwarmSolver>(config);
    pso->set_bounds(-side, side);

    auto solver_opt = std::make_unique<SolverOptimize>(
        std::move(pso),
        max_optimize,
        num_samples,
        false  // verbose
    );
    solver_opt->set_problem(&problem);
    result.solver_creation_time_ms = Duration(Clock::now() - t0).count();

    // Initialize state
    GlobalState global_state(seed, eval);
    global_state.set_mu(1e4f);
    std::any opt_state = solver_opt->init_state(eval);

    // Profile optimization loop
    std::vector<double> iteration_times;
    iteration_times.reserve(n_iterations);

    auto t_total_start = Clock::now();
    for (int i = 0; i < n_iterations; ++i) {
        auto t_iter_start = Clock::now();

        RNG rng(global_state.split_rng());
        solver_opt->apply(eval, opt_state, global_state, rng);
        global_state.maybe_update_best(problem, eval);
        global_state.next();

        iteration_times.push_back(Duration(Clock::now() - t_iter_start).count());
    }
    result.total_optimization_time_ms = Duration(Clock::now() - t_total_start).count();

    // Compute timing stats
    result.avg_iteration_time_ms = std::accumulate(iteration_times.begin(), iteration_times.end(), 0.0) / iteration_times.size();
    result.min_iteration_time_ms = *std::min_element(iteration_times.begin(), iteration_times.end());
    result.max_iteration_time_ms = *std::max_element(iteration_times.begin(), iteration_times.end());
    result.iterations_per_second = n_iterations / (result.total_optimization_time_ms / 1000.0);

    // Throughput
    result.total_solver_calls = n_iterations * max_optimize * num_samples;
    result.solver_calls_per_second = result.total_solver_calls / (result.total_optimization_time_ms / 1000.0);

    // Final metrics
    result.final_objective = eval.objective;
    result.final_violation = eval.total_violation();
    result.best_score = global_state.best_score();

    return result;
}

void print_result(const ProfileResult& r, const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << std::setw(40) << title << "\n";
    std::cout << std::string(60, '=') << "\n";

    std::cout << "\n[Setup]\n";
    std::cout << "  Setup time:           " << std::fixed << std::setprecision(2) << r.setup_time_ms << " ms\n";
    std::cout << "  Solver creation:      " << r.solver_creation_time_ms << " ms\n";
    std::cout << "  Initial objective:    " << std::setprecision(4) << r.initial_objective << "\n";
    std::cout << "  Initial violation:    " << r.initial_violation << "\n";

    std::cout << "\n[Timing]\n";
    std::cout << "  Total optimization:   " << std::setprecision(1) << r.total_optimization_time_ms << " ms\n";
    std::cout << "  Avg iteration:        " << std::setprecision(3) << r.avg_iteration_time_ms << " ms\n";
    std::cout << "  Min iteration:        " << r.min_iteration_time_ms << " ms\n";
    std::cout << "  Max iteration:        " << r.max_iteration_time_ms << " ms\n";
    std::cout << "  Iterations/sec:       " << std::setprecision(1) << r.iterations_per_second << "\n";

    std::cout << "\n[Throughput]\n";
    std::cout << "  Total solver calls:   " << r.total_solver_calls << "\n";
    std::cout << "  Solver calls/sec:     " << std::setprecision(1) << r.solver_calls_per_second << "\n";

    std::cout << "\n[Results]\n";
    std::cout << "  Final objective:      " << std::setprecision(4) << r.final_objective << "\n";
    std::cout << "  Final violation:      " << r.final_violation << "\n";
    std::cout << "  Best score:           " << std::setprecision(2) << r.best_score << "\n";
    std::cout << std::string(60, '=') << "\n";
}

int main() {
    std::cout << "=== SolverOptimize Profiling ===\n";

    // Configuration (reduced for faster profiling)
    constexpr int n_trees = 30;
    constexpr float side = 5.0f;
    constexpr int n_iterations = 20;
    constexpr int n_particles = 16;
    constexpr int pso_iterations = 20;
    constexpr int max_optimize = 2;
    constexpr int num_samples = 2;
    constexpr uint64_t seed = 42;

    std::cout << "\nConfiguration:\n";
    std::cout << "  Trees:          " << n_trees << "\n";
    std::cout << "  Iterations:     " << n_iterations << "\n";
    std::cout << "  PSO particles:  " << n_particles << "\n";
    std::cout << "  PSO iterations: " << pso_iterations << "\n";
    std::cout << "  Max optimize:   " << max_optimize << "\n";
    std::cout << "  Num samples:    " << num_samples << "\n";

    // Warmup
    std::cout << "\nWarming up...\n";
    profile_solver_optimize(10, 4.0f, 5, 8, 5, 1, 1, true, seed);

    // Profile cell-constrained (fast)
    std::cout << "\nProfiling CELL-CONSTRAINED search...\n";
    auto result_cell = profile_solver_optimize(
        n_trees, side, n_iterations,
        n_particles, pso_iterations,
        max_optimize, num_samples,
        true,  // constrain_to_cell
        seed
    );
    print_result(result_cell, "Cell-Constrained PSO");

    // Profile global search (slower)
    std::cout << "\nProfiling GLOBAL search...\n";
    auto result_global = profile_solver_optimize(
        n_trees, side, n_iterations,
        n_particles, pso_iterations,
        max_optimize, num_samples,
        false,  // constrain_to_cell
        seed
    );
    print_result(result_global, "Global Search PSO");

    // Comparison
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << std::setw(40) << "COMPARISON" << "\n";
    std::cout << std::string(60, '=') << "\n";

    double speedup = result_global.total_optimization_time_ms / result_cell.total_optimization_time_ms;
    std::cout << "\nCell-constrained speedup: " << std::fixed << std::setprecision(2) << speedup << "x faster\n";
    std::cout << "Cell-constrained:  " << std::setprecision(1) << result_cell.iterations_per_second << " iter/s, "
              << result_cell.solver_calls_per_second << " solver calls/s\n";
    std::cout << "Global search:     " << result_global.iterations_per_second << " iter/s, "
              << result_global.solver_calls_per_second << " solver calls/s\n";

    std::cout << "\nTime per solver call:\n";
    std::cout << "  Cell-constrained: " << std::setprecision(3)
              << (result_cell.total_optimization_time_ms / result_cell.total_solver_calls) << " ms\n";
    std::cout << "  Global search:    "
              << (result_global.total_optimization_time_ms / result_global.total_solver_calls) << " ms\n";

    std::cout << std::string(60, '=') << "\n";

    return 0;
}
