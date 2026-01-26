#include <iostream>
#include <string>
#include <chrono>
#include "tree_packing/tree_packing.hpp"

using namespace tree_packing;

int main(int argc, char** argv) {
    int num_iterations = 50000000;
    if (argc > 1) {
        try {
            num_iterations = std::stoi(argv[1]);
        } catch (...) {
            std::cerr << "Invalid iteration count, using default.\n";
        }
    }

    Problem problem = Problem::create_tree_packing_problem();

    const int num_trees = 32;
    const float side = 6.0f;
    const uint64_t seed = 42;

    Solution initial = Solution::init_random(num_trees, side, seed);
    SolutionEval eval = problem.eval(initial);

    std::cout << "[Initial]\n";
    std::cout << "Objective:       " << eval.objective << "\n";
    std::cout << "Violation:       " << eval.total_violation() << "\n";

    auto make_alns = []() {
        std::vector<OptimizerPtr> ruin_ops;
        std::vector<OptimizerPtr> recreate_ops;

        ruin_ops.push_back(std::make_unique<RandomRuin>(1));
        recreate_ops.push_back(std::make_unique<RandomRecreate>(1));

        return std::make_unique<ALNS>(
            std::move(ruin_ops),
            std::move(recreate_ops),
            0.01f,
            1.0f,
            0.0f,
            1e-3f
        );
    };

    auto benchmark = [&](const std::string& name, Optimizer& optimizer) {
        optimizer.set_problem(&problem);

        SolutionEval eval = problem.eval(initial);
        std::any state = optimizer.init_state(eval);
        GlobalState global_state(seed, eval);

        // Warmup
        optimizer.run(eval, state, global_state, 1000);

        // Reset for actual benchmark
        eval = problem.eval(initial);
        state = optimizer.init_state(eval);
        global_state = GlobalState(seed + 1, eval);

        auto start = std::chrono::high_resolution_clock::now();
        optimizer.run(eval, state, global_state, num_iterations);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double seconds = duration.count() / 1e6;
        double iters_per_sec = num_iterations / seconds;

        std::cout << "\n[" << name << "]\n";
        std::cout << "Iterations:      " << num_iterations << "\n";
        std::cout << "Time:            " << seconds << " s\n";
        std::cout << "Iterations/sec:  " << static_cast<int>(iters_per_sec) << "\n";
        std::cout << "us/iteration:    " << (duration.count() / static_cast<double>(num_iterations)) << "\n";
        std::cout << "Best feasible:   " << global_state.best_feasible_score() << "\n";
        std::cout << "Best score:      " << global_state.best_score() << "\n";
        std::cout << "Final objective: " << eval.objective << "\n";
        std::cout << "Final violation: " << eval.total_violation() << "\n";
    };

    {
        auto alns = make_alns();
        SimulatedAnnealing sa(
            std::move(alns),
            1000.0f,
            1e-6f,
            CoolingSchedule::Exponential,
            0.9995f,
            -1,
            false
        );
        benchmark("SimulatedAnnealing(ALNS)", sa);
    }

    {
        auto alns = make_alns();
        benchmark("ALNS", *alns);
    }

    {
        NoiseOptimizer noise(0.01f, 1, false);
        benchmark("NoiseOptimizer", noise);
    }

    {
        auto noise = std::make_unique<NoiseOptimizer>(0.01f, 1, false);
        SimulatedAnnealing sa(
            std::move(noise),
            1000.0f,
            1e-6f,
            CoolingSchedule::Exponential,
            0.9995f,
            -1,
            false
        );
        benchmark("SimulatedAnnealing(Noise)", sa);
    }

    return 0;
}
