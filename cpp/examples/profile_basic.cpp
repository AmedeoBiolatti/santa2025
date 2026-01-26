#include <iostream>
#include <string>
#include "tree_packing/tree_packing.hpp"

#ifdef COUNT_INTERSECTIONS
extern void print_intersection_stats();
#endif

using namespace tree_packing;

int main(int argc, char** argv) {
    int num_iterations = 20000000;
    if (argc > 1) {
        try {
            num_iterations = std::stoi(argv[1]);
        } catch (...) {
            std::cerr << "Invalid iteration count, using default.\n";
        }
    }

    Problem problem = Problem::create_tree_packing_problem();

    const int num_trees = 199;
    const float side = 1.0f;
    const uint64_t seed = 42;

    Solution initial = Solution::init_random(num_trees, side, seed);
    SolutionEval eval = problem.eval(initial);

    std::vector<OptimizerPtr> ruin_ops;
    std::vector<OptimizerPtr> recreate_ops;

    ruin_ops.push_back(std::make_unique<RandomRuin>(1));
    //ruin_ops.push_back(std::make_unique<CellRuin>(2));
    recreate_ops.push_back(std::make_unique<RandomRecreate>(1));
    //recreate_ops.push_back(std::make_unique<RandomRecreate>(2));

    auto alns = std::make_unique<ALNS>(
        std::move(ruin_ops),
        std::move(recreate_ops),
        0.01f,
        1.0f,
        0.0f,
        1e-3f
    );

    SimulatedAnnealing sa(
        std::move(alns),
        1000.0f,
        1e-6f,
        CoolingSchedule::Exponential,
        0.9995f,
        -1,
        false
    );

    sa.set_problem(&problem);

    std::any state = sa.init_state(eval);
    GlobalState global_state(seed, eval);

    sa.run(eval, state, global_state, num_iterations);

    std::cout << "Done: iterations=" << global_state.iteration()
              << " best_feasible=" << global_state.best_feasible_score()
              << " best=" << global_state.best_score()
              << "\n";

#ifdef COUNT_INTERSECTIONS
    print_intersection_stats();
#endif

    return 0;
}
