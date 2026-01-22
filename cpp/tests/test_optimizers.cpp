#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "tree_packing/tree_packing.hpp"

using namespace tree_packing;
using Catch::Approx;

TEST_CASE("RNG", "[optimizers]") {
    SECTION("Reproducibility") {
        RNG rng1(42);
        RNG rng2(42);

        for (int i = 0; i < 100; ++i) {
            REQUIRE(rng1.next() == rng2.next());
        }
    }

    SECTION("Uniform distribution") {
        RNG rng(123);
        float sum = 0.0f;
        int n = 10000;

        for (int i = 0; i < n; ++i) {
            float v = rng.uniform();
            REQUIRE(v >= 0.0f);
            REQUIRE(v < 1.0f);
            sum += v;
        }

        // Mean should be close to 0.5
        float mean = sum / n;
        REQUIRE(mean == Approx(0.5f).margin(0.05f));
    }

    SECTION("Permutation") {
        RNG rng(42);
        auto perm = rng.permutation(10);

        REQUIRE(perm.size() == 10);

        // Check that all elements are present
        std::vector<bool> found(10, false);
        for (int i : perm) {
            REQUIRE(i >= 0);
            REQUIRE(i < 10);
            found[i] = true;
        }
        for (bool f : found) {
            REQUIRE(f);
        }
    }

    SECTION("Weighted choice") {
        RNG rng(42);
        std::vector<float> weights = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<int> counts(4, 0);

        int n = 10000;
        for (int i = 0; i < n; ++i) {
            int idx = rng.weighted_choice(weights);
            REQUIRE(idx >= 0);
            REQUIRE(idx < 4);
            counts[idx]++;
        }

        // Check rough proportions (higher weights should have more counts)
        REQUIRE(counts[3] > counts[2]);
        REQUIRE(counts[2] > counts[1]);
        REQUIRE(counts[1] > counts[0]);
    }
}

TEST_CASE("Problem", "[optimizers]") {
    SECTION("Create problem") {
        Problem problem = Problem::create_tree_packing_problem();
        REQUIRE(problem.min_pos() < 0);
        REQUIRE(problem.max_pos() > 0);
    }

    SECTION("Evaluate solution") {
        Problem problem = Problem::create_tree_packing_problem();
        Solution sol = Solution::init_random(10, 5.0f, 42);

        SolutionEval eval = problem.eval(sol);

        REQUIRE(eval.objective >= 0.0f);
        REQUIRE(eval.intersection_violation >= 0.0f);
        REQUIRE(eval.bounds_violation >= 0.0f);
    }

    SECTION("Objective decreases with tighter packing") {
        Problem problem = Problem::create_tree_packing_problem();

        // Wide spread
        Solution sol1 = Solution::init_random(10, 20.0f, 42);
        // Tighter
        Solution sol2 = Solution::init_random(10, 5.0f, 43);

        float obj1 = problem.objective(sol1);
        float obj2 = problem.objective(sol2);

        // Smaller spread should give smaller objective
        REQUIRE(obj2 < obj1);
    }
}

TEST_CASE("RandomRuin", "[optimizers]") {
    SECTION("Removes specified number of trees") {
        Problem problem = Problem::create_tree_packing_problem();
        Solution sol = Solution::init_random(10, 5.0f, 42);
        SolutionEval eval = problem.eval(sol);

        RandomRuin ruin(3);
        ruin.set_problem(&problem);

        RNG rng(42);
        std::any state;
        GlobalState global_state(42);

        SolutionEval new_eval;
        ruin.apply(eval, state, global_state, rng, new_eval);

        REQUIRE(new_eval.n_missing() == 3);
    }
}

TEST_CASE("RandomRecreate", "[optimizers]") {
    SECTION("Recreates removed trees") {
        Problem problem = Problem::create_tree_packing_problem();
        Solution sol = Solution::init_random(10, 5.0f, 42);

        // Remove some trees
        sol.set_nan(2);
        sol.set_nan(5);
        sol.set_nan(7);

        SolutionEval eval = problem.eval(sol);
        REQUIRE(eval.n_missing() == 3);

        RandomRecreate recreate(2);
        recreate.set_problem(&problem);

        RNG rng(42);
        std::any state = recreate.init_state(eval);
        GlobalState global_state(42);

        SolutionEval new_eval;
        recreate.apply(eval, state, global_state, rng, new_eval);

        // Should have recreated 2 trees
        REQUIRE(new_eval.n_missing() == 1);
    }
}

TEST_CASE("ALNS", "[optimizers]") {
    SECTION("Single iteration") {
        Problem problem = Problem::create_tree_packing_problem();
        Solution sol = Solution::init_random(10, 5.0f, 42);
        SolutionEval eval = problem.eval(sol);

        std::vector<OptimizerPtr> ruin_ops;
        std::vector<OptimizerPtr> recreate_ops;
        ruin_ops.push_back(std::make_unique<RandomRuin>(1));
        recreate_ops.push_back(std::make_unique<RandomRecreate>(1));

        ALNS alns(std::move(ruin_ops), std::move(recreate_ops));
        alns.set_problem(&problem);

        std::any state = alns.init_state(eval);
        GlobalState global_state(42);
        RNG rng(42);

        SolutionEval new_eval;
        alns.apply(eval, state, global_state, rng, new_eval);

        // Should have same number of trees (ruin + recreate)
        REQUIRE(new_eval.n_missing() == 0);
    }
}

TEST_CASE("SimulatedAnnealing", "[optimizers]") {
    SECTION("Temperature decrease") {
        Problem problem = Problem::create_tree_packing_problem();
        Solution sol = Solution::init_random(10, 5.0f, 42);
        SolutionEval eval = problem.eval(sol);

        std::vector<OptimizerPtr> ruin_ops;
        std::vector<OptimizerPtr> recreate_ops;
        ruin_ops.push_back(std::make_unique<RandomRuin>(1));
        recreate_ops.push_back(std::make_unique<RandomRecreate>(1));

        auto alns = std::make_unique<ALNS>(std::move(ruin_ops), std::move(recreate_ops));
        SimulatedAnnealing sa(std::move(alns), 1000.0f, 1e-6f,
            CoolingSchedule::Exponential, 0.9f);
        sa.set_problem(&problem);

        std::any state = sa.init_state(eval);
        GlobalState global_state(42);

        // Run several iterations
        for (int i = 0; i < 10; ++i) {
            RNG rng(global_state.split_rng());
            SolutionEval new_eval;
            sa.apply(eval, state, global_state, rng, new_eval);
            eval = new_eval;
        }

        // Check that state has updated temperature
        auto sa_state = std::any_cast<SAState>(state);
        REQUIRE(sa_state.temperature < 1000.0f);
        REQUIRE(sa_state.iteration == 10);
    }
}

TEST_CASE("Integration test", "[optimizers]") {
    SECTION("Full optimization run") {
        Problem problem = Problem::create_tree_packing_problem();
        Solution sol = Solution::init_random(20, 8.0f, 42);
        SolutionEval eval = problem.eval(sol);

        std::vector<OptimizerPtr> ruin_ops;
        std::vector<OptimizerPtr> recreate_ops;
        ruin_ops.push_back(std::make_unique<RandomRuin>(1));
        recreate_ops.push_back(std::make_unique<SpatialRuin>(2));
        recreate_ops.push_back(std::make_unique<RandomRecreate>(1));
        recreate_ops.push_back(std::make_unique<RandomRecreate>(2));

        auto alns = std::make_unique<ALNS>(std::move(ruin_ops), std::move(recreate_ops));
        SimulatedAnnealing sa(std::move(alns), 1000.0f, 1e-6f,
            CoolingSchedule::Exponential, 0.995f);
        sa.set_problem(&problem);

        std::any state = sa.init_state(eval);
        GlobalState global_state(42, eval);

        float initial_score = global_state.best_score();

        // Run optimization
        for (int i = 0; i < 100; ++i) {
            RNG rng(global_state.split_rng());
            SolutionEval new_eval;
            sa.apply(eval, state, global_state, rng, new_eval);

            global_state.maybe_update_best(problem, new_eval);
            global_state.next();

            eval = new_eval;
        }

        // Score should improve or stay same (not necessarily improve in 100 iters)
        REQUIRE(global_state.best_score() <= initial_score + 1e-6f);
    }
}
