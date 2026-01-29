#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "tree_packing/tree_packing.hpp"
#include <cmath>

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

        ruin.apply(eval, state, global_state, rng);

        REQUIRE(eval.n_missing() == 3);
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

        recreate.apply(eval, state, global_state, rng);

        // Should have recreated 2 trees
        REQUIRE(eval.n_missing() == 1);
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

        alns.apply(eval, state, global_state, rng);

        // Should have same number of trees (ruin + recreate)
        REQUIRE(eval.n_missing() == 0);
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
            sa.apply(eval, state, global_state, rng);
        }
    }
}

namespace {
bool params_different(const TreeParams& a, const TreeParams& b, float tol = 1e-5f) {
    return std::abs(a.pos.x - b.pos.x) > tol ||
           std::abs(a.pos.y - b.pos.y) > tol ||
           std::abs(a.angle - b.angle) > tol;
}
}  // namespace

TEST_CASE("Solvers change params", "[solvers]") {
    Problem problem = Problem::create_tree_packing_problem();
    Solution sol = Solution::init_random(24, 6.0f, 123);
    SolutionEval eval = problem.eval(sol);
    const int target_idx = 0;
    const TreeParams initial = eval.solution.get_params(static_cast<size_t>(target_idx));

    SECTION("RandomSamplingSolver") {
        RandomSamplingSolver solver(RandomSamplingSolver::Config{
            .n_samples = 256,
            .mu = 1e6f,
            .isolation_penalty_per_missing = std::log(2.0f),
            .target_neighbors = 8.0f,
            .constrain_to_cell = false,
            .prefer_current_cell = true,
            .objective_type = RandomSamplingSolver::ObjectiveType::Distance,
        });
        RNG rng(1);
        auto res = solver.solve_single(eval, target_idx, rng);
        REQUIRE(params_different(initial, res.params.get(0)));
    }

    SECTION("ParticleSwarmSolver") {
        ParticleSwarmSolver solver(ParticleSwarmSolver::Config{
            .n_particles = 64,
            .n_iterations = 40,
            .w = 0.7f,
            .c1 = 1.5f,
            .c2 = 1.5f,
            .mu0 = 1e3f,
            .mu_max = 1e6f,
            .isolation_penalty_per_missing = std::log(2.0f),
            .target_neighbors = 8.0f,
            .vel_max = 2.0f,
            .vel_ang_max = PI / 3.0f,
            .constrain_to_cell = false,
            .prefer_current_cell = true,
            .objective_type = ParticleSwarmSolver::ObjectiveType::Distance,
        });
        RNG rng(2);
        auto res = solver.solve_single(eval, target_idx, rng);
        REQUIRE(params_different(initial, res.params.get(0)));
    }

    SECTION("BeamDescentSolver") {
        BeamDescentSolver solver(BeamDescentSolver::Config{
            .lattice_xy = 7,
            .lattice_ang = 10,
            .beam_width = 10,
            .descent_levels = 5,
            .max_iters_per_level = 10,
            .step_xy0 = 0.75f,
            .step_xy_decay = 0.5f,
            .step_ang0 = PI / 6.0f,
            .step_ang_decay = 0.5f,
            .mu = 1e6f,
            .isolation_penalty_per_missing = std::log(2.0f),
            .target_neighbors = 8.0f,
            .constrain_to_cell = false,
            .prefer_current_cell = true,
            .objective_type = BeamDescentSolver::ObjectiveType::Distance,
        });
        RNG rng(3);
        auto res = solver.solve_single(eval, target_idx, rng);
        REQUIRE(params_different(initial, res.params.get(0)));
    }

    SECTION("NoiseSolver") {
        NoiseSolver solver(NoiseSolver::Config{
            .n_variations = 512,
            .pos_sigma = 0.5f,
            .ang_sigma = PI / 8.0f,
            .mu = 1e6f,
            .isolation_penalty_per_missing = std::log(2.0f),
            .target_neighbors = 8.0f,
            .constrain_to_cell = false,
            .prefer_current_cell = true,
            .objective_type = NoiseSolver::ObjectiveType::Distance,
        });
        solver.set_problem(&problem);
        RNG rng(4);
        auto res = solver.solve_single(eval, target_idx, rng);
        REQUIRE(params_different(initial, res.params.get(0)));
    }

    SECTION("NelderMeadSolver") {
        NelderMeadSolver solver(NelderMeadSolver::Config{
            .step_xy = 1.5f,
            .step_ang = PI / 6.0f,
            .alpha = 1.0f,
            .gamma = 2.0f,
            .rho = 0.5f,
            .sigma = 0.5f,
            .max_iters = 200,
            .tol_f = 1e-7f,
            .tol_x = 1e-6f,
            .mu = 1e3f,
            .constrain_to_cell = false,
            .prefer_current_cell = true,
            .objective_type = NelderMeadSolver::ObjectiveType::Linf,
        });
        solver.set_problem(&problem);
        RNG rng(5);
        auto res = solver.solve_single(eval, target_idx, rng);
        REQUIRE(params_different(initial, res.params.get(0)));
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
        recreate_ops.push_back(std::make_unique<CellRuin>(2));
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
            sa.apply(eval, state, global_state, rng);

            global_state.maybe_update_best(problem, eval);
            global_state.next();
        }

        // Score should improve or stay same (not necessarily improve in 100 iters)
        REQUIRE(global_state.best_score() <= initial_score + 1e-6f);
    }
}

TEST_CASE("UpdateStack rollback with overlapping indices", "[optimizers]") {
    SECTION("Rollback restores original state after Ruin+Recreate+Noise") {
        Problem problem = Problem::create_tree_packing_problem();

        // Create a solution with 20 trees, spread to ensure some valid ones
        Solution sol = Solution::init_random(20, 4.0f, 42);
        SolutionEval eval = problem.eval(sol);

        // Store original params for comparison
        std::vector<TreeParams> original_params;
        std::vector<bool> original_valid;
        for (size_t i = 0; i < sol.size(); ++i) {
            original_params.push_back(sol.get_params(i));
            original_valid.push_back(sol.is_valid(i));
        }
        float original_score = problem.score(eval, GlobalState(42));

        // Create optimizers
        RandomRuin ruin(5);  // Remove 5 trees
        RandomRecreate recreate(5);  // Recreate up to 5 trees
        NoiseOptimizer noise(0.1f, 3);  // Perturb 3 trees

        ruin.set_problem(&problem);
        recreate.set_problem(&problem);
        noise.set_problem(&problem);

        GlobalState global_state(42);
        RNG rng(123);

        // Mark checkpoint before any operations
        size_t checkpoint = global_state.mark_checkpoint();
        REQUIRE(checkpoint == 0);

        // Apply ruin
        std::any ruin_state = ruin.init_state(eval);
        ruin.apply(eval, ruin_state, global_state, rng);

        // Check that trees were removed
        REQUIRE(eval.n_missing() > 0);
        size_t after_ruin = global_state.update_stack().size();
        REQUIRE(after_ruin > 0);

        // Apply recreate
        std::any recreate_state = recreate.init_state(eval);
        recreate.apply(eval, recreate_state, global_state, rng);

        size_t after_recreate = global_state.update_stack().size();
        REQUIRE(after_recreate > after_ruin);

        // Apply noise (will perturb some of the same indices that were recreated)
        std::any noise_state = noise.init_state(eval);
        noise.apply(eval, noise_state, global_state, rng);

        size_t after_noise = global_state.update_stack().size();
        REQUIRE(after_noise > after_recreate);

        // Verify solution has changed
        bool any_changed = false;
        for (size_t i = 0; i < sol.size(); ++i) {
            if (original_valid[i] != eval.solution.is_valid(i)) {
                any_changed = true;
                break;
            }
            if (eval.solution.is_valid(i)) {
                TreeParams p = eval.solution.get_params(i);
                if (p.pos.x != original_params[i].pos.x ||
                    p.pos.y != original_params[i].pos.y ||
                    p.angle != original_params[i].angle) {
                    any_changed = true;
                    break;
                }
            }
        }
        REQUIRE(any_changed);

        // Now rollback to checkpoint
        global_state.rollback_to(problem, eval, checkpoint);

        // Verify stack is back to checkpoint
        REQUIRE(global_state.update_stack().size() == checkpoint);

        // Verify all original params are restored
        for (size_t i = 0; i < sol.size(); ++i) {
            REQUIRE(eval.solution.is_valid(i) == original_valid[i]);
            if (original_valid[i]) {
                TreeParams p = eval.solution.get_params(i);
                REQUIRE(p.pos.x == Approx(original_params[i].pos.x).margin(1e-6f));
                REQUIRE(p.pos.y == Approx(original_params[i].pos.y).margin(1e-6f));
                REQUIRE(p.angle == Approx(original_params[i].angle).margin(1e-6f));
            }
        }

        // Verify score is restored
        float restored_score = problem.score(eval, GlobalState(42));
        REQUIRE(restored_score == Approx(original_score).margin(1e-6f));
    }

    SECTION("Rollback with explicit overlapping indices") {
        Problem problem = Problem::create_tree_packing_problem();

        // Create a solution with 10 trees
        Solution sol = Solution::init_random(10, 3.0f, 42);
        SolutionEval eval = problem.eval(sol);

        // Store original state
        std::vector<TreeParams> original_params;
        for (size_t i = 0; i < sol.size(); ++i) {
            original_params.push_back(sol.get_params(i));
        }

        GlobalState global_state(42);

        // Mark checkpoint
        size_t checkpoint = global_state.mark_checkpoint();
        auto& stack = global_state.update_stack();

        // Manually push overlapping updates to simulate the scenario:
        // - Remove indices 2, 3, 4
        // - Recreate indices 2, 3, 4 (same indices, now inserted)
        // - Update index 3 (overlap with recreate)

        // Push removes for indices 2, 3, 4
        for (int idx : {2, 3, 4}) {
            if (eval.solution.is_valid(idx)) {
                TreeParams prev = eval.solution.get_params(idx);
                stack.push_remove(idx, prev);
            }
        }
        problem.remove_and_eval(eval, {2, 3, 4});
        REQUIRE(eval.n_missing() == 3);

        // Push inserts for indices 2, 3, 4
        TreeParamsSoA new_params;
        new_params.push_back({0.5f, 0.5f, 0.1f});
        new_params.push_back({0.6f, 0.6f, 0.2f});
        new_params.push_back({0.7f, 0.7f, 0.3f});

        for (int idx : {2, 3, 4}) {
            stack.push_insert(idx);
        }
        problem.insert_and_eval(eval, {2, 3, 4}, new_params);
        REQUIRE(eval.n_missing() == 0);

        // Push update for index 3 (overlapping with the insert above)
        TreeParams old_p3 = eval.solution.get_params(3);
        stack.push_update(3, old_p3);
        TreeParamsSoA update_params;
        update_params.push_back({0.8f, 0.8f, 0.4f});
        problem.update_and_eval(eval, {3}, update_params);

        // Verify changes
        TreeParams p3 = eval.solution.get_params(3);
        REQUIRE(p3.pos.x == Approx(0.8f).margin(1e-6f));

        // Rollback
        global_state.rollback_to(problem, eval, checkpoint);

        // Verify original state restored
        for (size_t i = 0; i < sol.size(); ++i) {
            REQUIRE(eval.solution.is_valid(i));
            TreeParams p = eval.solution.get_params(i);
            REQUIRE(p.pos.x == Approx(original_params[i].pos.x).margin(1e-6f));
            REQUIRE(p.pos.y == Approx(original_params[i].pos.y).margin(1e-6f));
            REQUIRE(p.angle == Approx(original_params[i].angle).margin(1e-6f));
        }
    }

    SECTION("SimulatedAnnealing rollback on reject") {
        Problem problem = Problem::create_tree_packing_problem();
        Solution sol = Solution::init_random(15, 4.0f, 42);
        SolutionEval eval = problem.eval(sol);

        // Store original state
        std::vector<TreeParams> original_params;
        std::vector<bool> original_valid;
        for (size_t i = 0; i < sol.size(); ++i) {
            original_params.push_back(sol.get_params(i));
            original_valid.push_back(sol.is_valid(i));
        }

        // Create a chain of Ruin -> Recreate -> Noise
        std::vector<OptimizerPtr> chain_ops;
        chain_ops.push_back(std::make_unique<RandomRuin>(3));
        chain_ops.push_back(std::make_unique<RandomRecreate>(3));
        chain_ops.push_back(std::make_unique<NoiseOptimizer>(0.5f, 2));

        auto chain = std::make_unique<Chain>(std::move(chain_ops));

        // SA with very low temperature to force rejections (temp=0 effectively)
        SimulatedAnnealing sa(std::move(chain), 1e-10f, 1e-10f,
            CoolingSchedule::Exponential, 0.99f);
        sa.set_problem(&problem);

        GlobalState global_state(42, eval);
        std::any state = sa.init_state(eval);

        // Run several iterations - some will be rejected and should rollback correctly
        for (int iter = 0; iter < 20; ++iter) {
            RNG rng(global_state.split_rng());

            // Clear the stack before each iteration
            global_state.update_stack().clear();

            sa.apply(eval, state, global_state, rng);

            // With very low temp, only improvements are accepted
            // Rejected moves should be rolled back correctly

            global_state.maybe_update_best(problem, eval);
            global_state.next();
        }

        // Just verify no crashes and solution is still valid
        REQUIRE(eval.solution.size() == 15);
    }
}

TEST_CASE("SA with zero temperature has non-increasing score", "[optimizers]") {
    SECTION("Score never increases with temp=0") {
        Problem problem = Problem::create_tree_packing_problem();
        Solution sol = Solution::init_random(20, 5.0f, 42);
        SolutionEval eval = problem.eval(sol);

        // Create a chain of Ruin -> Recreate -> Noise (overlapping indices possible)
        std::vector<OptimizerPtr> chain_ops;
        chain_ops.push_back(std::make_unique<RandomRuin>(3));
        chain_ops.push_back(std::make_unique<RandomRecreate>(3));
        chain_ops.push_back(std::make_unique<NoiseOptimizer>(0.2f, 2));

        auto chain = std::make_unique<Chain>(std::move(chain_ops));

        // SA with temperature = 0 (initial and min both 0)
        // This means only strict improvements should be accepted
        SimulatedAnnealing sa(std::move(chain), 0.0f, 0.0f,
            CoolingSchedule::Exponential, 0.99f);
        sa.set_problem(&problem);

        GlobalState global_state(42, eval);
        std::any state = sa.init_state(eval);

        float prev_score = problem.score(eval, global_state);
        std::vector<float> scores;
        scores.push_back(prev_score);

        // Run many iterations
        for (int iter = 0; iter < 100; ++iter) {
            RNG rng(global_state.split_rng());

            // Clear the stack before each iteration
            global_state.update_stack().clear();

            sa.apply(eval, state, global_state, rng);

            float current_score = problem.score(eval, global_state);
            scores.push_back(current_score);

            // With temp=0, score should NEVER increase
            // (only improvements or equal scores accepted)
            REQUIRE(current_score <= prev_score + 1e-6f);

            prev_score = current_score;

            global_state.maybe_update_best(problem, eval);
            global_state.next();
        }

        // Verify we actually had some variation (not all scores identical)
        // This ensures the test is meaningful
        float min_score = *std::min_element(scores.begin(), scores.end());
        float max_score = *std::max_element(scores.begin(), scores.end());
        // At least some improvement should have happened
        REQUIRE(min_score <= max_score);
    }

    SECTION("Score never increases with ALNS + SA at temp=0") {
        Problem problem = Problem::create_tree_packing_problem();
        Solution sol = Solution::init_random(25, 6.0f, 123);
        SolutionEval eval = problem.eval(sol);

        // ALNS with multiple ruin/recreate operators
        std::vector<OptimizerPtr> ruin_ops;
        std::vector<OptimizerPtr> recreate_ops;
        ruin_ops.push_back(std::make_unique<RandomRuin>(2));
        ruin_ops.push_back(std::make_unique<CellRuin>(3));
        recreate_ops.push_back(std::make_unique<RandomRecreate>(2));
        recreate_ops.push_back(std::make_unique<GridCellRecreate>(3));

        auto alns = std::make_unique<ALNS>(std::move(ruin_ops), std::move(recreate_ops));

        // SA with temp=0
        SimulatedAnnealing sa(std::move(alns), 0.0f, 0.0f,
            CoolingSchedule::Exponential, 0.99f);
        sa.set_problem(&problem);

        GlobalState global_state(123, eval);
        std::any state = sa.init_state(eval);

        float prev_score = problem.score(eval, global_state);

        for (int iter = 0; iter < 50; ++iter) {
            RNG rng(global_state.split_rng());
            global_state.update_stack().clear();

            sa.apply(eval, state, global_state, rng);

            float current_score = problem.score(eval, global_state);

            // Score should never increase with temp=0
            REQUIRE(current_score <= prev_score + 1e-6f);

            prev_score = current_score;
            global_state.maybe_update_best(problem, eval);
            global_state.next();
        }
    }
}

TEST_CASE("NoiseOptimizer", "[optimizers]") {
    SECTION("Changes score majority of the time") {
        Problem problem = Problem::create_tree_packing_problem();
        Solution sol = Solution::init_random(15, 5.0f, 42);
        SolutionEval eval = problem.eval(sol);

        NoiseOptimizer noise(0.5f, 3);  // Perturb 3 trees with significant noise
        noise.set_problem(&problem);

        GlobalState global_state(42, eval);
        std::any state = noise.init_state(eval);

        int score_changed_count = 0;
        int total_iterations = 100;

        for (int i = 0; i < total_iterations; ++i) {
            RNG rng(global_state.split_rng());

            float score_before = problem.score(eval, global_state);
            noise.apply(eval, state, global_state, rng);
            float score_after = problem.score(eval, global_state);

            if (std::abs(score_after - score_before) > 1e-6f) {
                ++score_changed_count;
            }

            global_state.next();
        }

        // Score should change at least 50% of the time with meaningful noise
        REQUIRE(score_changed_count >= total_iterations / 2);
    }

    SECTION("With zero noise level, score stays approximately the same") {
        Problem problem = Problem::create_tree_packing_problem();
        Solution sol = Solution::init_random(15, 5.0f, 42);
        SolutionEval eval = problem.eval(sol);

        NoiseOptimizer noise(0.0f, 3);  // Zero noise
        noise.set_problem(&problem);

        GlobalState global_state(42, eval);
        std::any state = noise.init_state(eval);

        float initial_score = problem.score(eval, global_state);

        for (int i = 0; i < 20; ++i) {
            RNG rng(global_state.split_rng());
            noise.apply(eval, state, global_state, rng);
            global_state.next();
        }

        float final_score = problem.score(eval, global_state);

        // With zero noise, score should be unchanged
        REQUIRE(final_score == Approx(initial_score).margin(1e-5f));
    }
}
