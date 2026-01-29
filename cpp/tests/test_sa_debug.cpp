#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "tree_packing/tree_packing.hpp"
#include "tree_packing/constraints/intersection_tree_filter.hpp"
#include "tree_packing/geometry/sat.hpp"
#include <iostream>
#include <iomanip>
#include <map>
#include <set>

using namespace tree_packing;
using Catch::Approx;

static std::map<std::pair<int,int>, float> build_violation_map_sa(const SolutionEval::IntersectionMap& imap) {
    std::map<std::pair<int,int>, float> result;
    for (size_t i = 0; i < imap.size(); ++i) {
        for (const auto& entry : imap[i]) {
            int j = entry.neighbor;
            if (j >= 0) {
                int a = std::min((int)i, j);
                int b = std::max((int)i, j);
                result[{a, b}] = entry.score;
            }
        }
    }
    return result;
}

TEST_CASE("Debug SA ALNS zero temp", "[debug-sa]") {
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
    auto prev_violations = build_violation_map_sa(eval.intersection_map);

    std::cout << "\n=== INITIAL STATE ===" << std::endl;
    std::cout << "Score: " << prev_score << std::endl;
    std::cout << "Intersection pairs: " << prev_violations.size() << std::endl;

    IntersectionTreeFilter filter;

    for (int iter = 0; iter < 50; ++iter) {
        RNG rng(global_state.split_rng());
        global_state.update_stack().clear();

        sa.apply(eval, state, global_state, rng);

        float current_score = problem.score(eval, global_state);
        auto current_violations = build_violation_map_sa(eval.intersection_map);

        if (current_score > prev_score + 1e-6f) {
            std::cout << std::fixed << std::setprecision(8);
            std::cout << "\n=== SCORE INCREASED at iter " << iter << " ===" << std::endl;
            std::cout << "Prev score: " << prev_score << ", Current score: " << current_score << std::endl;
            std::cout << "Difference: " << (current_score - prev_score) << std::endl;

            // Find differing violations
            std::set<std::pair<int,int>> all_pairs;
            for (const auto& [pair, _] : prev_violations) all_pairs.insert(pair);
            for (const auto& [pair, _] : current_violations) all_pairs.insert(pair);

            for (const auto& pair : all_pairs) {
                float pv = prev_violations.count(pair) ? prev_violations.at(pair) : 0.0f;
                float cv = current_violations.count(pair) ? current_violations.at(pair) : 0.0f;

                if (std::abs(pv - cv) > 1e-6f) {
                    int i = pair.first;
                    int j = pair.second;

                    std::cout << "\nPair (" << i << ", " << j << "):" << std::endl;
                    std::cout << "  Prev violation: " << pv << ", Current: " << cv << std::endl;

                    if (eval.solution.is_valid(i) && eval.solution.is_valid(j)) {
                        TreeParams pa = eval.solution.get_params(i);
                        TreeParams pb = eval.solution.get_params(j);
                        std::cout << "  Tree " << i << ": pos=(" << pa.pos.x << ", " << pa.pos.y << "), angle=" << pa.angle << std::endl;
                        std::cout << "  Tree " << j << ": pos=(" << pb.pos.x << ", " << pb.pos.y << "), angle=" << pb.angle << std::endl;

                        int leaf_ij = filter.leaf_index_for(eval.solution, i, j);
                        int leaf_ji = filter.leaf_index_for(eval.solution, j, i);
                        std::cout << "  Leaf (i,j)=" << leaf_ij << ", Leaf (j,i)=" << leaf_ji << std::endl;

                        std::array<uint8_t, 16> pred_ij, pred_ji;
                        filter.leaf_pred_for(eval.solution, i, j, pred_ij);
                        filter.leaf_pred_for(eval.solution, j, i, pred_ji);

                        std::cout << "  Targets (i,j):";
                        for (int t = 0; t < 16; ++t) if (pred_ij[t]) std::cout << " " << t;
                        std::cout << std::endl;

                        std::cout << "  Targets (j,i):";
                        for (int t = 0; t < 16; ++t) if (pred_ji[t]) std::cout << " " << t;
                        std::cout << std::endl;

                        // Ground truth
                        Figure fig_i = params_to_figure(pa);
                        Figure fig_j = params_to_figure(pb);
                        float ground_truth = 0.0f;
                        for (int ti = 0; ti < 5; ++ti) {
                            for (int tj = 0; tj < 5; ++tj) {
                                float score = triangles_intersection_score(fig_i[ti], fig_j[tj]);
                                if (score > 0.0f) {
                                    ground_truth += score;
                                    std::cout << "  tri[" << ti << "] x tri[" << tj << "] = " << score << std::endl;
                                }
                            }
                        }
                        std::cout << "  Ground truth: " << ground_truth << " (capped: " << std::min(ground_truth, 0.15f) << ")" << std::endl;
                    }
                }
            }

            REQUIRE(current_score <= prev_score + 1e-6f);
        }

        prev_score = current_score;
        prev_violations = current_violations;
        global_state.maybe_update_best(problem, eval);
        global_state.next();
    }

    std::cout << "\n=== ALL 50 ITERATIONS PASSED ===" << std::endl;
}
