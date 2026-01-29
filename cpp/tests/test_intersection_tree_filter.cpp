#include <catch2/catch_test_macros.hpp>
#include "tree_packing/core/solution.hpp"
#include "tree_packing/constraints/intersection_tree_filter.hpp"
#include "tree_packing/geometry/sat.hpp"

#include <random>

using namespace tree_packing;

namespace {
constexpr int kTargetCount = 16;
constexpr int kMaxPairsPerTarget = 4;
constexpr int kTargetPairCounts[kTargetCount] = {
    1, 1, 1, 2, 1, 1, 1, 2,
    1, 1, 1, 2, 2, 2, 2, 4,
};
constexpr int kTargetPairs[kTargetCount][kMaxPairsPerTarget][2] = {
    {{0, 0}, {-1, -1}, {-1, -1}, {-1, -1}},
    {{0, 1}, {-1, -1}, {-1, -1}, {-1, -1}},
    {{0, 2}, {-1, -1}, {-1, -1}, {-1, -1}},
    {{0, 3}, {0, 4}, {-1, -1}, {-1, -1}},
    {{1, 0}, {-1, -1}, {-1, -1}, {-1, -1}},
    {{1, 1}, {-1, -1}, {-1, -1}, {-1, -1}},
    {{1, 2}, {-1, -1}, {-1, -1}, {-1, -1}},
    {{1, 3}, {1, 4}, {-1, -1}, {-1, -1}},
    {{2, 0}, {-1, -1}, {-1, -1}, {-1, -1}},
    {{2, 1}, {-1, -1}, {-1, -1}, {-1, -1}},
    {{2, 2}, {-1, -1}, {-1, -1}, {-1, -1}},
    {{2, 3}, {2, 4}, {-1, -1}, {-1, -1}},
    {{3, 0}, {4, 0}, {-1, -1}, {-1, -1}},
    {{3, 1}, {4, 1}, {-1, -1}, {-1, -1}},
    {{3, 2}, {4, 2}, {-1, -1}, {-1, -1}},
    {{3, 3}, {3, 4}, {4, 3}, {4, 4}},
};
}  // namespace

TEST_CASE("IntersectionTreeFilter covers intersecting triangle pairs", "[intersection][filter]") {
    constexpr float kXMin = 0.0f;
    constexpr float kXMax = 1.2f;
    constexpr float kYMin = -1.2f;
    constexpr float kYMax = 1.7f;
    constexpr float kAngleMin = 0.0f;
    constexpr float kAngleMax = TWO_PI;
    constexpr int kSamples = 200;

    IntersectionTreeFilter filter;
    TreeParamsSoA params;
    params.resize(2);
    params.set(0, TreeParams(0.0f, 0.0f, 0.0f));
    params.set(1, TreeParams(0.0f, 0.0f, 0.0f));
    Solution sol = Solution::init(params);

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist_x(kXMin, kXMax);
    std::uniform_real_distribution<float> dist_y(kYMin, kYMax);
    std::uniform_real_distribution<float> dist_ang(kAngleMin, kAngleMax);

    for (int s = 0; s < kSamples; ++s) {
        TreeParams b(dist_x(rng), dist_y(rng), dist_ang(rng));
        sol.set_params(1, b);

        const Figure& f0 = sol.figures()[0];
        const Figure& f1 = sol.figures()[1];
        const auto& n0 = sol.normals()[0];
        const auto& n1 = sol.normals()[1];

        bool expected[TREE_NUM_TRIANGLES][TREE_NUM_TRIANGLES] = {};
        for (int i = 0; i < static_cast<int>(TREE_NUM_TRIANGLES); ++i) {
            for (int j = 0; j < static_cast<int>(TREE_NUM_TRIANGLES); ++j) {
                float score = triangles_intersection_score_from_normals(
                    f0[i], f1[j], n0[i], n1[j], EPSILON
                );
                expected[i][j] = score > 0.0f;
            }
        }

        bool seen[TREE_NUM_TRIANGLES][TREE_NUM_TRIANGLES] = {};
        const auto* pairs = filter.triangle_pairs_for(sol, 0, 1);
        for (const auto& pair : *pairs) {
            seen[pair.first][pair.second] = true;
        }

        std::array<uint8_t, kTargetCount> leaf_pred{};
        int leaf_idx = -1;
        const bool leaf_ok = filter.leaf_pred_for(sol, 0, 1, leaf_pred, &leaf_idx);

        INFO("sample=" << s << " x=" << b.pos.x << " y=" << b.pos.y << " ang=" << b.angle);
        for (int i = 0; i < static_cast<int>(TREE_NUM_TRIANGLES); ++i) {
            for (int j = 0; j < static_cast<int>(TREE_NUM_TRIANGLES); ++j) {
                if (expected[i][j]) {
                    if (!seen[i][j]) {
                        INFO("missing_pair=(" << i << "," << j << ") leaf=" << leaf_idx << " leaf_ok=" << leaf_ok);
                        std::string bits;
                        bits.reserve(kTargetCount);
                        for (int t = 0; t < kTargetCount; ++t) {
                            bits.push_back(leaf_pred[static_cast<size_t>(t)] ? '1' : '0');
                        }
                        INFO("leaf_pred_bits=" << bits);
                        bool any_target = false;
                        for (int t = 0; t < kTargetCount; ++t) {
                            const int count = kTargetPairCounts[t];
                            for (int p = 0; p < count; ++p) {
                                const int ti = kTargetPairs[t][p][0];
                                const int tj = kTargetPairs[t][p][1];
                                if (ti == i && tj == j) {
                                    any_target = true;
                                    INFO("target=" << t << " leaf_pred=" << static_cast<int>(leaf_pred[static_cast<size_t>(t)]));
                                }
                            }
                        }
                        if (!any_target) {
                            INFO("target=none");
                        }
                        CHECK(seen[i][j]);
                    }
                }
            }
        }
    }
}
