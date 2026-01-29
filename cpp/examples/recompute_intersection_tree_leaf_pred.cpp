#include "tree_packing/constraints/intersection_tree_filter.hpp"
#include "tree_packing/core/solution.hpp"
#include "tree_packing/geometry/sat.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

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

constexpr float kXMin = 0.0f;
constexpr float kXMax = 1.2f;
constexpr float kYMin = -1.2f;
constexpr float kYMax = 1.7f;
constexpr float kAngleMin = 0.0f;
constexpr float kAngleMax = TWO_PI;

int parse_int(const char* arg, int fallback) {
    if (!arg) return fallback;
    char* end = nullptr;
    long v = std::strtol(arg, &end, 10);
    if (!end || *end != '\0' || v <= 0) return fallback;
    return static_cast<int>(v);
}
}  // namespace

int main(int argc, char** argv) {
    int nx = (argc > 1) ? parse_int(argv[1], 121) : 121;
    int ny = (argc > 2) ? parse_int(argv[2], 121) : 121;
    int na = (argc > 3) ? parse_int(argv[3], 180) : 180;

    IntersectionTreeFilter filter;
    constexpr int kNumLeaves = 1 << 8;
    std::vector<uint8_t> leaf_pred(static_cast<size_t>(kNumLeaves * kTargetCount), 0);
    std::vector<int> leaf_hits(static_cast<size_t>(kNumLeaves), 0);

    TreeParamsSoA params;
    params.resize(2);
    params.set(0, TreeParams(0.0f, 0.0f, 0.0f));
    params.set(1, TreeParams(0.0f, 0.0f, 0.0f));
    Solution sol = Solution::init(params);

    const float dx = (nx > 1) ? (kXMax - kXMin) / static_cast<float>(nx - 1) : 0.0f;
    const float dy = (ny > 1) ? (kYMax - kYMin) / static_cast<float>(ny - 1) : 0.0f;
    const float da = (na > 0) ? (kAngleMax - kAngleMin) / static_cast<float>(na) : 0.0f;

    for (int ix = 0; ix < nx; ++ix) {
        const float x = kXMin + dx * static_cast<float>(ix);
        for (int iy = 0; iy < ny; ++iy) {
            const float y = kYMin + dy * static_cast<float>(iy);
            for (int ia = 0; ia < na; ++ia) {
                const float ang = kAngleMin + da * static_cast<float>(ia);
                sol.set_params(1, TreeParams(x, y, ang));

                const int leaf = filter.leaf_index_for(sol, 0, 1);
                if (leaf < 0 || leaf >= kNumLeaves) continue;
                leaf_hits[static_cast<size_t>(leaf)]++;

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

                const size_t leaf_base = static_cast<size_t>(leaf * kTargetCount);
                for (int t = 0; t < kTargetCount; ++t) {
                    if (leaf_pred[leaf_base + static_cast<size_t>(t)] != 0) continue;
                    const int count = kTargetPairCounts[t];
                    bool hit = false;
                    for (int p = 0; p < count; ++p) {
                        const int ti = kTargetPairs[t][p][0];
                        const int tj = kTargetPairs[t][p][1];
                        if (expected[ti][tj]) {
                            hit = true;
                            break;
                        }
                    }
                    if (hit) {
                        leaf_pred[leaf_base + static_cast<size_t>(t)] = 1;
                    }
                }
            }
        }
    }

    std::cerr << "Grid: nx=" << nx << " ny=" << ny << " na=" << na << "\n";
    int empty = 0;
    for (int leaf = 0; leaf < kNumLeaves; ++leaf) {
        if (leaf_hits[static_cast<size_t>(leaf)] == 0) {
            empty++;
            const size_t base = static_cast<size_t>(leaf * kTargetCount);
            for (int t = 0; t < kTargetCount; ++t) {
                leaf_pred[base + static_cast<size_t>(t)] = 1;
            }
        }
    }
    std::cerr << "Leaves with no samples (set to all ones): " << empty
              << " / " << kNumLeaves << "\n";

    std::cout << "const std::array<uint8_t, kTreeNumLeaves * kTreeNumTargets> kTreeLeafPred = {\n";
    for (int leaf = 0; leaf < kNumLeaves; ++leaf) {
        const size_t base = static_cast<size_t>(leaf * kTargetCount);
        std::cout << "    ";
        for (int t = 0; t < kTargetCount; ++t) {
            std::cout << static_cast<int>(leaf_pred[base + static_cast<size_t>(t)]);
            if (leaf != kNumLeaves - 1 || t != kTargetCount - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "\n";
    }
    std::cout << "};\n";
    return 0;
}
