#include "tree_packing/optimizers/tighten.hpp"
#include "tree_packing/core/problem.hpp"
#include "tree_packing/core/tree.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

namespace tree_packing {
namespace {

struct BoundsMetrics {
    float min_x{0.0f};
    float max_x{0.0f};
    float min_y{0.0f};
    float max_y{0.0f};
    float length{0.0f};
    float perim{0.0f};
    Vec2 center{};
    float objective{0.0f};
};

inline BoundsMetrics metrics_from_eval(const SolutionEval& eval) {
    BoundsMetrics m;
    m.min_x = eval.min_x;
    m.max_x = eval.max_x;
    m.min_y = eval.min_y;
    m.max_y = eval.max_y;
    const float dx = m.max_x - m.min_x;
    const float dy = m.max_y - m.min_y;
    m.length = std::max(dx, dy);
    m.perim = dx + dy;
    m.center = Vec2{(m.min_x + m.max_x) * 0.5f, (m.min_y + m.max_y) * 0.5f};
    m.objective = eval.objective;
    return m;
}

inline float angle_key(float angle) {
    float a = std::fmod(angle, PI * 0.5f);
    if (a < 0.0f) a += PI * 0.5f;
    return (a <= PI * 0.25f) ? a : (PI * 0.5f - a);
}

inline float wrap_angle(float angle) {
    while (angle > PI) angle -= TWO_PI;
    while (angle < -PI) angle += TWO_PI;
    return angle;
}

inline bool accept_move(
    const BoundsMetrics& old_m,
    const BoundsMetrics& new_m,
    float old_r2,
    float new_r2,
    float old_ang_key,
    float new_ang_key,
    bool allow_angle_tie
) {
    const float scale_obj = std::max(1.0f, old_m.objective);
    const float scale_per = std::max(1.0f, old_m.perim);

    const float eps_improve = 1e-6f * scale_obj;
    const float eps_flat = 2e-6f * scale_obj;
    const float eps_perim = 1e-6f * scale_per;
    const float eps_r2 = 1e-8f;

    if (new_m.objective < old_m.objective - eps_improve) {
        return true;
    }

    if (new_m.objective <= old_m.objective + eps_flat) {
        if (new_m.perim < old_m.perim - eps_perim) {
            return true;
        }
        if (new_r2 < old_r2 - eps_r2) {
            return true;
        }
        if (allow_angle_tie && (new_ang_key < old_ang_key - 1e-6f)) {
            return true;
        }
    }

    return false;
}

inline bool violation_not_worse(double old_v, double new_v) {
    return new_v <= old_v + 1e-7;
}

// Fast AABB computation for translated tree (without full figure transform)
inline AABB translate_aabb(const AABB& aabb, const Vec2& delta) {
    return AABB{
        Vec2{aabb.min.x + delta.x, aabb.min.y + delta.y},
        Vec2{aabb.max.x + delta.x, aabb.max.y + delta.y}
    };
}

// Check if tree idx defines any of the current bounds
inline bool tree_on_boundary(const SolutionEval& eval, int idx) {
    return eval.min_x_idx == idx || eval.max_x_idx == idx ||
           eval.min_y_idx == idx || eval.max_y_idx == idx;
}

// Fast check: can this translation potentially improve bounds?
// Returns true if move might help, false if definitely won't help
inline bool translation_can_improve_bounds(
    const SolutionEval& eval,
    int idx,
    const AABB& old_aabb,
    const Vec2& delta
) {
    // If tree is on boundary, moving it might shrink bounds
    if (tree_on_boundary(eval, idx)) {
        return true;
    }

    // Tree is interior - check if new position could become a new boundary
    AABB new_aabb = translate_aabb(old_aabb, delta);

    // If new AABB is strictly inside current bounds, move can't improve
    const float eps = 1e-7f;
    if (new_aabb.min.x > eval.min_x + eps &&
        new_aabb.max.x < eval.max_x - eps &&
        new_aabb.min.y > eval.min_y + eps &&
        new_aabb.max.y < eval.max_y - eps) {
        return false;  // Interior move - can't improve bounds
    }

    return true;  // Might touch boundary
}

// Fast check: does tree have any current intersections?
inline bool tree_has_intersections(const SolutionEval& eval, size_t idx) {
    if (idx >= eval.intersection_map.size()) return false;
    return !eval.intersection_map[idx].empty();
}

// Combined fast rejection: can we skip this move entirely?
// Returns true if we should SKIP (reject) this move without full eval
inline bool fast_reject_translation(
    const SolutionEval& eval,
    int idx,
    const AABB& old_aabb,
    const Vec2& delta,
    double current_violation,
    double tol
) {
    // If we have violations, don't fast-reject (move might help resolve them)
    if (current_violation > tol) {
        return false;
    }

    // If tree has intersections, don't fast-reject (move might resolve them)
    if (tree_has_intersections(eval, static_cast<size_t>(idx))) {
        return false;
    }

    // If move can't improve bounds and tree has no intersections, skip it
    if (!translation_can_improve_bounds(eval, idx, old_aabb, delta)) {
        return true;
    }

    return false;
}

}  // namespace

// ============================================================================
// SqueezeOptimizer
// ============================================================================

SqueezeOptimizer::SqueezeOptimizer(
    float min_scale,
    float shrink,
    int bisect_iters,
    int axis_rounds,
    bool verbose
)
    : min_scale_(min_scale)
    , shrink_(shrink)
    , bisect_iters_(bisect_iters)
    , axis_rounds_(axis_rounds)
    , verbose_(verbose)
{}

std::any SqueezeOptimizer::init_state(const SolutionEval& solution) {
    SqueezeState state;
    state.indices.reserve(solution.solution.size());
    state.base_params.reserve(solution.solution.size());
    state.scratch_params.reserve(solution.solution.size());
    state.dx.reserve(solution.solution.size());
    state.dy.reserve(solution.solution.size());
    return state;
}

void SqueezeOptimizer::apply(
    SolutionEval& solution,
    std::any& state,
    GlobalState& global_state,
    RNG& /*rng*/
) {
    auto* sq_state = std::any_cast<SqueezeState>(&state);
    if (!sq_state) {
        state = init_state(solution);
        sq_state = std::any_cast<SqueezeState>(&state);
    }

    auto& indices = sq_state->indices;
    indices.clear();

    const size_t n = solution.solution.size();
    for (size_t i = 0; i < n; ++i) {
        if (solution.solution.is_valid(i)) {
            indices.push_back(static_cast<int>(i));
        }
    }

    if (indices.empty()) {
        return;
    }

    BoundsMetrics base_m = metrics_from_eval(solution);
    Vec2 center = base_m.center;
    const double tol = global_state.tolerance();
    double base_violation = solution.total_violation();

    auto& base_params = sq_state->base_params;
    auto& scratch_params = sq_state->scratch_params;
    auto& dx = sq_state->dx;
    auto& dy = sq_state->dy;

    base_params.resize(indices.size());
    scratch_params.resize(indices.size());
    dx.resize(indices.size());
    dy.resize(indices.size());

    auto rebuild_base = [&]() {
        base_m = metrics_from_eval(solution);
        center = base_m.center;
        base_violation = solution.total_violation();
        for (size_t k = 0; k < indices.size(); ++k) {
            const int idx = indices[k];
            const TreeParams p = solution.solution.get_params(static_cast<size_t>(idx));
            base_params.set(k, p);
            dx[k] = p.pos.x - center.x;
            dy[k] = p.pos.y - center.y;
        }
    };

    rebuild_base();
    if (base_violation > tol) {
        return;
    }

    auto try_scale = [&](float sx, float sy, float& obj_out, double& violation_out) -> bool {
        const size_t count = indices.size();
        for (size_t k = 0; k < count; ++k) {
            TreeParams p = base_params.get(k);
            p.pos.x = center.x + dx[k] * sx;
            p.pos.y = center.y + dy[k] * sy;
            scratch_params.set(k, p);
        }

        auto& stack = global_state.update_stack();
        const size_t checkpoint = global_state.mark_checkpoint();

        for (size_t k = 0; k < count; ++k) {
            stack.push_update(indices[k], base_params.get(k));
        }

        problem_->update_and_eval(solution, indices, scratch_params);

        const double new_violation = solution.total_violation();
        bool ok = violation_not_worse(base_violation, new_violation);
        if (base_violation <= tol) {
            ok = ok && (new_violation <= tol);
        }
        obj_out = solution.objective;
        violation_out = new_violation;

        global_state.rollback_to(*problem_, solution, checkpoint);

        if (!ok) {
            return false;
        }
        if (base_violation <= tol) {
            return obj_out <= base_m.objective + 1e-6f * std::max(1.0f, base_m.objective);
        }
        return true;
    };

    float best_obj = base_m.objective;
    double best_violation = base_violation;
    float best_sx = 1.0f;
    float best_sy = 1.0f;

    auto run_squeeze = [&](float sx_fixed, float sy_fixed, bool vary_x, bool vary_y) {
        auto pack = [&](float s) -> std::pair<float, float> {
            return {vary_x ? s : sx_fixed, vary_y ? s : sy_fixed};
        };

        float good = 1.0f;
        float bad = -1.0f;
        float best_local_s = 1.0f;
        float best_local_obj = (base_violation > tol)
            ? std::numeric_limits<float>::infinity()
            : best_obj;

        float cand = 1.0f;
        for (int k = 0; k < 40; ++k) {
            cand *= shrink_;
            if (cand < min_scale_) break;
            auto [sx, sy] = pack(cand);
            float obj = 0.0f;
            double violation = 0.0;
            if (try_scale(sx, sy, obj, violation)) {
                good = cand;
                if (base_violation > tol) {
                    if (violation < best_violation - 1e-7 ||
                        (violation_not_worse(best_violation, violation) && obj < best_local_obj)) {
                        best_local_obj = obj;
                        best_local_s = cand;
                        best_violation = violation;
                    }
                } else if (obj < best_local_obj) {
                    best_local_obj = obj;
                    best_local_s = cand;
                }
            } else {
                bad = cand;
                break;
            }
        }

        if (bad < 0.0f) {
            if (best_local_s < 1.0f && best_local_obj < best_obj) {
                best_obj = best_local_obj;
                auto [sx, sy] = pack(best_local_s);
                best_sx = sx;
                best_sy = sy;
            }
            return;
        }

        float lo = bad;
        float hi = good;
        for (int k = 0; k < bisect_iters_; ++k) {
            const float mid = 0.5f * (lo + hi);
            auto [sx, sy] = pack(mid);
            float obj = 0.0f;
            double violation = 0.0;
            if (try_scale(sx, sy, obj, violation)) {
                hi = mid;
                if (base_violation > tol) {
                    if (violation < best_violation - 1e-7 ||
                        (violation_not_worse(best_violation, violation) && obj < best_local_obj)) {
                        best_local_obj = obj;
                        best_local_s = mid;
                        best_violation = violation;
                    }
                } else if (obj < best_local_obj) {
                    best_local_obj = obj;
                    best_local_s = mid;
                }
            } else {
                lo = mid;
            }
        }

        if (best_local_s < 1.0f && (best_local_obj < best_obj || base_violation > tol)) {
            best_obj = best_local_obj;
            auto [sx, sy] = pack(best_local_s);
            best_sx = sx;
            best_sy = sy;
        }
    };

    auto apply_best_scale = [&]() {
        const size_t count = indices.size();
        for (size_t k = 0; k < count; ++k) {
            TreeParams p = base_params.get(k);
            p.pos.x = center.x + dx[k] * best_sx;
            p.pos.y = center.y + dy[k] * best_sy;
            scratch_params.set(k, p);
        }

        auto& stack = global_state.update_stack();
        for (size_t k = 0; k < count; ++k) {
            stack.push_update(indices[k], base_params.get(k));
        }

        problem_->update_and_eval(solution, indices, scratch_params);
    };

    // A) isotropic squeeze
    run_squeeze(1.0f, 1.0f, true, true);

    if (best_obj < base_m.objective || base_violation > tol) {
        apply_best_scale();
        if (verbose_) {
            std::cout << "[Squeeze] objective " << base_m.objective << " -> " << solution.objective
                      << " sx=" << best_sx << " sy=" << best_sy << "\n";
        }
        rebuild_base();
        best_obj = base_m.objective;
        best_violation = base_violation;
        best_sx = 1.0f;
        best_sy = 1.0f;
    }

    // B) axis squeeze (favor the longer axis)
    for (int round = 0; round < axis_rounds_; ++round) {
        const float w = base_m.max_x - base_m.min_x;
        const float h = base_m.max_y - base_m.min_y;
        const float old_obj = base_m.objective;

        if (w > h * 1.0000001f) {
            run_squeeze(1.0f, 1.0f, true, false);
        } else if (h > w * 1.0000001f) {
            run_squeeze(1.0f, 1.0f, false, true);
        } else {
            break;
        }

        if (best_obj < old_obj || base_violation > tol) {
            apply_best_scale();
            if (verbose_) {
                std::cout << "[Squeeze] objective " << old_obj << " -> " << solution.objective
                          << " sx=" << best_sx << " sy=" << best_sy << "\n";
            }
            rebuild_base();
            best_obj = base_m.objective;
            best_violation = base_violation;
            best_sx = 1.0f;
            best_sy = 1.0f;
            continue;
        }

        break;
    }

    if (best_obj < base_m.objective || base_violation > tol) {
        apply_best_scale();
        if (verbose_) {
            std::cout << "[Squeeze] objective " << base_m.objective << " -> " << solution.objective
                      << " sx=" << best_sx << " sy=" << best_sy << "\n";
        }
    }
}

OptimizerPtr SqueezeOptimizer::clone() const {
    return std::make_unique<SqueezeOptimizer>(*this);
}

// ============================================================================
// CompactionOptimizer
// ============================================================================

CompactionOptimizer::CompactionOptimizer(int iters_per_tree, bool verbose)
    : iters_per_tree_(iters_per_tree)
    , verbose_(verbose)
{}

std::any CompactionOptimizer::init_state(const SolutionEval& solution) {
    CompactionState state;
    state.indices.reserve(solution.solution.size());
    state.single_index.reserve(1);
    state.single_index.resize(1);
    state.single_params.reserve(1);
    state.single_params.resize(1);
    return state;
}

void CompactionOptimizer::apply(
    SolutionEval& solution,
    std::any& state,
    GlobalState& global_state,
    RNG& /*rng*/
) {
    auto* comp_state = std::any_cast<CompactionState>(&state);
    if (!comp_state) {
        state = init_state(solution);
        comp_state = std::any_cast<CompactionState>(&state);
    }

    auto& indices = comp_state->indices;
    indices.clear();

    const size_t n = solution.solution.size();
    for (size_t i = 0; i < n; ++i) {
        if (solution.solution.is_valid(i)) {
            indices.push_back(static_cast<int>(i));
        }
    }
    if (indices.empty()) {
        return;
    }

    const int max_iters = std::max(1, iters_per_tree_) * static_cast<int>(indices.size());
    const double tol = global_state.tolerance();

    static constexpr float steps[] = {0.02f, 0.008f, 0.003f, 0.001f, 0.0004f};

    int accepted_total = 0;
    for (int it = 0; it < max_iters; ++it) {
        bool any_accepted = false;

        for (int idx : indices) {
            TreeParams p = solution.solution.get_params(static_cast<size_t>(idx));

            for (float st : steps) {
                for (;;) {
                    const BoundsMetrics old_m = metrics_from_eval(solution);
                    const double old_violation = solution.total_violation();
                    const Vec2 center = old_m.center;
                    const Vec2 delta = Vec2{center.x - p.pos.x, center.y - p.pos.y};
                    const float d2 = delta.length_squared();
                    if (d2 < 1e-12f) {
                        break;
                    }
                    const float invd = 1.0f / std::sqrt(d2);
                    const Vec2 new_pos = Vec2{p.pos.x + delta.x * invd * st, p.pos.y + delta.y * invd * st};

                    auto& stack = global_state.update_stack();
                    const size_t checkpoint = global_state.mark_checkpoint();
                    stack.push_update(idx, p);

                    comp_state->single_index[0] = idx;
                    TreeParams new_p = p;
                    new_p.pos = new_pos;
                    comp_state->single_params.set(0, new_p);
                    problem_->update_and_eval(solution, comp_state->single_index, comp_state->single_params);

                    if (!violation_not_worse(old_violation, solution.total_violation())) {
                        global_state.rollback_to(*problem_, solution, checkpoint);
                        break;
                    }

                    const BoundsMetrics new_m = metrics_from_eval(solution);
                    const float old_r2 = Vec2{p.pos.x - center.x, p.pos.y - center.y}.length_squared();
                    const float new_r2 = Vec2{new_p.pos.x - center.x, new_p.pos.y - center.y}.length_squared();

                    if (old_violation > tol ||
                        accept_move(old_m, new_m, old_r2, new_r2, 0.0f, 0.0f, false)) {
                        p = new_p;
                        any_accepted = true;
                        ++accepted_total;
                        continue;
                    }

                    global_state.rollback_to(*problem_, solution, checkpoint);
                    break;
                }
            }
        }

        if (!any_accepted) {
            break;
        }
    }

    if (verbose_) {
        std::cout << "[Compaction] accepted=" << accepted_total << "\n";
    }
}

OptimizerPtr CompactionOptimizer::clone() const {
    return std::make_unique<CompactionOptimizer>(*this);
}

// ============================================================================
// LocalSearchOptimizer
// ============================================================================

// Compute minimum AABB gap from tree idx to nearby trees using spatial grid
// This is O(k) where k is the number of nearby trees, instead of O(N)
inline float compute_min_gap(
    const SolutionEval& eval,
    int idx,
    std::vector<Index>& candidates_buffer  // Reusable buffer to avoid allocations
) {
    const auto& grid = eval.solution.grid();
    const auto& aabbs = eval.solution.aabbs();
    const AABB& aabb = aabbs[static_cast<size_t>(idx)];

    // Get candidates from spatial grid (nearby trees only)
    grid.get_candidates(idx, candidates_buffer);

    float min_gap = std::numeric_limits<float>::max();
    for (Index other_idx : candidates_buffer) {
        if (other_idx == -1 || other_idx == idx) continue;
        if (!eval.solution.is_valid(static_cast<size_t>(other_idx))) continue;
        const AABB& other_aabb = aabbs[static_cast<size_t>(other_idx)];
        float gap = aabb.min_gap(other_aabb);
        min_gap = std::min(min_gap, gap);
    }
    return min_gap;
}

// Lipschitz pre-check for translations:
// If tree has no intersections AND min_gap > step_size, translation can't create intersections
inline bool lipschitz_skip_translation(
    const SolutionEval& eval,
    int idx,
    float step_size,
    float min_gap,
    double current_violation,
    double tol
) {
    // If we have violations, don't skip (move might help resolve them)
    if (current_violation > tol) {
        return false;
    }

    // If tree has current intersections, don't skip
    if (idx < static_cast<int>(eval.intersection_map.size()) &&
        !eval.intersection_map[static_cast<size_t>(idx)].empty()) {
        return false;
    }

    // Lipschitz bound: if min_gap > step_size, translation can't create new overlaps
    // Add small epsilon for numerical safety
    return min_gap > step_size + 1e-6f;
}

LocalSearchOptimizer::LocalSearchOptimizer(int iters_per_tree, bool verbose)
    : iters_per_tree_(iters_per_tree)
    , verbose_(verbose)
{}

std::any LocalSearchOptimizer::init_state(const SolutionEval& solution) {
    LocalSearchState state;
    state.indices.reserve(solution.solution.size());
    state.single_index.reserve(1);
    state.single_index.resize(1);
    state.single_params.reserve(1);
    state.single_params.resize(1);
    state.min_gaps.reserve(solution.solution.size());
    state.candidates_buffer.reserve(64);  // Typically ~9 cells * ~8 items/cell max
    return state;
}

void LocalSearchOptimizer::apply(
    SolutionEval& solution,
    std::any& state,
    GlobalState& global_state,
    RNG& /*rng*/
) {
    auto* ls_state = std::any_cast<LocalSearchState>(&state);
    if (!ls_state) {
        state = init_state(solution);
        ls_state = std::any_cast<LocalSearchState>(&state);
    }

    auto& indices = ls_state->indices;
    indices.clear();

    const size_t n = solution.solution.size();
    for (size_t i = 0; i < n; ++i) {
        if (solution.solution.is_valid(i)) {
            indices.push_back(static_cast<int>(i));
        }
    }
    if (indices.empty()) {
        return;
    }

    const int max_iters = std::max(1, iters_per_tree_) * static_cast<int>(indices.size());
    const double tol = global_state.tolerance();

    static constexpr float steps[] = {0.01f, 0.004f, 0.0015f, 0.0006f, 0.00025f, 0.0001f};
    static constexpr float rots_deg[] = {5.0f, 2.0f, 0.8f, 0.3f, 0.1f};

    float rots[sizeof(rots_deg) / sizeof(rots_deg[0])];
    for (size_t i = 0; i < sizeof(rots_deg) / sizeof(rots_deg[0]); ++i) {
        rots[i] = rots_deg[i] * (PI / 180.0f);
    }

    const int dx8[] = {1, -1, 0, 0, 1, 1, -1, -1};
    const int dy8[] = {0, 0, 1, -1, 1, -1, 1, -1};

    // Pre-allocate min_gaps and candidates buffer
    auto& min_gaps = ls_state->min_gaps;
    min_gaps.resize(n);
    auto& candidates_buffer = ls_state->candidates_buffer;

    int accepted_total = 0;
    for (int iter = 0; iter < max_iters; ++iter) {
        bool any_accepted = false;

        // Recompute min_gaps at the start of each outer iteration (now O(N*k) instead of O(N²))
        for (int idx : indices) {
            min_gaps[static_cast<size_t>(idx)] = compute_min_gap(solution, idx, candidates_buffer);
        }

        for (int idx : indices) {
            TreeParams p = solution.solution.get_params(static_cast<size_t>(idx));
            bool moved_i = false;
            float idx_min_gap = min_gaps[static_cast<size_t>(idx)];

            // 1) pull-to-center
            for (float st : steps) {
                // Lipschitz pre-check: if min_gap > step, translation can't create new intersections
                if (lipschitz_skip_translation(solution, idx, st, idx_min_gap, solution.total_violation(), tol)) {
                    continue;  // Skip this step size entirely
                }

                for (;;) {
                    const BoundsMetrics old_m = metrics_from_eval(solution);
                    const double old_violation = solution.total_violation();
                    const Vec2 center = old_m.center;
                    const Vec2 delta = Vec2{center.x - p.pos.x, center.y - p.pos.y};
                    const float d2 = delta.length_squared();
                    if (d2 < 1e-12f) {
                        break;
                    }
                    const float invd = 1.0f / std::sqrt(d2);
                    const Vec2 step_delta = Vec2{delta.x * invd * st, delta.y * invd * st};

                    // Fast rejection: skip if move can't help
                    const AABB& cur_aabb = solution.solution.aabbs()[static_cast<size_t>(idx)];
                    if (fast_reject_translation(solution, idx, cur_aabb, step_delta, old_violation, tol)) {
                        break;
                    }

                    const Vec2 new_pos = Vec2{p.pos.x + step_delta.x, p.pos.y + step_delta.y};

                    auto& stack = global_state.update_stack();
                    const size_t checkpoint = global_state.mark_checkpoint();
                    stack.push_update(idx, p);

                    ls_state->single_index[0] = idx;
                    TreeParams new_p = p;
                    new_p.pos = new_pos;
                    ls_state->single_params.set(0, new_p);
                    problem_->update_and_eval(solution, ls_state->single_index, ls_state->single_params);

                    if (!violation_not_worse(old_violation, solution.total_violation())) {
                        global_state.rollback_to(*problem_, solution, checkpoint);
                        break;
                    }

                    const BoundsMetrics new_m = metrics_from_eval(solution);
                    const float old_r2 = Vec2{p.pos.x - center.x, p.pos.y - center.y}.length_squared();
                    const float new_r2 = Vec2{new_p.pos.x - center.x, new_p.pos.y - center.y}.length_squared();

                    if (old_violation > tol ||
                        accept_move(old_m, new_m, old_r2, new_r2, 0.0f, 0.0f, false)) {
                        p = new_p;
                        moved_i = true;
                        any_accepted = true;
                        ++accepted_total;
                        // Update min_gap after successful move
                        idx_min_gap = compute_min_gap(solution, idx, candidates_buffer);
                        continue;
                    }

                    global_state.rollback_to(*problem_, solution, checkpoint);
                    break;
                }
            }

            // 2) pruned 8-dir tiny moves
            {
                const BoundsMetrics old_m = metrics_from_eval(solution);
                const double phase2_violation = solution.total_violation();
                const Vec2 center = old_m.center;
                const Vec2 delta = Vec2{center.x - p.pos.x, center.y - p.pos.y};
                const AABB& cur_aabb = solution.solution.aabbs()[static_cast<size_t>(idx)];

                float dot[8];
                int ord[8];
                for (int d = 0; d < 8; ++d) {
                    dot[d] = static_cast<float>(dx8[d]) * delta.x + static_cast<float>(dy8[d]) * delta.y;
                    ord[d] = d;
                }
                for (int k = 0; k < 3; ++k) {
                    int best = k;
                    for (int j = k + 1; j < 8; ++j) {
                        if (dot[ord[j]] > dot[ord[best]]) best = j;
                    }
                    std::swap(ord[k], ord[best]);
                }

                for (float st : steps) {
                    // Lipschitz pre-check: skip this step size if it can't create intersections
                    // For 8-dir moves, max displacement is sqrt(2)*st for diagonal moves
                    const float max_disp = st * 1.42f;  // sqrt(2) ≈ 1.414
                    if (lipschitz_skip_translation(solution, idx, max_disp, idx_min_gap, phase2_violation, tol)) {
                        continue;
                    }

                    bool accepted_step = false;

                    for (int t = 0; t < 3; ++t) {
                        const int d = ord[t];
                        const Vec2 step_delta = Vec2{
                            static_cast<float>(dx8[d]) * st,
                            static_cast<float>(dy8[d]) * st
                        };

                        // Fast rejection check
                        if (fast_reject_translation(solution, idx, cur_aabb, step_delta, phase2_violation, tol)) {
                            continue;
                        }

                        const Vec2 new_pos = Vec2{p.pos.x + step_delta.x, p.pos.y + step_delta.y};
                        const double old_violation = solution.total_violation();
                        auto& stack = global_state.update_stack();
                        const size_t checkpoint = global_state.mark_checkpoint();
                        stack.push_update(idx, p);

                        ls_state->single_index[0] = idx;
                        TreeParams new_p = p;
                        new_p.pos = new_pos;
                        ls_state->single_params.set(0, new_p);
                        problem_->update_and_eval(solution, ls_state->single_index, ls_state->single_params);

                        if (!violation_not_worse(old_violation, solution.total_violation())) {
                            global_state.rollback_to(*problem_, solution, checkpoint);
                            continue;
                        }

                        const BoundsMetrics new_m = metrics_from_eval(solution);
                        const float old_r2 = Vec2{p.pos.x - center.x, p.pos.y - center.y}.length_squared();
                        const float new_r2 = Vec2{new_p.pos.x - center.x, new_p.pos.y - center.y}.length_squared();

                        if (old_violation > tol ||
                            accept_move(old_m, new_m, old_r2, new_r2, 0.0f, 0.0f, false)) {
                            p = new_p;
                            moved_i = true;
                            any_accepted = true;
                            ++accepted_total;
                            accepted_step = true;
                            // Update min_gap after successful move
                            idx_min_gap = compute_min_gap(solution, idx, candidates_buffer);
                            break;
                        }

                        global_state.rollback_to(*problem_, solution, checkpoint);
                    }

                    if (accepted_step) {
                        continue;
                    }

                    if (st <= 0.0015f) {
                        for (int t = 3; t < 8; ++t) {
                            const int d = ord[t];
                            const Vec2 step_delta = Vec2{
                                static_cast<float>(dx8[d]) * st,
                                static_cast<float>(dy8[d]) * st
                            };

                            // Fast rejection check
                            if (fast_reject_translation(solution, idx, cur_aabb, step_delta, phase2_violation, tol)) {
                                continue;
                            }

                            const Vec2 new_pos = Vec2{p.pos.x + step_delta.x, p.pos.y + step_delta.y};
                        const double old_violation = solution.total_violation();
                            auto& stack = global_state.update_stack();
                            const size_t checkpoint = global_state.mark_checkpoint();
                            stack.push_update(idx, p);

                            ls_state->single_index[0] = idx;
                            TreeParams new_p = p;
                            new_p.pos = new_pos;
                            ls_state->single_params.set(0, new_p);
                            problem_->update_and_eval(solution, ls_state->single_index, ls_state->single_params);

                            if (!violation_not_worse(old_violation, solution.total_violation())) {
                                global_state.rollback_to(*problem_, solution, checkpoint);
                                continue;
                            }

                            const BoundsMetrics new_m = metrics_from_eval(solution);
                            const float old_r2 = Vec2{p.pos.x - center.x, p.pos.y - center.y}.length_squared();
                            const float new_r2 = Vec2{new_p.pos.x - center.x, new_p.pos.y - center.y}.length_squared();

                            if (old_violation > tol ||
                                accept_move(old_m, new_m, old_r2, new_r2, 0.0f, 0.0f, false)) {
                                p = new_p;
                                moved_i = true;
                                any_accepted = true;
                                ++accepted_total;
                                // Update min_gap after successful move
                                idx_min_gap = compute_min_gap(solution, idx, candidates_buffer);
                                break;
                            }

                            global_state.rollback_to(*problem_, solution, checkpoint);
                        }
                    }
                }
            }

            // 3) angle tweaks
            {
                const BoundsMetrics m = metrics_from_eval(solution);
                const double old_violation = solution.total_violation();
                const float w = m.max_x - m.min_x;
                const float h = m.max_y - m.min_y;
                const float side = std::max(w, h);
                const float margin = std::max(1e-4f, 0.05f * side);

                const bool near_edge =
                    (std::abs(p.pos.x - m.min_x) < margin) ||
                    (std::abs(p.pos.x - m.max_x) < margin) ||
                    (std::abs(p.pos.y - m.min_y) < margin) ||
                    (std::abs(p.pos.y - m.max_y) < margin);

                if (near_edge || !moved_i) {
                    const bool allow_tie = !moved_i;
                    const float old_ang_key = angle_key(p.angle);

                    for (float rt : rots) {
                        bool accepted_rt = false;
                        for (float da : {rt, -rt}) {
                            auto& stack = global_state.update_stack();
                            const size_t checkpoint = global_state.mark_checkpoint();
                            stack.push_update(idx, p);

                            ls_state->single_index[0] = idx;
                            TreeParams new_p = p;
                            new_p.angle = wrap_angle(p.angle + da);
                            ls_state->single_params.set(0, new_p);
                            problem_->update_and_eval(solution, ls_state->single_index, ls_state->single_params);

                            if (!violation_not_worse(old_violation, solution.total_violation())) {
                                global_state.rollback_to(*problem_, solution, checkpoint);
                                continue;
                            }

                            const BoundsMetrics new_m = metrics_from_eval(solution);
                            const float new_ang_key = angle_key(new_p.angle);

                            if (old_violation > tol ||
                                accept_move(m, new_m, 0.0f, 0.0f, old_ang_key, new_ang_key, allow_tie)) {
                                p = new_p;
                                moved_i = true;
                                any_accepted = true;
                                ++accepted_total;
                                accepted_rt = true;
                                break;
                            }

                            global_state.rollback_to(*problem_, solution, checkpoint);
                        }
                        if (accepted_rt) {
                            break;
                        }
                    }
                }
            }
        }

        if (!any_accepted) {
            break;
        }
    }

    if (verbose_) {
        std::cout << "[LocalSearch] accepted=" << accepted_total << "\n";
    }
}

OptimizerPtr LocalSearchOptimizer::clone() const {
    return std::make_unique<LocalSearchOptimizer>(*this);
}

}  // namespace tree_packing
