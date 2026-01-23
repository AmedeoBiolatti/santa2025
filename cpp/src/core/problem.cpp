#include "tree_packing/core/problem.hpp"
#include "tree_packing/core/tree.hpp"
#include "tree_packing/core/types.hpp"
#include "tree_packing/core/solution.hpp"
#include "tree_packing/geometry/sat.hpp"
#include "tree_packing/constraints/intersection.hpp"
#include "tree_packing/constraints/bounds.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tree_packing {
namespace {
struct ObjectiveBounds {
    int16_t valid_count{0};
    float min_x{std::numeric_limits<float>::max()};
    float max_x{std::numeric_limits<float>::lowest()};
    float min_y{std::numeric_limits<float>::max()};
    float max_y{std::numeric_limits<float>::lowest()};
    Index min_x_idx{-1};
    Index max_x_idx{-1};
    Index min_y_idx{-1};
    Index max_y_idx{-1};
};

void update_bounds_with_point(ObjectiveBounds& b, float x, float y, int idx) {
    if (x < b.min_x) {
        b.min_x = x;
        b.min_x_idx = static_cast<Index>(idx);
    }
    if (x > b.max_x) {
        b.max_x = x;
        b.max_x_idx = static_cast<Index>(idx);
    }
    if (y < b.min_y) {
        b.min_y = y;
        b.min_y_idx = static_cast<Index>(idx);
    }
    if (y > b.max_y) {
        b.max_y = y;
        b.max_y_idx = static_cast<Index>(idx);
    }
}

void update_bounds_with_aabb(ObjectiveBounds& b, const AABB& aabb, int idx) {
    if (aabb.min.x < b.min_x) {
        b.min_x = aabb.min.x;
        b.min_x_idx = static_cast<Index>(idx);
    }
    if (aabb.max.x > b.max_x) {
        b.max_x = aabb.max.x;
        b.max_x_idx = static_cast<Index>(idx);
    }
    if (aabb.min.y < b.min_y) {
        b.min_y = aabb.min.y;
        b.min_y_idx = static_cast<Index>(idx);
    }
    if (aabb.max.y > b.max_y) {
        b.max_y = aabb.max.y;
        b.max_y_idx = static_cast<Index>(idx);
    }
}

ObjectiveBounds compute_bounds_full(const Solution& solution) {
    ObjectiveBounds b;
    const auto& aabbs = solution.aabbs();
    for (size_t i = 0; i < aabbs.size(); ++i) {
        if (!solution.is_valid(i)) continue;
        b.valid_count = static_cast<int16_t>(b.valid_count + 1);
        update_bounds_with_aabb(b, aabbs[i], static_cast<int>(i));
    }

    return b;
}

}  // namespace

Problem Problem::create_tree_packing_problem(float side) {
    Problem problem;

    if (side < 0) {
        side = 16.0f * THR;
    }

    problem.min_pos_ = -side / 2.0f;
    problem.max_pos_ = side / 2.0f;

    return problem;
}

float Problem::objective(const Solution& solution) const {
    const auto& aabbs = solution.aabbs();
    size_t n = aabbs.size();

    // Compute bounding box of all trees
    float min_x = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float min_y = std::numeric_limits<float>::max();
    float max_y = std::numeric_limits<float>::lowest();

    int valid_count = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!solution.is_valid(i)) continue;
        const auto& aabb = aabbs[i];
        if (aabb.min.x < min_x) min_x = aabb.min.x;
        if (aabb.max.x > max_x) max_x = aabb.max.x;
        if (aabb.min.y < min_y) min_y = aabb.min.y;
        if (aabb.max.y > max_y) max_y = aabb.max.y;
        ++valid_count;
    }

    if (valid_count == 0) {
        return 0.0f;
    }

    float delta_x = max_x - min_x;
    float delta_y = max_y - min_y;
    float length = std::max(delta_x, delta_y);

    return (length * length) / static_cast<float>(n);
}

float Problem::intersection_constraint(const Solution& solution) const {
    IntersectionConstraint constraint;
    SolutionEval::IntersectionMap map;
    return constraint.eval(solution, map);
}

float Problem::bounds_constraint(const Solution& solution) const {
    BoundsConstraint constraint(min_pos_, max_pos_);
    return constraint.eval(solution);
}

SolutionEval Problem::eval(const Solution& solution) const {
    SolutionEval eval;
    eval_inplace(solution, eval);
    return eval;
}

void Problem::eval_inplace(const Solution& solution, SolutionEval& eval) const {
    if (&eval.solution != &solution) {
        eval.solution.copy_from(solution);
    }

    auto obj_bounds = compute_bounds_full(solution);
    eval.valid_count = obj_bounds.valid_count;
    eval.min_x = obj_bounds.min_x;
    eval.max_x = obj_bounds.max_x;
    eval.min_y = obj_bounds.min_y;
    eval.max_y = obj_bounds.max_y;
    eval.min_x_idx = obj_bounds.min_x_idx;
    eval.max_x_idx = obj_bounds.max_x_idx;
    eval.min_y_idx = obj_bounds.min_y_idx;
    eval.max_y_idx = obj_bounds.max_y_idx;

    if (obj_bounds.valid_count == 0) {
        eval.objective = 0.0f;
    } else {
        float delta_x = obj_bounds.max_x - obj_bounds.min_x;
        float delta_y = obj_bounds.max_y - obj_bounds.min_y;
        float length = std::max(delta_x, delta_y);
        eval.objective = (length * length) / static_cast<float>(solution.size());
    }

    // Compute intersection constraint with sparse map
    IntersectionConstraint intersection;
    eval.intersection_violation = intersection.eval(
        solution, eval.intersection_map, &eval.intersection_count
    );

    // Compute bounds constraint
    BoundsConstraint bounds(min_pos_, max_pos_);
    eval.bounds_violation = bounds.eval(solution);
}

void Problem::update_and_eval(
    SolutionEval& eval,
    const std::vector<int>& indices,
    const TreeParamsSoA& new_params
) const {
    size_t n = indices.size();
    if (new_params.size() != n) {
        throw std::runtime_error("update_and_eval: indices and new_params size mismatch");
    }

    bool recompute_max_abs = false;
    bool recompute_min_x = false;
    bool recompute_max_x = false;
    bool recompute_min_y = false;
    bool recompute_max_y = false;

    for (size_t k=0; k<indices.size(); k++){
        auto i = indices[k];
        TreeParams p = new_params.get(k);

        bool old_valid = eval.solution.valid_[i];
        eval.solution.params_.x[i] = p.pos.x;
        eval.solution.params_.y[i] = p.pos.y;
        eval.solution.params_.angle[i] = p.angle;
        eval.solution.update_cache_for(static_cast<size_t>(i));
        bool new_valid = eval.solution.valid_[i];
        if (old_valid != new_valid) {
            eval.valid_count += new_valid ? 1 : -1;
        }
        if (eval.min_x_idx == i) {
            recompute_min_x = true;
        } else if (new_valid && eval.solution.aabbs_[i].min.x < eval.min_x) {
            eval.min_x = eval.solution.aabbs_[i].min.x;
            eval.min_x_idx = static_cast<Index>(i);
        }
        if (eval.max_x_idx == i) {
            recompute_max_x = true;
        } else if (new_valid && eval.solution.aabbs_[i].max.x > eval.max_x) {
            eval.max_x = eval.solution.aabbs_[i].max.x;
            eval.max_x_idx = static_cast<Index>(i);
        }
        if (eval.min_y_idx == i) {
            recompute_min_y = true;
        } else if (new_valid && eval.solution.aabbs_[i].min.y < eval.min_y) {
            eval.min_y = eval.solution.aabbs_[i].min.y;
            eval.min_y_idx = static_cast<Index>(i);
        }
        if (eval.max_y_idx == i) {
            recompute_max_y = true;
        } else if (new_valid && eval.solution.aabbs_[i].max.y > eval.max_y) {
            eval.max_y = eval.solution.aabbs_[i].max.y;
            eval.max_y_idx = static_cast<Index>(i);
        }
        if (eval.solution.max_max_abs_idx_ == static_cast<size_t>(i)) {
            recompute_max_abs = true;
        }
    }
    if (recompute_max_abs) {
        float max_abs = 0.0f;
        size_t idx = static_cast<size_t>(-1);
        for (size_t i = 0; i < eval.solution.max_abs_.size(); ++i) {
            float v = eval.solution.max_abs_[i];
            if (v > max_abs) {
                max_abs = v;
                idx = i;
            }
        }
        eval.solution.max_max_abs_ = max_abs;
        eval.solution.max_max_abs_idx_ = idx;
    }
    if (recompute_min_x || recompute_max_x || recompute_min_y || recompute_max_y) {
        const auto& aabbs = eval.solution.aabbs();
        if (recompute_min_x) {
            eval.min_x = std::numeric_limits<float>::max();
            eval.min_x_idx = -1;
        }
        if (recompute_max_x) {
            eval.max_x = std::numeric_limits<float>::lowest();
            eval.max_x_idx = -1;
        }
        if (recompute_min_y) {
            eval.min_y = std::numeric_limits<float>::max();
            eval.min_y_idx = -1;
        }
        if (recompute_max_y) {
            eval.max_y = std::numeric_limits<float>::lowest();
            eval.max_y_idx = -1;
        }
        for (size_t i = 0; i < aabbs.size(); ++i) {
            if (!eval.solution.valid_[i]) continue;
            if (recompute_min_x && aabbs[i].min.x < eval.min_x) {
                eval.min_x = aabbs[i].min.x;
                eval.min_x_idx = static_cast<Index>(i);
            }
            if (recompute_max_x && aabbs[i].max.x > eval.max_x) {
                eval.max_x = aabbs[i].max.x;
                eval.max_x_idx = static_cast<Index>(i);
            }
            if (recompute_min_y && aabbs[i].min.y < eval.min_y) {
                eval.min_y = aabbs[i].min.y;
                eval.min_y_idx = static_cast<Index>(i);
            }
            if (recompute_max_y && aabbs[i].max.y > eval.max_y) {
                eval.max_y = aabbs[i].max.y;
                eval.max_y_idx = static_cast<Index>(i);
            }
        }
    }
    float delta_x = eval.max_x - eval.min_x;
    float delta_y = eval.max_y - eval.min_y;
    float length = std::max(delta_x, delta_y);
    eval.objective = (length * length) / static_cast<float>(eval.solution.size());
    // bounds constraint
    BoundsConstraint bounds(min_pos_, max_pos_);
    eval.bounds_violation = bounds.eval_update(eval.min_x, eval.max_x, eval.min_y, eval.max_y);
    // const constraint
    IntersectionConstraint intersection;
    eval.intersection_violation = intersection.eval_update(
        eval.solution,
        eval.intersection_map,
        indices,
        eval.intersection_violation,
        eval.intersection_count,
        &eval.intersection_count
    );
    eval.solution.revision_ = Solution::next_revision_.fetch_add(1, std::memory_order_relaxed);
}

void Problem::remove_and_eval(
    SolutionEval& eval,
    const std::vector<int>& indices
) const {
    bool recompute_max_abs = false;
    bool recompute_min_x = false;
    bool recompute_max_x = false;
    bool recompute_min_y = false;
    bool recompute_max_y = false;

    for (int idx : indices) {
        size_t i = static_cast<size_t>(idx);
        bool was_valid = eval.solution.valid_[i];
        eval.solution.set_nan(i);
        if (was_valid) {
            eval.valid_count -= 1;
        }
        if (eval.min_x_idx == idx) {
            recompute_min_x = true;
        }
        if (eval.max_x_idx == idx) {
            recompute_max_x = true;
        }
        if (eval.min_y_idx == idx) {
            recompute_min_y = true;
        }
        if (eval.max_y_idx == idx) {
            recompute_max_y = true;
        }
        if (eval.solution.max_max_abs_idx_ == i) {
            recompute_max_abs = true;
        }
    }

    if (recompute_max_abs) {
        float max_abs = 0.0f;
        size_t idx = static_cast<size_t>(-1);
        for (size_t i = 0; i < eval.solution.max_abs_.size(); ++i) {
            float v = eval.solution.max_abs_[i];
            if (v > max_abs) {
                max_abs = v;
                idx = i;
            }
        }
        eval.solution.max_max_abs_ = max_abs;
        eval.solution.max_max_abs_idx_ = idx;
    }

    if (recompute_min_x || recompute_max_x || recompute_min_y || recompute_max_y) {
        const auto& aabbs = eval.solution.aabbs();
        if (recompute_min_x) {
            eval.min_x = std::numeric_limits<float>::max();
            eval.min_x_idx = -1;
        }
        if (recompute_max_x) {
            eval.max_x = std::numeric_limits<float>::lowest();
            eval.max_x_idx = -1;
        }
        if (recompute_min_y) {
            eval.min_y = std::numeric_limits<float>::max();
            eval.min_y_idx = -1;
        }
        if (recompute_max_y) {
            eval.max_y = std::numeric_limits<float>::lowest();
            eval.max_y_idx = -1;
        }
        for (size_t i = 0; i < aabbs.size(); ++i) {
            if (!eval.solution.valid_[i]) continue;
            if (recompute_min_x && aabbs[i].min.x < eval.min_x) {
                eval.min_x = aabbs[i].min.x;
                eval.min_x_idx = static_cast<Index>(i);
            }
            if (recompute_max_x && aabbs[i].max.x > eval.max_x) {
                eval.max_x = aabbs[i].max.x;
                eval.max_x_idx = static_cast<Index>(i);
            }
            if (recompute_min_y && aabbs[i].min.y < eval.min_y) {
                eval.min_y = aabbs[i].min.y;
                eval.min_y_idx = static_cast<Index>(i);
            }
            if (recompute_max_y && aabbs[i].max.y > eval.max_y) {
                eval.max_y = aabbs[i].max.y;
                eval.max_y_idx = static_cast<Index>(i);
            }
        }
    }
    if (eval.valid_count == 0) {
        eval.objective = 0.0f;
    } else {
        float delta_x = eval.max_x - eval.min_x;
        float delta_y = eval.max_y - eval.min_y;
        float length = std::max(delta_x, delta_y);
        eval.objective = (length * length) / static_cast<float>(eval.solution.size());
    }

    BoundsConstraint bounds(min_pos_, max_pos_);
    eval.bounds_violation = bounds.eval_update(eval.min_x, eval.max_x, eval.min_y, eval.max_y);

    IntersectionConstraint intersection;
    eval.intersection_violation = intersection.eval_remove(
        eval.solution,
        eval.intersection_map,
        indices,
        eval.intersection_violation,
        eval.intersection_count,
        &eval.intersection_count
    );
    eval.solution.revision_ = Solution::next_revision_.fetch_add(1, std::memory_order_relaxed);
}

float Problem::score(const SolutionEval& solution_eval, const GlobalState& global_state) const {
    uint64_t revision = solution_eval.solution.revision();
    float mu = global_state.mu();
    if (solution_eval.score_cache_valid &&
        solution_eval.score_cache_revision == revision &&
        solution_eval.score_cache_mu == mu) {
        return solution_eval.score_cache_value;
    }

    float violation = solution_eval.total_violation();
    int n_missing = solution_eval.n_missing();
    float reg = solution_eval.reg();

    float score = solution_eval.objective +
           global_state.mu() * violation +
           1.0f * static_cast<float>(n_missing) +
           1e-6 * reg;
    solution_eval.score_cache_revision = revision;
    solution_eval.score_cache_mu = mu;
    solution_eval.score_cache_value = score;
    solution_eval.score_cache_valid = true;
    return score;
}

void Problem::update_intersection_matrix(
    const Solution& solution,
    std::vector<float>& matrix,
    const std::vector<int>& modified_indices,
    [[maybe_unused]] const Grid2D* old_grid
) const {
    // Simple update: recompute rows/columns for modified indices
    // Call the free function from sat.hpp (use :: to disambiguate)
    ::tree_packing::update_intersection_matrix(solution.figures(), matrix, modified_indices);
}

}  // namespace tree_packing
