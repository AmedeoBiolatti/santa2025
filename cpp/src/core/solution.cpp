#include "tree_packing/core/solution.hpp"
#include "tree_packing/random/rng.hpp"
#include <cmath>
#include <algorithm>

namespace tree_packing {
std::atomic<uint64_t> Solution::next_revision_{1};

namespace {
bool float_equal(float a, float b, float tol) {
    if (std::isnan(a) && std::isnan(b)) {
        return true;
    }
    if (std::isinf(a) || std::isinf(b)) {
        return std::isinf(a) && std::isinf(b) && (std::signbit(a) == std::signbit(b));
    }
    return std::abs(a - b) <= tol;
}

bool vec2_equal(const Vec2& a, const Vec2& b, float tol) {
    return float_equal(a.x, b.x, tol) && float_equal(a.y, b.y, tol);
}

bool triangle_equal(const Triangle& a, const Triangle& b, float tol) {
    return vec2_equal(a.v0, b.v0, tol) && vec2_equal(a.v1, b.v1, tol) &&
           vec2_equal(a.v2, b.v2, tol);
}

bool figure_equal(const Figure& a, const Figure& b, float tol) {
    for (size_t i = 0; i < TREE_NUM_TRIANGLES; ++i) {
        if (!triangle_equal(a.triangles[i], b.triangles[i], tol)) {
            return false;
        }
    }
    return true;
}

bool normals_equal(
    const std::array<std::array<Vec2, 3>, TREE_NUM_TRIANGLES>& a,
    const std::array<std::array<Vec2, 3>, TREE_NUM_TRIANGLES>& b,
    float tol
) {
    for (size_t i = 0; i < TREE_NUM_TRIANGLES; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            if (!vec2_equal(a[i][j], b[i][j], tol)) {
                return false;
            }
        }
    }
    return true;
}

bool aabb_is_default(const AABB& aabb) {
    return aabb.min.x == std::numeric_limits<float>::max() &&
           aabb.min.y == std::numeric_limits<float>::max() &&
           aabb.max.x == std::numeric_limits<float>::lowest() &&
           aabb.max.y == std::numeric_limits<float>::lowest();
}

bool aabb_equal(const AABB& a, const AABB& b, float tol) {
    return vec2_equal(a.min, b.min, tol) && vec2_equal(a.max, b.max, tol);
}

AABB triangle_aabb(const Triangle& tri) {
    AABB aabb;
    aabb.expand(tri);
    return aabb;
}

void recompute_max_abs_cache(const std::vector<float>& per_tree, float& out_max, size_t& out_idx) {
    out_max = 0.0f;
    out_idx = static_cast<size_t>(-1);
    for (size_t i = 0; i < per_tree.size(); ++i) {
        float v = per_tree[i];
        if (v > out_max) {
            out_max = v;
            out_idx = i;
        }
    }
}
}  // namespace

Solution::Solution(size_t num_trees) {
    params_.resize(num_trees);
    figures_.resize(num_trees);
    centers_.resize(num_trees);
    aabbs_.resize(num_trees);
    triangle_aabbs_.resize(num_trees);
    max_abs_.resize(num_trees, 0.0f);
    normals_.resize(num_trees);
    max_max_abs_ = 0.0f;
    max_max_abs_idx_ = static_cast<size_t>(-1);
    valid_.resize(num_trees, false);
    missing_count_ = static_cast<int>(num_trees);
    revision_ = next_revision_.fetch_add(1, std::memory_order_relaxed);
}

Solution Solution::init_random(size_t num_trees, float side, uint64_t seed) {
    Solution sol(num_trees);
    RNG rng(seed);

    float half = side / 2.0f;
    for (size_t i = 0; i < num_trees; ++i) {
        sol.params_.x[i] = rng.uniform(-half, half);
        sol.params_.y[i] = rng.uniform(-half, half);
        sol.params_.angle[i] = rng.uniform(-PI, PI);
        sol.valid_[i] = true;
    }

    sol.recompute_cache();
    sol.revision_ = next_revision_.fetch_add(1, std::memory_order_relaxed);
    return sol;
}

Solution Solution::init_empty(size_t num_trees) {
    Solution sol(num_trees);

    for (size_t i = 0; i < num_trees; ++i) {
        sol.params_.set_nan(i);
    }

    sol.recompute_cache();
    sol.revision_ = next_revision_.fetch_add(1, std::memory_order_relaxed);
    return sol;
}

Solution Solution::init(const TreeParamsSoA& params, int grid_n, float grid_size, int grid_capacity) {
    Solution sol;
    sol.params_ = params;
    sol.figures_.resize(params.size());
    sol.centers_.resize(params.size());
    sol.aabbs_.resize(params.size());
    sol.triangle_aabbs_.resize(params.size());
    sol.max_abs_.resize(params.size(), 0.0f);
    sol.normals_.resize(params.size());
    sol.valid_.resize(params.size(), false);

    // Compute figures and centers
    params_to_figures(sol.params_, sol.figures_);
    get_tree_centers(sol.params_, sol.centers_);
    for (size_t i = 0; i < params.size(); ++i) {
        sol.valid_[i] = !sol.params_.is_nan(i);
        get_tree_normals(sol.params_.angle[i], sol.normals_[i]);
        if (sol.valid_[i]) {
            sol.aabbs_[i] = compute_aabb(sol.figures_[i]);
            sol.max_abs_[i] = compute_aabb_max_abs(sol.aabbs_[i]);
            for (size_t t = 0; t < TREE_NUM_TRIANGLES; ++t) {
                sol.triangle_aabbs_[i][t] = triangle_aabb(sol.figures_[i].triangles[t]);
            }
        } else {
            sol.aabbs_[i] = AABB{};
            sol.max_abs_[i] = 0.0f;
            for (size_t t = 0; t < TREE_NUM_TRIANGLES; ++t) {
                sol.triangle_aabbs_[i][t] = AABB{};
            }
        }
    }
    sol.missing_count_ = static_cast<int>(params.size());
    sol.reg_sum_int_ = 0;
    for (size_t i = 0; i < params.size(); ++i) {
        if (sol.valid_[i]) {
            --sol.missing_count_;
            sol.reg_sum_int_ += (int) (1000.0f * sol.max_abs_[i]);
        }
    }
    recompute_max_abs_cache(sol.max_abs_, sol.max_max_abs_, sol.max_max_abs_idx_);

    // Initialize grid
    sol.grid_ = Grid2D::init(sol.centers_, grid_n, grid_capacity, grid_size);
    sol.revision_ = next_revision_.fetch_add(1, std::memory_order_relaxed);

    return sol;
}

void Solution::set_params(size_t i, const TreeParams& p) {
    params_.set(i, p);
    update_cache_for(i);
    revision_ = next_revision_.fetch_add(1, std::memory_order_relaxed);
}

void Solution::set_nan(size_t i) {
    revision_ = next_revision_.fetch_add(1, std::memory_order_relaxed);
    bool old_valid = valid_[i];
    float old_max_abs = max_abs_[i];
    params_.set_nan(i);
    valid_[i] = false;
    aabbs_[i] = AABB{};
    max_abs_[i] = 0.0f;
    if (old_valid) {
        reg_sum_int_ -= (int)(1000.0f * old_max_abs);
    }
    for (size_t t = 0; t < TREE_NUM_TRIANGLES; ++t) {
        triangle_aabbs_[i][t] = AABB{};
    }
    if (old_valid) {
        ++missing_count_;
    }
    if (i == max_max_abs_idx_) {
        recompute_max_abs_cache(max_abs_, max_max_abs_, max_max_abs_idx_);
    }
    // Update figure to NaN
    for (auto& tri : figures_[i].triangles) {
        tri.v0 = Vec2::nan();
        tri.v1 = Vec2::nan();
        tri.v2 = Vec2::nan();
    }
    for (auto& tri_normals : normals_[i]) {
        tri_normals[0] = Vec2::nan();
        tri_normals[1] = Vec2::nan();
        tri_normals[2] = Vec2::nan();
    }
    centers_[i] = Vec2::nan();
    grid_.update(static_cast<int>(i), Vec2::nan());
}

Solution Solution::update(const TreeParamsSoA& new_params, const std::vector<int>& indices) const {
    Solution sol = *this;
    sol.params_ = new_params;

    for (int idx : indices) {
        if (idx >= 0 && static_cast<size_t>(idx) < sol.size()) {
            sol.update_cache_for(static_cast<size_t>(idx));
        }
    }
    sol.revision_ = next_revision_.fetch_add(1, std::memory_order_relaxed);

    return sol;
}

void Solution::recompute_cache() {
    params_to_figures(params_, figures_);
    get_tree_centers(params_, centers_);
    if (valid_.size() != params_.size()) {
        valid_.resize(params_.size(), false);
    }
    if (aabbs_.size() != params_.size()) {
        aabbs_.resize(params_.size());
    }
    if (triangle_aabbs_.size() != params_.size()) {
        triangle_aabbs_.resize(params_.size());
    }
    if (max_abs_.size() != params_.size()) {
        max_abs_.resize(params_.size(), 0.0f);
    }
    if (normals_.size() != params_.size()) {
        normals_.resize(params_.size());
    }
    for (size_t i = 0; i < params_.size(); ++i) {
        valid_[i] = !params_.is_nan(i);
        get_tree_normals(params_.angle[i], normals_[i]);
        if (valid_[i]) {
            aabbs_[i] = compute_aabb(figures_[i]);
            max_abs_[i] = compute_aabb_max_abs(aabbs_[i]);
            for (size_t t = 0; t < TREE_NUM_TRIANGLES; ++t) {
                triangle_aabbs_[i][t] = triangle_aabb(figures_[i].triangles[t]);
            }
        } else {
            aabbs_[i] = AABB{};
            max_abs_[i] = 0.0f;
            for (size_t t = 0; t < TREE_NUM_TRIANGLES; ++t) {
                triangle_aabbs_[i][t] = AABB{};
            }
        }
    }
    missing_count_ = static_cast<int>(params_.size());
    reg_sum_int_ = 0;
    for (size_t i = 0; i < params_.size(); ++i) {
        if (valid_[i]) {
            --missing_count_;
            reg_sum_int_ += (int)(1000.0f * max_abs_[i]);
        }
    }
    recompute_max_abs_cache(max_abs_, max_max_abs_, max_max_abs_idx_);
    grid_ = Grid2D::init(centers_, 16, 8, THR);
}

void Solution::update_cache_for(size_t i) {
    bool old_valid = valid_[i];
    float old_max_abs = max_abs_[i];
    TreeParams p = params_.get(i);
    figures_[i] = params_to_figure(p);
    centers_[i] = get_tree_center(p);
    valid_[i] = !params_.is_nan(i);
    get_tree_normals(p.angle, normals_[i]);
    if (valid_[i]) {
        aabbs_[i] = compute_aabb(figures_[i]);
        max_abs_[i] = compute_aabb_max_abs(aabbs_[i]);
        for (size_t t = 0; t < TREE_NUM_TRIANGLES; ++t) {
            triangle_aabbs_[i][t] = triangle_aabb(figures_[i].triangles[t]);
        }
    } else {
        aabbs_[i] = AABB{};
        max_abs_[i] = 0.0f;
        for (size_t t = 0; t < TREE_NUM_TRIANGLES; ++t) {
            triangle_aabbs_[i][t] = AABB{};
        }
    }
    // Update reg_sum_int_ incrementally
    if (old_valid) reg_sum_int_ -= (int)(1000.0f * old_max_abs);
    if (valid_[i]) reg_sum_int_ += (int)(1000.0f * max_abs_[i]);

    if (old_valid != valid_[i]) {
        missing_count_ += old_valid ? 1 : -1;
    }
    if (max_abs_[i] > max_max_abs_) {
        max_max_abs_ = max_abs_[i];
        max_max_abs_idx_ = i;
    } else if (i == max_max_abs_idx_ && max_abs_[i] < max_max_abs_) {
        recompute_max_abs_cache(max_abs_, max_max_abs_, max_max_abs_idx_);
    }
    grid_.update(static_cast<int>(i), centers_[i]);
}

void Solution::copy_from(const Solution& other) {
    params_.x = other.params_.x;
    params_.y = other.params_.y;
    params_.angle = other.params_.angle;
    figures_ = other.figures_;
    centers_ = other.centers_;
    aabbs_ = other.aabbs_;
    triangle_aabbs_ = other.triangle_aabbs_;
    max_abs_ = other.max_abs_;
    normals_ = other.normals_;
    max_max_abs_ = other.max_max_abs_;
    max_max_abs_idx_ = other.max_max_abs_idx_;
    reg_sum_int_ = other.reg_sum_int_;
    valid_ = other.valid_;
    missing_count_ = other.missing_count_;
    grid_ = other.grid_;
    revision_ = other.revision_;
}

bool Solution::validate_cache(float tol, bool check_grid) const {
    size_t n = params_.size();
    if (figures_.size() != n || centers_.size() != n || aabbs_.size() != n ||
        triangle_aabbs_.size() != n || max_abs_.size() != n ||
        normals_.size() != n || valid_.size() != n) {
        return false;
    }

    std::vector<Figure> expected_figures(n);
    std::vector<Vec2> expected_centers(n);
    params_to_figures(params_, expected_figures);
    get_tree_centers(params_, expected_centers);
    std::vector<std::array<std::array<Vec2, 3>, TREE_NUM_TRIANGLES>> expected_normals(n);
    for (size_t i = 0; i < n; ++i) {
        get_tree_normals(params_.angle[i], expected_normals[i]);
    }

    int expected_missing = 0;
    for (size_t i = 0; i < n; ++i) {
        bool expected_valid = !params_.is_nan(i);
        if (valid_[i] != expected_valid) {
            return false;
        }
        if (!figure_equal(figures_[i], expected_figures[i], tol)) {
            return false;
        }
        if (!vec2_equal(centers_[i], expected_centers[i], tol)) {
            return false;
        }
        if (!normals_equal(normals_[i], expected_normals[i], tol)) {
            return false;
        }

        if (expected_valid) {
            AABB expected_aabb = compute_aabb(expected_figures[i]);
            if (!aabb_equal(aabbs_[i], expected_aabb, tol)) {
                return false;
            }
            float expected_max_abs = compute_aabb_max_abs(expected_aabb);
            if (!float_equal(max_abs_[i], expected_max_abs, tol)) {
                return false;
            }
            for (size_t t = 0; t < TREE_NUM_TRIANGLES; ++t) {
                AABB expected_tri = triangle_aabb(expected_figures[i].triangles[t]);
                if (!aabb_equal(triangle_aabbs_[i][t], expected_tri, tol)) {
                    return false;
                }
            }
        } else {
            expected_missing++;
            if (!aabb_is_default(aabbs_[i])) {
                return false;
            }
            if (!float_equal(max_abs_[i], 0.0f, tol)) {
                return false;
            }
            for (size_t t = 0; t < TREE_NUM_TRIANGLES; ++t) {
                if (!aabb_is_default(triangle_aabbs_[i][t])) {
                    return false;
                }
            }
        }
    }

    if (missing_count_ != expected_missing) {
        return false;
    }

    float expected_max = 0.0f;
    size_t expected_idx = static_cast<size_t>(-1);
    recompute_max_abs_cache(max_abs_, expected_max, expected_idx);
    if (!float_equal(max_max_abs_, expected_max, tol)) {
        return false;
    }
    if (max_max_abs_idx_ != expected_idx) {
        return false;
    }

    if (check_grid) {
        std::vector<Index> candidates;
        for (size_t i = 0; i < n; ++i) {
            if (!valid_[i]) continue;
            grid_.get_candidates(static_cast<int>(i), candidates);
            bool found = false;
            for (Index idx : candidates) {
                if (idx == static_cast<Index>(i)) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                return false;
            }
        }
    }

    return true;
}

float Solution::reg() const {
    // Return cached sum for O(1) performance
    return 0.001f * ((float) reg_sum_int_);
}

}  // namespace tree_packing
