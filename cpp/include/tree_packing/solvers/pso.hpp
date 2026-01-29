#pragma once

#include "solver.hpp"
#include "fitness_utils.hpp"
#include "../core/tree.hpp"
#include "../geometry/sat.hpp"
#include "../spatial/grid2d.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tree_packing {

/**
 * Particle Swarm Optimization solver for single tree placement.
 *
 * PSO maintains a swarm of candidate solutions (particles) that explore
 * the search space. Each particle has:
 * - Position: current candidate solution (x, y, angle)
 * - Velocity: direction and speed of movement
 * - Personal best: best position this particle has found
 * - Global best: best position any particle has found
 *
 * Particles update their velocities based on:
 * - Inertia (w): continue in current direction
 * - Cognitive (c1): move toward personal best
 * - Social (c2): move toward global best
 */
class ParticleSwarmSolver : public Solver {
public:
    // Objective function type
    enum class ObjectiveType {
        Distance,  // Distance-based score using neighbor intersections (original)
        Linf,      // L-infinity norm: max(max(abs(x)), max(abs(y))) of figure vertices
        Zero       // No objective, only constraint satisfaction + ceiling check
    };

    struct Config {
        int n_particles{32};        // Number of particles in swarm
        int n_iterations{50};       // Number of PSO iterations
        float w{0.7f};              // Inertia weight (0.4-0.9 typical)
        float c1{1.5f};             // Cognitive coefficient
        float c2{1.5f};             // Social coefficient
        float mu0{1e6f};            // Initial penalty multiplier
        float mu_max{1e6f};         // Maximum penalty multiplier
        // Isolation penalty for Distance objective: discourages escaping to empty space.
        // We charge an upper-bound per missing neighbor.
        float isolation_penalty_per_missing{std::log(2.0f)};  // Upper bound per missing neighbor
        float target_neighbors{8.0f};                         // Neighbor count that removes isolation penalty
        float vel_max{1.0f};        // Maximum velocity for position
        float vel_ang_max{0.785f};  // Maximum velocity for angle (pi/4)
        bool constrain_to_cell{true};   // If true, constrain search to a grid cell
        bool prefer_current_cell{true}; // Kept for API compatibility (current cell is always used when valid)
        ObjectiveType objective_type{ObjectiveType::Distance};  // Which objective to use
    };

    ParticleSwarmSolver() = default;
    explicit ParticleSwarmSolver(const Config& config) : config_(config) {}

    [[nodiscard]] SolverResult solve(
        const SolutionEval& eval,
        std::span<const int> indices,
        RNG& rng,
        const GlobalState* global_state = nullptr
    ) const override;

    [[nodiscard]] SolverPtr clone() const override {
        auto ptr = std::make_unique<ParticleSwarmSolver>(config_);
        ptr->set_bounds(min_pos_, max_pos_);
        ptr->set_problem(problem_);
        return ptr;
    }

    [[nodiscard]] int max_group_size() const override { return 2; }

    // Accessors for config
    [[nodiscard]] const Config& config() const { return config_; }
    Config& config() { return config_; }

private:
    Config config_;

    // Particle state
    struct Particle {
        float x, y, angle;          // Current position
        float vx, vy, vangle;       // Velocity
        float best_x, best_y, best_angle;  // Personal best position
        float best_cost;            // Personal best cost
    };

    // Compute fitness (lower is better): f + mu * max(0, g)
    // If candidates is provided, uses those instead of querying grid
    [[nodiscard]] float compute_fitness(
        const TreeParams& params,
        const SolutionEval& eval,
        int target_index,
        float mu,
        const std::vector<Index>* candidates = nullptr,
        const GlobalState* global_state = nullptr
    ) const;

    // Compute objective (distance-based score)
    // If candidates is provided, uses those instead of querying grid
    [[nodiscard]] float compute_objective(
        const Figure& figure,
        const std::array<std::array<Vec2, 3>, TREE_NUM_TRIANGLES>& figure_normals,
        const AABB& figure_aabb,
        const std::array<AABB, TREE_NUM_TRIANGLES>& figure_tri_aabbs,
        const SolutionEval& eval,
        int target_index,
        const std::vector<Index>* candidates = nullptr
    ) const;

    // Compute L-infinity objective: max(max(abs(x)), max(abs(y))) of figure vertices
    // Encourages placing trees closer to origin (minimizing bounding box)
    [[nodiscard]] float compute_objective_linf(const Figure& figure) const;

    // Compute constraint violation (intersection score, positive = overlap)
    // If candidates is provided, uses those instead of querying grid
    [[nodiscard]] float compute_constraint(
        const Figure& figure,
        const std::array<std::array<Vec2, 3>, TREE_NUM_TRIANGLES>& figure_normals,
        const AABB& figure_aabb,
        const std::array<AABB, TREE_NUM_TRIANGLES>& figure_tri_aabbs,
        const SolutionEval& eval,
        int target_index,
        const std::vector<Index>* candidates = nullptr
    ) const;

    // Compute both distance objective and constraint in a single pass
    [[nodiscard]] std::pair<float, float> compute_objective_and_constraint_distance(
        const Figure& figure,
        const std::array<std::array<Vec2, 3>, TREE_NUM_TRIANGLES>& figure_normals,
        const AABB& figure_aabb,
        const std::array<AABB, TREE_NUM_TRIANGLES>& figure_tri_aabbs,
        const SolutionEval& eval,
        int target_index,
        const std::vector<Index>* candidates = nullptr
    ) const;

    // Clip velocity to max bounds
    void clip_velocity(Particle& p) const;

    // Clip position to bounds and wrap angle
    void clip_position(Particle& p) const;

    // Wrap angle to [-pi, pi]
    [[nodiscard]] static float wrap_angle(float angle);
};

// Implementation

inline float ParticleSwarmSolver::wrap_angle(float angle) {
    while (angle > PI) angle -= TWO_PI;
    while (angle < -PI) angle += TWO_PI;
    return angle;
}

inline void ParticleSwarmSolver::clip_velocity(Particle& p) const {
    p.vx = std::clamp(p.vx, -config_.vel_max, config_.vel_max);
    p.vy = std::clamp(p.vy, -config_.vel_max, config_.vel_max);
    p.vangle = std::clamp(p.vangle, -config_.vel_ang_max, config_.vel_ang_max);
}

inline void ParticleSwarmSolver::clip_position(Particle& p) const {
    p.x = std::clamp(p.x, min_pos_, max_pos_);
    p.y = std::clamp(p.y, min_pos_, max_pos_);
    p.angle = wrap_angle(p.angle);
}

inline float ParticleSwarmSolver::compute_objective(
    const Figure& figure,
    const std::array<std::array<Vec2, 3>, TREE_NUM_TRIANGLES>& figure_normals,
    const AABB& figure_aabb,
    const std::array<AABB, TREE_NUM_TRIANGLES>& figure_tri_aabbs,
    const SolutionEval& eval,
    int target_index,
    const std::vector<Index>* candidates_ptr
) const {
    const auto& solution = eval.solution;

    // Use provided candidates or query grid
    std::vector<Index> candidates_storage;
    const std::vector<Index>* candidates;
    if (candidates_ptr) {
        candidates = candidates_ptr;
    } else {
        const auto& grid = solution.grid();
        Vec2 center{figure[0].centroid()};
        auto [ci, cj] = grid.compute_ij(center);
        const size_t needed = NEIGHBOR_DELTAS.size() * static_cast<size_t>(grid.capacity());
        candidates_storage.resize(needed);
        const size_t n_candidates = grid.get_candidates_by_cell(ci, cj, candidates_storage);
        candidates_storage.resize(n_candidates);
        candidates = &candidates_storage;
    }

    const auto& all_normals = solution.normals();
    const auto& all_aabbs = solution.aabbs();
    const auto& all_tri_aabbs = solution.triangle_aabbs();
    float total_score = 0.0f;
    int count = 0;

    for (int neighbor_idx : *candidates) {
        if (neighbor_idx < 0 || neighbor_idx == target_index) continue;
        if (!solution.is_valid(neighbor_idx)) continue;

        const auto& neighbor_aabb = all_aabbs[neighbor_idx];

        // Figure-level AABB culling
        if (!figure_aabb.intersects(neighbor_aabb)) {
            // No overlap possible, use a default distance score
            total_score += std::log(2.0f);  // log(1 + 1) as approximation
            ++count;
            continue;
        }

        const auto& neighbor_fig = solution.figures()[neighbor_idx];
        const auto& neighbor_normals = all_normals[neighbor_idx];
        const auto& neighbor_tri_aabbs = all_tri_aabbs[neighbor_idx];

        // Distance-based score: log(1 + mean(relu(-intersection_score)))
        // intersection_score is positive when overlapping, negative when separated
        float min_score = std::numeric_limits<float>::max();

        // Triangle-level AABB culling
        for (size_t i = 0; i < TREE_NUM_TRIANGLES; ++i) {
            if (!figure_tri_aabbs[i].intersects(neighbor_aabb)) continue;
            for (size_t j = 0; j < TREE_NUM_TRIANGLES; ++j) {
                if (!figure_tri_aabbs[i].intersects(neighbor_tri_aabbs[j])) continue;
                float score = triangles_intersection_score_from_normals(
                    figure[i], neighbor_fig[j],
                    figure_normals[i], neighbor_normals[j]
                );
                min_score = std::min(min_score, score);
            }
        }

        // If no triangles overlapped, use default
        if (min_score == std::numeric_limits<float>::max()) {
            min_score = -1.0f;  // Approximate separation
        }

        // When separated, min_score < 0, so relu(-min_score) gives positive distance
        // When overlapping, min_score > 0, so relu(-min_score) = 0
        float dist_score = std::log(1.0f + std::max(0.0f, -min_score));
        total_score += dist_score;
        ++count;
    }

    return count > 0 ? total_score / static_cast<float>(count) : 0.0f;
}

inline float ParticleSwarmSolver::compute_constraint(
    const Figure& figure,
    const std::array<std::array<Vec2, 3>, TREE_NUM_TRIANGLES>& figure_normals,
    const AABB& figure_aabb,
    const std::array<AABB, TREE_NUM_TRIANGLES>& figure_tri_aabbs,
    const SolutionEval& eval,
    int target_index,
    const std::vector<Index>* candidates_ptr
) const {
    const auto& solution = eval.solution;

    // Use provided candidates or query grid
    std::vector<Index> candidates_storage;
    const std::vector<Index>* candidates;
    if (candidates_ptr) {
        candidates = candidates_ptr;
    } else {
        const auto& grid = solution.grid();
        Vec2 center{figure[0].centroid()};
        candidates_storage = grid.get_candidates_by_pos(center);
        candidates = &candidates_storage;
    }

    const auto& all_normals = solution.normals();
    const auto& all_aabbs = solution.aabbs();
    const auto& all_tri_aabbs = solution.triangle_aabbs();
    float max_violation = 0.0f;

    for (int neighbor_idx : *candidates) {
        if (neighbor_idx < 0 || neighbor_idx == target_index) continue;
        if (!solution.is_valid(neighbor_idx)) continue;

        const auto& neighbor_aabb = all_aabbs[neighbor_idx];

        // Figure-level AABB culling - no overlap means no violation
        if (!figure_aabb.intersects(neighbor_aabb)) continue;

        const auto& neighbor_fig = solution.figures()[neighbor_idx];
        const auto& neighbor_normals = all_normals[neighbor_idx];
        const auto& neighbor_tri_aabbs = all_tri_aabbs[neighbor_idx];

        // Compute max intersection score across triangle pairs with AABB culling
        // Positive score = overlap, negative = separated
        for (size_t i = 0; i < TREE_NUM_TRIANGLES; ++i) {
            if (!figure_tri_aabbs[i].intersects(neighbor_aabb)) continue;
            for (size_t j = 0; j < TREE_NUM_TRIANGLES; ++j) {
                if (!figure_tri_aabbs[i].intersects(neighbor_tri_aabbs[j])) continue;
                float score = triangles_intersection_score_from_normals(
                    figure[i], neighbor_fig[j],
                    figure_normals[i], neighbor_normals[j]
                );
                max_violation = std::max(max_violation, score);
            }
        }
    }

    return max_violation;
}

inline std::pair<float, float> ParticleSwarmSolver::compute_objective_and_constraint_distance(
    const Figure& figure,
    const std::array<std::array<Vec2, 3>, TREE_NUM_TRIANGLES>& figure_normals,
    const AABB& figure_aabb,
    const std::array<AABB, TREE_NUM_TRIANGLES>& figure_tri_aabbs,
    const SolutionEval& eval,
    int target_index,
    const std::vector<Index>* candidates_ptr
) const {
    const auto& solution = eval.solution;

    std::vector<Index> candidates_storage;
    const std::vector<Index>* candidates = candidates_ptr;
    if (!candidates) {
        const auto& grid = solution.grid();
        Vec2 center{figure[0].centroid()};
        auto [ci, cj] = grid.compute_ij(center);
        const size_t needed = NEIGHBOR_DELTAS.size() * static_cast<size_t>(grid.capacity());
        candidates_storage.resize(needed);
        const size_t n_candidates = grid.get_candidates_by_cell(ci, cj, candidates_storage);
        candidates_storage.resize(n_candidates);
        candidates = &candidates_storage;
    }

    const auto& all_normals = solution.normals();
    const auto& all_aabbs = solution.aabbs();
    const auto& all_tri_aabbs = solution.triangle_aabbs();

    float total_score = 0.0f;
    int count = 0;
    float max_violation = 0.0f;

    for (int neighbor_idx : *candidates) {
        if (neighbor_idx < 0 || neighbor_idx == target_index) continue;
        if (!solution.is_valid(neighbor_idx)) continue;

        const auto& neighbor_aabb = all_aabbs[neighbor_idx];

        if (!figure_aabb.intersects(neighbor_aabb)) {
            total_score += std::log(2.0f);
            ++count;
            continue;
        }

        const auto& neighbor_fig = solution.figures()[neighbor_idx];
        const auto& neighbor_normals = all_normals[neighbor_idx];
        const auto& neighbor_tri_aabbs = all_tri_aabbs[neighbor_idx];

        float min_score = std::numeric_limits<float>::max();

        for (size_t i = 0; i < TREE_NUM_TRIANGLES; ++i) {
            if (!figure_tri_aabbs[i].intersects(neighbor_aabb)) continue;
            for (size_t j = 0; j < TREE_NUM_TRIANGLES; ++j) {
                if (!figure_tri_aabbs[i].intersects(neighbor_tri_aabbs[j])) continue;
                const float score = triangles_intersection_score_from_normals(
                    figure[i], neighbor_fig[j], figure_normals[i], neighbor_normals[j]
                );
                min_score = std::min(min_score, score);
                max_violation = std::max(max_violation, score);
            }
        }

        if (min_score == std::numeric_limits<float>::max()) {
            min_score = -1.0f;
        }

        const float dist_score = std::log(1.0f + std::max(0.0f, -min_score));
        total_score += dist_score;
        ++count;
    }

    const float mean_score = count > 0 ? total_score / static_cast<float>(count) : 0.0f;
    const float tgt = std::max(1.0f, config_.target_neighbors);
    const float missing = std::max(0.0f, tgt - static_cast<float>(count));
    const float isolation = config_.isolation_penalty_per_missing * missing;
    const float f = mean_score + isolation;
    const float g = max_violation;
    return {f, g};
}

inline float ParticleSwarmSolver::compute_objective_linf(const Figure& figure) const {
    // L-infinity norm: max(max(abs(x)), max(abs(y))) across convex hull vertices
    float max_abs = 0.0f;
    for (const auto& [ti, vi] : CONVEX_HULL_INDEXES) {
        const auto& tri = figure[static_cast<size_t>(ti)];
        const Vec2& v = (vi == 0) ? tri.v0 : (vi == 1) ? tri.v1 : tri.v2;
        max_abs = std::max(max_abs, std::abs(v.x));
        max_abs = std::max(max_abs, std::abs(v.y));
    }
    return max_abs;
}

inline float ParticleSwarmSolver::compute_fitness(
    const TreeParams& params,
    const SolutionEval& eval,
    int target_index,
    float mu,
    const std::vector<Index>* candidates_ptr,
    const GlobalState* global_state
) const {
    Figure figure = params_to_figure(params);

    // Compute normals once for the optimized tree
    std::array<std::array<Vec2, 3>, TREE_NUM_TRIANGLES> figure_normals;
    get_tree_normals(params.angle, figure_normals);

    // Compute AABBs for the figure
    AABB figure_aabb;
    std::array<AABB, TREE_NUM_TRIANGLES> figure_tri_aabbs;
    compute_triangle_aabbs_and_figure_aabb(figure, figure_tri_aabbs, figure_aabb);

    // L-infinity objective does not depend on neighbors; keep the existing
    // split path for clarity and determinism.
    if (config_.objective_type == ObjectiveType::Linf) {
        const float f = compute_objective_linf(figure);
        const float g = compute_constraint(
            figure, figure_normals, figure_aabb, figure_tri_aabbs, eval, target_index, candidates_ptr
        );
        return f + compute_penalty(g, mu);
    }

    // Zero objective: satisfy constraints, but apply objective ceiling penalty.
    if (config_.objective_type == ObjectiveType::Zero) {
        const float g = compute_constraint(
            figure, figure_normals, figure_aabb, figure_tri_aabbs, eval, target_index, candidates_ptr
        );
        return compute_penalty(g, mu);
    }

    const auto [f, g] = compute_objective_and_constraint_distance(
        figure, figure_normals, figure_aabb, figure_tri_aabbs, eval, target_index, candidates_ptr
    );

    // Penalized objective
    return f + compute_penalty(g, mu);
}

inline SolverResult ParticleSwarmSolver::solve(
    const SolutionEval& eval,
    std::span<const int> indices,
    RNG& rng,
    const GlobalState* global_state
) const {
    SolverResult result;
    result.params.resize(indices.size());

    if (indices.empty()) {
        return result;
    }

    // Joint two-tree path: PSO over a 6D state (x,y,ang for each tree).
    if (indices.size() == 2) {
        if (problem_ == nullptr) {
            throw std::runtime_error("ParticleSwarmSolver requires problem pointer for 2-tree solve");
        }

        const int idx0 = indices[0];
        const int idx1 = indices[1];
        const auto& solution = eval.solution;

        SearchRegion region0 = compute_search_region(
            solution, idx0, config_.constrain_to_cell, config_.prefer_current_cell, rng
        );
        SearchRegion region1 = compute_search_region(
            solution, idx1, config_.constrain_to_cell, config_.prefer_current_cell, rng
        );

        struct JointParticle {
            float x0, y0, a0;
            float x1, y1, a1;
            float vx0, vy0, va0;
            float vx1, vy1, va1;
            float bx0, by0, ba0;
            float bx1, by1, ba1;
            float best_cost;
        };

        auto clamp_joint = [&](JointParticle& p) {
            p.x0 = std::clamp(p.x0, region0.min_x, region0.max_x);
            p.y0 = std::clamp(p.y0, region0.min_y, region0.max_y);
            p.a0 = wrap_angle(p.a0);
            p.x1 = std::clamp(p.x1, region1.min_x, region1.max_x);
            p.y1 = std::clamp(p.y1, region1.min_y, region1.max_y);
            p.a1 = wrap_angle(p.a1);
        };

        auto eval_joint = [&](const JointParticle& p, float mu, float* out_f, float* out_g) {
            const TreeParams params0{p.x0, p.y0, p.a0};
            const TreeParams params1{p.x1, p.y1, p.a1};

            const Figure fig0 = params_to_figure(params0);
            const Figure fig1 = params_to_figure(params1);

            std::array<std::array<Vec2, 3>, TREE_NUM_TRIANGLES> normals0;
            std::array<std::array<Vec2, 3>, TREE_NUM_TRIANGLES> normals1;
            get_tree_normals(params0.angle, normals0);
            get_tree_normals(params1.angle, normals1);

            AABB aabb0;
            AABB aabb1;
            std::array<AABB, TREE_NUM_TRIANGLES> tri_aabbs0;
            std::array<AABB, TREE_NUM_TRIANGLES> tri_aabbs1;
            compute_triangle_aabbs_and_figure_aabb(fig0, tri_aabbs0, aabb0);
            compute_triangle_aabbs_and_figure_aabb(fig1, tri_aabbs1, aabb1);

            float f = 0.0f;
            float g = 0.0f;

            if (config_.objective_type == ObjectiveType::Linf) {
                float max_abs = 0.0f;
                for (const auto& [ti, vi] : CONVEX_HULL_INDEXES) {
                    const auto& tri0 = fig0[static_cast<size_t>(ti)];
                    const auto& tri1 = fig1[static_cast<size_t>(ti)];
                    const Vec2& v0 = (vi == 0) ? tri0.v0 : (vi == 1) ? tri0.v1 : tri0.v2;
                    const Vec2& v1 = (vi == 0) ? tri1.v0 : (vi == 1) ? tri1.v1 : tri1.v2;
                    max_abs = std::max(max_abs, std::abs(v0.x));
                    max_abs = std::max(max_abs, std::abs(v0.y));
                    max_abs = std::max(max_abs, std::abs(v1.x));
                    max_abs = std::max(max_abs, std::abs(v1.y));
                }
                f = max_abs;
            } else if (config_.objective_type == ObjectiveType::Distance) {
                const auto fg0 = compute_objective_and_constraint_distance(
                    fig0, normals0, aabb0, tri_aabbs0, eval, idx0, nullptr
                );
                const auto fg1 = compute_objective_and_constraint_distance(
                    fig1, normals1, aabb1, tri_aabbs1, eval, idx1, nullptr
                );
                f = fg0.first + fg1.first;
                g = std::max(fg0.second, fg1.second);
            }

            if (config_.objective_type == ObjectiveType::Zero) {
                g = std::max(
                    compute_constraint(fig0, normals0, aabb0, tri_aabbs0, eval, idx0, nullptr),
                    compute_constraint(fig1, normals1, aabb1, tri_aabbs1, eval, idx1, nullptr)
                );
            }

            if (aabb0.intersects(aabb1)) {
                for (size_t i = 0; i < TREE_NUM_TRIANGLES; ++i) {
                    if (!tri_aabbs0[i].intersects(aabb1)) continue;
                    for (size_t j = 0; j < TREE_NUM_TRIANGLES; ++j) {
                        if (!tri_aabbs0[i].intersects(tri_aabbs1[j])) continue;
                        const float score = triangles_intersection_score_from_normals(
                            fig0[i], fig1[j], normals0[i], normals1[j]
                        );
                        g = std::max(g, score);
                    }
                }
            }

            if (out_f) *out_f = f;
            if (out_g) *out_g = g;
            return f + compute_penalty(g, mu);
        };

        const int n_particles = config_.n_particles;
        std::vector<JointParticle> particles(n_particles);
        for (int i = 0; i < n_particles; ++i) {
            auto& p = particles[i];
            p.x0 = rng.uniform(region0.min_x, region0.max_x);
            p.y0 = rng.uniform(region0.min_y, region0.max_y);
            p.a0 = rng.uniform(-PI, PI);
            p.x1 = rng.uniform(region1.min_x, region1.max_x);
            p.y1 = rng.uniform(region1.min_y, region1.max_y);
            p.a1 = rng.uniform(-PI, PI);

            p.vx0 = rng.uniform(-config_.vel_max, config_.vel_max);
            p.vy0 = rng.uniform(-config_.vel_max, config_.vel_max);
            p.va0 = rng.uniform(-config_.vel_ang_max, config_.vel_ang_max);
            p.vx1 = rng.uniform(-config_.vel_max, config_.vel_max);
            p.vy1 = rng.uniform(-config_.vel_max, config_.vel_max);
            p.va1 = rng.uniform(-config_.vel_ang_max, config_.vel_ang_max);

            p.bx0 = p.x0; p.by0 = p.y0; p.ba0 = p.a0;
            p.bx1 = p.x1; p.by1 = p.y1; p.ba1 = p.a1;
            p.best_cost = std::numeric_limits<float>::max();
        }

        float mu = config_.mu0;
        float global_best_cost = std::numeric_limits<float>::max();
        JointParticle global_best = particles[0];

        for (auto& p : particles) {
            float f = 0.0f, g = 0.0f;
            const float cost = eval_joint(p, mu, &f, &g);
            p.best_cost = cost;
            if (cost < global_best_cost) {
                global_best_cost = cost;
                global_best = p;
                result.start_f = f;
                result.start_g = g;
            }
        }

        for (int iter = 0; iter < config_.n_iterations; ++iter) {
            mu = std::min(mu * 1.2f, config_.mu_max);
            for (auto& p : particles) {
                const float r1 = rng.uniform();
                const float r2 = rng.uniform();
                // Update velocities for both trees.
                p.vx0 = config_.w * p.vx0 + config_.c1 * r1 * (p.bx0 - p.x0) + config_.c2 * r2 * (global_best.bx0 - p.x0);
                p.vy0 = config_.w * p.vy0 + config_.c1 * r1 * (p.by0 - p.y0) + config_.c2 * r2 * (global_best.by0 - p.y0);
                p.va0 = config_.w * p.va0 + config_.c1 * r1 * (p.ba0 - p.a0) + config_.c2 * r2 * (global_best.ba0 - p.a0);
                p.vx1 = config_.w * p.vx1 + config_.c1 * r1 * (p.bx1 - p.x1) + config_.c2 * r2 * (global_best.bx1 - p.x1);
                p.vy1 = config_.w * p.vy1 + config_.c1 * r1 * (p.by1 - p.y1) + config_.c2 * r2 * (global_best.by1 - p.y1);
                p.va1 = config_.w * p.va1 + config_.c1 * r1 * (p.ba1 - p.a1) + config_.c2 * r2 * (global_best.ba1 - p.a1);

                // Clip velocities.
                p.vx0 = std::clamp(p.vx0, -config_.vel_max, config_.vel_max);
                p.vy0 = std::clamp(p.vy0, -config_.vel_max, config_.vel_max);
                p.va0 = std::clamp(p.va0, -config_.vel_ang_max, config_.vel_ang_max);
                p.vx1 = std::clamp(p.vx1, -config_.vel_max, config_.vel_max);
                p.vy1 = std::clamp(p.vy1, -config_.vel_max, config_.vel_max);
                p.va1 = std::clamp(p.va1, -config_.vel_ang_max, config_.vel_ang_max);

                // Update positions.
                p.x0 += p.vx0; p.y0 += p.vy0; p.a0 += p.va0;
                p.x1 += p.vx1; p.y1 += p.vy1; p.a1 += p.va1;
                clamp_joint(p);

                float f = 0.0f, g = 0.0f;
                const float cost = eval_joint(p, mu, &f, &g);
                if (cost < p.best_cost) {
                    p.best_cost = cost;
                    p.bx0 = p.x0; p.by0 = p.y0; p.ba0 = p.a0;
                    p.bx1 = p.x1; p.by1 = p.y1; p.ba1 = p.a1;
                }
                if (cost < global_best_cost) {
                    global_best_cost = cost;
                    global_best = p;
                    result.final_f = f;
                    result.final_g = g;
                }
            }
        }

        // Ensure final metrics are populated even if no improvement occurred.
        {
            float f = 0.0f, g = 0.0f;
            (void)eval_joint(global_best, mu, &f, &g);
            result.final_f = f;
            result.final_g = g;
        }
        result.params.set(0, TreeParams{global_best.bx0, global_best.by0, global_best.ba0});
        result.params.set(1, TreeParams{global_best.bx1, global_best.by1, global_best.ba1});
        return result;
    }

    // Single-index path.
    int target_index = indices[0];

    const auto& solution = eval.solution;

    // Compute search region using common helper
    SearchRegion region = compute_search_region(
        solution, target_index, config_.constrain_to_cell, config_.prefer_current_cell, rng
    );
    const float search_min_x = region.min_x;
    const float search_max_x = region.max_x;
    const float search_min_y = region.min_y;
    const float search_max_y = region.max_y;
    const std::vector<Index>* candidates_ptr = region.has_candidates ? &region.candidates : nullptr;

    // Initialize particles
    const int n_particles = config_.n_particles;
    std::vector<Particle> particles(n_particles);

    for (int i = 0; i < n_particles; ++i) {
        auto& p = particles[i];

        // Random initial position within search bounds
        p.x = rng.uniform(search_min_x, search_max_x);
        p.y = rng.uniform(search_min_y, search_max_y);
        p.angle = rng.uniform(-PI, PI);

        // Random initial velocity
        p.vx = rng.uniform(-config_.vel_max, config_.vel_max);
        p.vy = rng.uniform(-config_.vel_max, config_.vel_max);
        p.vangle = rng.uniform(-config_.vel_ang_max, config_.vel_ang_max);

        // Initialize personal best to current position
        p.best_x = p.x;
        p.best_y = p.y;
        p.best_angle = p.angle;
        p.best_cost = std::numeric_limits<float>::max();
    }

    // Lambda to clip position to cell bounds (or global bounds)
    auto clip_to_bounds = [&](Particle& p) {
        p.x = std::clamp(p.x, search_min_x, search_max_x);
        p.y = std::clamp(p.y, search_min_y, search_max_y);
        p.angle = wrap_angle(p.angle);
    };

    // Initialize global best
    float global_best_cost = std::numeric_limits<float>::max();
    float global_best_x = particles[0].x;
    float global_best_y = particles[0].y;
    float global_best_angle = particles[0].angle;

    // Evaluate initial positions
    float mu = config_.mu0;
    for (int i = 0; i < n_particles; ++i) {
        auto& p = particles[i];
        TreeParams params{Vec2{p.x, p.y}, p.angle};
        float cost = compute_fitness(params, eval, target_index, mu, candidates_ptr, global_state);

        p.best_cost = cost;
        if (cost < global_best_cost) {
            global_best_cost = cost;
            global_best_x = p.x;
            global_best_y = p.y;
            global_best_angle = p.angle;
        }
    }

    // Record start values
    {
        TreeParams start_params{Vec2{global_best_x, global_best_y}, global_best_angle};
        Figure start_figure = params_to_figure(start_params);
        std::array<std::array<Vec2, 3>, TREE_NUM_TRIANGLES> start_normals;
        get_tree_normals(start_params.angle, start_normals);
        AABB start_aabb;
        std::array<AABB, TREE_NUM_TRIANGLES> start_tri_aabbs;
        compute_triangle_aabbs_and_figure_aabb(start_figure, start_tri_aabbs, start_aabb);
        if (config_.objective_type == ObjectiveType::Zero) {
            result.start_f = 0.0f;
            result.start_g = compute_constraint(
                start_figure, start_normals, start_aabb, start_tri_aabbs, eval, target_index, candidates_ptr
            );
        } else if (config_.objective_type == ObjectiveType::Linf) {
            result.start_f = compute_objective_linf(start_figure);
            result.start_g = compute_constraint(start_figure, start_normals, start_aabb, start_tri_aabbs, eval, target_index, candidates_ptr);
        } else {
            const auto [start_f, start_g] = compute_objective_and_constraint_distance(
                start_figure, start_normals, start_aabb, start_tri_aabbs, eval, target_index, candidates_ptr
            );
            result.start_f = start_f;
            result.start_g = start_g;
        }
    }

    // Penalty increment per iteration
    float d_mu = (config_.mu_max - config_.mu0) / std::max(1, config_.n_iterations - 1);

    // PSO main loop
    for (int iter = 0; iter < config_.n_iterations; ++iter) {
        mu = config_.mu0 + iter * d_mu;

        for (auto& p : particles) {
            // Random coefficients
            float r1_x = rng.uniform();
            float r1_y = rng.uniform();
            float r1_a = rng.uniform();
            float r2_x = rng.uniform();
            float r2_y = rng.uniform();
            float r2_a = rng.uniform();

            // Update velocity
            // v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
            p.vx = config_.w * p.vx
                 + config_.c1 * r1_x * (p.best_x - p.x)
                 + config_.c2 * r2_x * (global_best_x - p.x);
            p.vy = config_.w * p.vy
                 + config_.c1 * r1_y * (p.best_y - p.y)
                 + config_.c2 * r2_y * (global_best_y - p.y);
            p.vangle = config_.w * p.vangle
                     + config_.c1 * r1_a * wrap_angle(p.best_angle - p.angle)
                     + config_.c2 * r2_a * wrap_angle(global_best_angle - p.angle);

            clip_velocity(p);

            // Update position
            p.x += p.vx;
            p.y += p.vy;
            p.angle += p.vangle;

            // Clip to cell bounds (or global bounds)
            clip_to_bounds(p);

            // Evaluate new position
            TreeParams params{Vec2{p.x, p.y}, p.angle};
            float cost = compute_fitness(params, eval, target_index, mu, candidates_ptr, global_state);

            // Update personal best
            if (cost < p.best_cost) {
                p.best_cost = cost;
                p.best_x = p.x;
                p.best_y = p.y;
                p.best_angle = p.angle;

                // Update global best
                if (cost < global_best_cost) {
                    global_best_cost = cost;
                    global_best_x = p.x;
                    global_best_y = p.y;
                    global_best_angle = p.angle;
                }
            }
        }

    }

    // Store final result
    TreeParams final_params{Vec2{global_best_x, global_best_y}, global_best_angle};
    result.params.set(0, final_params);

    // Record final values
    Figure final_figure = params_to_figure(final_params);
    std::array<std::array<Vec2, 3>, TREE_NUM_TRIANGLES> final_normals;
    get_tree_normals(final_params.angle, final_normals);
    AABB final_aabb;
    std::array<AABB, TREE_NUM_TRIANGLES> final_tri_aabbs;
    compute_triangle_aabbs_and_figure_aabb(final_figure, final_tri_aabbs, final_aabb);
    if (config_.objective_type == ObjectiveType::Zero) {
        result.final_f = 0.0f;
        result.final_g = compute_constraint(
            final_figure, final_normals, final_aabb, final_tri_aabbs, eval, target_index, candidates_ptr
        );
    } else if (config_.objective_type == ObjectiveType::Linf) {
        result.final_f = compute_objective_linf(final_figure);
        result.final_g = compute_constraint(final_figure, final_normals, final_aabb, final_tri_aabbs, eval, target_index, candidates_ptr);
    } else {
        const auto [final_f, final_g] = compute_objective_and_constraint_distance(
            final_figure, final_normals, final_aabb, final_tri_aabbs, eval, target_index, candidates_ptr
        );
        result.final_f = final_f;
        result.final_g = final_g;
    }

    // If solving for multiple indices, copy the result (placeholder for future)
    for (size_t i = 1; i < indices.size(); ++i) {
        result.params.set(i, result.params.get(0));
    }

    return result;
}

}  // namespace tree_packing
