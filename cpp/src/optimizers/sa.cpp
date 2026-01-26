#include "tree_packing/optimizers/sa.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <utility>
#include <cassert>

namespace tree_packing {

#ifndef NDEBUG
// Debug helper to compare solution params
static bool solutions_match(const Solution& a, const Solution& b, float tol = 1e-6f) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a.is_valid(i) != b.is_valid(i)) return false;
        if (a.is_valid(i)) {
            TreeParams pa = a.get_params(i);
            TreeParams pb = b.get_params(i);
            if (std::abs(pa.pos.x - pb.pos.x) > tol ||
                std::abs(pa.pos.y - pb.pos.y) > tol ||
                std::abs(pa.angle - pb.angle) > tol) {
                return false;
            }
        }
    }
    return true;
}

// Debug struct to save full state for validation
struct DebugSavedState {
    Solution solution;
    float intersection_violation{0.0f};
    float bounds_violation{0.0f};
    float objective{0.0f};
    int intersection_count{0};

    void save_from(const SolutionEval& eval) {
        solution.copy_from(eval.solution);
        intersection_violation = eval.intersection_violation;
        bounds_violation = eval.bounds_violation;
        objective = eval.objective;
        intersection_count = eval.intersection_count;
    }

    bool compare_scores(const SolutionEval& eval, float tol = 1e-4f) const {
        if (std::abs(intersection_violation - eval.intersection_violation) > tol) {
            std::cerr << "[SA DEBUG] intersection_violation mismatch: "
                      << intersection_violation << " vs " << eval.intersection_violation
                      << " (diff=" << std::abs(intersection_violation - eval.intersection_violation) << ")\n";
            return false;
        }
        if (std::abs(bounds_violation - eval.bounds_violation) > tol) {
            std::cerr << "[SA DEBUG] bounds_violation mismatch: "
                      << bounds_violation << " vs " << eval.bounds_violation << "\n";
            return false;
        }
        if (std::abs(objective - eval.objective) > tol) {
            std::cerr << "[SA DEBUG] objective mismatch: "
                      << objective << " vs " << eval.objective << "\n";
            return false;
        }
        return true;
    }
};
#endif

SimulatedAnnealing::SimulatedAnnealing(
    OptimizerPtr inner_optimizer,
    float initial_temp,
    float min_temp,
    CoolingSchedule cooling_schedule,
    float cooling_rate,
    int patience,
    bool verbose
)
    : inner_optimizer_(std::move(inner_optimizer))
    , initial_temp_(initial_temp)
    , min_temp_(min_temp)
    , cooling_schedule_(cooling_schedule)
    , cooling_rate_(cooling_rate)
    , patience_(patience)
    , verbose_(verbose)
{}

void SimulatedAnnealing::set_problem(Problem* problem) {
    Optimizer::set_problem(problem);
    inner_optimizer_->set_problem(problem);
}

std::any SimulatedAnnealing::init_state(const SolutionEval& solution) {
    SAState state;
    state.inner_state = inner_optimizer_->init_state(solution);
    return state;
}

float SimulatedAnnealing::compute_temperature(int iteration) const {
    float temp;

    switch (cooling_schedule_) {
        case CoolingSchedule::Exponential:
            temp = initial_temp_ * std::pow(cooling_rate_, static_cast<float>(iteration));
            break;
        case CoolingSchedule::Linear:
            temp = initial_temp_ - cooling_rate_ * static_cast<float>(iteration);
            break;
        case CoolingSchedule::Logarithmic:
            temp = initial_temp_ / std::log(static_cast<float>(iteration) + std::exp(1.0f));
            break;
        default:
            temp = initial_temp_;
    }

    return std::max(temp, min_temp_);
}

inline float fast_exp(float x) {
      // Schraudolph's approximation
      union { float f; int32_t i; } u;
      u.i = (int32_t)(12102203.0f * x + 1064866805.0f);
      return u.f;
  }

void SimulatedAnnealing::apply(
    SolutionEval& solution,
    std::any& state,
    GlobalState& global_state,
    RNG& rng
) {
    auto* sa_state = std::any_cast<SAState>(&state);
    if (!sa_state) {
        state = init_state(solution);
        sa_state = std::any_cast<SAState>(&state);
    }

    // Compute current score before mutation
    float current_score = problem_->score(solution, global_state);

#ifndef NDEBUG
    // Save debug copy of full state before mutation
    thread_local DebugSavedState debug_state;
    debug_state.save_from(solution);
#endif
    // Mark checkpoint before applying inner optimizer
    size_t checkpoint = global_state.mark_checkpoint();

    // Apply inner optimizer
    RNG inner_rng = rng.split();
    inner_optimizer_->apply(
        solution, sa_state->inner_state, global_state, inner_rng
    );

    // Compute acceptance probability
    float candidate_score = problem_->score(solution, global_state);
    float delta = candidate_score - current_score;

    if (patience_ >= 0) {
        if (global_state.iters_since_improvement() <= 1) {
            counter_ = 0;
        }
        if (counter_ > patience_) {
            iteration_ = 0;
            counter_ = 0;
        }
    }

    bool accept;
    float temperature, accept_prob;
    if (delta <= 0.0f) {
        accept = true;
        temperature = -1.0f;
        accept_prob = 1.0f;
        ++n_accepted_;
    } else {
        temperature = compute_temperature(iteration_);
        accept_prob = std::min(fast_exp(-delta / temperature), 1.0f);
        accept = rng.uniform() < accept_prob;
        if (!accept) {
            // Rollback using checkpoint
            global_state.rollback_to(*problem_, solution, checkpoint);
#ifndef NDEBUG
            // Verify solution params match after rollback
            if (!solutions_match(debug_state.solution, solution.solution)) {
                std::cerr << "[SimulatedAnnealing] ERROR: Solution params not restored after rollback!\n";
                for (size_t i = 0; i < debug_state.solution.size(); ++i) {
                    if (debug_state.solution.is_valid(i) != solution.solution.is_valid(i)) {
                        std::cerr << "  idx=" << i << " valid mismatch\n";
                    } else if (debug_state.solution.is_valid(i)) {
                        TreeParams pa = debug_state.solution.get_params(i);
                        TreeParams pb = solution.solution.get_params(i);
                        if (std::abs(pa.pos.x - pb.pos.x) > 1e-6f ||
                            std::abs(pa.pos.y - pb.pos.y) > 1e-6f ||
                            std::abs(pa.angle - pb.angle) > 1e-6f) {
                            std::cerr << "  idx=" << i << " params: ("
                                << pa.pos.x << "," << pa.pos.y << "," << pa.angle << ") vs ("
                                << pb.pos.x << "," << pb.pos.y << "," << pb.angle << ")\n";
                        }
                    }
                }
                assert(false && "Solution params not restored after rollback");
            }
            // Verify cache is consistent with params
            if (!solution.solution.validate_cache(1e-5f, false)) {
                std::cerr << "[SimulatedAnnealing] ERROR: Cache invalid after rollback!\n";
                assert(false && "Cache invalid after rollback");
            }
            // Verify scores match
            if (!debug_state.compare_scores(solution)) {
                std::cerr << "[SimulatedAnnealing] ERROR: Scores don't match after rollback!\n";
                std::cerr << "  saved intersection_violation=" << debug_state.intersection_violation
                          << ", current=" << solution.intersection_violation << "\n";
                std::cerr << "  saved bounds_violation=" << debug_state.bounds_violation
                          << ", current=" << solution.bounds_violation << "\n";
                std::cerr << "  saved objective=" << debug_state.objective
                          << ", current=" << solution.objective << "\n";
                assert(false && "Scores don't match after rollback");
            }
#endif
            ++n_rejected_;
        } else {
            ++n_accepted_;
        }
    }
    last_accept_ = accept;
    last_checkpoint_ = checkpoint;

    // Update state
    ++iteration_;
    ++counter_;
    if (verbose_) {
        float rate = ((float) n_accepted_) / ((float) n_accepted_ + n_rejected_);
        std::cout << "[SimulatedAnnealing] temp=" << temperature << " acc-rate=" << rate;
        std::cout << " | score: " << current_score << "->" << candidate_score << " prob=" << accept_prob;
        std::cout << (accept ? "👍" : "👎") << "\n";
    }
}

OptimizerPtr SimulatedAnnealing::clone() const {
    return std::make_unique<SimulatedAnnealing>(
        inner_optimizer_->clone(),
        initial_temp_,
        min_temp_,
        cooling_schedule_,
        cooling_rate_,
        patience_,
        verbose_
    );
}

}  // namespace tree_packing
