#include "tree_packing/optimizers/sa.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <utility>

namespace tree_packing {

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
    state.iteration = 0;
    state.counter = 0;
    state.temperature = initial_temp_;
    state.inner_state = inner_optimizer_->init_state(solution);
    state.n_accepted = 0;
    state.n_rejected = 0;
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

void SimulatedAnnealing::apply(
    const SolutionEval& solution,
    std::any& state,
    GlobalState& global_state,
    RNG& rng,
    SolutionEval& out
) {
    auto* sa_state = std::any_cast<SAState>(&state);
    if (!sa_state) {
        state = init_state(solution);
        sa_state = std::any_cast<SAState>(&state);
    }

    // Handle reheating with patience
    if (patience_ > 0) {
        if (global_state.iters_since_improvement() > static_cast<uint64_t>(patience_)) {
            // Reset temperature
            sa_state->iteration = 0;
            sa_state->counter = 0;
        }
    }

    // Apply inner optimizer
    RNG inner_rng = rng.split();
    inner_optimizer_->apply(
        solution, sa_state->inner_state, global_state, inner_rng, out
    );

    // Compute acceptance probability
    float current_score = problem_->score(solution, global_state);
    float candidate_score = problem_->score(out, global_state);
    float delta = candidate_score - current_score;

    float temperature = compute_temperature(sa_state->iteration);
    float accept_prob = std::min(std::exp(-delta / temperature), 1.0f);
    bool accept = rng.uniform() < accept_prob;

    if (accept) {
        sa_state->n_accepted++;
    } else {
        out = solution;
        sa_state->n_rejected++;
    }

    // Update state
    sa_state->iteration++;
    sa_state->counter++;
    sa_state->temperature = temperature;

    if (verbose_) {
        int total = sa_state->n_accepted + sa_state->n_rejected;
        float rate = (total > 0) ? static_cast<float>(sa_state->n_accepted) / total : 0.0f;
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
