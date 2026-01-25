#include "tree_packing/optimizers/alns.hpp"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <utility>

namespace tree_packing {

ALNS::ALNS(
    std::vector<OptimizerPtr> ruin_operators,
    std::vector<OptimizerPtr> recreate_operators,
    float reaction_factor,
    float reward_improve,
    float reward_no_improve,
    float min_weight,
    bool verbose
)
    : ruin_operators_(std::move(ruin_operators))
    , recreate_operators_(std::move(recreate_operators))
    , reaction_factor_(reaction_factor)
    , reward_improve_(reward_improve)
    , reward_no_improve_(reward_no_improve)
    , min_weight_(min_weight)
    , verbose_(verbose)
{}

void ALNS::set_problem(Problem* problem) {
    Optimizer::set_problem(problem);
    for (auto& opt : ruin_operators_) {
        opt->set_problem(problem);
    }
    for (auto& opt : recreate_operators_) {
        opt->set_problem(problem);
    }
}

std::any ALNS::init_state(const SolutionEval& solution) {
    ALNSState state;
    state.iteration = 0;

    // Initialize ruin states
    for (auto& opt : ruin_operators_) {
        state.ruin_states.push_back(opt->init_state(solution));
    }

    // Initialize recreate states
    for (auto& opt : recreate_operators_) {
        state.recreate_states.push_back(opt->init_state(solution));
    }

    // Initialize weights to 1.0
    state.ruin_weights.resize(ruin_operators_.size(), 1.0f);
    state.recreate_weights.resize(recreate_operators_.size(), 1.0f);

    return state;
}

int ALNS::select_index(const std::vector<float>& weights, RNG& rng) {
    return rng.weighted_choice(weights);
}

void ALNS::update_weights(std::vector<float>& weights, int index, float reward) {
    for (size_t i = 0; i < weights.size(); ++i) {
        if (static_cast<int>(i) == index) {
            weights[i] = (1.0f - reaction_factor_) * weights[i] + reaction_factor_ * reward;
        } else {
            weights[i] = (1.0f - reaction_factor_) * weights[i];
        }
        weights[i] = std::max(weights[i], min_weight_);
    }
}

void ALNS::apply(
    SolutionEval& solution,
    std::any& state,
    GlobalState& global_state,
    RNG& rng
) {
    auto* alns_state = std::any_cast<ALNSState>(&state);
    if (!alns_state) {
        state = init_state(solution);
        alns_state = std::any_cast<ALNSState>(&state);
    }

    // Evaluate current solution to compare
    float current_score = problem_->score(solution, global_state);

    // Select ruin operator
    int ruin_idx = select_index(alns_state->ruin_weights, rng);
    RNG ruin_rng = rng.split();
    alns_state->last_ruin_idx = ruin_idx;

    // Apply ruin in-place
    ruin_operators_[ruin_idx]->apply(
        solution, alns_state->ruin_states[ruin_idx], global_state, ruin_rng
    );

    // Select recreate operator
    int recreate_idx = select_index(alns_state->recreate_weights, rng);
    RNG recreate_rng = rng.split();
    alns_state->last_recreate_idx = recreate_idx;

    // Apply recreate in-place
    recreate_operators_[recreate_idx]->apply(
        solution, alns_state->recreate_states[recreate_idx], global_state, recreate_rng
    );

    // Evaluate both solutions to compare
    float new_score = problem_->score(solution, global_state);

    // Compute reward
    float reward = (new_score < current_score) ? reward_improve_ : reward_no_improve_;

    // Update weights
    update_weights(alns_state->ruin_weights, ruin_idx, reward);
    update_weights(alns_state->recreate_weights, recreate_idx, reward);

    alns_state->iteration++;

    if (verbose_) {
        std::cout << "[ALNS] iter=" << alns_state->iteration
                  << " ruin=" << ruin_idx
                  << " recreate=" << recreate_idx
                  << " score: " << current_score << "->" << new_score
                  << " reward=" << reward << "\n";
    }
}

OptimizerPtr ALNS::clone() const {
    std::vector<OptimizerPtr> ruin_clones;
    std::vector<OptimizerPtr> recreate_clones;

    ruin_clones.reserve(ruin_operators_.size());
    for (const auto& opt : ruin_operators_) {
        ruin_clones.push_back(opt->clone());
    }

    recreate_clones.reserve(recreate_operators_.size());
    for (const auto& opt : recreate_operators_) {
        recreate_clones.push_back(opt->clone());
    }

    return std::make_unique<ALNS>(
        std::move(ruin_clones),
        std::move(recreate_clones),
        reaction_factor_,
        reward_improve_,
        reward_no_improve_,
        min_weight_,
        verbose_
    );
}

}  // namespace tree_packing
