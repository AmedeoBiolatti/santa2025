#include "tree_packing/optimizers/combine.hpp"
#include <numeric>

namespace tree_packing {

// Chain implementation
Chain::Chain(std::vector<OptimizerPtr> optimizers)
    : optimizers_(std::move(optimizers))
{}

void Chain::set_problem(Problem* problem) {
    Optimizer::set_problem(problem);
    for (auto& opt : optimizers_) {
        opt->set_problem(problem);
    }
}

std::any Chain::init_state(const SolutionEval& solution) {
    std::vector<std::any> states;
    for (auto& opt : optimizers_) {
        states.push_back(opt->init_state(solution));
    }
    return states;
}

std::pair<SolutionEval, std::any> Chain::apply(
    const SolutionEval& solution,
    std::any& state,
    GlobalState& global_state,
    RNG& rng
) {
    auto states = std::any_cast<std::vector<std::any>>(state);
    SolutionEval current = solution;

    for (size_t i = 0; i < optimizers_.size(); ++i) {
        RNG step_rng = rng.split();
        auto [new_sol, new_state] = optimizers_[i]->apply(
            current, states[i], global_state, step_rng
        );
        current = new_sol;
        states[i] = new_state;
    }

    return {current, states};
}

OptimizerPtr Chain::clone() const {
    std::vector<OptimizerPtr> clones;
    clones.reserve(optimizers_.size());
    for (const auto& opt : optimizers_) {
        clones.push_back(opt->clone());
    }
    return std::make_unique<Chain>(std::move(clones));
}

// Repeat implementation
Repeat::Repeat(OptimizerPtr optimizer, int n)
    : optimizer_(std::move(optimizer)), n_(n)
{
    if (n_ < 0) n_ = 0;
}

void Repeat::set_problem(Problem* problem) {
    Optimizer::set_problem(problem);
    optimizer_->set_problem(problem);
}

std::any Repeat::init_state(const SolutionEval& solution) {
    return optimizer_->init_state(solution);
}

std::pair<SolutionEval, std::any> Repeat::apply(
    const SolutionEval& solution,
    std::any& state,
    GlobalState& global_state,
    RNG& rng
) {
    if (n_ == 0) {
        return {solution, state};
    }

    SolutionEval current = solution;
    std::any current_state = state;

    for (int i = 0; i < n_; ++i) {
        RNG step_rng = rng.split();
        auto [new_sol, new_state] = optimizer_->apply(
            current, current_state, global_state, step_rng
        );
        current = new_sol;
        current_state = new_state;
    }

    return {current, current_state};
}

OptimizerPtr Repeat::clone() const {
    return std::make_unique<Repeat>(optimizer_->clone(), n_);
}

// RestoreBest implementation
RestoreBest::RestoreBest(OptimizerPtr optimizer, int patience)
    : optimizer_(std::move(optimizer)), patience_(patience)
{
    if (patience_ < 0) patience_ = 0;
}

void RestoreBest::set_problem(Problem* problem) {
    Optimizer::set_problem(problem);
    optimizer_->set_problem(problem);
}

std::any RestoreBest::init_state(const SolutionEval& solution) {
    return optimizer_->init_state(solution);
}

std::pair<SolutionEval, std::any> RestoreBest::apply(
    const SolutionEval& solution,
    std::any& state,
    GlobalState& global_state,
    RNG& rng
) {
    SolutionEval input = solution;

    // Restore best if no improvement for patience iterations
    if (global_state.iters_since_improvement() >= static_cast<uint64_t>(patience_)) {
        const auto* best = global_state.best_solution();
        if (best != nullptr) {
            input = *best;
        }
    }

    return optimizer_->apply(input, state, global_state, rng);
}

OptimizerPtr RestoreBest::clone() const {
    return std::make_unique<RestoreBest>(optimizer_->clone(), patience_);
}

// RandomChoice implementation
RandomChoice::RandomChoice(
    std::vector<OptimizerPtr> optimizers,
    std::vector<float> probabilities
)
    : optimizers_(std::move(optimizers))
    , probabilities_(std::move(probabilities))
{
    // If no probabilities given, use uniform
    if (probabilities_.empty()) {
        probabilities_.resize(optimizers_.size(), 1.0f / optimizers_.size());
    }

    // Normalize probabilities
    float sum = std::accumulate(probabilities_.begin(), probabilities_.end(), 0.0f);
    for (auto& p : probabilities_) {
        p /= sum;
    }
}

void RandomChoice::set_problem(Problem* problem) {
    Optimizer::set_problem(problem);
    for (auto& opt : optimizers_) {
        opt->set_problem(problem);
    }
}

std::any RandomChoice::init_state(const SolutionEval& solution) {
    std::vector<std::any> states;
    for (auto& opt : optimizers_) {
        states.push_back(opt->init_state(solution));
    }
    return states;
}

std::pair<SolutionEval, std::any> RandomChoice::apply(
    const SolutionEval& solution,
    std::any& state,
    GlobalState& global_state,
    RNG& rng
) {
    auto states = std::any_cast<std::vector<std::any>>(state);

    // Select optimizer based on probabilities
    int idx = rng.weighted_choice(probabilities_);

    // Apply selected optimizer
    RNG step_rng = rng.split();
    auto [new_sol, new_state] = optimizers_[idx]->apply(
        solution, states[idx], global_state, step_rng
    );
    states[idx] = new_state;

    return {new_sol, states};
}

OptimizerPtr RandomChoice::clone() const {
    std::vector<OptimizerPtr> clones;
    clones.reserve(optimizers_.size());
    for (const auto& opt : optimizers_) {
        clones.push_back(opt->clone());
    }
    return std::make_unique<RandomChoice>(std::move(clones), probabilities_);
}

}  // namespace tree_packing
