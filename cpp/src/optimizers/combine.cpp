#include "tree_packing/optimizers/combine.hpp"
#include <iostream>
#include <numeric>

namespace tree_packing {

// Chain implementation
Chain::Chain(std::vector<OptimizerPtr> optimizers, bool verbose)
    : optimizers_(std::move(optimizers))
    , verbose_(verbose)
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

void Chain::apply(
    const SolutionEval& solution,
    std::any& state,
    GlobalState& global_state,
    RNG& rng,
    SolutionEval& out
) {
    auto* states = std::any_cast<std::vector<std::any>>(&state);
    if (!states) {
        state = init_state(solution);
        states = std::any_cast<std::vector<std::any>>(&state);
    }
    SolutionEval current = solution;
    SolutionEval next;

    for (size_t i = 0; i < optimizers_.size(); ++i) {
        RNG step_rng = rng.split();
        optimizers_[i]->apply(current, (*states)[i], global_state, step_rng, next);
        current = next;
        if (verbose_) {
            std::cout << "[Chain] step=" << i << "\n";
        }
    }

    out = current;
}

OptimizerPtr Chain::clone() const {
    std::vector<OptimizerPtr> clones;
    clones.reserve(optimizers_.size());
    for (const auto& opt : optimizers_) {
        clones.push_back(opt->clone());
    }
    return std::make_unique<Chain>(std::move(clones), verbose_);
}

// Repeat implementation
Repeat::Repeat(OptimizerPtr optimizer, int n, bool verbose)
    : optimizer_(std::move(optimizer))
    , n_(n)
    , verbose_(verbose)
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

void Repeat::apply(
    const SolutionEval& solution,
    std::any& state,
    GlobalState& global_state,
    RNG& rng,
    SolutionEval& out
) {
    if (n_ == 0) {
        out = solution;
        return;
    }

    SolutionEval current = solution;
    SolutionEval next;

    for (int i = 0; i < n_; ++i) {
        RNG step_rng = rng.split();
        optimizer_->apply(current, state, global_state, step_rng, next);
        current = next;
        if (verbose_) {
            std::cout << "[Repeat] iter=" << i + 1 << "/" << n_ << "\n";
        }
    }

    out = current;
}

OptimizerPtr Repeat::clone() const {
    return std::make_unique<Repeat>(optimizer_->clone(), n_, verbose_);
}

// RestoreBest implementation
RestoreBest::RestoreBest(OptimizerPtr optimizer, int patience, bool verbose)
    : optimizer_(std::move(optimizer))
    , patience_(patience)
    , verbose_(verbose)
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

void RestoreBest::apply(
    const SolutionEval& solution,
    std::any& state,
    GlobalState& global_state,
    RNG& rng,
    SolutionEval& out
) {
    SolutionEval input = solution;

    // Restore best if no improvement for patience iterations
    if (global_state.iters_since_improvement() >= static_cast<uint64_t>(patience_)) {
        const auto* best = global_state.best_solution();
        if (best != nullptr) {
            input = *best;
            if (verbose_) {
                std::cout << "[RestoreBest] restored best\n";
            }
        } else if (verbose_) {
            std::cout << "[RestoreBest] no best to restore\n";
        }
    }

    optimizer_->apply(input, state, global_state, rng, out);
}

OptimizerPtr RestoreBest::clone() const {
    return std::make_unique<RestoreBest>(optimizer_->clone(), patience_, verbose_);
}

// RandomChoice implementation
RandomChoice::RandomChoice(
    std::vector<OptimizerPtr> optimizers,
    std::vector<float> probabilities,
    bool verbose
)
    : optimizers_(std::move(optimizers))
    , probabilities_(std::move(probabilities))
    , verbose_(verbose)
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

void RandomChoice::apply(
    const SolutionEval& solution,
    std::any& state,
    GlobalState& global_state,
    RNG& rng,
    SolutionEval& out
) {
    auto* states = std::any_cast<std::vector<std::any>>(&state);
    if (!states) {
        state = init_state(solution);
        states = std::any_cast<std::vector<std::any>>(&state);
    }

    // Select optimizer based on probabilities
    int idx = rng.weighted_choice(probabilities_);

    // Apply selected optimizer
    RNG step_rng = rng.split();
    optimizers_[idx]->apply(solution, (*states)[idx], global_state, step_rng, out);

    if (verbose_) {
        std::cout << "[RandomChoice] idx=" << idx << "\n";
    }

    return;
}

OptimizerPtr RandomChoice::clone() const {
    std::vector<OptimizerPtr> clones;
    clones.reserve(optimizers_.size());
    for (const auto& opt : optimizers_) {
        clones.push_back(opt->clone());
    }
    return std::make_unique<RandomChoice>(std::move(clones), probabilities_, verbose_);
}

}  // namespace tree_packing
