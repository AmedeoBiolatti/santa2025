#include "tree_packing/optimizers/combine.hpp"
#include <iostream>
#include <limits>
#include <numeric>

namespace tree_packing {
namespace {
struct RepeatState {
    std::any inner_state;
    std::vector<std::any> history;
    std::any initial_state;
};

struct RestoreBestState {
    bool restored{false};
    uint64_t last_iteration{std::numeric_limits<uint64_t>::max()};
};

struct RandomChoiceState {
    std::vector<std::any> states;
    int last_idx{-1};
};
}  // namespace

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
    SolutionEval& solution,
    std::any& state,
    GlobalState& global_state,
    RNG& rng
) {
    auto* states = std::any_cast<std::vector<std::any>>(&state);
    if (!states) {
        state = init_state(solution);
        states = std::any_cast<std::vector<std::any>>(&state);
    }

    for (size_t i = 0; i < optimizers_.size(); ++i) {
        RNG step_rng = rng.split();
        optimizers_[i]->apply(solution, (*states)[i], global_state, step_rng);
        if (verbose_) {
            std::cout << "[Chain] step=" << i << "\n";
        }
    }
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
    RepeatState state;
    state.inner_state = optimizer_->init_state(solution);
    state.history.clear();
    state.initial_state = state.inner_state;
    return state;
}

void Repeat::apply(
    SolutionEval& solution,
    std::any& state,
    GlobalState& global_state,
    RNG& rng
) {
    auto* rep_state = std::any_cast<RepeatState>(&state);
    if (!rep_state) {
        state = RepeatState{optimizer_->init_state(solution), {}, {}};
        rep_state = std::any_cast<RepeatState>(&state);
    }
    if (n_ == 0) {
        return;
    }
    rep_state->initial_state = rep_state->inner_state;
    rep_state->history.clear();

    for (int i = 0; i < n_; ++i) {
        RNG step_rng = rng.split();
        optimizer_->apply(solution, rep_state->inner_state, global_state, step_rng);
        rep_state->history.push_back(rep_state->inner_state);
        if (verbose_) {
            std::cout << "[Repeat] iter=" << i + 1 << "/" << n_ << "\n";
        }
    }
}

OptimizerPtr Repeat::clone() const {
    return std::make_unique<Repeat>(optimizer_->clone(), n_, verbose_);
}

// RestoreBest implementation
RestoreBest::RestoreBest(int interval, bool verbose)
    : interval_(interval)
    , verbose_(verbose)
{
    if (interval_ < 0) interval_ = 0;
}

std::any RestoreBest::init_state([[maybe_unused]] const SolutionEval& solution) {
    RestoreBestState state;
    state.restored = false;
    state.last_iteration = std::numeric_limits<uint64_t>::max();
    return state;
}

void RestoreBest::apply(
    SolutionEval& solution,
    std::any& state,
    GlobalState& global_state,
    RNG& rng
) {
    auto* rb_state = std::any_cast<RestoreBestState>(&state);
    if (!rb_state) {
        state = init_state(solution);
        rb_state = std::any_cast<RestoreBestState>(&state);
    }
    rb_state->restored = false;

    uint64_t iteration = global_state.iteration();
    if (interval_ > 0 && rb_state->last_iteration != iteration) {
        rb_state->last_iteration = iteration;
        if ((iteration + 1) % static_cast<uint64_t>(interval_) == 0) {
            const auto* best_params = global_state.best_params();
            if (best_params != nullptr && problem_ != nullptr) {
                problem_->restore_from_params(solution, *best_params);
                rb_state->restored = true;
                if (verbose_) {
                    std::cout << "[RestoreBest] restored best\n";
                }
            } else if (verbose_) {
                std::cout << "[RestoreBest] no best to restore\n";
            }
        }
    }
    (void)rng;
}

OptimizerPtr RestoreBest::clone() const {
    return std::make_unique<RestoreBest>(interval_, verbose_);
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
    RandomChoiceState state;
    for (auto& opt : optimizers_) {
        state.states.push_back(opt->init_state(solution));
    }
    return state;
}

void RandomChoice::apply(
    SolutionEval& solution,
    std::any& state,
    GlobalState& global_state,
    RNG& rng
) {
    auto* rc_state = std::any_cast<RandomChoiceState>(&state);
    if (!rc_state) {
        state = init_state(solution);
        rc_state = std::any_cast<RandomChoiceState>(&state);
    }

    // Select optimizer based on probabilities
    int idx = rng.weighted_choice(probabilities_);
    rc_state->last_idx = idx;

    // Apply selected optimizer
    RNG step_rng = rng.split();
    optimizers_[idx]->apply(solution, rc_state->states[idx], global_state, step_rng);

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
