#pragma once

#include "optimizer.hpp"
#include <string>
#include <memory>

namespace tree_packing {

// State for Simulated Annealing
struct SAState {
    std::any inner_state;
};

// Cooling schedule type
enum class CoolingSchedule {
    Exponential,
    Linear,
    Logarithmic
};

// Simulated Annealing wrapper around another optimizer
class SimulatedAnnealing : public Optimizer {
public:
    SimulatedAnnealing(
        OptimizerPtr inner_optimizer,
        float initial_temp = 1.0f,
        float min_temp = 1e-6f,
        CoolingSchedule cooling_schedule = CoolingSchedule::Exponential,
        float cooling_rate = 0.995f,
        int patience = -1,  // -1 means no patience (no reheating)
        bool verbose = false
    );

    void set_problem(Problem* problem) override;

    std::any init_state(const SolutionEval& solution) override;

    void apply(
        SolutionEval& solution,
        std::any& state,
        GlobalState& global_state,
        RNG& rng
    ) override;

    [[nodiscard]] OptimizerPtr clone() const override;

    [[nodiscard]] float last_temperature() const { return last_temperature_; }
    [[nodiscard]] float accept_rate() const {
        float total = static_cast<float>(n_accepted_ + n_rejected_);
        return total > 0.0f ? static_cast<float>(n_accepted_) / total : 0.0f;
    }
    [[nodiscard]] float last_accept_prob() const { return last_accept_prob_; }
    [[nodiscard]] size_t accepted_count() const { return n_accepted_; }
    [[nodiscard]] size_t rejected_count() const { return n_rejected_; }
    [[nodiscard]] int iteration() const { return iteration_; }
    [[nodiscard]] bool last_accept() const { return last_accept_; }

private:
    OptimizerPtr inner_optimizer_;
    float initial_temp_;
    float min_temp_;
    CoolingSchedule cooling_schedule_;
    float cooling_rate_;
    int patience_;
    bool verbose_;

    int iteration_ = 0;
    int counter_ = 0;
    bool last_accept_ = false;
    size_t last_checkpoint_ = 0;
    size_t n_accepted_ = 0;
    size_t n_rejected_ = 0;
    float last_temperature_{0.0f};
    float last_accept_prob_{0.0f};

    // Compute temperature based on iteration
    float compute_temperature(int iteration) const;
};

}  // namespace tree_packing
