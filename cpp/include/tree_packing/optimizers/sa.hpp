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

    // Compute temperature based on iteration
    float compute_temperature(int iteration) const;
};

}  // namespace tree_packing
