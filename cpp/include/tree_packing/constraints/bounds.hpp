#pragma once

#include "../core/solution.hpp"

namespace tree_packing {

// Bounds constraint: penalizes trees outside the allowed region
class BoundsConstraint {
public:
    BoundsConstraint(float min_pos = -10.0f, float max_pos = 10.0f);

    // Evaluate bounds violation
    [[nodiscard]] float eval(const Solution& solution) const;
    // Evaluate bounds violation from precomputed min/max
    [[nodiscard]] float eval_update(float min_x, float max_x, float min_y, float max_y) const;

private:
    float min_pos_;
    float max_pos_;
};

}  // namespace tree_packing
