#pragma once

#include "../core/solution.hpp"

namespace tree_packing {

// Bounds constraint: penalizes trees outside the allowed region
class BoundsConstraint {
public:
    BoundsConstraint(float min_pos = -10.0f, float max_pos = 10.0f);

    // Evaluate bounds violation
    [[nodiscard]] float eval(const Solution& solution) const;

private:
    float min_pos_;
    float max_pos_;
};

}  // namespace tree_packing
