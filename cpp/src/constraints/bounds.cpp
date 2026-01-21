#include "tree_packing/constraints/bounds.hpp"
#include <algorithm>
#include <cmath>

namespace tree_packing {

BoundsConstraint::BoundsConstraint(float min_pos, float max_pos)
    : min_pos_(min_pos), max_pos_(max_pos) {}

float BoundsConstraint::eval(const Solution& solution) const {
    float violation = 0.0f;

    const auto& params = solution.params();
    for (size_t i = 0; i < params.size(); ++i) {
        if (!solution.is_valid(i)) continue;

        // Violation for x
        float x = params.x[i];
        if (x > max_pos_) violation += (x - max_pos_);
        if (x < min_pos_) violation += (min_pos_ - x);

        // Violation for y
        float y = params.y[i];
        if (y > max_pos_) violation += (y - max_pos_);
        if (y < min_pos_) violation += (min_pos_ - y);
    }

    return violation;
}

}  // namespace tree_packing
