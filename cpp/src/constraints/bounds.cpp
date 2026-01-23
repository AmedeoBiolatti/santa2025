#include "tree_packing/constraints/bounds.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace tree_packing {

BoundsConstraint::BoundsConstraint(float min_pos, float max_pos)
    : min_pos_(min_pos), max_pos_(max_pos) {}

float BoundsConstraint::eval(const Solution& solution) const {
    const auto& params = solution.params();
    float min_x = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float min_y = std::numeric_limits<float>::max();
    float max_y = std::numeric_limits<float>::lowest();
    bool has_valid = false;
    for (size_t i = 0; i < params.size(); ++i) {
        if (!solution.is_valid(i)) continue;
        has_valid = true;
        min_x = std::min(min_x, params.x[i]);
        max_x = std::max(max_x, params.x[i]);
        min_y = std::min(min_y, params.y[i]);
        max_y = std::max(max_y, params.y[i]);
    }

    if (!has_valid) {
        return 0.0f;
    }

    float violation = 0.0f;
    violation = std::max(violation, max_x - max_pos_);
    violation = std::max(violation, min_pos_ - min_x);
    violation = std::max(violation, max_y - max_pos_);
    violation = std::max(violation, min_pos_ - min_y);
    return std::max(0.0f, violation);
}

float BoundsConstraint::eval_update(float min_x, float max_x, float min_y, float max_y) const {
    float violation = 0.0f;
    violation = std::max(violation, max_x - max_pos_);
    violation = std::max(violation, min_pos_ - min_x);
    violation = std::max(violation, max_y - max_pos_);
    violation = std::max(violation, min_pos_ - min_y);
    return std::max(0.0f, violation);
}

}  // namespace tree_packing
