#include "tree_packing/core/tree.hpp"

namespace tree_packing {

void params_to_figures(const TreeParamsSoA& params, std::vector<Figure>& figures) {
    size_t n = params.size();
    figures.resize(n);

    for (size_t i = 0; i < n; ++i) {
        TreeParams p = params.get(i);
        figures[i] = params_to_figure(p);
    }
}

void get_tree_centers(const TreeParamsSoA& params, std::vector<Vec2>& centers) {
    size_t n = params.size();
    centers.resize(n);

    for (size_t i = 0; i < n; ++i) {
        centers[i] = get_tree_center(params.x[i], params.y[i], params.angle[i]);
    }
}

}  // namespace tree_packing
