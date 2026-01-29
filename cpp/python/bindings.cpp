#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <any>

#include "tree_packing/tree_packing.hpp"
#include "tree_packing/constraints/intersection_tree_filter.hpp"

namespace py = pybind11;

namespace {
struct OptimizerState {
    std::any value;
};

template <typename T>
py::array_t<T> make_array_view(const std::vector<T>& data, py::handle base) {
    return py::array_t<T>(
        {static_cast<py::ssize_t>(data.size())},
        {static_cast<py::ssize_t>(sizeof(T))},
        data.data(),
        base
    );
}

tree_packing::OptimizerPtr clone_optimizer(
    const std::shared_ptr<tree_packing::Optimizer>& optimizer
) {
    if (!optimizer) {
        throw py::value_error("Optimizer is null");
    }
    return optimizer->clone();
}
}  // namespace

PYBIND11_MODULE(tree_packing_cpp, m) {
    m.doc() = "C++ implementation of tree packing optimization";

    // Vec2
    py::class_<tree_packing::Vec2>(m, "Vec2")
        .def(py::init<>())
        .def(py::init<float, float>())
        .def_readwrite("x", &tree_packing::Vec2::x)
        .def_readwrite("y", &tree_packing::Vec2::y)
        .def("__repr__", [](const tree_packing::Vec2& v) {
            return "Vec2(" + std::to_string(v.x) + ", " + std::to_string(v.y) + ")";
        });

    py::class_<tree_packing::AABB>(m, "AABB")
        .def(py::init<>())
        .def(py::init<const tree_packing::Vec2&, const tree_packing::Vec2&>())
        .def_readwrite("min", &tree_packing::AABB::min)
        .def_readwrite("max", &tree_packing::AABB::max)
        .def("center", &tree_packing::AABB::center)
        .def("size", &tree_packing::AABB::size)
        .def("contains", &tree_packing::AABB::contains)
        .def("intersects", &tree_packing::AABB::intersects);

    py::class_<tree_packing::Grid2D>(m, "Grid2D")
        .def(py::init<>())
        .def_static("empty", &tree_packing::Grid2D::empty,
            py::arg("num_items"),
            py::arg("n") = 20,
            py::arg("size") = 1.04f,
            py::arg("capacity") = 16,
            py::arg("center") = 0.0f)
        .def_static("init", &tree_packing::Grid2D::init,
            py::arg("centers"),
            py::arg("n") = 16,
            py::arg("capacity") = 8,
            py::arg("size") = 1.04f,
            py::arg("center") = 0.0f)
        .def("grid_n", &tree_packing::Grid2D::grid_n)
        .def("grid_N", &tree_packing::Grid2D::grid_N)
        .def("capacity", &tree_packing::Grid2D::capacity)
        .def("cell_size", &tree_packing::Grid2D::cell_size)
        .def("center", &tree_packing::Grid2D::center)
        .def("compute_ij", &tree_packing::Grid2D::compute_ij)
        .def("get_item_cell", &tree_packing::Grid2D::get_item_cell)
        .def("cell_bounds", &tree_packing::Grid2D::cell_bounds)
        .def("cell_bounds_expanded", &tree_packing::Grid2D::cell_bounds_expanded)
        .def("cell_count", &tree_packing::Grid2D::cell_count)
        .def("get_candidates", [](const tree_packing::Grid2D& self, int k) {
            return self.get_candidates(k);
        })
        .def("get_candidates_by_pos", [](const tree_packing::Grid2D& self, const tree_packing::Vec2& pos) {
            return self.get_candidates_by_pos(pos);
        })
        .def("get_candidates_by_cell", [](const tree_packing::Grid2D& self, int i, int j) {
            return self.get_candidates_by_cell(i, j);
        })
        .def("get_items_in_cell", [](const tree_packing::Grid2D& self, int i, int j) {
            return self.get_items_in_cell(i, j);
        })
        .def("non_empty_cells", [](const tree_packing::Grid2D& self) {
            return py::cast(self.non_empty_cells());
        })
        .def_property_readonly("i2n", [](const tree_packing::Grid2D& self) {
            return make_array_view(self.i2n(), py::cast(&self));
        })
        .def_property_readonly("j2n", [](const tree_packing::Grid2D& self) {
            return make_array_view(self.j2n(), py::cast(&self));
        })
        .def_property_readonly("ij2k", [](const tree_packing::Grid2D& self) {
            return make_array_view(self.ij2k(), py::cast(&self));
        })
        .def_property_readonly("ij2n", [](const tree_packing::Grid2D& self) {
            return make_array_view(self.ij2n(), py::cast(&self));
        })
        .def_property_readonly("k2ij", [](const tree_packing::Grid2D& self) {
            return make_array_view(self.k2ij(), py::cast(&self));
        })
        .def_property_readonly("cell_to_non_empty_idx", [](const tree_packing::Grid2D& self) {
            return make_array_view(self.cell_to_non_empty_idx(), py::cast(&self));
        })
        .def_property_readonly("cell_bounds_data", [](const tree_packing::Grid2D& self) {
            return py::cast(self.cell_bounds_list());
        })
        .def_property_readonly("cell_bounds_expanded_data", [](const tree_packing::Grid2D& self) {
            return py::cast(self.cell_bounds_expanded_list());
        })
        .def_property_readonly("min_i", &tree_packing::Grid2D::min_i)
        .def_property_readonly("max_i", &tree_packing::Grid2D::max_i)
        .def_property_readonly("min_j", &tree_packing::Grid2D::min_j)
        .def_property_readonly("max_j", &tree_packing::Grid2D::max_j);

    // TreeParams
    py::class_<tree_packing::TreeParams>(m, "TreeParams")
        .def(py::init<>())
        .def(py::init<tree_packing::Vec2, float>())
        .def(py::init<float, float, float>())
        .def_readwrite("pos", &tree_packing::TreeParams::pos)
        .def_readwrite("angle", &tree_packing::TreeParams::angle)
        .def("is_nan", &tree_packing::TreeParams::is_nan);

    // TreeParamsSoA
    py::class_<tree_packing::TreeParamsSoA>(m, "TreeParamsSoA")
        .def(py::init<>())
        .def(py::init<size_t>())
        .def("size", &tree_packing::TreeParamsSoA::size)
        .def("get", &tree_packing::TreeParamsSoA::get)
        .def("set", &tree_packing::TreeParamsSoA::set)
        .def("set_nan", &tree_packing::TreeParamsSoA::set_nan)
        .def("is_nan", &tree_packing::TreeParamsSoA::is_nan)
        .def("count_nan", &tree_packing::TreeParamsSoA::count_nan)
        .def("to_numpy", [](const tree_packing::TreeParamsSoA& self) {
            py::ssize_t n = static_cast<py::ssize_t>(self.size());
            std::vector<py::ssize_t> shape = {n, 2};
            std::vector<py::ssize_t> strides = {
                static_cast<py::ssize_t>(2 * sizeof(float)),
                static_cast<py::ssize_t>(sizeof(float))
            };
            py::array_t<float> pos(shape, strides);
            auto buf = pos.mutable_unchecked<2>();
            for (py::ssize_t i = 0; i < n; ++i) {
                buf(i, 0) = self.x[static_cast<size_t>(i)];
                buf(i, 1) = self.y[static_cast<size_t>(i)];
            }
            return py::make_tuple(pos, make_array_view(self.angle, py::cast(&self)));
        })
        .def_property("x",
            [](const tree_packing::TreeParamsSoA& self) {
                return make_array_view(self.x, py::cast(&self));
            },
            [](tree_packing::TreeParamsSoA& self, py::array_t<float> arr) {
                auto buf = arr.request();
                float* ptr = static_cast<float*>(buf.ptr);
                self.x.assign(ptr, ptr + buf.size);
            })
        .def_property("y",
            [](const tree_packing::TreeParamsSoA& self) {
                return make_array_view(self.y, py::cast(&self));
            },
            [](tree_packing::TreeParamsSoA& self, py::array_t<float> arr) {
                auto buf = arr.request();
                float* ptr = static_cast<float*>(buf.ptr);
                self.y.assign(ptr, ptr + buf.size);
            })
        .def_property("angle",
            [](const tree_packing::TreeParamsSoA& self) {
                return make_array_view(self.angle, py::cast(&self));
            },
            [](tree_packing::TreeParamsSoA& self, py::array_t<float> arr) {
                auto buf = arr.request();
                float* ptr = static_cast<float*>(buf.ptr);
                self.angle.assign(ptr, ptr + buf.size);
            });

    m.def("params_to_figures", [](const tree_packing::TreeParamsSoA& params) {
        std::vector<tree_packing::Figure> figures;
        figures.resize(params.size());
        tree_packing::params_to_figures(params, figures);

        py::ssize_t n = static_cast<py::ssize_t>(figures.size());
        py::ssize_t t = static_cast<py::ssize_t>(tree_packing::TREE_NUM_TRIANGLES);
        py::array_t<float> arr({n, t, static_cast<py::ssize_t>(3), static_cast<py::ssize_t>(2)});
        auto buf = arr.mutable_unchecked<4>();
        for (py::ssize_t i = 0; i < n; ++i) {
            const auto& fig = figures[static_cast<size_t>(i)];
            for (py::ssize_t ti = 0; ti < t; ++ti) {
                const auto& tri = fig.triangles[static_cast<size_t>(ti)];
                buf(i, ti, 0, 0) = tri.v0.x;
                buf(i, ti, 0, 1) = tri.v0.y;
                buf(i, ti, 1, 0) = tri.v1.x;
                buf(i, ti, 1, 1) = tri.v1.y;
                buf(i, ti, 2, 0) = tri.v2.x;
                buf(i, ti, 2, 1) = tri.v2.y;
            }
        }
        return arr;
    });

    m.def("triangle_pair_intersection_scores", [](
        const tree_packing::TreeParams& a,
        const tree_packing::TreeParams& b,
        float eps
    ) {
        constexpr py::ssize_t t = static_cast<py::ssize_t>(tree_packing::TREE_NUM_TRIANGLES);
        py::array_t<float> arr({t, t});
        auto buf = arr.mutable_unchecked<2>();

        const auto scores = tree_packing::triangle_pair_intersection_scores(a, b, eps);
        for (py::ssize_t i = 0; i < t; ++i) {
            for (py::ssize_t j = 0; j < t; ++j) {
                buf(i, j) = scores[static_cast<size_t>(i)][static_cast<size_t>(j)];
            }
        }
        return arr;
    }, py::arg("a"), py::arg("b"), py::arg("eps") = tree_packing::EPSILON,
       "Return a (5, 5) numpy array of per-triangle intersection scores.");

    // Solution
    py::class_<tree_packing::Solution>(m, "Solution")
        .def(py::init<>())
        .def(py::init<size_t>())
        .def_static("init_random", &tree_packing::Solution::init_random,
            py::arg("num_trees"), py::arg("side") = 10.0f, py::arg("seed") = 42)
        .def_static("init_empty", &tree_packing::Solution::init_empty)
        .def_static("from_numpy", [](
            py::array_t<float> positions,
            py::array_t<float> angles,
            int grid_n,
            float grid_size,
            int grid_capacity
        ) {
            auto pos_buf = positions.unchecked<2>();
            auto ang_buf = angles.unchecked<1>();

            py::ssize_t n = pos_buf.shape(0);
            if (pos_buf.shape(1) != 2) {
                throw py::value_error("positions must have shape (N, 2)");
            }
            if (ang_buf.shape(0) != n) {
                throw py::value_error("angles must have same length as positions");
            }

            tree_packing::TreeParamsSoA params(static_cast<size_t>(n));
            for (py::ssize_t i = 0; i < n; ++i) {
                params.x[static_cast<size_t>(i)] = pos_buf(i, 0);
                params.y[static_cast<size_t>(i)] = pos_buf(i, 1);
                params.angle[static_cast<size_t>(i)] = ang_buf(i);
            }
            return tree_packing::Solution::init(params, grid_n, grid_size, grid_capacity);
        },
            py::arg("positions"),
            py::arg("angles"),
            py::arg("grid_n") = 16,
            py::arg("grid_size") = tree_packing::THR,
            py::arg("grid_capacity") = 8,
            "Create Solution from numpy arrays. positions: (N, 2), angles: (N,)")
        .def("size", &tree_packing::Solution::size)
        .def("n_missing", &tree_packing::Solution::n_missing)
        .def("reg", &tree_packing::Solution::reg)
        .def("params", py::overload_cast<>(&tree_packing::Solution::params),
            py::return_value_policy::reference_internal)
        .def("params", py::overload_cast<>(&tree_packing::Solution::params, py::const_),
            py::return_value_policy::reference_internal)
        .def("get_params", &tree_packing::Solution::get_params)
        .def("set_params", &tree_packing::Solution::set_params)
        .def("set_nan", &tree_packing::Solution::set_nan)
        .def("recompute_cache", &tree_packing::Solution::recompute_cache)
        .def("grid", [](const tree_packing::Solution& self) -> const tree_packing::Grid2D& {
            return self.grid();
        }, py::return_value_policy::reference_internal)
        .def("centers", [](const tree_packing::Solution& self) {
            py::ssize_t n = static_cast<py::ssize_t>(self.centers().size());
            py::array_t<float> arr({n, static_cast<py::ssize_t>(2)});
            auto buf = arr.mutable_unchecked<2>();
            for (py::ssize_t i = 0; i < n; ++i) {
                const auto& c = self.centers()[static_cast<size_t>(i)];
                buf(i, 0) = c.x;
                buf(i, 1) = c.y;
            }
            return arr;
        })
        .def("aabbs", [](const tree_packing::Solution& self) {
            py::ssize_t n = static_cast<py::ssize_t>(self.aabbs().size());
            py::array_t<float> arr({n, static_cast<py::ssize_t>(4)});
            auto buf = arr.mutable_unchecked<2>();
            for (py::ssize_t i = 0; i < n; ++i) {
                const auto& aabb = self.aabbs()[static_cast<size_t>(i)];
                buf(i, 0) = aabb.min.x;
                buf(i, 1) = aabb.min.y;
                buf(i, 2) = aabb.max.x;
                buf(i, 3) = aabb.max.y;
            }
            return arr;
        })
        .def("max_abs", [](const tree_packing::Solution& self) {
            return make_array_view(self.max_abs(), py::cast(&self));
        })
        .def("figures", [](const tree_packing::Solution& self) {
            py::ssize_t n = static_cast<py::ssize_t>(self.figures().size());
            py::ssize_t t = static_cast<py::ssize_t>(tree_packing::TREE_NUM_TRIANGLES);
            py::array_t<float> arr({n, t, static_cast<py::ssize_t>(3), static_cast<py::ssize_t>(2)});
            auto buf = arr.mutable_unchecked<4>();
            for (py::ssize_t i = 0; i < n; ++i) {
                const auto& fig = self.figures()[static_cast<size_t>(i)];
                for (py::ssize_t ti = 0; ti < t; ++ti) {
                    const auto& tri = fig.triangles[static_cast<size_t>(ti)];
                    buf(i, ti, 0, 0) = tri.v0.x;
                    buf(i, ti, 0, 1) = tri.v0.y;
                    buf(i, ti, 1, 0) = tri.v1.x;
                    buf(i, ti, 1, 1) = tri.v1.y;
                    buf(i, ti, 2, 0) = tri.v2.x;
                    buf(i, ti, 2, 1) = tri.v2.y;
                }
            }
            return arr;
        })
        .def("set_params_from_numpy", [](
            tree_packing::Solution& self,
            py::array_t<float> positions,
            py::array_t<float> angles
        ) {
            auto pos_buf = positions.unchecked<2>();
            auto ang_buf = angles.unchecked<1>();

            py::ssize_t n = pos_buf.shape(0);
            if (pos_buf.shape(1) != 2) {
                throw py::value_error("positions must have shape (N, 2)");
            }
            if (ang_buf.shape(0) != n) {
                throw py::value_error("angles must have same length as positions");
            }
            if (static_cast<size_t>(n) != self.size()) {
                throw py::value_error("array size must match solution size");
            }

            auto& params = self.params();
            for (py::ssize_t i = 0; i < n; ++i) {
                params.x[static_cast<size_t>(i)] = pos_buf(i, 0);
                params.y[static_cast<size_t>(i)] = pos_buf(i, 1);
                params.angle[static_cast<size_t>(i)] = ang_buf(i);
            }
            self.recompute_cache();
        },
            py::arg("positions"),
            py::arg("angles"),
            "Set all parameters from numpy arrays. positions: (N, 2), angles: (N,)")
        .def("get_params_as_numpy", [](const tree_packing::Solution& self) {
            const auto& params = self.params();
            py::ssize_t n = static_cast<py::ssize_t>(params.size());

            // Create positions array (N, 2)
            py::array_t<float> positions({n, static_cast<py::ssize_t>(2)});
            auto pos_buf = positions.mutable_unchecked<2>();
            for (py::ssize_t i = 0; i < n; ++i) {
                pos_buf(i, 0) = params.x[static_cast<size_t>(i)];
                pos_buf(i, 1) = params.y[static_cast<size_t>(i)];
            }

            // Create angles array (N,) - use explicit shape
            std::vector<py::ssize_t> ang_shape = {n};
            py::array_t<float> angles(ang_shape);
            float* ang_ptr = angles.mutable_data();
            for (py::ssize_t i = 0; i < n; ++i) {
                ang_ptr[i] = params.angle[static_cast<size_t>(i)];
            }

            return py::make_tuple(positions, angles);
        }, "Get parameters as numpy arrays. Returns (positions, angles) where positions is (N, 2) and angles is (N,)");

    // IntersectionTreeFilter
    py::class_<tree_packing::IntersectionTreeFilter>(m, "IntersectionTreeFilter")
        .def(py::init<>())
        .def("ready", &tree_packing::IntersectionTreeFilter::ready)
        .def("triangle_pairs_for", [](
            const tree_packing::IntersectionTreeFilter& self,
            const tree_packing::Solution& solution,
            size_t idx_a,
            size_t idx_b
        ) {
            py::list out;
            const auto* pairs = self.triangle_pairs_for(solution, idx_a, idx_b);
            if (!pairs) {
                return out;
            }
            for (const auto& pair : *pairs) {
                out.append(py::make_tuple(pair.first, pair.second));
            }
            return out;
        }, py::arg("solution"), py::arg("idx_a"), py::arg("idx_b"))
        .def("leaf_index_for", &tree_packing::IntersectionTreeFilter::leaf_index_for,
             py::arg("solution"), py::arg("idx_a"), py::arg("idx_b"))
        .def("leaf_pred_for", [](
            const tree_packing::IntersectionTreeFilter& self,
            const tree_packing::Solution& solution,
            size_t idx_a,
            size_t idx_b
        ) {
            std::array<uint8_t, 16> pred{};
            int leaf_idx = -1;
            bool ok = self.leaf_pred_for(solution, idx_a, idx_b, pred, &leaf_idx);
            py::array_t<uint8_t> arr({static_cast<py::ssize_t>(pred.size())});
            auto buf = arr.mutable_unchecked<1>();
            for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(pred.size()); ++i) {
                buf(i) = pred[static_cast<size_t>(i)];
            }
            return py::make_tuple(ok, leaf_idx, arr);
        }, py::arg("solution"), py::arg("idx_a"), py::arg("idx_b"))
        .def("features_for", [](
            const tree_packing::IntersectionTreeFilter& self,
            const tree_packing::Solution& solution,
            size_t idx_a,
            size_t idx_b
        ) {
            std::array<float, 10> feat{};
            bool ok = self.features_for(solution, idx_a, idx_b, feat);
            py::array_t<float> arr({static_cast<py::ssize_t>(feat.size())});
            auto buf = arr.mutable_unchecked<1>();
            for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(feat.size()); ++i) {
                buf(i) = feat[static_cast<size_t>(i)];
            }
            return py::make_tuple(ok, arr);
        }, py::arg("solution"), py::arg("idx_a"), py::arg("idx_b"));

    // SolutionEval
    py::class_<tree_packing::SolutionEval>(m, "SolutionEval")
        .def(py::init<>())
        .def_readwrite("solution", &tree_packing::SolutionEval::solution)
        .def_readwrite("objective", &tree_packing::SolutionEval::objective)
        .def_readwrite("intersection_violation", &tree_packing::SolutionEval::intersection_violation)
        .def_readwrite("bounds_violation", &tree_packing::SolutionEval::bounds_violation)
        .def("intersection_map", [](const tree_packing::SolutionEval& self) {
            py::list out;
            for (const auto& row : self.intersection_map) {
                py::list row_list;
                for (const auto& entry : row) {
                    row_list.append(py::make_tuple(entry.neighbor, entry.score));
                }
                out.append(row_list);
            }
            return out;
        })
        .def("intersection_matrix", [](const tree_packing::SolutionEval& self) {
            size_t n = self.intersection_map.size();
            py::array_t<float> mat({static_cast<py::ssize_t>(n),
                                    static_cast<py::ssize_t>(n)});
            auto buf = mat.mutable_unchecked<2>();
            for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(n); ++i) {
                for (py::ssize_t j = 0; j < static_cast<py::ssize_t>(n); ++j) {
                    buf(i, j) = 0.0f;
                }
            }
            for (size_t i = 0; i < n; ++i) {
                for (const auto& entry : self.intersection_map[i]) {
                    if (entry.neighbor < 0) continue;
                    size_t j = static_cast<size_t>(entry.neighbor);
                    if (j >= n) continue;
                    buf(static_cast<py::ssize_t>(i), static_cast<py::ssize_t>(j)) = entry.score;
                }
            }
            return mat;
        })
        .def("copy_from", &tree_packing::SolutionEval::copy_from)
        .def("total_violation", &tree_packing::SolutionEval::total_violation)
        .def("n_missing", &tree_packing::SolutionEval::n_missing)
        .def("reg", &tree_packing::SolutionEval::reg);

    // ConstraintPenaltyType enum
    py::enum_<tree_packing::ConstraintPenaltyType>(m, "ConstraintPenaltyType")
        .value("Linear", tree_packing::ConstraintPenaltyType::Linear)
        .value("Quadratic", tree_packing::ConstraintPenaltyType::Quadratic)
        .value("Tolerant", tree_packing::ConstraintPenaltyType::Tolerant)
        .value("LogBarrier", tree_packing::ConstraintPenaltyType::LogBarrier)
        .value("Exponential", tree_packing::ConstraintPenaltyType::Exponential)
        .export_values();

    // Problem
    py::class_<tree_packing::Problem>(m, "Problem")
        .def(py::init<>())
        .def_static("create_tree_packing_problem", &tree_packing::Problem::create_tree_packing_problem,
            py::arg("side") = -1.0f)
        .def("eval", &tree_packing::Problem::eval)
        .def("score", &tree_packing::Problem::score)
        .def("objective", &tree_packing::Problem::objective)
        .def("min_pos", &tree_packing::Problem::min_pos)
        .def("max_pos", &tree_packing::Problem::max_pos)
        .def("update_and_eval", &tree_packing::Problem::update_and_eval,
            py::arg("eval"), py::arg("indices"), py::arg("new_params"),
            "Update solution params in-place and evaluate objective/constraints.")
        // Constraint penalty settings
        .def("set_constraint_penalty_type", [](tree_packing::Problem& self, py::object type_obj) {
            if (py::isinstance<py::str>(type_obj)) {
                std::string s = type_obj.cast<std::string>();
                if (s == "Linear" || s == "linear") {
                    self.set_constraint_penalty_type(tree_packing::ConstraintPenaltyType::Linear);
                } else if (s == "Quadratic" || s == "quadratic") {
                    self.set_constraint_penalty_type(tree_packing::ConstraintPenaltyType::Quadratic);
                } else if (s == "Tolerant" || s == "tolerant") {
                    self.set_constraint_penalty_type(tree_packing::ConstraintPenaltyType::Tolerant);
                } else if (s == "LogBarrier" || s == "logbarrier" || s == "log_barrier") {
                    self.set_constraint_penalty_type(tree_packing::ConstraintPenaltyType::LogBarrier);
                } else if (s == "Exponential" || s == "exponential") {
                    self.set_constraint_penalty_type(tree_packing::ConstraintPenaltyType::Exponential);
                } else {
                    throw std::invalid_argument("Unknown constraint penalty type: " + s);
                }
            } else {
                self.set_constraint_penalty_type(type_obj.cast<tree_packing::ConstraintPenaltyType>());
            }
        }, py::arg("type"), "Set constraint penalty type (Linear, Quadratic, Tolerant, LogBarrier, Exponential)")
        .def("set_constraint_tolerance", &tree_packing::Problem::set_constraint_tolerance,
            py::arg("tolerance"), "Set tolerance for Tolerant/LogBarrier penalty types")
        .def("set_constraint_mu_low", &tree_packing::Problem::set_constraint_mu_low,
            py::arg("mu_low"), "Set low penalty multiplier for Tolerant type (when violation < tolerance)")
        .def("set_constraint_exp_scale", &tree_packing::Problem::set_constraint_exp_scale,
            py::arg("scale"), "Set scale for Exponential penalty type")
        .def("constraint_penalty_type", &tree_packing::Problem::constraint_penalty_type)
        .def("constraint_tolerance", &tree_packing::Problem::constraint_tolerance)
        .def("constraint_mu_low", &tree_packing::Problem::constraint_mu_low)
        .def("constraint_exp_scale", &tree_packing::Problem::constraint_exp_scale)
        // Objective ceiling settings
        .def("set_objective_ceiling", &tree_packing::Problem::set_objective_ceiling,
            py::arg("ceiling"), "Set objective ceiling directly")
        .def("set_objective_ceiling_delta", &tree_packing::Problem::set_objective_ceiling_delta,
            py::arg("delta"), "Set objective ceiling as best_feasible + delta")
        .def("set_objective_ceiling_mu", &tree_packing::Problem::set_objective_ceiling_mu,
            py::arg("mu"), "Set penalty multiplier for exceeding ceiling")
        .def("objective_ceiling", &tree_packing::Problem::objective_ceiling)
        .def("objective_ceiling_delta", &tree_packing::Problem::objective_ceiling_delta)
        .def("objective_ceiling_mu", &tree_packing::Problem::objective_ceiling_mu)
        .def("effective_ceiling", &tree_packing::Problem::effective_ceiling,
            py::arg("global_state"), "Get effective ceiling value given current global state");

    // GlobalState
    py::class_<tree_packing::GlobalState>(m, "GlobalState")
        .def(py::init<uint64_t>())
        .def(py::init<uint64_t, const tree_packing::SolutionEval&>())
        .def("next", &tree_packing::GlobalState::next)
        .def("maybe_update_best", &tree_packing::GlobalState::maybe_update_best)
        .def("iteration", &tree_packing::GlobalState::iteration)
        .def("iters_since_improvement", &tree_packing::GlobalState::iters_since_improvement)
        .def("iters_since_feasible_improvement", &tree_packing::GlobalState::iters_since_feasible_improvement)
        .def("best_score", &tree_packing::GlobalState::best_score)
        .def("best_feasible_score", &tree_packing::GlobalState::best_feasible_score)
        .def("best_feasible_objective", &tree_packing::GlobalState::best_feasible_objective)
        .def("best_params", [](const tree_packing::GlobalState& self) -> py::object {
            const auto* best = self.best_params();
            if (!best) {
                return py::none();
            }
            return py::cast(*best);
        })
        .def("best_feasible_params", [](const tree_packing::GlobalState& self) -> py::object {
            const auto* best = self.best_feasible_params();
            if (!best) {
                return py::none();
            }
            return py::cast(*best);
        })
        .def("mu", &tree_packing::GlobalState::mu)
        .def("set_mu", &tree_packing::GlobalState::set_mu)
        .def("tolerance", &tree_packing::GlobalState::tolerance)
        .def("set_tolerance", &tree_packing::GlobalState::set_tolerance)
        .def("random_float", py::overload_cast<>(&tree_packing::GlobalState::random_float))
        .def("random_int", &tree_packing::GlobalState::random_int);

    // RNG
    py::class_<tree_packing::RNG>(m, "RNG")
        .def(py::init<>())
        .def(py::init<uint64_t>())
        .def("uniform", py::overload_cast<>(&tree_packing::RNG::uniform))
        .def("uniform", py::overload_cast<float, float>(&tree_packing::RNG::uniform))
        .def("randint", &tree_packing::RNG::randint)
        .def("permutation", py::overload_cast<int>(&tree_packing::RNG::permutation))
        .def("choice", py::overload_cast<int, int>(&tree_packing::RNG::choice))
        .def("split", &tree_packing::RNG::split);

    py::class_<OptimizerState>(m, "OptimizerState")
        .def(py::init<>());

    // Optimizer base class (for type hints)
    py::class_<tree_packing::Optimizer, std::shared_ptr<tree_packing::Optimizer>>(m, "Optimizer")
        .def("set_problem", &tree_packing::Optimizer::set_problem)
        .def("init_state", [](tree_packing::Optimizer& self, const tree_packing::SolutionEval& solution) {
            return OptimizerState{self.init_state(solution)};
        })
        .def("apply", [](
            tree_packing::Optimizer& self,
            const tree_packing::SolutionEval& solution,
            OptimizerState& state,
            tree_packing::GlobalState& global_state,
            tree_packing::RNG& rng
        ) {
            tree_packing::SolutionEval out = solution;
            self.apply(out, state.value, global_state, rng);
            return py::make_tuple(out, OptimizerState{state.value});
        })
        .def("step", [](
            tree_packing::Optimizer& self,
            const tree_packing::SolutionEval& solution,
            OptimizerState& state,
            tree_packing::GlobalState& global_state
        ) {
            tree_packing::SolutionEval out = solution;
            self.step(out, state.value, global_state);
            return py::make_tuple(out, OptimizerState{state.value}, global_state);
        })
        .def("run", [](
            tree_packing::Optimizer& self,
            const tree_packing::SolutionEval& solution,
            OptimizerState& state,
            tree_packing::GlobalState& global_state,
            int n
        ) {
            tree_packing::SolutionEval out = solution;
            self.run(out, state.value, global_state, n);
            return py::make_tuple(out, OptimizerState{state.value}, global_state);
        })
        .def("apply_inplace", [](
            tree_packing::Optimizer& self,
            tree_packing::SolutionEval& solution,
            OptimizerState& state,
            tree_packing::GlobalState& global_state,
            tree_packing::RNG& rng
        ) {
            self.apply(solution, state.value, global_state, rng);
        })
        .def("step_inplace", [](
            tree_packing::Optimizer& self,
            tree_packing::SolutionEval& solution,
            OptimizerState& state,
            tree_packing::GlobalState& global_state
        ) {
            self.step(solution, state.value, global_state);
        })
        .def("run_inplace", [](
            tree_packing::Optimizer& self,
            tree_packing::SolutionEval& solution,
            OptimizerState& state,
            tree_packing::GlobalState& global_state,
            int n
        ) {
            self.run(solution, state.value, global_state, n);
        });

    // RandomRuin
    py::class_<tree_packing::RandomRuin, tree_packing::Optimizer, std::shared_ptr<tree_packing::RandomRuin>>(m, "RandomRuin")
        .def(py::init<int, bool>(), py::arg("n_remove") = 1, py::arg("verbose") = false);

    // CellRuin
    py::class_<tree_packing::CellRuin, tree_packing::Optimizer, std::shared_ptr<tree_packing::CellRuin>>(m, "CellRuin")
        .def(py::init<int, bool>(), py::arg("n_remove") = 1, py::arg("verbose") = false);

    // RandomRecreate
    py::class_<tree_packing::RandomRecreate, tree_packing::Optimizer, std::shared_ptr<tree_packing::RandomRecreate>>(m, "RandomRecreate")
        .def(py::init<int, float, float, bool>(),
            py::arg("max_recreate") = 1,
            py::arg("box_size") = 5.0f,
            py::arg("delta") = 0.35f,
            py::arg("verbose") = false);

    // GridCellRecreate
    py::class_<tree_packing::GridCellRecreate, tree_packing::Optimizer, std::shared_ptr<tree_packing::GridCellRecreate>>(m, "GridCellRecreate")
        .def(py::init<int, int, int, int, int, bool>(),
            py::arg("max_recreate") = 1,
            py::arg("cell_min") = 0,
            py::arg("cell_max") = 4,
            py::arg("neighbor_min") = 1,
            py::arg("neighbor_max") = -1,
            py::arg("verbose") = false);

    // NoiseOptimizer
    py::class_<tree_packing::NoiseOptimizer, tree_packing::Optimizer, std::shared_ptr<tree_packing::NoiseOptimizer>>(m, "NoiseOptimizer")
        .def(py::init<float, int, bool>(),
            py::arg("noise_level") = 0.01f,
            py::arg("n_change") = 1,
            py::arg("verbose") = false);

    // SqueezeOptimizer
    py::class_<tree_packing::SqueezeOptimizer, tree_packing::Optimizer, std::shared_ptr<tree_packing::SqueezeOptimizer>>(m, "SqueezeOptimizer")
        .def(py::init<float, float, int, int, bool>(),
            py::arg("min_scale") = 0.05f,
            py::arg("shrink") = 0.92f,
            py::arg("bisect_iters") = 18,
            py::arg("axis_rounds") = 3,
            py::arg("verbose") = false);

    // CompactionOptimizer
    py::class_<tree_packing::CompactionOptimizer, tree_packing::Optimizer, std::shared_ptr<tree_packing::CompactionOptimizer>>(m, "CompactionOptimizer")
        .def(py::init<int, bool>(),
            py::arg("iters_per_tree") = 8,
            py::arg("verbose") = false);

    // LocalSearchOptimizer
    py::class_<tree_packing::LocalSearchOptimizer, tree_packing::Optimizer, std::shared_ptr<tree_packing::LocalSearchOptimizer>>(m, "LocalSearchOptimizer")
        .def(py::init<int, bool>(),
            py::arg("iters_per_tree") = 18,
            py::arg("verbose") = false);

    // RestoreBest
    py::class_<tree_packing::RestoreBest, tree_packing::Optimizer, std::shared_ptr<tree_packing::RestoreBest>>(m, "RestoreBest")
        .def(py::init<int, bool>(), py::arg("interval"), py::arg("verbose") = false);

    // Chain
    py::class_<tree_packing::Chain, tree_packing::Optimizer, std::shared_ptr<tree_packing::Chain>>(m, "Chain")
        .def(py::init([](
            std::vector<std::shared_ptr<tree_packing::Optimizer>> optimizers,
            bool verbose
        ) {
            std::vector<tree_packing::OptimizerPtr> optimizer_ptrs;
            optimizer_ptrs.reserve(optimizers.size());
            for (auto& op : optimizers) {
                optimizer_ptrs.push_back(clone_optimizer(op));
            }

            return std::make_shared<tree_packing::Chain>(std::move(optimizer_ptrs), verbose);
        }),
            py::arg("optimizers"),
            py::arg("verbose") = false);

    // Alternate
    py::class_<tree_packing::Alternate, tree_packing::Optimizer, std::shared_ptr<tree_packing::Alternate>>(m, "Alternate")
        .def(py::init([](
            std::vector<std::shared_ptr<tree_packing::Optimizer>> optimizers,
            bool verbose
        ) {
            std::vector<tree_packing::OptimizerPtr> optimizer_ptrs;
            optimizer_ptrs.reserve(optimizers.size());
            for (auto& op : optimizers) {
                optimizer_ptrs.push_back(clone_optimizer(op));
            }

            return std::make_shared<tree_packing::Alternate>(std::move(optimizer_ptrs), verbose);
        }),
            py::arg("optimizers"),
            py::arg("verbose") = false);

    // ALNS
    py::class_<tree_packing::ALNS, tree_packing::Optimizer, std::shared_ptr<tree_packing::ALNS>>(m, "ALNS")
        .def(py::init([](
            std::vector<std::shared_ptr<tree_packing::Optimizer>> ruin_operators,
            std::vector<std::shared_ptr<tree_packing::Optimizer>> recreate_operators,
            float reaction_factor,
            float reward_improve,
            float reward_no_improve,
            float min_weight,
            bool verbose
        ) {
            std::vector<tree_packing::OptimizerPtr> ruin_ptrs;
            std::vector<tree_packing::OptimizerPtr> recreate_ptrs;

            ruin_ptrs.reserve(ruin_operators.size());
            for (auto& op : ruin_operators) {
                ruin_ptrs.push_back(clone_optimizer(op));
            }
            recreate_ptrs.reserve(recreate_operators.size());
            for (auto& op : recreate_operators) {
                recreate_ptrs.push_back(clone_optimizer(op));
            }

            return std::make_shared<tree_packing::ALNS>(
                std::move(ruin_ptrs),
                std::move(recreate_ptrs),
                reaction_factor,
                reward_improve,
                reward_no_improve,
                min_weight,
                verbose
            );
        }),
            py::arg("ruin_operators"),
            py::arg("recreate_operators"),
            py::arg("reaction_factor") = 0.01f,
            py::arg("reward_improve") = 1.0f,
            py::arg("reward_no_improve") = 0.0f,
            py::arg("min_weight") = 1e-3f,
            py::arg("verbose") = false);

    // CoolingSchedule enum
    py::enum_<tree_packing::CoolingSchedule>(m, "CoolingSchedule")
        .value("Exponential", tree_packing::CoolingSchedule::Exponential)
        .value("Linear", tree_packing::CoolingSchedule::Linear)
        .value("Logarithmic", tree_packing::CoolingSchedule::Logarithmic);

    // SimulatedAnnealing
    py::class_<tree_packing::SimulatedAnnealing, tree_packing::Optimizer, std::shared_ptr<tree_packing::SimulatedAnnealing>>(m, "SimulatedAnnealing")
        .def(py::init([](
            std::shared_ptr<tree_packing::Optimizer> inner_optimizer,
            float initial_temp,
            float min_temp,
            tree_packing::CoolingSchedule cooling_schedule,
            float cooling_rate,
            int patience,
            bool verbose
        ) {
            tree_packing::OptimizerPtr inner = clone_optimizer(inner_optimizer);
            return std::make_shared<tree_packing::SimulatedAnnealing>(
                std::move(inner),
                initial_temp,
                min_temp,
                cooling_schedule,
                cooling_rate,
                patience,
                verbose
            );
        }),
            py::arg("inner_optimizer"),
            py::arg("initial_temp") = 1.0f,
            py::arg("min_temp") = 1e-6f,
            py::arg("cooling_schedule") = tree_packing::CoolingSchedule::Exponential,
            py::arg("cooling_rate") = 0.995f,
            py::arg("patience") = -1,
            py::arg("verbose") = false);

    // Conditional: applies optimizer only when ALL conditions are met
    py::class_<tree_packing::Conditional, tree_packing::Optimizer, std::shared_ptr<tree_packing::Conditional>>(m, "Conditional")
        .def(py::init([](
            std::shared_ptr<tree_packing::Optimizer> inner_optimizer,
            uint64_t every_n,
            uint64_t min_iters_since_improvement,
            uint64_t min_iters_since_feasible_improvement,
            bool verbose
        ) {
            tree_packing::OptimizerPtr inner = clone_optimizer(inner_optimizer);
            return std::make_shared<tree_packing::Conditional>(
                std::move(inner),
                every_n,
                min_iters_since_improvement,
                min_iters_since_feasible_improvement,
                nullptr,  // custom_condition not exposed to Python
                verbose
            );
        }),
            py::arg("inner_optimizer"),
            py::arg("every_n") = 0,
            py::arg("min_iters_since_improvement") = 0,
            py::arg("min_iters_since_feasible_improvement") = 0,
            py::arg("verbose") = false,
            R"doc(
            Conditional meta-optimizer that applies inner optimizer only when ALL conditions are met.

            All conditions use logical AND - the optimizer runs only if all enabled conditions are true.
            Set a condition to 0 to disable it.

            Args:
                inner_optimizer: The optimizer to conditionally apply.
                every_n: Run when iteration % every_n == 0 (0 = disabled).
                min_iters_since_improvement: Run when iters_since_improvement >= this value (0 = disabled).
                min_iters_since_feasible_improvement: Run when iters_since_feasible_improvement >= this value (0 = disabled).
                verbose: Print debug messages.

            Examples:
                # Run expensive optimizer every 100 iterations
                opt = Conditional(expensive_opt, every_n=100)

                # Run diversification when stuck for 50 iterations
                opt = Conditional(diversify_opt, min_iters_since_improvement=50)

                # Run every 10 iterations AND when stuck for 20 iterations
                opt = Conditional(opt, every_n=10, min_iters_since_improvement=20)
            )doc"
        )
        .def_property_readonly("every_n", &tree_packing::Conditional::every_n)
        .def_property_readonly("min_iters_since_improvement", &tree_packing::Conditional::min_iters_since_improvement)
        .def_property_readonly("min_iters_since_feasible_improvement", &tree_packing::Conditional::min_iters_since_feasible_improvement);

    // =========================================================================
    // Solvers
    // =========================================================================

    // SolverResult
    py::class_<tree_packing::SolverResult>(m, "SolverResult")
        .def(py::init<>())
        .def_readwrite("params", &tree_packing::SolverResult::params)
        .def_readwrite("start_f", &tree_packing::SolverResult::start_f)
        .def_readwrite("start_g", &tree_packing::SolverResult::start_g)
        .def_readwrite("final_f", &tree_packing::SolverResult::final_f)
        .def_readwrite("final_g", &tree_packing::SolverResult::final_g);

    // Solver base class
    py::class_<tree_packing::Solver, std::shared_ptr<tree_packing::Solver>>(m, "Solver")
        .def("solve", [](
            tree_packing::Solver& self,
            const tree_packing::SolutionEval& eval,
            std::vector<int> indices,
            tree_packing::RNG& rng
        ) {
            return self.solve(eval, indices, rng);
        }, py::arg("eval"), py::arg("indices"), py::arg("rng"))
        .def("solve_single", [](
            tree_packing::Solver& self,
            const tree_packing::SolutionEval& eval,
            int index,
            tree_packing::RNG& rng
        ) {
            return self.solve_single(eval, index, rng);
        }, py::arg("eval"), py::arg("index"), py::arg("rng"))
        .def("set_problem", [](tree_packing::Solver& self, tree_packing::Problem& problem) {
            self.set_problem(&problem);
        }, py::arg("problem"))
        .def("min_pos", &tree_packing::Solver::min_pos)
        .def("max_pos", &tree_packing::Solver::max_pos)
        .def("set_bounds", &tree_packing::Solver::set_bounds,
            py::arg("min_pos"), py::arg("max_pos"));

    // ParticleSwarmSolver::ObjectiveType enum
    py::enum_<tree_packing::ParticleSwarmSolver::ObjectiveType>(m, "PSOObjectiveType")
        .value("Distance", tree_packing::ParticleSwarmSolver::ObjectiveType::Distance)
        .value("Linf", tree_packing::ParticleSwarmSolver::ObjectiveType::Linf)
        .value("Zero", tree_packing::ParticleSwarmSolver::ObjectiveType::Zero)
        .export_values();

    // ParticleSwarmSolver::Config
    py::class_<tree_packing::ParticleSwarmSolver::Config>(m, "PSOConfig")
        .def(py::init<>())
        .def_readwrite("n_particles", &tree_packing::ParticleSwarmSolver::Config::n_particles)
        .def_readwrite("n_iterations", &tree_packing::ParticleSwarmSolver::Config::n_iterations)
        .def_readwrite("w", &tree_packing::ParticleSwarmSolver::Config::w)
        .def_readwrite("c1", &tree_packing::ParticleSwarmSolver::Config::c1)
        .def_readwrite("c2", &tree_packing::ParticleSwarmSolver::Config::c2)
        .def_readwrite("mu0", &tree_packing::ParticleSwarmSolver::Config::mu0)
        .def_readwrite("mu_max", &tree_packing::ParticleSwarmSolver::Config::mu_max)
        .def_readwrite("isolation_penalty_per_missing", &tree_packing::ParticleSwarmSolver::Config::isolation_penalty_per_missing)
        .def_readwrite("target_neighbors", &tree_packing::ParticleSwarmSolver::Config::target_neighbors)
        .def_readwrite("vel_max", &tree_packing::ParticleSwarmSolver::Config::vel_max)
        .def_readwrite("vel_ang_max", &tree_packing::ParticleSwarmSolver::Config::vel_ang_max)
        .def_readwrite("constrain_to_cell", &tree_packing::ParticleSwarmSolver::Config::constrain_to_cell)
        .def_readwrite("prefer_current_cell", &tree_packing::ParticleSwarmSolver::Config::prefer_current_cell)
        .def_readwrite("objective_type", &tree_packing::ParticleSwarmSolver::Config::objective_type);

    // ParticleSwarmSolver
    py::class_<tree_packing::ParticleSwarmSolver, tree_packing::Solver, std::shared_ptr<tree_packing::ParticleSwarmSolver>>(m, "ParticleSwarmSolver")
        .def(py::init<>())
        .def(py::init<const tree_packing::ParticleSwarmSolver::Config&>(), py::arg("config"))
        .def(py::init([](
            int n_particles,
            int n_iterations,
            float w,
            float c1,
            float c2,
            float mu0,
            float mu_max,
            float isolation_penalty_per_missing,
            float target_neighbors,
            float vel_max,
            float vel_ang_max,
            bool constrain_to_cell,
            bool prefer_current_cell,
            py::object objective_type_obj
        ) {
            tree_packing::ParticleSwarmSolver::Config config;
            config.n_particles = n_particles;
            config.n_iterations = n_iterations;
            config.w = w;
            config.c1 = c1;
            config.c2 = c2;
            config.mu0 = mu0;
            config.mu_max = mu_max;
            config.isolation_penalty_per_missing = isolation_penalty_per_missing;
            config.target_neighbors = target_neighbors;
            config.vel_max = vel_max;
            config.vel_ang_max = vel_ang_max;
            config.constrain_to_cell = constrain_to_cell;
            config.prefer_current_cell = prefer_current_cell;
            // Accept string or enum
            if (py::isinstance<py::str>(objective_type_obj)) {
                std::string s = objective_type_obj.cast<std::string>();
                if (s == "Distance" || s == "distance") {
                    config.objective_type = tree_packing::ParticleSwarmSolver::ObjectiveType::Distance;
                } else if (s == "Linf" || s == "linf") {
                    config.objective_type = tree_packing::ParticleSwarmSolver::ObjectiveType::Linf;
                } else if (s == "Zero" || s == "zero") {
                    config.objective_type = tree_packing::ParticleSwarmSolver::ObjectiveType::Zero;
                } else {
                    throw std::invalid_argument("objective_type must be 'Distance', 'Linf', or 'Zero'");
                }
            } else {
                config.objective_type = objective_type_obj.cast<tree_packing::ParticleSwarmSolver::ObjectiveType>();
            }
            return std::make_shared<tree_packing::ParticleSwarmSolver>(config);
        }),
            py::arg("n_particles") = 32,
            py::arg("n_iterations") = 50,
            py::arg("w") = 0.7f,
            py::arg("c1") = 1.5f,
            py::arg("c2") = 1.5f,
            py::arg("mu0") = 1.0f,
            py::arg("mu_max") = 1e6f,
            py::arg("isolation_penalty_per_missing") = std::log(2.0f),
            py::arg("target_neighbors") = 8.0f,
            py::arg("vel_max") = 1.0f,
            py::arg("vel_ang_max") = 0.785f,
            py::arg("constrain_to_cell") = true,
            py::arg("prefer_current_cell") = true,
            py::arg("objective_type") = "Distance"
        )
        .def("config", py::overload_cast<>(&tree_packing::ParticleSwarmSolver::config),
            py::return_value_policy::reference_internal);

    // RandomSamplingSolver::ObjectiveType enum
    py::enum_<tree_packing::RandomSamplingSolver::ObjectiveType>(m, "RandomSamplingObjectiveType")
        .value("Distance", tree_packing::RandomSamplingSolver::ObjectiveType::Distance)
        .value("Linf", tree_packing::RandomSamplingSolver::ObjectiveType::Linf)
        .value("Zero", tree_packing::RandomSamplingSolver::ObjectiveType::Zero)
        .export_values();

    // RandomSamplingSolver::Config
    py::class_<tree_packing::RandomSamplingSolver::Config>(m, "RandomSamplingConfig")
        .def(py::init<>())
        .def_readwrite("n_samples", &tree_packing::RandomSamplingSolver::Config::n_samples)
        .def_readwrite("mu", &tree_packing::RandomSamplingSolver::Config::mu)
        .def_readwrite("isolation_penalty_per_missing", &tree_packing::RandomSamplingSolver::Config::isolation_penalty_per_missing)
        .def_readwrite("target_neighbors", &tree_packing::RandomSamplingSolver::Config::target_neighbors)
        .def_readwrite("constrain_to_cell", &tree_packing::RandomSamplingSolver::Config::constrain_to_cell)
        .def_readwrite("prefer_current_cell", &tree_packing::RandomSamplingSolver::Config::prefer_current_cell)
        .def_readwrite("objective_type", &tree_packing::RandomSamplingSolver::Config::objective_type);

    // RandomSamplingSolver
    py::class_<tree_packing::RandomSamplingSolver, tree_packing::Solver, std::shared_ptr<tree_packing::RandomSamplingSolver>>(m, "RandomSamplingSolver")
        .def(py::init<>())
        .def(py::init<const tree_packing::RandomSamplingSolver::Config&>(), py::arg("config"))
        .def(py::init([](
            int n_samples,
            float mu,
            float isolation_penalty_per_missing,
            float target_neighbors,
            bool constrain_to_cell,
            bool prefer_current_cell,
            py::object objective_type_obj
        ) {
            tree_packing::RandomSamplingSolver::Config config;
            config.n_samples = n_samples;
            config.mu = mu;
            config.isolation_penalty_per_missing = isolation_penalty_per_missing;
            config.target_neighbors = target_neighbors;
            config.constrain_to_cell = constrain_to_cell;
            config.prefer_current_cell = prefer_current_cell;
            // Accept string or enum
            if (py::isinstance<py::str>(objective_type_obj)) {
                std::string s = objective_type_obj.cast<std::string>();
                if (s == "Distance" || s == "distance") {
                    config.objective_type = tree_packing::RandomSamplingSolver::ObjectiveType::Distance;
                } else if (s == "Linf" || s == "linf") {
                    config.objective_type = tree_packing::RandomSamplingSolver::ObjectiveType::Linf;
                } else if (s == "Zero" || s == "zero") {
                    config.objective_type = tree_packing::RandomSamplingSolver::ObjectiveType::Zero;
                } else {
                    throw std::invalid_argument("objective_type must be 'Distance', 'Linf', or 'Zero'");
                }
            } else {
                config.objective_type = objective_type_obj.cast<tree_packing::RandomSamplingSolver::ObjectiveType>();
            }
            return std::make_shared<tree_packing::RandomSamplingSolver>(config);
        }),
            py::arg("n_samples") = 100,
            py::arg("mu") = 1e6f,
            py::arg("isolation_penalty_per_missing") = std::log(2.0f),
            py::arg("target_neighbors") = 8.0f,
            py::arg("constrain_to_cell") = true,
            py::arg("prefer_current_cell") = true,
            py::arg("objective_type") = "Linf"
        )
        .def("config", py::overload_cast<>(&tree_packing::RandomSamplingSolver::config),
            py::return_value_policy::reference_internal);

    // BeamDescentSolver::ObjectiveType enum
    py::enum_<tree_packing::BeamDescentSolver::ObjectiveType>(m, "BeamDescentObjectiveType")
        .value("Distance", tree_packing::BeamDescentSolver::ObjectiveType::Distance)
        .value("Linf", tree_packing::BeamDescentSolver::ObjectiveType::Linf)
        .value("Zero", tree_packing::BeamDescentSolver::ObjectiveType::Zero)
        .export_values();

    // BeamDescentSolver::Config
    py::class_<tree_packing::BeamDescentSolver::Config>(m, "BeamDescentConfig")
        .def(py::init<>())
        .def_readwrite("lattice_xy", &tree_packing::BeamDescentSolver::Config::lattice_xy)
        .def_readwrite("lattice_ang", &tree_packing::BeamDescentSolver::Config::lattice_ang)
        .def_readwrite("beam_width", &tree_packing::BeamDescentSolver::Config::beam_width)
        .def_readwrite("descent_levels", &tree_packing::BeamDescentSolver::Config::descent_levels)
        .def_readwrite("max_iters_per_level", &tree_packing::BeamDescentSolver::Config::max_iters_per_level)
        .def_readwrite("step_xy0", &tree_packing::BeamDescentSolver::Config::step_xy0)
        .def_readwrite("step_xy_decay", &tree_packing::BeamDescentSolver::Config::step_xy_decay)
        .def_readwrite("step_ang0", &tree_packing::BeamDescentSolver::Config::step_ang0)
        .def_readwrite("step_ang_decay", &tree_packing::BeamDescentSolver::Config::step_ang_decay)
        .def_readwrite("mu", &tree_packing::BeamDescentSolver::Config::mu)
        .def_readwrite("isolation_penalty_per_missing", &tree_packing::BeamDescentSolver::Config::isolation_penalty_per_missing)
        .def_readwrite("target_neighbors", &tree_packing::BeamDescentSolver::Config::target_neighbors)
        .def_readwrite("constrain_to_cell", &tree_packing::BeamDescentSolver::Config::constrain_to_cell)
        .def_readwrite("prefer_current_cell", &tree_packing::BeamDescentSolver::Config::prefer_current_cell)
        .def_readwrite("objective_type", &tree_packing::BeamDescentSolver::Config::objective_type);

    // BeamDescentSolver
    py::class_<tree_packing::BeamDescentSolver, tree_packing::Solver, std::shared_ptr<tree_packing::BeamDescentSolver>>(m, "BeamDescentSolver")
        .def(py::init<>())
        .def(py::init<const tree_packing::BeamDescentSolver::Config&>(), py::arg("config"))
        .def(py::init([](
            int lattice_xy,
            int lattice_ang,
            int beam_width,
            int descent_levels,
            int max_iters_per_level,
            float step_xy0,
            float step_xy_decay,
            float step_ang0,
            float step_ang_decay,
            float mu,
            float isolation_penalty_per_missing,
            float target_neighbors,
            bool constrain_to_cell,
            bool prefer_current_cell,
            py::object objective_type_obj
        ) {
            tree_packing::BeamDescentSolver::Config config;
            config.lattice_xy = lattice_xy;
            config.lattice_ang = lattice_ang;
            config.beam_width = beam_width;
            config.descent_levels = descent_levels;
            config.max_iters_per_level = max_iters_per_level;
            config.step_xy0 = step_xy0;
            config.step_xy_decay = step_xy_decay;
            config.step_ang0 = step_ang0;
            config.step_ang_decay = step_ang_decay;
            config.mu = mu;
            config.isolation_penalty_per_missing = isolation_penalty_per_missing;
            config.target_neighbors = target_neighbors;
            config.constrain_to_cell = constrain_to_cell;
            config.prefer_current_cell = prefer_current_cell;
            // Accept string or enum
            if (py::isinstance<py::str>(objective_type_obj)) {
                std::string s = objective_type_obj.cast<std::string>();
                if (s == "Distance" || s == "distance") {
                    config.objective_type = tree_packing::BeamDescentSolver::ObjectiveType::Distance;
                } else if (s == "Linf" || s == "linf") {
                    config.objective_type = tree_packing::BeamDescentSolver::ObjectiveType::Linf;
                } else if (s == "Zero" || s == "zero") {
                    config.objective_type = tree_packing::BeamDescentSolver::ObjectiveType::Zero;
                } else {
                    throw std::invalid_argument("objective_type must be 'Distance', 'Linf', or 'Zero'");
                }
            } else {
                config.objective_type = objective_type_obj.cast<tree_packing::BeamDescentSolver::ObjectiveType>();
            }
            return std::make_shared<tree_packing::BeamDescentSolver>(config);
        }),
            py::arg("lattice_xy") = 5,
            py::arg("lattice_ang") = 8,
            py::arg("beam_width") = 8,
            py::arg("descent_levels") = 4,
            py::arg("max_iters_per_level") = 8,
            py::arg("step_xy0") = 0.5f,
            py::arg("step_xy_decay") = 0.5f,
            py::arg("step_ang0") = tree_packing::PI / 8.0f,
            py::arg("step_ang_decay") = 0.5f,
            py::arg("mu") = 1e6f,
            py::arg("isolation_penalty_per_missing") = std::log(2.0f),
            py::arg("target_neighbors") = 8.0f,
            py::arg("constrain_to_cell") = true,
            py::arg("prefer_current_cell") = true,
            py::arg("objective_type") = "Distance"
        )
        .def("config", py::overload_cast<>(&tree_packing::BeamDescentSolver::config),
            py::return_value_policy::reference_internal);

    // NoiseSolver::ObjectiveType enum
    py::enum_<tree_packing::NoiseSolver::ObjectiveType>(m, "NoiseSolverObjectiveType")
        .value("Distance", tree_packing::NoiseSolver::ObjectiveType::Distance)
        .value("Linf", tree_packing::NoiseSolver::ObjectiveType::Linf)
        .value("Zero", tree_packing::NoiseSolver::ObjectiveType::Zero)
        .export_values();

    // NoiseSolver::Config
    py::class_<tree_packing::NoiseSolver::Config>(m, "NoiseSolverConfig")
        .def(py::init<>())
        .def_readwrite("n_variations", &tree_packing::NoiseSolver::Config::n_variations)
        .def_readwrite("pos_sigma", &tree_packing::NoiseSolver::Config::pos_sigma)
        .def_readwrite("ang_sigma", &tree_packing::NoiseSolver::Config::ang_sigma)
        .def_readwrite("mu", &tree_packing::NoiseSolver::Config::mu)
        .def_readwrite("isolation_penalty_per_missing", &tree_packing::NoiseSolver::Config::isolation_penalty_per_missing)
        .def_readwrite("target_neighbors", &tree_packing::NoiseSolver::Config::target_neighbors)
        .def_readwrite("constrain_to_cell", &tree_packing::NoiseSolver::Config::constrain_to_cell)
        .def_readwrite("prefer_current_cell", &tree_packing::NoiseSolver::Config::prefer_current_cell)
        .def_readwrite("objective_type", &tree_packing::NoiseSolver::Config::objective_type);

    // NoiseSolver
    py::class_<tree_packing::NoiseSolver, tree_packing::Solver, std::shared_ptr<tree_packing::NoiseSolver>>(m, "NoiseSolver")
        .def(py::init<>())
        .def(py::init<const tree_packing::NoiseSolver::Config&>(), py::arg("config"))
        .def(py::init([](
            int n_variations,
            float pos_sigma,
            float ang_sigma,
            float mu,
            float isolation_penalty_per_missing,
            float target_neighbors,
            bool constrain_to_cell,
            bool prefer_current_cell,
            py::object objective_type_obj
        ) {
            tree_packing::NoiseSolver::Config config;
            config.n_variations = n_variations;
            config.pos_sigma = pos_sigma;
            config.ang_sigma = ang_sigma;
            config.mu = mu;
            config.isolation_penalty_per_missing = isolation_penalty_per_missing;
            config.target_neighbors = target_neighbors;
            config.constrain_to_cell = constrain_to_cell;
            config.prefer_current_cell = prefer_current_cell;
            if (py::isinstance<py::str>(objective_type_obj)) {
                std::string s = objective_type_obj.cast<std::string>();
                if (s == "Distance" || s == "distance") {
                    config.objective_type = tree_packing::NoiseSolver::ObjectiveType::Distance;
                } else if (s == "Linf" || s == "linf") {
                    config.objective_type = tree_packing::NoiseSolver::ObjectiveType::Linf;
                } else if (s == "Zero" || s == "zero") {
                    config.objective_type = tree_packing::NoiseSolver::ObjectiveType::Zero;
                } else {
                    throw std::invalid_argument("objective_type must be 'Distance', 'Linf', or 'Zero'");
                }
            } else {
                config.objective_type = objective_type_obj.cast<tree_packing::NoiseSolver::ObjectiveType>();
            }
            return std::make_shared<tree_packing::NoiseSolver>(config);
        }),
            py::arg("n_variations") = 128,
            py::arg("pos_sigma") = 0.25f,
            py::arg("ang_sigma") = tree_packing::PI / 16.0f,
            py::arg("mu") = 1e6f,
            py::arg("isolation_penalty_per_missing") = std::log(2.0f),
            py::arg("target_neighbors") = 8.0f,
            py::arg("constrain_to_cell") = true,
            py::arg("prefer_current_cell") = true,
            py::arg("objective_type") = "Distance"
        )
        .def("config", py::overload_cast<>(&tree_packing::NoiseSolver::config),
            py::return_value_policy::reference_internal);

    // NelderMeadSolver::Config
    py::enum_<tree_packing::NelderMeadSolver::ObjectiveType>(m, "NelderMeadObjectiveType")
        .value("Distance", tree_packing::NelderMeadSolver::ObjectiveType::Distance)
        .value("Linf", tree_packing::NelderMeadSolver::ObjectiveType::Linf)
        .value("Zero", tree_packing::NelderMeadSolver::ObjectiveType::Zero)
        .export_values();

    py::class_<tree_packing::NelderMeadSolver::Config>(m, "NelderMeadConfig")
        .def(py::init<>())
        .def_readwrite("step_xy", &tree_packing::NelderMeadSolver::Config::step_xy)
        .def_readwrite("step_ang", &tree_packing::NelderMeadSolver::Config::step_ang)
        .def_readwrite("randomize_simplex", &tree_packing::NelderMeadSolver::Config::randomize_simplex)
        .def_readwrite("jitter_xy", &tree_packing::NelderMeadSolver::Config::jitter_xy)
        .def_readwrite("jitter_ang", &tree_packing::NelderMeadSolver::Config::jitter_ang)
        .def_readwrite("alpha", &tree_packing::NelderMeadSolver::Config::alpha)
        .def_readwrite("gamma", &tree_packing::NelderMeadSolver::Config::gamma)
        .def_readwrite("rho", &tree_packing::NelderMeadSolver::Config::rho)
        .def_readwrite("sigma", &tree_packing::NelderMeadSolver::Config::sigma)
        .def_readwrite("max_iters", &tree_packing::NelderMeadSolver::Config::max_iters)
        .def_readwrite("tol_f", &tree_packing::NelderMeadSolver::Config::tol_f)
        .def_readwrite("tol_x", &tree_packing::NelderMeadSolver::Config::tol_x)
        .def_readwrite("mu", &tree_packing::NelderMeadSolver::Config::mu)
        .def_readwrite("constrain_to_cell", &tree_packing::NelderMeadSolver::Config::constrain_to_cell)
        .def_readwrite("prefer_current_cell", &tree_packing::NelderMeadSolver::Config::prefer_current_cell)
        .def_readwrite("objective_type", &tree_packing::NelderMeadSolver::Config::objective_type);

    // NelderMeadSolver
    py::class_<tree_packing::NelderMeadSolver, tree_packing::Solver, std::shared_ptr<tree_packing::NelderMeadSolver>>(m, "NelderMeadSolver")
        .def(py::init<>())
        .def(py::init<const tree_packing::NelderMeadSolver::Config&>(), py::arg("config"))
        .def(py::init([](
            float step_xy,
            float step_ang,
            bool randomize_simplex,
            float jitter_xy,
            float jitter_ang,
            float alpha,
            float gamma,
            float rho,
            float sigma,
            int max_iters,
            float tol_f,
            float tol_x,
            float mu,
            bool constrain_to_cell,
            bool prefer_current_cell,
            py::object objective_type_obj
        ) {
            tree_packing::NelderMeadSolver::Config config;
            config.step_xy = step_xy;
            config.step_ang = step_ang;
            config.randomize_simplex = randomize_simplex;
            config.jitter_xy = jitter_xy;
            config.jitter_ang = jitter_ang;
            config.alpha = alpha;
            config.gamma = gamma;
            config.rho = rho;
            config.sigma = sigma;
            config.max_iters = max_iters;
            config.tol_f = tol_f;
            config.tol_x = tol_x;
            config.mu = mu;
            config.constrain_to_cell = constrain_to_cell;
            config.prefer_current_cell = prefer_current_cell;
            if (py::isinstance<py::str>(objective_type_obj)) {
                std::string s = objective_type_obj.cast<std::string>();
                if (s == "Distance" || s == "distance") {
                    config.objective_type = tree_packing::NelderMeadSolver::ObjectiveType::Distance;
                } else if (s == "Linf" || s == "linf") {
                    config.objective_type = tree_packing::NelderMeadSolver::ObjectiveType::Linf;
                } else if (s == "Zero" || s == "zero") {
                    config.objective_type = tree_packing::NelderMeadSolver::ObjectiveType::Zero;
                } else {
                    throw std::invalid_argument("objective_type must be 'Distance', 'Linf', or 'Zero'");
                }
            } else {
                config.objective_type = objective_type_obj.cast<tree_packing::NelderMeadSolver::ObjectiveType>();
            }
            return std::make_shared<tree_packing::NelderMeadSolver>(config);
        }),
            py::arg("step_xy") = 0.25f,
            py::arg("step_ang") = tree_packing::PI / 24.0f,
            py::arg("randomize_simplex") = true,
            py::arg("jitter_xy") = 0.5f,
            py::arg("jitter_ang") = tree_packing::PI / 12.0f,
            py::arg("alpha") = 1.0f,
            py::arg("gamma") = 2.0f,
            py::arg("rho") = 0.5f,
            py::arg("sigma") = 0.5f,
            py::arg("max_iters") = 80,
            py::arg("tol_f") = 1e-5f,
            py::arg("tol_x") = 1e-4f,
            py::arg("mu") = 1e6f,
            py::arg("constrain_to_cell") = true,
            py::arg("prefer_current_cell") = true,
            py::arg("objective_type") = "Distance"
        )
        .def("config", py::overload_cast<>(&tree_packing::NelderMeadSolver::config),
            py::return_value_policy::reference_internal);

    // SolverRecreate optimizer
    py::class_<tree_packing::SolverRecreate, tree_packing::Optimizer, std::shared_ptr<tree_packing::SolverRecreate>>(m, "SolverRecreate")
        .def(py::init([](
            std::shared_ptr<tree_packing::Solver> solver,
            int max_recreate,
            int num_samples,
            bool verbose
        ) {
            // Clone the solver to transfer ownership
            tree_packing::SolverPtr solver_ptr = solver->clone();
            return std::make_shared<tree_packing::SolverRecreate>(
                std::move(solver_ptr),
                max_recreate,
                num_samples,
                verbose
            );
        }),
            py::arg("solver"),
            py::arg("max_recreate") = 1,
            py::arg("num_samples") = 16,
            py::arg("verbose") = false)
        .def("solver", py::overload_cast<>(&tree_packing::SolverRecreate::solver),
            py::return_value_policy::reference_internal);

    // SolverOptimize optimizer
    py::class_<tree_packing::SolverOptimize, tree_packing::Optimizer, std::shared_ptr<tree_packing::SolverOptimize>>(m, "SolverOptimize")
        .def(py::init([](
            std::shared_ptr<tree_packing::Solver> solver,
            int max_optimize,
            int group_size,
            bool same_cell_pairs,
            int num_samples,
            bool verbose
        ) {
            // Clone the solver to transfer ownership
            tree_packing::SolverPtr solver_ptr = solver->clone();
            return std::make_shared<tree_packing::SolverOptimize>(
                std::move(solver_ptr),
                max_optimize,
                group_size,
                same_cell_pairs,
                num_samples,
                verbose
            );
        }),
            py::arg("solver"),
            py::arg("max_optimize") = 1,
            py::arg("group_size") = 1,
            py::arg("same_cell_pairs") = false,
            py::arg("num_samples") = 8,
            py::arg("verbose") = false)
        .def("solver", py::overload_cast<>(&tree_packing::SolverOptimize::solver),
            py::return_value_policy::reference_internal);

    // Helper function to run optimization
    m.def("run_optimization", [](
        tree_packing::Problem& problem,
        tree_packing::Solution& initial_solution,
        int num_iterations,
        uint64_t seed,
        int n_remove,
        int max_recreate,
        float initial_temp,
        float cooling_rate,
        bool verbose
    ) {
        // Create optimizers
        auto ruin = std::make_unique<tree_packing::RandomRuin>(n_remove);
        auto recreate = std::make_unique<tree_packing::RandomRecreate>(max_recreate);

        std::vector<tree_packing::OptimizerPtr> ruin_ops;
        std::vector<tree_packing::OptimizerPtr> recreate_ops;
        ruin_ops.push_back(std::move(ruin));
        recreate_ops.push_back(std::move(recreate));

        auto alns = std::make_unique<tree_packing::ALNS>(
            std::move(ruin_ops),
            std::move(recreate_ops)
        );

        auto sa = tree_packing::SimulatedAnnealing(
            std::move(alns),
            initial_temp,
            1e-6f,
            tree_packing::CoolingSchedule::Exponential,
            cooling_rate,
            -1,
            false
        );

        sa.set_problem(&problem);

        // Initialize
        auto solution_eval = problem.eval(initial_solution);
        auto state = sa.init_state(solution_eval);
        auto global_state = tree_packing::GlobalState(seed, solution_eval);

        // Run optimization
        for (int i = 0; i < num_iterations; ++i) {
            tree_packing::RNG rng(global_state.split_rng());
            sa.apply(solution_eval, state, global_state, rng);

            global_state.maybe_update_best(problem, solution_eval);
            global_state.next();

            if (verbose && (i + 1) % 100 == 0) {
                py::print(
                "Iteration", i + 1,
                "best_score:", global_state.best_feasible_score(),
                "violations:", solution_eval.total_violation()
                );
            }
        }

        return py::make_tuple(
            global_state.best_feasible_score(),
            global_state.best_score()
        );
    },
        py::arg("problem"),
        py::arg("initial_solution"),
        py::arg("num_iterations") = 1000,
        py::arg("seed") = 42,
        py::arg("n_remove") = 1,
        py::arg("max_recreate") = 1,
        py::arg("initial_temp") = 1000.0f,
        py::arg("cooling_rate") = 0.9995f,
        py::arg("verbose") = false);

    // Constants
    m.attr("PI") = tree_packing::PI;
    m.attr("CENTER_Y") = tree_packing::CENTER_Y;
    m.attr("CENTER_R") = tree_packing::CENTER_R;
    m.attr("THR") = tree_packing::THR;
}
