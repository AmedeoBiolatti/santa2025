#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <any>

#include "tree_packing/tree_packing.hpp"

namespace py = pybind11;

namespace {
struct OptimizerState {
    std::any value;
};

py::array_t<float> make_array_view(const std::vector<float>& data, py::handle base) {
    return py::array_t<float>(
        {static_cast<py::ssize_t>(data.size())},
        {static_cast<py::ssize_t>(sizeof(float))},
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

    // Solution
    py::class_<tree_packing::Solution>(m, "Solution")
        .def(py::init<>())
        .def(py::init<size_t>())
        .def_static("init_random", &tree_packing::Solution::init_random,
            py::arg("num_trees"), py::arg("side") = 10.0f, py::arg("seed") = 42)
        .def_static("init_empty", &tree_packing::Solution::init_empty)
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
        });

    // SolutionEval
    py::class_<tree_packing::SolutionEval>(m, "SolutionEval")
        .def(py::init<>())
        .def_readwrite("solution", &tree_packing::SolutionEval::solution)
        .def_readwrite("objective", &tree_packing::SolutionEval::objective)
        .def_readwrite("intersection_violation", &tree_packing::SolutionEval::intersection_violation)
        .def_readwrite("bounds_violation", &tree_packing::SolutionEval::bounds_violation)
        .def("intersection_map", [](const tree_packing::SolutionEval& self) {
            py::list out;
            for (const auto& row_ptr : self.intersection_map) {
                if (!row_ptr) {
                    out.append(py::none());
                    continue;
                }
                py::list row;
                for (const auto& [idx, score] : *row_ptr) {
                    row.append(py::make_tuple(idx, score));
                }
                out.append(row);
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
                const auto& row_ptr = self.intersection_map[i];
                if (!row_ptr) continue;
                for (const auto& [idx, score] : *row_ptr) {
                    if (idx < 0) continue;
                    size_t j = static_cast<size_t>(idx);
                    if (j >= n) continue;
                    buf(static_cast<py::ssize_t>(i), static_cast<py::ssize_t>(j)) = score;
                }
            }
            return mat;
        })
        .def("copy_from", &tree_packing::SolutionEval::copy_from)
        .def("total_violation", &tree_packing::SolutionEval::total_violation)
        .def("n_missing", &tree_packing::SolutionEval::n_missing)
        .def("reg", &tree_packing::SolutionEval::reg);

    // Problem
    py::class_<tree_packing::Problem>(m, "Problem")
        .def(py::init<>())
        .def_static("create_tree_packing_problem", &tree_packing::Problem::create_tree_packing_problem,
            py::arg("side") = -1.0f)
        .def("eval", &tree_packing::Problem::eval)
        .def("eval_update", &tree_packing::Problem::eval_update)
        .def("score", &tree_packing::Problem::score)
        .def("objective", &tree_packing::Problem::objective)
        .def("min_pos", &tree_packing::Problem::min_pos)
        .def("max_pos", &tree_packing::Problem::max_pos);

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
        .def("best_solution", [](const tree_packing::GlobalState& self) -> py::object {
            const auto* best = self.best_solution();
            if (!best) {
                return py::none();
            }
            return py::cast(*best);
        })
        .def("best_feasible_solution", [](const tree_packing::GlobalState& self) -> py::object {
            const auto* best = self.best_feasible_solution();
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

    // SpatialRuin
    py::class_<tree_packing::SpatialRuin, tree_packing::Optimizer, std::shared_ptr<tree_packing::SpatialRuin>>(m, "SpatialRuin")
        .def(py::init<int, bool>(), py::arg("n_remove") = 1, py::arg("verbose") = false);

    // RandomRecreate
    py::class_<tree_packing::RandomRecreate, tree_packing::Optimizer, std::shared_ptr<tree_packing::RandomRecreate>>(m, "RandomRecreate")
        .def(py::init<int, float, float, bool>(),
            py::arg("max_recreate") = 1,
            py::arg("box_size") = 5.0f,
            py::arg("delta") = 0.35f,
            py::arg("verbose") = false);

    // NoiseOptimizer
    py::class_<tree_packing::NoiseOptimizer, tree_packing::Optimizer, std::shared_ptr<tree_packing::NoiseOptimizer>>(m, "NoiseOptimizer")
        .def(py::init<float, bool>(), py::arg("noise_level") = 0.01f, py::arg("verbose") = false);

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
