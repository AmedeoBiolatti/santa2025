#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cstdint>
#include <chrono>
#include <chrono>
#include <cstdio>
#include <deque>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <optional>
#include <sstream>
#include <optional>
#include <string>
#include <thread>
#include <cmath>
#include <limits>
#include <variant>
#include <vector>
#include <functional>

#include "tree_packing/tree_packing.hpp"
#include "tree_packing/solvers/noise_solver.hpp"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl2.h"

#include <GLFW/glfw3.h>

using namespace tree_packing;

namespace {

struct MetricsSnapshot {
    uint64_t iteration = 0;
    float objective = 0.0f;
    float intersection_violation = 0.0f;
    float bounds_violation = 0.0f;
    float total_violation = 0.0f;
    int n_missing = 0;
    float reg = 0.0f;
    float best_score = 0.0f;
    float best_feasible_score = 0.0f;
    bool has_sa = false;
    float sa_temperature = 0.0f;
    float sa_accept_rate = 0.0f;
    float sa_last_accept_prob = 0.0f;
    uint64_t sa_accepted = 0;
    uint64_t sa_rejected = 0;
    int sa_iteration = 0;
    bool sa_last_accept = false;
};

struct GeometrySnapshot {
    uint64_t iteration = 0;
    TreeParamsSoA current;
    TreeParamsSoA best;
    bool has_best = false;
    SolutionEval::IntersectionMap current_intersections;
    std::optional<SolutionEval::IntersectionMap> best_intersections;
};

struct SnapshotQueue {
    std::mutex mutex;
    std::deque<MetricsSnapshot> metrics;
    std::deque<GeometrySnapshot> geometry;
    size_t max_metrics = 4096;
    size_t max_geometry = 8;

    void push_metrics(const MetricsSnapshot& snap) {
        std::lock_guard<std::mutex> lock(mutex);
        metrics.push_back(snap);
        while (metrics.size() > max_metrics) {
            metrics.pop_front();
        }
    }

    void push_geometry(GeometrySnapshot&& snap) {
        std::lock_guard<std::mutex> lock(mutex);
        geometry.push_back(std::move(snap));
        while (geometry.size() > max_geometry) {
            geometry.pop_front();
        }
    }

    std::deque<MetricsSnapshot> pop_all_metrics() {
        std::lock_guard<std::mutex> lock(mutex);
        std::deque<MetricsSnapshot> out;
        out.swap(metrics);
        return out;
    }

    std::optional<GeometrySnapshot> pop_latest_geometry() {
        std::lock_guard<std::mutex> lock(mutex);
        if (geometry.empty()) {
            return std::nullopt;
        }
        GeometrySnapshot out = std::move(geometry.back());
        geometry.clear();
        return out;
    }
};

struct FloatRing {
    std::vector<float> data;
    int capacity = 0;
    int count = 0;
    int head = 0;

    explicit FloatRing(int cap = 0) : data(static_cast<size_t>(cap), 0.0f), capacity(cap) {}

    void push(float v) {
        if (capacity == 0) return;
        data[static_cast<size_t>(head)] = v;
        head = (head + 1) % capacity;
        if (count < capacity) {
            ++count;
        }
    }

    float get(int idx) const {
        if (count == 0) return 0.0f;
        int start = (head - count + capacity) % capacity;
        int pos = (start + idx) % capacity;
        return data[static_cast<size_t>(pos)];
    }
};

struct PlotSeries {
    FloatRing values;
    const char* label = "";

    explicit PlotSeries(int cap = 0, const char* lbl = "")
        : values(cap), label(lbl) {}
};

struct RenderCache {
    std::vector<Figure> figures;
    AABB bounds;
    bool has_bounds = false;
    uint64_t iteration = 0;
    std::vector<Vec2> centers;
};

struct ViewCache {
    RenderCache current;
    RenderCache best;
    TreeParamsSoA current_params;
    TreeParamsSoA best_params;
    SolutionEval::IntersectionMap current_intersections;
    SolutionEval::IntersectionMap best_intersections;
    bool has_best = false;
};

struct RunConfig {
    int num_trees = 27;
    float side = 10.0f;
    uint64_t seed = 42;
    int metrics_stride = 100;
    int geometry_stride = 1000;
};

struct TreeDragCommand {
    int index = -1;
    TreeParams params{};
};

struct TreeDragState {
    bool active = false;
    int index = -1;
    Vec2 click_offset{};
};

struct ProblemParams {
    float objective_ceiling = std::numeric_limits<float>::infinity();
    float intersection_tolerance = 1e-6f;
    ConstraintPenaltyType constraint_penalty = ConstraintPenaltyType::Tolerant;
};

bool save_best_solution_to_csv(const TreeParamsSoA& params, int num_trees, std::string& status) {
    size_t tree_count = params.size();
    if (tree_count == 0) {
        status = "Best solution empty";
        return false;
    }
    if (static_cast<int>(tree_count) != num_trees) {
        status = "Best solution tree count mismatch";
        return false;
    }
    std::string filename = "submission_" + std::to_string(num_trees) + ".csv";
    std::ofstream out(filename, std::ios::trunc);
    if (!out.is_open()) {
        status = "Failed to open " + filename;
        return false;
    }
    out << "id,x,y,deg\n";
    std::ostringstream prefix_ss;
    prefix_ss << std::setfill('0') << std::setw(3) << num_trees;
    std::string prefix = prefix_ss.str();

    auto format_value = [](float v) {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(6) << v;
        return std::string("s") + ss.str();
    };

    for (int i = 0; i < num_trees; ++i) {
        out << prefix << "_" << i << ","
            << format_value(params.x[i]) << ","
            << format_value(params.y[i]) << ","
            << format_value(params.angle[i]) << "\n";
    }
    if (!out) {
        status = "Failed to write " + filename;
        return false;
    }
    status = "Saved best solution to " + filename;
    return true;
}

constexpr uint64_t kDashboardFullEvalInterval = 1'000'000;
constexpr float kDefaultSAInitialTemp = 1000.0f;
constexpr float kDefaultSAMinTemp = 1e-6f;

void configure_problem(Problem& problem, const ProblemParams& params) {
    problem.set_objective_ceiling(params.objective_ceiling);
    problem.set_constraint_tolerance(params.intersection_tolerance);
    problem.set_constraint_penalty_type(params.constraint_penalty);
}

enum class OptType {
    RandomRuin,
    CellRuin,
    RandomRecreate,
    GridCellRecreate,
    Noise,
    Squeeze,
    Compaction,
    LocalSearch,
    RestoreBest,
    Chain,
    Alternate,
    RandomChoice,
    Repeat,
    Conditional,
    ALNS,
    SimulatedAnnealing,
    SolverOptimize
};

struct RandomRuinParams { int n_remove = 1; };
struct CellRuinParams { int n_remove = 1; };
struct RandomRecreateParams { int max_recreate = 1; float box_size = 5.0f; float delta = 0.35f; };
struct GridCellRecreateParams { int max_recreate = 1; int cell_min = 0; int cell_max = 4; int neighbor_min = 1; int neighbor_max = -1; };
struct NoiseParams { float noise_level = 0.01f; int n_change = 1; };
struct SqueezeParams { float min_scale = 0.05f; float shrink = 0.92f; int bisect_iters = 18; int axis_rounds = 3; };
struct CompactionParams { int iters_per_tree = 8; };
struct LocalSearchParams { int iters_per_tree = 18; };
struct RestoreBestParams { int interval = 100; };
struct RepeatParams { int n = 1; };
struct ConditionalParams { uint64_t every_n = 0; uint64_t min_iters_since_improvement = 0; uint64_t min_iters_since_feasible_improvement = 0; };
struct ALNSParams { float reaction_factor = 0.01f; float reward_improve = 1.0f; float reward_no_improve = 0.0f; float min_weight = 1e-3f; };
struct SAParams {
    float initial_temp = 1000.0f;
    float min_temp = 1e-6f;
    CoolingSchedule schedule = CoolingSchedule::Exponential;
    float cooling_rate = 0.9995f;
    int patience = -1;
};
struct SolverOptimizeParams {
    int max_optimize = 1;
    int group_size = 1;
    bool same_cell_pairs = false;
    int num_samples = 8;
};

using OptParams = std::variant<
    std::monostate,
    RandomRuinParams,
    CellRuinParams,
    RandomRecreateParams,
    GridCellRecreateParams,
    NoiseParams,
    SqueezeParams,
    CompactionParams,
    LocalSearchParams,
    RestoreBestParams,
    RepeatParams,
    ConditionalParams,
    ALNSParams,
    SAParams,
    SolverOptimizeParams
>;

struct OptConfig {
    OptType type = OptType::Chain;
    bool verbose = false;
    OptParams params{};
    std::vector<OptConfig> children;
    std::vector<OptConfig> ruin_children;
    std::vector<OptConfig> recreate_children;
};

void update_sa_initial_temp(OptConfig& cfg, float temp) {
    if (cfg.type == OptType::SimulatedAnnealing) {
        auto& p = std::get<SAParams>(cfg.params);
        p.initial_temp = temp;
    }
    for (auto& child : cfg.children) {
        update_sa_initial_temp(child, temp);
    }
    for (auto& child : cfg.ruin_children) {
        update_sa_initial_temp(child, temp);
    }
    for (auto& child : cfg.recreate_children) {
        update_sa_initial_temp(child, temp);
    }
}

void update_sa_temperature_params(OptConfig& cfg, float initial_temp, float min_temp) {
    if (cfg.type == OptType::SimulatedAnnealing) {
        auto& p = std::get<SAParams>(cfg.params);
        p.initial_temp = initial_temp;
        p.min_temp = min_temp;
    }
    for (auto& child : cfg.children) {
        update_sa_temperature_params(child, initial_temp, min_temp);
    }
    for (auto& child : cfg.ruin_children) {
        update_sa_temperature_params(child, initial_temp, min_temp);
    }
    for (auto& child : cfg.recreate_children) {
        update_sa_temperature_params(child, initial_temp, min_temp);
    }
}

const char* opt_type_name(OptType type) {
    switch (type) {
        case OptType::RandomRuin: return "RandomRuin";
        case OptType::CellRuin: return "CellRuin";
        case OptType::RandomRecreate: return "RandomRecreate";
        case OptType::GridCellRecreate: return "GridCellRecreate";
        case OptType::Noise: return "NoiseOptimizer";
        case OptType::Squeeze: return "SqueezeOptimizer";
        case OptType::Compaction: return "CompactionOptimizer";
        case OptType::LocalSearch: return "LocalSearchOptimizer";
        case OptType::RestoreBest: return "RestoreBest";
        case OptType::Chain: return "Chain";
        case OptType::Alternate: return "Alternate";
        case OptType::RandomChoice: return "RandomChoice";
        case OptType::Repeat: return "Repeat";
        case OptType::Conditional: return "Conditional";
        case OptType::ALNS: return "ALNS";
        case OptType::SimulatedAnnealing: return "SimulatedAnnealing";
        case OptType::SolverOptimize: return "SolverOptimize";
        default: return "Unknown";
    }
}

OptConfig default_optimizer_config() {
    OptConfig ruin;
    ruin.type = OptType::RandomRuin;
    ruin.params = RandomRuinParams{1};
    OptConfig recreate;
    recreate.type = OptType::RandomRecreate;
    recreate.params = RandomRecreateParams{1, 5.0f, 0.35f};

    OptConfig alns;
    alns.type = OptType::ALNS;
    alns.params = ALNSParams{0.01f, 1.0f, 0.0f, 1e-3f};
    alns.ruin_children = {ruin};
    alns.recreate_children = {recreate};

    OptConfig noise;
    noise.type = OptType::Noise;
    noise.params = NoiseParams{0.01f, 1};

    OptConfig alternate;
    alternate.type = OptType::Alternate;
    alternate.children = {noise, alns};

    OptConfig sa;
    sa.type = OptType::SimulatedAnnealing;
    sa.params = SAParams{1000.0f, 1e-6f, CoolingSchedule::Exponential, 0.9995f, -1};
    sa.children = {alternate};

    return sa;
}

struct OptimizerPreset {
    const char* name = "Unknown";
    OptConfig config;
};

const std::vector<OptimizerPreset>& optimizer_presets() {
    static const std::vector<OptimizerPreset> presets = []() {
        std::vector<OptimizerPreset> out;

        OptimizerPreset sa_preset;
        sa_preset.name = "SimulatedAnnealing (default)";
        sa_preset.config = default_optimizer_config();
        out.push_back(sa_preset);

        OptimizerPreset alns_preset;
        {
            OptConfig ruin;
            ruin.type = OptType::RandomRuin;
            ruin.params = RandomRuinParams{1};
            OptConfig recreate;
            recreate.type = OptType::RandomRecreate;
            recreate.params = RandomRecreateParams{1, 5.0f, 0.35f};
            OptConfig alns;
            alns.type = OptType::ALNS;
            alns.params = ALNSParams{0.01f, 1.0f, 0.0f, 1e-3f};
            alns.ruin_children = {ruin};
            alns.recreate_children = {recreate};
            alns_preset.name = "ALNS (ruin + recreate)";
            alns_preset.config = alns;
        }
        out.push_back(alns_preset);

        OptimizerPreset noise_preset;
        noise_preset.name = "NoiseOptimizer";
        noise_preset.config.type = OptType::Noise;
        noise_preset.config.params = NoiseParams{0.01f, 1};
        out.push_back(noise_preset);

        OptimizerPreset compaction_preset;
        compaction_preset.name = "CompactionOptimizer";
        compaction_preset.config.type = OptType::Compaction;
        compaction_preset.config.params = CompactionParams{10};
        out.push_back(compaction_preset);

        OptimizerPreset local_search_preset;
        local_search_preset.name = "LocalSearchOptimizer";
        local_search_preset.config.type = OptType::LocalSearch;
        local_search_preset.config.params = LocalSearchParams{10};
        out.push_back(local_search_preset);

        OptimizerPreset solver_opt_preset;
        solver_opt_preset.name = "SolverOptimize (NoiseSolver)";
        solver_opt_preset.config.type = OptType::SolverOptimize;
        solver_opt_preset.config.params = SolverOptimizeParams{1, 1, false, 8};
        out.push_back(solver_opt_preset);

        return out;
    }();
    return presets;
}

using OptimizerPresetApplyCallback = std::function<void()>;
OptimizerPresetApplyCallback g_optimizer_preset_apply_callback;

void set_optimizer_preset_apply_callback(OptimizerPresetApplyCallback cb) {
    g_optimizer_preset_apply_callback = std::move(cb);
}

void clear_optimizer_preset_apply_callback() {
    g_optimizer_preset_apply_callback = nullptr;
}

void notify_optimizer_preset_applied() {
    if (g_optimizer_preset_apply_callback) {
        g_optimizer_preset_apply_callback();
    }
}

bool render_optimizer_preset_combo(const char* label, OptConfig& cfg, bool enabled) {
    const auto& presets = optimizer_presets();
    if (presets.empty()) {
        return false;
    }
    if (!enabled) {
        ImGui::BeginDisabled();
    }
    bool applied = false;
    if (ImGui::BeginCombo(label, "Select preset")) {
        for (const auto& preset : presets) {
            if (ImGui::Selectable(preset.name)) {
                cfg = preset.config;
                applied = true;
                notify_optimizer_preset_applied();
            }
        }
        ImGui::EndCombo();
    }
    if (!enabled) {
        ImGui::EndDisabled();
    }
    return applied;
}

OptimizerPtr build_optimizer(const OptConfig& cfg) {
    switch (cfg.type) {
        case OptType::RandomRuin: {
            const auto& p = std::get<RandomRuinParams>(cfg.params);
            return std::make_unique<RandomRuin>(p.n_remove, cfg.verbose);
        }
        case OptType::CellRuin: {
            const auto& p = std::get<CellRuinParams>(cfg.params);
            return std::make_unique<CellRuin>(p.n_remove, cfg.verbose);
        }
        case OptType::RandomRecreate: {
            const auto& p = std::get<RandomRecreateParams>(cfg.params);
            return std::make_unique<RandomRecreate>(p.max_recreate, p.box_size, p.delta, cfg.verbose);
        }
        case OptType::GridCellRecreate: {
            const auto& p = std::get<GridCellRecreateParams>(cfg.params);
            return std::make_unique<GridCellRecreate>(
                p.max_recreate, p.cell_min, p.cell_max, p.neighbor_min, p.neighbor_max, cfg.verbose
            );
        }
        case OptType::Noise: {
            const auto& p = std::get<NoiseParams>(cfg.params);
            return std::make_unique<NoiseOptimizer>(p.noise_level, p.n_change, cfg.verbose);
        }
        case OptType::Squeeze: {
            const auto& p = std::get<SqueezeParams>(cfg.params);
            return std::make_unique<SqueezeOptimizer>(p.min_scale, p.shrink, p.bisect_iters, p.axis_rounds, cfg.verbose);
        }
        case OptType::Compaction: {
            const auto& p = std::get<CompactionParams>(cfg.params);
            return std::make_unique<CompactionOptimizer>(p.iters_per_tree, cfg.verbose);
        }
        case OptType::LocalSearch: {
            const auto& p = std::get<LocalSearchParams>(cfg.params);
            return std::make_unique<LocalSearchOptimizer>(p.iters_per_tree, cfg.verbose);
        }
        case OptType::RestoreBest: {
            const auto& p = std::get<RestoreBestParams>(cfg.params);
            return std::make_unique<RestoreBest>(p.interval, cfg.verbose);
        }
        case OptType::SolverOptimize: {
            const auto& p = std::get<SolverOptimizeParams>(cfg.params);
            return std::make_unique<SolverOptimize>(
                std::make_unique<NoiseSolver>(),
                p.max_optimize,
                p.group_size,
                p.same_cell_pairs,
                p.num_samples,
                cfg.verbose
            );
        }
        case OptType::Chain: {
            std::vector<OptimizerPtr> children;
            children.reserve(cfg.children.size());
            for (const auto& child : cfg.children) {
                children.push_back(build_optimizer(child));
            }
            return std::make_unique<Chain>(std::move(children), cfg.verbose);
        }
        case OptType::Alternate: {
            std::vector<OptimizerPtr> children;
            children.reserve(cfg.children.size());
            for (const auto& child : cfg.children) {
                children.push_back(build_optimizer(child));
            }
            return std::make_unique<Alternate>(std::move(children), cfg.verbose);
        }
        case OptType::RandomChoice: {
            std::vector<OptimizerPtr> children;
            children.reserve(cfg.children.size());
            for (const auto& child : cfg.children) {
                children.push_back(build_optimizer(child));
            }
            return std::make_unique<RandomChoice>(std::move(children), std::vector<float>{}, cfg.verbose);
        }
        case OptType::Repeat: {
            const auto& p = std::get<RepeatParams>(cfg.params);
            if (cfg.children.empty()) {
                return std::make_unique<Repeat>(std::make_unique<NoiseOptimizer>(), p.n, cfg.verbose);
            }
            return std::make_unique<Repeat>(build_optimizer(cfg.children.front()), p.n, cfg.verbose);
        }
        case OptType::Conditional: {
            const auto& p = std::get<ConditionalParams>(cfg.params);
            if (cfg.children.empty()) {
                return std::make_unique<Conditional>(std::make_unique<NoiseOptimizer>(), p.every_n, p.min_iters_since_improvement, p.min_iters_since_feasible_improvement, nullptr, cfg.verbose);
            }
            return std::make_unique<Conditional>(
                build_optimizer(cfg.children.front()),
                p.every_n,
                p.min_iters_since_improvement,
                p.min_iters_since_feasible_improvement,
                nullptr,
                cfg.verbose
            );
        }
        case OptType::ALNS: {
            const auto& p = std::get<ALNSParams>(cfg.params);
            std::vector<OptimizerPtr> ruin_ops;
            std::vector<OptimizerPtr> recreate_ops;
            ruin_ops.reserve(cfg.ruin_children.size());
            recreate_ops.reserve(cfg.recreate_children.size());
            for (const auto& child : cfg.ruin_children) {
                ruin_ops.push_back(build_optimizer(child));
            }
            for (const auto& child : cfg.recreate_children) {
                recreate_ops.push_back(build_optimizer(child));
            }
            return std::make_unique<ALNS>(
                std::move(ruin_ops),
                std::move(recreate_ops),
                p.reaction_factor,
                p.reward_improve,
                p.reward_no_improve,
                p.min_weight,
                cfg.verbose
            );
        }
        case OptType::SimulatedAnnealing: {
            const auto& p = std::get<SAParams>(cfg.params);
            if (cfg.children.empty()) {
                return std::make_unique<SimulatedAnnealing>(
                    std::make_unique<NoiseOptimizer>(),
                    p.initial_temp,
                    p.min_temp,
                    p.schedule,
                    p.cooling_rate,
                    p.patience,
                    cfg.verbose
                );
            }
            return std::make_unique<SimulatedAnnealing>(
                build_optimizer(cfg.children.front()),
                p.initial_temp,
                p.min_temp,
                p.schedule,
                p.cooling_rate,
                p.patience,
                cfg.verbose
            );
        }
        default:
            return std::make_unique<NoiseOptimizer>();
    }
}

struct MetricsSnapshot;
bool edit_optimizer_config(OptConfig& cfg, const MetricsSnapshot* metrics, bool* reset_sa_temp = nullptr, bool allow_preset_combo = true, bool presets_enabled = true);

bool edit_children(
    const char* label,
    std::vector<OptConfig>& children,
    const MetricsSnapshot* metrics,
    bool show_root = true,
    bool* reset_sa_temp = nullptr,
    bool presets_enabled = true
) {
    bool changed = false;
    auto render_children = [&]() {
        for (size_t i = 0; i < children.size(); ++i) {
            std::string node_label = std::string(opt_type_name(children[i].type)) + "##" + std::to_string(i);
            if (ImGui::TreeNode(node_label.c_str())) {
                changed |= edit_optimizer_config(children[i], metrics, reset_sa_temp, true, presets_enabled);
                ImGui::TreePop();
            }
        }
    };

    if (show_root) {
        if (ImGui::TreeNode(label)) {
            render_children();
            ImGui::TreePop();
        }
    } else {
        ImGui::Indent(ImGui::GetTreeNodeToLabelSpacing());
        render_children();
        ImGui::Unindent(ImGui::GetTreeNodeToLabelSpacing());
    }
    return changed;
}

bool apply_mouse_wheel_int(int& value, int step = 1, int min_value = std::numeric_limits<int>::min(), int max_value = std::numeric_limits<int>::max()) {
    ImGuiIO& io = ImGui::GetIO();
    if (io.MouseWheel == 0.0f) {
        return false;
    }
    if (!ImGui::IsItemHovered(ImGuiHoveredFlags_None) || ImGui::IsItemActive()) {
        return false;
    }
    int delta = static_cast<int>(std::round(io.MouseWheel)) * step;
    if (delta == 0) {
        return false;
    }
    int new_value = std::clamp(value + delta, min_value, max_value);
    if (new_value == value) {
        return false;
    }
    value = new_value;
    return true;
}

bool apply_mouse_wheel_float(float& value, float step = 0.1f, float min_value = -std::numeric_limits<float>::max(), float max_value = std::numeric_limits<float>::max()) {
    ImGuiIO& io = ImGui::GetIO();
    if (io.MouseWheel == 0.0f) {
        return false;
    }
    if (!ImGui::IsItemHovered(ImGuiHoveredFlags_None) || ImGui::IsItemActive()) {
        return false;
    }
    float delta = io.MouseWheel * step;
    if (delta == 0.0f) {
        return false;
    }
    float new_value = std::clamp(value + delta, min_value, max_value);
    if (new_value == value) {
        return false;
    }
    value = new_value;
    return true;
}

bool apply_mouse_wheel_uint64(uint64_t& value, uint64_t step = 1, uint64_t min_value = 0, uint64_t max_value = std::numeric_limits<uint64_t>::max()) {
    ImGuiIO& io = ImGui::GetIO();
    if (io.MouseWheel == 0.0f) {
        return false;
    }
    if (!ImGui::IsItemHovered(ImGuiHoveredFlags_None) || ImGui::IsItemActive()) {
        return false;
    }
    int delta = static_cast<int>(std::round(io.MouseWheel));
    if (delta == 0) {
        return false;
    }
    uint64_t new_value = value;
    if (delta > 0) {
        uint64_t increment = static_cast<uint64_t>(delta) * step;
        new_value = std::min(max_value, value + increment);
    } else {
        uint64_t magnitude = static_cast<uint64_t>(-delta) * step;
        uint64_t diff = value - min_value;
        if (magnitude >= diff) {
            new_value = min_value;
        } else {
            new_value = value - magnitude;
        }
    }
    if (new_value == value) {
        return false;
    }
    value = new_value;
    return true;
}

bool edit_optimizer_config(OptConfig& cfg, const MetricsSnapshot* metrics, bool* reset_sa_temp, bool allow_preset_combo, bool presets_enabled) {
    bool changed = false;
    ImGui::PushID(&cfg);
    ImGui::TextUnformatted(opt_type_name(cfg.type));
    if (allow_preset_combo) {
        bool applied = render_optimizer_preset_combo("Preset", cfg, presets_enabled);
        changed |= applied;
        ImGui::Spacing();
    }
    changed |= ImGui::Checkbox("Verbose", &cfg.verbose);

    switch (cfg.type) {
        case OptType::RandomRuin: {
            auto& p = std::get<RandomRuinParams>(cfg.params);
            changed |= ImGui::InputInt("n_remove", &p.n_remove);
            changed |= apply_mouse_wheel_int(p.n_remove, 1, 1);
            break;
        }
        case OptType::CellRuin: {
            auto& p = std::get<CellRuinParams>(cfg.params);
            changed |= ImGui::InputInt("n_remove", &p.n_remove);
            changed |= apply_mouse_wheel_int(p.n_remove, 1, 1);
            break;
        }
        case OptType::RandomRecreate: {
            auto& p = std::get<RandomRecreateParams>(cfg.params);
            changed |= ImGui::InputInt("max_recreate", &p.max_recreate);
            changed |= apply_mouse_wheel_int(p.max_recreate, 1, 1);
            changed |= ImGui::InputFloat("box_size", &p.box_size, 0.0f, 0.0f, "%.6f");
            changed |= apply_mouse_wheel_float(p.box_size, 0.1f, 0.0f);
            changed |= ImGui::InputFloat("delta", &p.delta, 0.0f, 0.0f, "%.6f");
            changed |= apply_mouse_wheel_float(p.delta, 0.01f, 0.0f, 1.0f);
            break;
        }
        case OptType::GridCellRecreate: {
            auto& p = std::get<GridCellRecreateParams>(cfg.params);
            changed |= ImGui::InputInt("max_recreate", &p.max_recreate);
            changed |= ImGui::InputInt("cell_min", &p.cell_min);
            changed |= ImGui::InputInt("cell_max", &p.cell_max);
            changed |= ImGui::InputInt("neighbor_min", &p.neighbor_min);
            changed |= ImGui::InputInt("neighbor_max", &p.neighbor_max);
            changed |= apply_mouse_wheel_int(p.max_recreate, 1, 1);
            changed |= apply_mouse_wheel_int(p.cell_min, 1, 0);
            changed |= apply_mouse_wheel_int(p.cell_max, 1, 0);
            changed |= apply_mouse_wheel_int(p.neighbor_min, 1, -10, 10);
            changed |= apply_mouse_wheel_int(p.neighbor_max, 1, -10, 10);
            break;
        }
        case OptType::Noise: {
            auto& p = std::get<NoiseParams>(cfg.params);
            changed |= ImGui::InputFloat("noise_level", &p.noise_level, 0.0f, 0.0f, "%.6f");
            changed |= apply_mouse_wheel_float(p.noise_level, 0.01f, 0.0f, 10.0f);
            changed |= ImGui::InputInt("n_change", &p.n_change);
            changed |= apply_mouse_wheel_int(p.n_change, 1, 1);
            break;
        }
        case OptType::Squeeze: {
            auto& p = std::get<SqueezeParams>(cfg.params);
            changed |= ImGui::InputFloat("min_scale", &p.min_scale, 0.0f, 0.0f, "%.6f");
            changed |= apply_mouse_wheel_float(p.min_scale, 0.01f, 0.0f, 1.0f);
            changed |= ImGui::InputFloat("shrink", &p.shrink, 0.0f, 0.0f, "%.6f");
            changed |= apply_mouse_wheel_float(p.shrink, 0.01f, 0.0f, 1.0f);
            changed |= ImGui::InputInt("bisect_iters", &p.bisect_iters);
            changed |= apply_mouse_wheel_int(p.bisect_iters, 1, 1);
            changed |= ImGui::InputInt("axis_rounds", &p.axis_rounds);
            changed |= apply_mouse_wheel_int(p.axis_rounds, 1, 1);
            break;
        }
        case OptType::Compaction: {
            auto& p = std::get<CompactionParams>(cfg.params);
            changed |= ImGui::InputInt("iters_per_tree", &p.iters_per_tree);
            changed |= apply_mouse_wheel_int(p.iters_per_tree, 1, 1);
            break;
        }
        case OptType::LocalSearch: {
            auto& p = std::get<LocalSearchParams>(cfg.params);
            changed |= ImGui::InputInt("iters_per_tree", &p.iters_per_tree);
            changed |= apply_mouse_wheel_int(p.iters_per_tree, 1, 1);
            break;
        }
        case OptType::SolverOptimize: {
            auto& p = std::get<SolverOptimizeParams>(cfg.params);
            changed |= ImGui::InputInt("max_optimize", &p.max_optimize);
            changed |= apply_mouse_wheel_int(p.max_optimize, 1, 1);
            changed |= ImGui::InputInt("group_size", &p.group_size);
            changed |= apply_mouse_wheel_int(p.group_size, 1, 1, 2);
            changed |= ImGui::Checkbox("same_cell_pairs", &p.same_cell_pairs);
            changed |= ImGui::InputInt("num_samples", &p.num_samples);
            changed |= apply_mouse_wheel_int(p.num_samples, 1, 1);
            break;
        }
        case OptType::RestoreBest: {
            auto& p = std::get<RestoreBestParams>(cfg.params);
            changed |= ImGui::InputInt("interval", &p.interval);
            changed |= apply_mouse_wheel_int(p.interval, 1, 0);
            break;
        }
        case OptType::Repeat: {
            auto& p = std::get<RepeatParams>(cfg.params);
            changed |= ImGui::InputInt("n", &p.n);
            changed |= apply_mouse_wheel_int(p.n, 1, 1);
            changed |= edit_children("Inner", cfg.children, metrics, true, reset_sa_temp, presets_enabled);
            break;
        }
        case OptType::Conditional: {
            auto& p = std::get<ConditionalParams>(cfg.params);
            changed |= ImGui::InputScalar("every_n", ImGuiDataType_U64, &p.every_n);
            changed |= apply_mouse_wheel_uint64(p.every_n);
            changed |= ImGui::InputScalar("min_iters_since_improvement", ImGuiDataType_U64, &p.min_iters_since_improvement);
            changed |= apply_mouse_wheel_uint64(p.min_iters_since_improvement);
            changed |= ImGui::InputScalar("min_iters_since_feasible_improvement", ImGuiDataType_U64, &p.min_iters_since_feasible_improvement);
            changed |= apply_mouse_wheel_uint64(p.min_iters_since_feasible_improvement);
            changed |= edit_children("Inner", cfg.children, metrics, true, reset_sa_temp, presets_enabled);
            break;
        }
        case OptType::ALNS: {
            auto& p = std::get<ALNSParams>(cfg.params);
            changed |= ImGui::InputFloat("reaction_factor", &p.reaction_factor, 0.0f, 0.0f, "%.6f");
            changed |= apply_mouse_wheel_float(p.reaction_factor, 0.01f, 0.0f);
            changed |= ImGui::InputFloat("reward_improve", &p.reward_improve, 0.0f, 0.0f, "%.6f");
            changed |= apply_mouse_wheel_float(p.reward_improve, 0.1f, 0.0f);
            changed |= ImGui::InputFloat("reward_no_improve", &p.reward_no_improve, 0.0f, 0.0f, "%.6f");
            changed |= apply_mouse_wheel_float(p.reward_no_improve, 0.1f, 0.0f);
            changed |= ImGui::InputFloat("min_weight", &p.min_weight, 0.0f, 0.0f, "%.6f");
            changed |= apply_mouse_wheel_float(p.min_weight, 0.01f, 0.0f);
            changed |= edit_children("Ruin Operators", cfg.ruin_children, metrics, true, reset_sa_temp, presets_enabled);
            changed |= edit_children("Recreate Operators", cfg.recreate_children, metrics, true, reset_sa_temp, presets_enabled);
            break;
        }
        case OptType::SimulatedAnnealing: {
            auto& p = std::get<SAParams>(cfg.params);
            changed |= ImGui::InputFloat("initial_temp", &p.initial_temp, 0.0f, 0.0f, "%.6f");
            changed |= apply_mouse_wheel_float(p.initial_temp, 10.0f, 0.0f);
            changed |= ImGui::InputFloat("min_temp", &p.min_temp, 0.0f, 0.0f, "%.8f");
            changed |= apply_mouse_wheel_float(p.min_temp, 0.01f, 0.0f);
            changed |= ImGui::InputFloat("cooling_rate", &p.cooling_rate, 0.0f, 0.0f, "%.6f");
            changed |= apply_mouse_wheel_float(p.cooling_rate, 0.001f, 0.0f, 1.0f);
            changed |= ImGui::InputInt("patience", &p.patience);
            changed |= apply_mouse_wheel_int(p.patience, 1);
            const char* schedules[] = {"Exponential", "Linear", "Logarithmic"};
            int schedule_idx = static_cast<int>(p.schedule);
            if (ImGui::Combo("schedule", &schedule_idx, schedules, 3)) {
                p.schedule = static_cast<CoolingSchedule>(schedule_idx);
                changed = true;
            }
            if (ImGui::Button("Reset temperature")) {
                p.initial_temp = kDefaultSAInitialTemp;
                p.min_temp = kDefaultSAMinTemp;
                changed = true;
                if (reset_sa_temp) {
                    *reset_sa_temp = true;
                }
            }
    if (metrics && metrics->has_sa) {
        ImGui::Separator();
        ImGui::TextUnformatted("Runtime stats");
        ImGui::Text("Temp: %.6f", metrics->sa_temperature);
        ImGui::Text("Accept rate: %.2f%%", metrics->sa_accept_rate * 100.0f);
        ImGui::Text("Last prob: %.3f", metrics->sa_last_accept_prob);
        ImGui::Text("Last: %s", metrics->sa_last_accept ? "accept" : "reject");
        ImGui::Text("Accepted: %llu", static_cast<unsigned long long>(metrics->sa_accepted));
        ImGui::Text("Rejected: %llu", static_cast<unsigned long long>(metrics->sa_rejected));
                ImGui::Text("SA iter: %d", metrics->sa_iteration);
            }
            changed |= edit_children("Inner", cfg.children, metrics, false, reset_sa_temp, presets_enabled);
            break;
        }
        default:
            changed |= edit_children("Children", cfg.children, metrics, true, reset_sa_temp, presets_enabled);
            break;
    }
    ImGui::PopID();
    return changed;
}

void build_figures_cache(const TreeParamsSoA& params, RenderCache& cache) {
    cache.figures.clear();
    if (params.size() == 0) {
        cache.has_bounds = false;
        return;
    }
    cache.figures.resize(params.size());
    params_to_figures(params, cache.figures);
    cache.centers.resize(params.size());
    get_tree_centers(params, cache.centers);

    AABB bounds;
    bool has_bounds = false;
    for (size_t i = 0; i < cache.figures.size(); ++i) {
        const auto& fig = cache.figures[i];
        if (fig.is_nan()) {
            continue;
        }
        for (size_t t = 0; t < fig.size(); ++t) {
            const Triangle& tri = fig[t];
            for (size_t v = 0; v < 3; ++v) {
                const Vec2& pt = tri[v];
                if (!pt.is_finite()) continue;
                if (!has_bounds) {
                    bounds.min = pt;
                    bounds.max = pt;
                    has_bounds = true;
                } else {
                    bounds.expand(pt);
                }
            }
        }
    }
    cache.bounds = bounds;
    cache.has_bounds = has_bounds;
}

static inline ImU32 with_alpha(ImU32 col, int a) {
    // ImGui packs color as ABGR in ImU32 via IM_COL32.
    // Preserve RGB, override A.
    return (col & ~IM_COL32_A_MASK) | ((ImU32)a << IM_COL32_A_SHIFT);
}

static void add_line_with_glow(
    ImDrawList* dl,
    const ImVec2& p0,
    const ImVec2& p1,
    ImU32 core_col,
    float core_thickness = 1.0f,
    float glow_thickness = 7.0f,
    int glow_alpha = 40
) {
    // Glow pass (thicker, low alpha)
    dl->AddLine(p0, p1, with_alpha(core_col, glow_alpha), glow_thickness);
    // Core pass (thin, full color)
    dl->AddLine(p0, p1, core_col, core_thickness);
}

ImVec2 world_to_screen(const Vec2& pt, const AABB& bounds, const ImVec2& origin, const ImVec2& size) {
    float width = bounds.max.x - bounds.min.x;
    float height = bounds.max.y - bounds.min.y;
    float scale = 1.0f;
    if (width > 1e-6f && height > 1e-6f) {
        scale = std::min(size.x / width, size.y / height);
    }
float x = (pt.x - bounds.min.x) * scale;
float y = (bounds.max.y - pt.y) * scale;
return ImVec2(origin.x + x, origin.y + y);
}

Vec2 screen_to_world(const ImVec2& screen, const AABB& bounds, const ImVec2& origin, const ImVec2& size) {
    float width = bounds.max.x - bounds.min.x;
    float height = bounds.max.y - bounds.min.y;
    float scale = 1.0f;
    if (width > 1e-6f && height > 1e-6f) {
        scale = std::min(size.x / width, size.y / height);
    }
    float tx = (screen.x - origin.x) / scale;
    float ty = (screen.y - origin.y) / scale;
    return Vec2{bounds.min.x + tx, bounds.max.y - ty};
}

inline float cross(const Vec2& a, const Vec2& b) {
    return a.x * b.y - a.y * b.x;
}

bool point_in_triangle(const Vec2& point, const Triangle& tri) {
    Vec2 ab = tri.v1 - tri.v0;
    Vec2 bc = tri.v2 - tri.v1;
    Vec2 ca = tri.v0 - tri.v2;
    Vec2 ap = point - tri.v0;
    Vec2 bp = point - tri.v1;
    Vec2 cp = point - tri.v2;
    float c0 = cross(ab, ap);
    float c1 = cross(bc, bp);
    float c2 = cross(ca, cp);
    return (c0 >= 0.0f && c1 >= 0.0f && c2 >= 0.0f) ||
           (c0 <= 0.0f && c1 <= 0.0f && c2 <= 0.0f);
}

int find_tree_at_point(const RenderCache& cache, const Vec2& point) {
    for (size_t i = 0; i < cache.figures.size(); ++i) {
        const auto& fig = cache.figures[i];
        if (fig.is_nan()) {
            continue;
        }
        for (size_t t = 0; t < fig.size(); ++t) {
            if (point_in_triangle(point, fig[t])) {
                return static_cast<int>(i);
            }
        }
    }
    return -1;
}

void draw_intersection_segments(
    const RenderCache& cache,
    const ImVec2& origin,
    const ImVec2& size,
    const SolutionEval::IntersectionMap& intersections,
    ImU32 color = IM_COL32(255, 120, 45, 200)
) {
    if (!cache.has_bounds || cache.centers.empty() || intersections.empty()) {
        return;
    }
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    size_t center_count = cache.centers.size();
    for (size_t i = 0; i < intersections.size(); ++i) {
        if (i >= center_count) break;
        const Vec2& center_i = cache.centers[i];
        if (!center_i.is_finite()) continue;
        for (const auto& entry : intersections[i]) {
            int neighbor = entry.neighbor;
            if (neighbor <= static_cast<int>(i)) {
                continue;
            }
            if (neighbor < 0 || static_cast<size_t>(neighbor) >= center_count) {
                continue;
            }
            const Vec2& center_j = cache.centers[static_cast<size_t>(neighbor)];
            if (!center_j.is_finite()) {
                continue;
            }
            ImVec2 p0 = world_to_screen(center_i, cache.bounds, origin, size);
            ImVec2 p1 = world_to_screen(center_j, cache.bounds, origin, size);

            add_line_with_glow(draw_list, p0, p1, color);
        }
    }
}

void draw_figures(
    const RenderCache& cache,
    const ImVec2& size,
    ImU32 color,
    const std::optional<AABB>& highlight = std::nullopt,
    ImU32 highlight_color = IM_COL32(255, 0, 0, 160),
    const SolutionEval::IntersectionMap* intersections = nullptr
) {
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 origin = ImGui::GetCursorScreenPos();
    ImVec2 max = ImVec2(origin.x + size.x, origin.y + size.y);
    draw_list->AddRect(origin, max, IM_COL32(70, 160, 95, 140));
    draw_list->AddRect(ImVec2(origin.x+1, origin.y+1), ImVec2(max.x-1, max.y-1), IM_COL32(20, 60, 35, 160));


    if (!cache.has_bounds) {
        ImGui::Dummy(size);
        return;
    }

    for (const auto& fig : cache.figures) {
        if (fig.is_nan()) continue;
        for (size_t t = 0; t < fig.size(); ++t) {
            const Triangle& tri = fig[t];
            ImVec2 p0 = world_to_screen(tri.v0, cache.bounds, origin, size);
            ImVec2 p1 = world_to_screen(tri.v1, cache.bounds, origin, size);
            ImVec2 p2 = world_to_screen(tri.v2, cache.bounds, origin, size);
            add_line_with_glow(draw_list, p0, p1, color);
            add_line_with_glow(draw_list, p1, p2, color);
            add_line_with_glow(draw_list, p2, p0, color);
        }
    }

    if (highlight && cache.has_bounds) {
        const AABB& bbox = *highlight;
        Vec2 top_left = Vec2(bbox.min.x, bbox.max.y);
        Vec2 bottom_right = Vec2(bbox.max.x, bbox.min.y);
        ImVec2 screen_tl = world_to_screen(top_left, cache.bounds, origin, size);
        ImVec2 screen_br = world_to_screen(bottom_right, cache.bounds, origin, size);
        draw_list->AddRect(screen_tl, screen_br, highlight_color, 0.0f, 0, 2.0f);
    }
    if (intersections) {
        draw_intersection_segments(cache, origin, size, *intersections);
    }

    ImGui::Dummy(size);
}

bool handle_tree_drag(
    RenderCache& cache,
    TreeParamsSoA& params,
    const ImVec2& origin,
    const ImVec2& size,
    TreeDragState& state,
    std::mutex& drag_mutex,
    TreeDragCommand& drag_command,
    std::atomic<bool>& drag_pending,
    bool is_paused
) {
    if (!is_paused || !cache.has_bounds || size.x <= 0.0f || size.y <= 0.0f) {
        state.active = false;
        return false;
    }

    ImVec2 mouse_pos = ImGui::GetMousePos();
    ImVec2 max = ImVec2(origin.x + size.x, origin.y + size.y);
    bool hovered = ImGui::IsMouseHoveringRect(origin, max, false);

    Vec2 mouse_world = screen_to_world(mouse_pos, cache.bounds, origin, size);
    int best_idx = find_tree_at_point(cache, mouse_world);
    if (!state.active && ImGui::IsMouseClicked(0) && hovered && best_idx >= 0) {
        state.active = true;
        state.index = best_idx;
        TreeParams tree = params.get(static_cast<size_t>(best_idx));
        state.click_offset = tree.pos - mouse_world;
    }
    float wheel_delta = ImGui::GetIO().MouseWheel;
    if (wheel_delta != 0.0f && hovered && best_idx >= 0) {
        float rotation_step = wheel_delta * 0.025f;
        TreeParams tree = params.get(static_cast<size_t>(best_idx));
        tree.angle += rotation_step;
        params.set(static_cast<size_t>(best_idx), tree);
        build_figures_cache(params, cache);
        std::lock_guard<std::mutex> lock(drag_mutex);
        drag_command.index = best_idx;
        drag_command.params = tree;
        drag_pending.store(true, std::memory_order_relaxed);
        state.index = best_idx;
    }

    if (state.active) {
        if (!ImGui::IsMouseDown(0)) {
            state.active = false;
            return true;
        }
        Vec2 mouse_world = screen_to_world(mouse_pos, cache.bounds, origin, size);
        Vec2 target = mouse_world + state.click_offset;
        if (state.index < 0 || static_cast<size_t>(state.index) >= params.size()) {
            state.active = false;
            return false;
        }
        TreeParams tree = params.get(static_cast<size_t>(state.index));
        tree.pos = target;
        params.set(static_cast<size_t>(state.index), tree);
        build_figures_cache(params, cache);
        {
            std::lock_guard<std::mutex> lock(drag_mutex);
            drag_command.index = state.index;
            drag_command.params = tree;
            drag_pending.store(true, std::memory_order_relaxed);
        }
        return true;
    }

    return false;
}

float plot_getter(void* data, int idx) {
    const FloatRing* ring = static_cast<const FloatRing*>(data);
    return ring->get(idx);
}

void optimization_loop(
    SnapshotQueue& queue,
    std::atomic<bool>& shutdown,
    std::atomic<bool>& rebuild_requested,
    std::atomic<bool>& restore_best_requested,
    std::atomic<bool>& paused,
    std::mutex& config_mutex,
    OptConfig& current_config,
    ProblemParams& current_problem_params,
    std::mutex& drag_mutex,
    TreeDragCommand& drag_command,
    std::atomic<bool>& drag_pending,
    std::atomic<float>& sa_resume_temperature,
    const RunConfig& run_cfg
) {
    Problem problem = Problem::create_tree_packing_problem();
    Solution initial = Solution::init_random(run_cfg.num_trees, run_cfg.side, run_cfg.seed);
    SolutionEval eval = problem.eval(initial);
    GlobalState global_state(run_cfg.seed, eval);

    OptConfig local_config;
    ProblemParams local_problem_params;
    {
        std::lock_guard<std::mutex> lock(config_mutex);
        local_config = current_config;
        local_problem_params = current_problem_params;
    }
    float resume_temp = sa_resume_temperature.exchange(std::numeric_limits<float>::quiet_NaN(), std::memory_order_relaxed);
    if (std::isfinite(resume_temp)) {
        update_sa_initial_temp(local_config, resume_temp);
    }
    configure_problem(problem, local_problem_params);
    std::unique_ptr<Optimizer> optimizer = build_optimizer(local_config);
    optimizer->set_problem(&problem);
    std::any opt_state = optimizer->init_state(eval);

    uint64_t last_iter = 0;
    uint64_t last_geometry_push_iter = 0;
    uint64_t full_eval_counter = 0;
    std::optional<SolutionEval> forced_best_eval;
    while (!shutdown.load(std::memory_order_relaxed)) {
        if (rebuild_requested.load(std::memory_order_relaxed)) {
            paused.store(true, std::memory_order_relaxed);
            {
                std::lock_guard<std::mutex> lock(config_mutex);
                local_config = current_config;
                local_problem_params = current_problem_params;
            }
            float resume_temp_rebuild = sa_resume_temperature.exchange(std::numeric_limits<float>::quiet_NaN(), std::memory_order_relaxed);
            if (std::isfinite(resume_temp_rebuild)) {
                update_sa_initial_temp(local_config, resume_temp_rebuild);
            }
            configure_problem(problem, local_problem_params);
            optimizer = build_optimizer(local_config);
            optimizer->set_problem(&problem);
            opt_state = optimizer->init_state(eval);
            rebuild_requested.store(false, std::memory_order_relaxed);
            paused.store(false, std::memory_order_relaxed);
        }

        if (restore_best_requested.load(std::memory_order_relaxed)) {
            bool was_paused = paused.load(std::memory_order_relaxed);
            paused.store(true, std::memory_order_relaxed);
            if (const auto* best = global_state.best_params()) {
                int grid_n = eval.solution.grid().grid_n();
                float grid_size = eval.solution.grid().cell_size();
                int grid_capacity = eval.solution.grid().capacity();
                Solution best_solution = Solution::init(*best, grid_n, grid_size, grid_capacity);
                eval = problem.eval(best_solution);
                global_state.update_stack().clear();
                GeometrySnapshot snap;
                snap.iteration = global_state.iteration();
                snap.current = eval.solution.params();
                snap.current_intersections = eval.intersection_map;
                snap.best = *best;
                snap.best_intersections = eval.intersection_map;
                snap.has_best = true;
                queue.push_geometry(std::move(snap));
            }
            restore_best_requested.store(false, std::memory_order_relaxed);
            if (!was_paused) {
                paused.store(false, std::memory_order_relaxed);
            }
        }

        if (drag_pending.load(std::memory_order_relaxed)) {
            TreeDragCommand command;
            {
                std::lock_guard<std::mutex> lock(drag_mutex);
                command = drag_command;
            }
            drag_pending.store(false, std::memory_order_relaxed);
            if (command.index >= 0 && static_cast<size_t>(command.index) < eval.solution.size()) {
                std::vector<int> indices = {command.index};
                TreeParamsSoA update_params(1);
                update_params.set(0, command.params);
                problem.update_and_eval(eval, indices, update_params);
                global_state.maybe_update_best(problem, eval);
                GeometrySnapshot snap;
                snap.iteration = global_state.iteration();
                snap.current = eval.solution.params();
                snap.current_intersections = eval.intersection_map;
                queue.push_geometry(std::move(snap));
            }
        }

        if (paused.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        RNG rng(global_state.split_rng());
        optimizer->apply(eval, opt_state, global_state, rng);
        ++full_eval_counter;
        bool force_full_eval = false;
        global_state.maybe_update_best(problem, eval);
        global_state.next();
        last_iter = global_state.iteration();

        if (run_cfg.metrics_stride > 0 && (last_iter % run_cfg.metrics_stride == 0)) {
            MetricsSnapshot snap;
            snap.iteration = last_iter;
            snap.objective = eval.objective;
            snap.intersection_violation = eval.intersection_violation;
            snap.bounds_violation = eval.bounds_violation;
            snap.total_violation = eval.total_violation();
            snap.n_missing = eval.n_missing();
            snap.reg = eval.reg();
            snap.best_score = global_state.best_score();
            snap.best_feasible_score = global_state.best_feasible_score();
            if (auto* sa = dynamic_cast<SimulatedAnnealing*>(optimizer.get())) {
                snap.has_sa = true;
                snap.sa_temperature = sa->last_temperature();
                snap.sa_accept_rate = sa->accept_rate();
                snap.sa_last_accept_prob = sa->last_accept_prob();
                snap.sa_accepted = sa->accepted_count();
                snap.sa_rejected = sa->rejected_count();
                snap.sa_iteration = sa->iteration();
                snap.sa_last_accept = sa->last_accept();
            }
            queue.push_metrics(snap);
        }

        bool force_geometry_snapshot = force_full_eval;
        bool pushed_geometry = false;
        if (run_cfg.geometry_stride > 0 && (last_iter % run_cfg.geometry_stride == 0)) {
            GeometrySnapshot snap;
            snap.iteration = last_iter;
            snap.current = eval.solution.params();
            snap.current_intersections = eval.intersection_map;
            if (const auto* best = global_state.best_params()) {
                snap.best = *best;
                snap.has_best = true;
                if (forced_best_eval) {
                    snap.best_intersections = forced_best_eval->intersection_map;
                } else {
                    int grid_n = eval.solution.grid().grid_n();
                    float grid_size = eval.solution.grid().cell_size();
                    int grid_capacity = eval.solution.grid().capacity();
                    Solution best_solution = Solution::init(*best, grid_n, grid_size, grid_capacity);
                    SolutionEval best_eval = problem.eval(best_solution);
                    snap.best_intersections = best_eval.intersection_map;
                }
            }
            queue.push_geometry(std::move(snap));
            last_geometry_push_iter = last_iter;
            pushed_geometry = true;
            force_geometry_snapshot = false;
            forced_best_eval.reset();
        }
        if (!pushed_geometry && (force_geometry_snapshot || eval.intersection_violation > 0.0f)) {
            uint64_t min_interval = run_cfg.metrics_stride > 0 ? static_cast<uint64_t>(run_cfg.metrics_stride) : 1;
            if (last_iter == 0 || (last_iter - last_geometry_push_iter >= min_interval)) {
                GeometrySnapshot snap;
                snap.iteration = last_iter;
                snap.current = eval.solution.params();
                snap.current_intersections = eval.intersection_map;
                // No best data to avoid re-evaluating unless scheduled
                queue.push_geometry(std::move(snap));
                last_geometry_push_iter = last_iter;
                forced_best_eval.reset();
            }
        }
    }
}

static void ApplyCRTStyle() {
    ImGuiStyle& s = ImGui::GetStyle();
    ImVec4* c = s.Colors;

    // Tight, terminal-ish geometry
    s.WindowRounding = 0.0f;
    s.ChildRounding  = 0.0f;
    s.FrameRounding  = 0.0f;
    s.ScrollbarRounding = 0.0f;
    s.GrabRounding   = 0.0f;

    s.WindowBorderSize = 1.0f;
    s.FrameBorderSize  = 1.0f;

    // Slightly denser layout looks more "old UI"
    s.WindowPadding = ImVec2(8, 6);
    s.FramePadding  = ImVec2(6, 4);
    s.ItemSpacing   = ImVec2(8, 6);

    // CRT-ish palette (dark + green phosphor)
    auto col = [](int r,int g,int b,int a){ return ImVec4(r/255.f,g/255.f,b/255.f,a/255.f); };

    c[ImGuiCol_Text]                 = col(160, 255, 180, 230);
    c[ImGuiCol_TextDisabled]         = col(120, 160, 130, 180);

    c[ImGuiCol_WindowBg]             = col(  5,  10,   7, 255);
    c[ImGuiCol_ChildBg]              = col(  5,  10,   7, 220);
    c[ImGuiCol_PopupBg]              = col(  6,  12,   8, 240);

    c[ImGuiCol_Border]               = col( 50, 110,  70, 170);
    c[ImGuiCol_BorderShadow]         = col(  0,   0,   0,   0);

    c[ImGuiCol_FrameBg]              = col( 10,  20,  14, 220);
    c[ImGuiCol_FrameBgHovered]       = col( 18,  40,  26, 230);
    c[ImGuiCol_FrameBgActive]        = col( 24,  56,  34, 240);

    c[ImGuiCol_TitleBg]              = col(  5,  12,   8, 255);
    c[ImGuiCol_TitleBgActive]        = col(  8,  20,  12, 255);
    c[ImGuiCol_TitleBgCollapsed]     = col(  5,  12,   8, 200);

    c[ImGuiCol_Button]               = col( 12,  28,  18, 220);
    c[ImGuiCol_ButtonHovered]        = col( 18,  44,  28, 235);
    c[ImGuiCol_ButtonActive]         = col( 26,  70,  42, 245);

    c[ImGuiCol_Header]               = col( 12,  28,  18, 220);
    c[ImGuiCol_HeaderHovered]        = col( 18,  44,  28, 235);
    c[ImGuiCol_HeaderActive]         = col( 26,  70,  42, 245);

    c[ImGuiCol_Separator]            = col( 50, 110,  70, 150);
    c[ImGuiCol_SeparatorHovered]     = col( 80, 170, 110, 190);
    c[ImGuiCol_SeparatorActive]      = col(100, 220, 140, 220);

    // Plots pick these up (your PlotLines etc.)
    c[ImGuiCol_PlotLines]            = col(120, 255, 160, 220);
    c[ImGuiCol_PlotLinesHovered]     = col(200, 255, 210, 240);
    c[ImGuiCol_PlotHistogram]        = col(120, 255, 160, 200);
    c[ImGuiCol_PlotHistogramHovered] = col(200, 255, 210, 220);
}

static void DrawCRTOverlay(const ImVec2& pos, const ImVec2& size, float t) {
    ImDrawList* dl = ImGui::GetForegroundDrawList();

    // Very subtle flicker (keep tiny)
    float flicker = 0.06f + 0.02f * std::sin(t * 13.0f) + 0.01f * std::sin(t * 37.0f);

    // Slight greenish screen tint
    dl->AddRectFilled(pos, ImVec2(pos.x + size.x, pos.y + size.y),
                      IM_COL32(20, 60, 30, 12));

    // Scanlines (every 2px)
    for (int y = 0; y < (int)size.y; y += 2) {
        int a = (int)(255.0f * (0.05f + flicker)); // low alpha
        dl->AddRectFilled(
            ImVec2(pos.x, pos.y + (float)y),
            ImVec2(pos.x + size.x, pos.y + (float)y + 1.0f),
            IM_COL32(0, 0, 0, a)
        );
    }

    // Simple vignette using 4 edge fades (cheap + convincing)
    const float edge = 80.0f; // thickness in px
    ImU32 dark = IM_COL32(0, 0, 0, 80);

    // top
    dl->AddRectFilledMultiColor(pos, ImVec2(pos.x + size.x, pos.y + edge),
        dark, dark, IM_COL32(0,0,0,0), IM_COL32(0,0,0,0));
    // bottom
    dl->AddRectFilledMultiColor(ImVec2(pos.x, pos.y + size.y - edge), ImVec2(pos.x + size.x, pos.y + size.y),
        IM_COL32(0,0,0,0), IM_COL32(0,0,0,0), dark, dark);
    // left
    dl->AddRectFilledMultiColor(pos, ImVec2(pos.x + edge, pos.y + size.y),
        dark, IM_COL32(0,0,0,0), IM_COL32(0,0,0,0), dark);
    // right
    dl->AddRectFilledMultiColor(ImVec2(pos.x + size.x - edge, pos.y), ImVec2(pos.x + size.x, pos.y + size.y),
        IM_COL32(0,0,0,0), dark, dark, IM_COL32(0,0,0,0));
}

}  // namespace

int main() {
    if (!glfwInit()) {
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    GLFWwindow* window = glfwCreateWindow(1280, 720, "Optimizer Dashboard", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->AddFontFromFileTTF("cpp/apps/DejaVuSansMono.ttf", 18.0f);
    io.FontGlobalScale = 1.8f; // then tune this
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.FontGlobalScale = 2.0f;
    ImGui::StyleColorsDark();
    ApplyCRTStyle();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL2_Init();

    SnapshotQueue queue;
    std::atomic<bool> shutdown{false};
    std::atomic<bool> rebuild_requested{false};
    std::atomic<bool> restore_best_requested{false};
    std::atomic<bool> paused{false};
    std::mutex config_mutex;

    RunConfig run_cfg;
    OptConfig current_config = default_optimizer_config();
    OptConfig pending_config = current_config;
    int optimizer_preset_idx = 0;
    ProblemParams current_problem_params;
    ProblemParams pending_problem_params = current_problem_params;

    std::mutex drag_mutex;
    TreeDragCommand drag_command;
    std::atomic<bool> drag_pending{false};
    std::atomic<float> sa_resume_temperature{std::numeric_limits<float>::quiet_NaN()};
    TreeDragState drag_state;

    std::thread worker(optimization_loop,
        std::ref(queue),
        std::ref(shutdown),
        std::ref(rebuild_requested),
        std::ref(restore_best_requested),
        std::ref(paused),
        std::ref(config_mutex),
        std::ref(current_config),
        std::ref(current_problem_params),
        std::ref(drag_mutex),
        std::ref(drag_command),
        std::ref(drag_pending),
        std::ref(sa_resume_temperature),
        std::cref(run_cfg)
    );

    auto apply_pending_config = [&](float resume_temp) {
        sa_resume_temperature.store(resume_temp, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(config_mutex);
        current_config = pending_config;
        current_problem_params = pending_problem_params;
        rebuild_requested.store(true, std::memory_order_relaxed);
    };

    const int history_iters = 10'000'000;
    const int history_points = std::max(1024, history_iters / std::max(1, run_cfg.metrics_stride));
    PlotSeries objective_series(history_points, "Objective");
    PlotSeries violation_series(history_points, "Total violation");
    PlotSeries intersection_series(history_points, "Intersection violation");
    PlotSeries bounds_series(history_points, "Bounds violation");
    PlotSeries best_series(history_points, "Best score");
    PlotSeries log_violation_series(history_points, "Total violation (log)");
    PlotSeries log_intersection_series(history_points, "Intersection violation (log)");

    ViewCache view_cache;
    MetricsSnapshot latest_metrics;
    bool has_metrics = false;
    std::string save_best_status;
    set_optimizer_preset_apply_callback([&]() {
        float temp = std::numeric_limits<float>::quiet_NaN();
        if (has_metrics && latest_metrics.has_sa) {
            temp = latest_metrics.sa_temperature;
        }
        apply_pending_config(temp);
        optimizer_preset_idx = -1;
    });

    const auto frame_interval = std::chrono::milliseconds(33);
    const auto metrics_interval = std::chrono::milliseconds(167);
    auto last_frame_time = std::chrono::steady_clock::now() - frame_interval;
    auto last_metrics_update = std::chrono::steady_clock::now() - metrics_interval;
    auto last_rate_time = std::chrono::steady_clock::now() - metrics_interval;
    uint64_t last_rate_iter = 0;
    float iter_rate = 0.0f;
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        auto now = std::chrono::steady_clock::now();
        if (now - last_frame_time < frame_interval) {
            std::this_thread::sleep_for(frame_interval - (now - last_frame_time));
            continue;
        }
        last_frame_time = now;

        ProblemParams display_problem_params;
        {
            std::lock_guard<std::mutex> lock(config_mutex);
            display_problem_params = current_problem_params;
        }
        float plot_tol = std::max(display_problem_params.intersection_tolerance, 1e-12f);
        float log_tol = std::log10(plot_tol);
        bool is_paused = paused.load(std::memory_order_relaxed);
        if (now - last_metrics_update >= metrics_interval) {
            last_metrics_update = now;
            auto metrics = queue.pop_all_metrics();
            for (const auto& snap : metrics) {
                objective_series.values.push(snap.objective);
                violation_series.values.push(snap.total_violation);
                intersection_series.values.push(snap.intersection_violation);
                bounds_series.values.push(snap.bounds_violation);
                best_series.values.push(snap.best_feasible_score);
                float total_clamped = std::max(snap.total_violation, plot_tol);
                float intersection_clamped = std::max(snap.intersection_violation, plot_tol);
                log_violation_series.values.push(std::log10(total_clamped));
                log_intersection_series.values.push(std::log10(intersection_clamped));
                latest_metrics = snap;
                has_metrics = true;
            }
            if (is_paused) {
                iter_rate = 0.0f;
                last_rate_time = now;
                last_rate_iter = latest_metrics.iteration;
            } else if (!metrics.empty()) {
                uint64_t latest_iter = metrics.back().iteration;
                float duration = std::chrono::duration<float>(now - last_rate_time).count();
                if (duration <= 0.0f) {
                    duration = std::chrono::duration<float>(metrics_interval).count();
                }
                iter_rate = static_cast<float>(latest_iter - last_rate_iter) / duration;
                last_rate_iter = latest_iter;
                last_rate_time = now;
            } else {
                iter_rate = 0.0f;
                last_rate_time = now;
            }
        }

        auto geometry = queue.pop_latest_geometry();
        if (geometry) {
            view_cache.current.iteration = geometry->iteration;
            build_figures_cache(geometry->current, view_cache.current);
            view_cache.current_intersections = geometry->current_intersections;
            view_cache.current_params = geometry->current;
            if (geometry->has_best) {
                view_cache.has_best = true;
                view_cache.best.iteration = geometry->iteration;
                build_figures_cache(geometry->best, view_cache.best);
                if (geometry->best_intersections) {
                    view_cache.best_intersections = *geometry->best_intersections;
                }
                view_cache.best_params = geometry->best;
            }
        }
        // reuse display_problem_params defined above for bbox calculations
        std::optional<AABB> objective_bbox_current;
        std::optional<AABB> objective_bbox_best;
        if (std::isfinite(display_problem_params.objective_ceiling) && display_problem_params.objective_ceiling > 0.0f) {
            float side = std::sqrt(display_problem_params.objective_ceiling * run_cfg.num_trees);
            float half = side * 0.5f;
            if (view_cache.current.has_bounds) {
                Vec2 center = view_cache.current.bounds.center();
                objective_bbox_current = AABB{Vec2{center.x - half, center.y - half}, Vec2{center.x + half, center.y + half}};
            }
            if (view_cache.has_best && view_cache.best.has_bounds) {
                Vec2 center = view_cache.best.bounds.center();
                objective_bbox_best = AABB{Vec2{center.x - half, center.y - half}, Vec2{center.x + half, center.y + half}};
            }
        }

        ImGui_ImplOpenGL2_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(io.DisplaySize);
        ImGui::Begin("Optimizer Dashboard", nullptr,
            ImGuiWindowFlags_NoTitleBar |
            ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoCollapse);
        DrawCRTOverlay(ImGui::GetWindowPos(), ImGui::GetWindowSize(), (float)glfwGetTime());
        ImGui::Text("Status: %s", is_paused ? "Applying changes..." : "Running");
        float rate_value = is_paused ? 0.0f : iter_rate;
        char rate_buf[64];
        std::snprintf(rate_buf, sizeof(rate_buf), "Iters/s: %.2f", rate_value);
        ImGui::SameLine();
        ImGuiStyle& style = ImGui::GetStyle();
        float text_width = ImGui::CalcTextSize(rate_buf).x;
        float window_right = ImGui::GetWindowContentRegionMax().x;
        float cursor_x = ImGui::GetCursorPosX();
        float target_x = window_right - text_width - style.ItemSpacing.x;
        if (target_x < cursor_x) {
            target_x = cursor_x;
        }
        ImGui::SetCursorPosX(target_x);
        ImGui::TextUnformatted(rate_buf);
        if (has_metrics) {
            ImGui::Text("Iter: %llu | Obj: %.6f | Viol: %.6f | Best: %.6f",
                static_cast<unsigned long long>(latest_metrics.iteration),
                latest_metrics.objective,
                latest_metrics.total_violation,
                latest_metrics.best_feasible_score
            );
        }

        auto calc_min_max = [](const FloatRing& ring, float& out_min, float& out_max) {
            if (ring.count == 0) {
                out_min = 0.0f;
                out_max = 1.0f;
                return;
            }
            out_min = ring.get(0);
            out_max = ring.get(0);
            for (int i = 1; i < ring.count; ++i) {
                float v = ring.get(i);
                out_min = std::min(out_min, v);
                out_max = std::max(out_max, v);
            }
            if (out_min == out_max) {
                out_max = out_min + 1.0f;
            }
        };

        ImVec2 plot_size = ImVec2(-1.0f, 143.0f);
        float min_v = 0.0f;
        float max_v = 1.0f;

        auto plot_metric_row = [&](const char* name, const FloatRing& ring, float min_clip = -std::numeric_limits<float>::infinity(), bool force_min = false) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::AlignTextToFramePadding();
            ImGui::TextUnformatted(name);
            ImGui::TableSetColumnIndex(1);
            calc_min_max(ring, min_v, max_v);
            if (force_min) {
                min_v = min_clip;
            } else if (min_clip > min_v) {
                min_v = min_clip;
            }
            ImGui::PlotLines("##metric", plot_getter, const_cast<FloatRing*>(&ring), ring.count, 0, nullptr, min_v, max_v, plot_size);
        };

        ImGui::Dummy(ImVec2(0.0f, 8.0f));
        ImGui::Indent(12.0f);
        if (ImGui::BeginTable("MetricsTable", 2, ImGuiTableFlags_SizingFixedFit)) {
            ImGui::TableSetupColumn("Metric", ImGuiTableColumnFlags_WidthFixed, 160.0f);
            ImGui::TableSetupColumn("Plot", ImGuiTableColumnFlags_WidthStretch);
            plot_metric_row("Objective", objective_series.values);
            plot_metric_row("Total violation (log)", log_violation_series.values, log_tol, true);
            plot_metric_row("Intersection violation (log)", log_intersection_series.values, log_tol, true);
            plot_metric_row("Bounds violation", bounds_series.values);
            plot_metric_row("Best score", best_series.values);
            ImGui::EndTable();
        }
        ImGui::Unindent(12.0f);
        ImGui::Dummy(ImVec2(0.0f, 4.0f));
        ImGui::BeginGroup();
        if (ImGui::Button(is_paused ? "Play" : "Pause")) {
            paused.store(!is_paused, std::memory_order_relaxed);
        }
        ImGui::SameLine();
        if (ImGui::Button("Restore Best")) {
            restore_best_requested.store(true, std::memory_order_relaxed);
        }
        ImGui::SameLine();
        if (ImGui::Button("Save best solution")) {
            if (!view_cache.has_best || static_cast<int>(view_cache.best_params.size()) != run_cfg.num_trees) {
                save_best_status = "No complete best solution yet";
            } else {
                save_best_solution_to_csv(view_cache.best_params, run_cfg.num_trees, save_best_status);
            }
        }
        ImGui::SameLine();
        ImGui::TextDisabled("Pause applying configs");
        ImGui::EndGroup();
        if (!save_best_status.empty()) {
            ImGui::TextWrapped("%s", save_best_status.c_str());
        }
        ImGui::Dummy(ImVec2(0.0f, 4.0f));
        ImGui::Separator();

        ImVec2 avail = ImGui::GetContentRegionAvail();
        float right_width = 720.0f * 1.3f;
        if (avail.x < right_width + 50.0f) {
            right_width = 520.0f * 1.3f;
        }
        ImVec2 left_size = ImVec2(avail.x - right_width - 8.0f, avail.y);
        ImVec2 right_size = ImVec2(right_width, avail.y);

        ImGui::BeginChild("TreesPane", left_size, true);
        ImVec2 trees_avail = ImGui::GetContentRegionAvail();
        ImVec2 half = ImVec2(trees_avail.x * 0.5f - 6.0f, trees_avail.y);

        ImGui::BeginChild("CurrentTree", half, true);
        ImGui::Text("Current (iter %llu)", static_cast<unsigned long long>(view_cache.current.iteration));
        ImVec2 canvas = ImGui::GetContentRegionAvail();
        draw_figures(
            view_cache.current,
            canvas,
            IM_COL32(0, 200, 255, 255),
            objective_bbox_current,
            IM_COL32(255, 0, 0, 160),
            &view_cache.current_intersections
        );
        ImVec2 current_origin = ImGui::GetItemRectMin();
        ImVec2 current_size = ImGui::GetItemRectSize();
        handle_tree_drag(view_cache.current, view_cache.current_params, current_origin, current_size, drag_state, drag_mutex, drag_command, drag_pending, is_paused);
        ImGui::EndChild();

        ImGui::SameLine();

        ImGui::BeginChild("BestTree", half, true);
        if (view_cache.has_best) {
            ImGui::Text("Best (iter %llu)", static_cast<unsigned long long>(view_cache.best.iteration));
            ImVec2 canvas_best = ImGui::GetContentRegionAvail();
            const SolutionEval::IntersectionMap* best_map = view_cache.has_best ? &view_cache.best_intersections : nullptr;
            draw_figures(
                view_cache.best,
                canvas_best,
                IM_COL32(0, 255, 140, 255),
                objective_bbox_best,
                IM_COL32(255, 0, 0, 160),
                best_map
            );
        } else {
            ImGui::TextUnformatted("Best (n/a)");
            ImVec2 canvas_best = ImGui::GetContentRegionAvail();
            ImGui::Dummy(canvas_best);
        }
        ImGui::EndChild();
        ImGui::EndChild();

        ImGui::SameLine();

        ImGui::BeginChild("OptimizerPane", right_size, true);
        ImGui::TextUnformatted("Optimizer");
        ImGui::Separator();
        if (ImGui::CollapsingHeader("Problem parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::InputFloat("Objective ceiling", &pending_problem_params.objective_ceiling, 0.0f, 0.0f, "%.6f");
            apply_mouse_wheel_float(pending_problem_params.objective_ceiling, 0.5f, 0.0f);
            ImGui::SameLine();
            if (ImGui::Button("Disable ceiling##ProblemParam")) {
                pending_problem_params.objective_ceiling = std::numeric_limits<float>::infinity();
            }
            if (std::isfinite(pending_problem_params.objective_ceiling) && pending_problem_params.objective_ceiling > 0.0f) {
                float derived_side = std::sqrt(pending_problem_params.objective_ceiling * run_cfg.num_trees);
                float half = derived_side * 0.5f;
                ImGui::Text("Derived bbox: side %.6f → [%.6f, %.6f]²", derived_side, -half, half);
            } else {
                ImGui::Text("Objective ceiling disabled; using base side %.6f", run_cfg.side);
            }
            ImGui::Spacing();
            ImGui::InputFloat("Intersection tolerance", &pending_problem_params.intersection_tolerance, 0.0f, 0.0f, "%.6f");
            apply_mouse_wheel_float(pending_problem_params.intersection_tolerance, 1e-6f, 0.0f);
            const char* penalty_names[] = {
                "Linear", "Quadratic", "Tolerant", "LogBarrier", "Exponential"
            };
            int penalty_idx = static_cast<int>(pending_problem_params.constraint_penalty);
            if (ImGui::Combo("Constraint mode", &penalty_idx, penalty_names, IM_ARRAYSIZE(penalty_names))) {
                pending_problem_params.constraint_penalty = static_cast<ConstraintPenaltyType>(penalty_idx);
            }
            if (pending_problem_params.constraint_penalty == ConstraintPenaltyType::Tolerant ||
                pending_problem_params.constraint_penalty == ConstraintPenaltyType::LogBarrier) {
                ImGui::Text("Tolerance is active for this penalty mode.");
            } else {
                ImGui::Text("Tolerance parameter is ignored outside Tolerant/LogBarrier modes.");
            }
            ImGui::Text("Objective formula: side² / num_trees (side %.6f, trees %d)", run_cfg.side, run_cfg.num_trees);
        }
        ImGui::Spacing();
        ImGui::TextUnformatted("Optimizer presets");
        const auto& presets = optimizer_presets();
        const char* preset_label = "Custom";
        if (optimizer_preset_idx >= 0 && optimizer_preset_idx < static_cast<int>(presets.size())) {
            preset_label = presets[optimizer_preset_idx].name;
        }
        ImGui::BeginDisabled(!is_paused);
        if (ImGui::BeginCombo("##OptimizerPreset", preset_label)) {
            for (int i = 0; i < static_cast<int>(presets.size()); ++i) {
                bool selected = optimizer_preset_idx == i;
                if (ImGui::Selectable(presets[i].name, selected)) {
                    pending_config = presets[i].config;
                    optimizer_preset_idx = i;
                    apply_pending_config(std::numeric_limits<float>::quiet_NaN());
                }
                if (selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }
        ImGui::EndDisabled();
        if (!is_paused) {
            ImGui::TextDisabled("(pause to swap optimizer presets)");
        }
        ImGui::Separator();
        bool reset_sa_temp_request = false;
        bool config_tree_changed = false;
        if (ImGui::TreeNode("Config")) {
            config_tree_changed = edit_optimizer_config(pending_config, has_metrics ? &latest_metrics : nullptr, &reset_sa_temp_request, false, is_paused);
            ImGui::TreePop();
        }
        if (config_tree_changed) {
            optimizer_preset_idx = -1;
        }
        if (reset_sa_temp_request) {
            std::lock_guard<std::mutex> reset_lock(config_mutex);
            update_sa_temperature_params(pending_config, kDefaultSAInitialTemp, kDefaultSAMinTemp);
            update_sa_temperature_params(current_config, kDefaultSAInitialTemp, kDefaultSAMinTemp);
            sa_resume_temperature.store(kDefaultSAInitialTemp, std::memory_order_relaxed);
            rebuild_requested.store(true, std::memory_order_relaxed);
        }
        if (ImGui::Button("Commit Changes")) {
            float temp = std::numeric_limits<float>::quiet_NaN();
            if (has_metrics && latest_metrics.has_sa) {
                temp = latest_metrics.sa_temperature;
            }
            apply_pending_config(temp);
            optimizer_preset_idx = -1;
        }
        ImGui::EndChild();

        ImGui::End();

        ImGui::Render();
        int display_w = 0;
        int display_h = 0;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.08f, 0.08f, 0.08f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    clear_optimizer_preset_apply_callback();

    shutdown.store(true, std::memory_order_relaxed);
    if (worker.joinable()) {
        worker.join();
    }

    ImGui_ImplOpenGL2_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
