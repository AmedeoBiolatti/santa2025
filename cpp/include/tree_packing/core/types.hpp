#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace tree_packing {

// Forward declarations
struct Vec2;
struct Mat2;
struct Triangle;
struct Figure;
struct TreeParams;

// 2D vector with SIMD-friendly alignment
struct alignas(16) Vec2 {
    float x{0.0f};
    float y{0.0f};

    constexpr Vec2() = default;
    constexpr Vec2(float x_, float y_) : x(x_), y(y_) {}

    [[nodiscard]] constexpr Vec2 operator+(const Vec2& other) const {
        return Vec2{x + other.x, y + other.y};
    }

    [[nodiscard]] constexpr Vec2 operator-(const Vec2& other) const {
        return Vec2{x - other.x, y - other.y};
    }

    [[nodiscard]] constexpr Vec2 operator*(float scalar) const {
        return Vec2{x * scalar, y * scalar};
    }

    [[nodiscard]] constexpr Vec2 operator/(float scalar) const {
        return Vec2{x / scalar, y / scalar};
    }

    constexpr Vec2& operator+=(const Vec2& other) {
        x += other.x;
        y += other.y;
        return *this;
    }

    constexpr Vec2& operator-=(const Vec2& other) {
        x -= other.x;
        y -= other.y;
        return *this;
    }

    [[nodiscard]] constexpr float dot(const Vec2& other) const {
        return x * other.x + y * other.y;
    }

    [[nodiscard]] constexpr float cross(const Vec2& other) const {
        return x * other.y - y * other.x;
    }

    [[nodiscard]] float length() const {
        return std::sqrt(x * x + y * y);
    }

    [[nodiscard]] float length_squared() const {
        return x * x + y * y;
    }

    [[nodiscard]] Vec2 normalized() const {
        float len = length();
        if (len < 1e-12f) return Vec2{0.0f, 0.0f};
        return Vec2{x / len, y / len};
    }

    [[nodiscard]] constexpr Vec2 perpendicular() const {
        return Vec2{-y, x};
    }

    [[nodiscard]] bool is_nan() const {
        return std::isnan(x) || std::isnan(y);
    }

    [[nodiscard]] bool is_finite() const {
        return std::isfinite(x) && std::isfinite(y);
    }

    static constexpr Vec2 nan() {
        return Vec2{std::numeric_limits<float>::quiet_NaN(),
                   std::numeric_limits<float>::quiet_NaN()};
    }
};

[[nodiscard]] inline constexpr Vec2 operator*(float scalar, const Vec2& v) {
    return v * scalar;
}

// 2x2 rotation matrix
struct alignas(16) Mat2 {
    float m00{1.0f}, m01{0.0f};
    float m10{0.0f}, m11{1.0f};

    constexpr Mat2() = default;
    constexpr Mat2(float m00_, float m01_, float m10_, float m11_)
        : m00(m00_), m01(m01_), m10(m10_), m11(m11_) {}

    [[nodiscard]] static Mat2 rotation(float angle) {
        float c = std::cos(angle);
        float s = std::sin(angle);
        return Mat2{c, -s, s, c};
    }

    [[nodiscard]] constexpr Vec2 operator*(const Vec2& v) const {
        return Vec2{m00 * v.x + m01 * v.y, m10 * v.x + m11 * v.y};
    }
};

// Triangle (3 vertices)
struct alignas(32) Triangle {
    Vec2 v0, v1, v2;

    constexpr Triangle() = default;
    constexpr Triangle(const Vec2& v0_, const Vec2& v1_, const Vec2& v2_)
        : v0(v0_), v1(v1_), v2(v2_) {}

    [[nodiscard]] constexpr const Vec2& operator[](size_t i) const {
        switch (i) {
            case 0: return v0;
            case 1: return v1;
            default: return v2;
        }
    }

    [[nodiscard]] constexpr Vec2& operator[](size_t i) {
        switch (i) {
            case 0: return v0;
            case 1: return v1;
            default: return v2;
        }
    }

    [[nodiscard]] Vec2 centroid() const {
        return Vec2{(v0.x + v1.x + v2.x) / 3.0f, (v0.y + v1.y + v2.y) / 3.0f};
    }

    [[nodiscard]] bool is_nan() const {
        return v0.is_nan() || v1.is_nan() || v2.is_nan();
    }
};

// Tree parameters (position + angle)
struct TreeParams {
    Vec2 pos{0.0f, 0.0f};
    float angle{0.0f};

    constexpr TreeParams() = default;
    constexpr TreeParams(const Vec2& pos_, float angle_)
        : pos(pos_), angle(angle_) {}
    constexpr TreeParams(float x, float y, float angle_)
        : pos(x, y), angle(angle_) {}

    [[nodiscard]] bool is_nan() const {
        return pos.is_nan() || std::isnan(angle);
    }

    void set_nan() {
        pos = Vec2::nan();
        angle = std::numeric_limits<float>::quiet_NaN();
    }
};

// Figure is a collection of triangles representing a tree
constexpr size_t TREE_NUM_TRIANGLES = 5;

struct Figure {
    std::array<Triangle, TREE_NUM_TRIANGLES> triangles;

    constexpr Figure() = default;

    [[nodiscard]] constexpr const Triangle& operator[](size_t i) const {
        return triangles[i];
    }

    [[nodiscard]] constexpr Triangle& operator[](size_t i) {
        return triangles[i];
    }

    [[nodiscard]] size_t size() const { return TREE_NUM_TRIANGLES; }

    [[nodiscard]] bool is_nan() const {
        for (const auto& tri : triangles) {
            if (tri.is_nan()) return true;
        }
        return false;
    }
};

// SoA (Structure of Arrays) for tree parameters - better for SIMD
struct TreeParamsSoA {
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> angle;

    TreeParamsSoA() = default;
    explicit TreeParamsSoA(size_t n) : x(n, 0.0f), y(n, 0.0f), angle(n, 0.0f) {}

    [[nodiscard]] size_t size() const { return x.size(); }

    void resize(size_t n) {
        x.resize(n, 0.0f);
        y.resize(n, 0.0f);
        angle.resize(n, 0.0f);
    }

    [[nodiscard]] TreeParams get(size_t i) const {
        return TreeParams{x[i], y[i], angle[i]};
    }

    void set(size_t i, const TreeParams& params) {
        x[i] = params.pos.x;
        y[i] = params.pos.y;
        angle[i] = params.angle;
    }

    void set_nan(size_t i) {
        x[i] = std::numeric_limits<float>::quiet_NaN();
        y[i] = std::numeric_limits<float>::quiet_NaN();
        angle[i] = std::numeric_limits<float>::quiet_NaN();
    }

    [[nodiscard]] bool is_nan(size_t i) const {
        return std::isnan(x[i]) || std::isnan(y[i]) || std::isnan(angle[i]);
    }

    [[nodiscard]] int count_nan() const {
        int count = 0;
        for (size_t i = 0; i < size(); ++i) {
            if (is_nan(i)) ++count;
        }
        return count;
    }
};

// Axis-aligned bounding box
struct AABB {
    Vec2 min{std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
    Vec2 max{std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest()};

    constexpr AABB() = default;
    constexpr AABB(const Vec2& min_, const Vec2& max_) : min(min_), max(max_) {}

    void expand(const Vec2& point) {
        min.x = std::min(min.x, point.x);
        min.y = std::min(min.y, point.y);
        max.x = std::max(max.x, point.x);
        max.y = std::max(max.y, point.y);
    }

    void expand(const Triangle& tri) {
        expand(tri.v0);
        expand(tri.v1);
        expand(tri.v2);
    }

    void expand(const Figure& fig) {
        for (const auto& tri : fig.triangles) {
            expand(tri);
        }
    }

    [[nodiscard]] Vec2 center() const {
        return Vec2{(min.x + max.x) * 0.5f, (min.y + max.y) * 0.5f};
    }

    [[nodiscard]] Vec2 size() const {
        return Vec2{max.x - min.x, max.y - min.y};
    }

    [[nodiscard]] bool contains(const Vec2& point) const {
        return point.x >= min.x && point.x <= max.x &&
               point.y >= min.y && point.y <= max.y;
    }

    [[nodiscard]] bool intersects(const AABB& other) const {
        return min.x <= other.max.x && max.x >= other.min.x &&
               min.y <= other.max.y && max.y >= other.min.y;
    }
};

// Constraint evaluation result
struct ConstraintEval {
    float violation{0.0f};
};

// Common constants
constexpr float PI = 3.14159265358979323846f;
constexpr float TWO_PI = 2.0f * PI;
constexpr float EPSILON = 1e-12f;

}  // namespace tree_packing
