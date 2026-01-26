#pragma once

#include "types.hpp"
#include <array>

namespace tree_packing {

// Tree shape constants (from Python TREE array)
constexpr std::array<Triangle, TREE_NUM_TRIANGLES> TREE_SHAPE = {{
    Triangle{Vec2{0.0f, 0.8f}, Vec2{-0.125f, 0.5f}, Vec2{0.125f, 0.5f}},
    Triangle{Vec2{0.2f, 0.25f}, Vec2{0.0f, 27.0f/44.0f}, Vec2{-0.2f, 0.25f}},
    Triangle{Vec2{0.35f, 0.0f}, Vec2{0.0f, 0.35f}, Vec2{-0.35f, 0.0f}},
    Triangle{Vec2{0.075f, 0.0f}, Vec2{-0.075f, -0.2f}, Vec2{0.075f, -0.2f}},
    Triangle{Vec2{0.075f, 0.0f}, Vec2{-0.075f, 0.0f}, Vec2{-0.075f, -0.2f}}
}};

// Precomputed edge normals for TREE_SHAPE (from santa.tree_packing.tree.TREE_NORMALS)
constexpr std::array<std::array<Vec2, 3>, TREE_NUM_TRIANGLES> TREE_NORMALS = {{
    {Vec2{0.9230769230769231f, -0.3846153846153846f},
     Vec2{-0.0f, 1.0f},
     Vec2{-0.9230769230769231f, -0.3846153846153846f}},
    {Vec2{-0.8762160353251135f, -0.4819187497721559f},
     Vec2{0.8762160353251135f, -0.4819187497721559f},
     Vec2{-0.0f, 1.0f}},
    {Vec2{-0.7071067811865476f, -0.7071067811865476f},
     Vec2{0.7071067811865476f, -0.7071067811865476f},
     Vec2{-0.0f, 1.0f}},
    {Vec2{0.8f, -0.6f},
     Vec2{-0.0f, 1.0f},
     Vec2{-1.0f, 0.0f}},
    {Vec2{-0.0f, -1.0f},
     Vec2{1.0f, 0.0f},
     Vec2{-0.8f, 0.6f}}
}};

// Triangles centers and radius
constexpr std::array<std::array<Vec2, 3>, TREE_NUM_TRIANGLES> TRIANGLE_CENTERS = {
  Vec2{ 0.0f,  0.62395835f },   // t1
  Vec2{ 0.0f,  0.37681818f },   // t2
  Vec2{ 0.0f,  0.0f        },   // t3
  Vec2{ 0.0f, -0.10000000f },   // t4
  Vec2{ 0.0f, -0.10000000f },   // t5
};
constexpr std::array<float, TREE_NUM_TRIANGLES> TRIANGLE_RADII = {
  0.17604166f,  // t1
  0.23681818f,  // t2
  0.34999999f,  // t3  (float32 representation of 0.35)
  0.12500000f,  // t4
  0.12500000f,  // t5
};

// Disjoint tree shape (for area computation)
constexpr size_t TREE_DISJOINT_NUM_TRIANGLES = 7;
constexpr std::array<Triangle, TREE_DISJOINT_NUM_TRIANGLES> TREE_DISJOINT_SHAPE = {{
    Triangle{Vec2{0.0f, 0.8f}, Vec2{-0.125f, 0.5f}, Vec2{0.125f, 0.5f}},
    Triangle{Vec2{0.2f, 0.25f}, Vec2{-0.0625f, 0.5f}, Vec2{-0.2f, 0.25f}},
    Triangle{Vec2{0.2f, 0.25f}, Vec2{0.0625f, 0.5f}, Vec2{-0.0625f, 0.5f}},
    Triangle{Vec2{0.35f, 0.0f}, Vec2{-0.1f, 0.25f}, Vec2{-0.35f, 0.0f}},
    Triangle{Vec2{0.35f, 0.0f}, Vec2{0.1f, 0.25f}, Vec2{-0.1f, 0.25f}},
    Triangle{Vec2{-0.075f, -0.2f}, Vec2{0.075f, -0.2f}, Vec2{-0.075f, 0.0f}},
    Triangle{Vec2{0.075f, 0.0f}, Vec2{-0.075f, 0.0f}, Vec2{0.075f, -0.2f}}
}};

// Tree center constants (for bounding sphere approximation)
constexpr float CENTER_Y = 0.2972f;
constexpr float CENTER_R = 0.5029f;
constexpr float THR = 2.0f * CENTER_R;
constexpr float THR2 = THR * THR;

// Rotate a point by angle
[[nodiscard]] inline Vec2 rotate_point(const Vec2& point, float angle) {
    float c = std::cos(angle);
    float s = std::sin(angle);
    return Vec2{c * point.x - s * point.y, s * point.x + c * point.y};
}

// Transform a point: offset + rotate
[[nodiscard]] inline Vec2 transform_point(const Vec2& offset, float angle, const Vec2& point) {
    return offset + rotate_point(point, angle);
}

// Transform a triangle
[[nodiscard]] inline Triangle transform_triangle(const Vec2& offset, float angle, const Triangle& tri) {
    return Triangle{
        transform_point(offset, angle, tri.v0),
        transform_point(offset, angle, tri.v1),
        transform_point(offset, angle, tri.v2)
    };
}

// Transform params to figure (tree shape)
[[nodiscard]] inline Figure params_to_figure(const TreeParams& params) {
    Figure fig;
    for (size_t i = 0; i < TREE_NUM_TRIANGLES; ++i) {
        fig.triangles[i] = transform_triangle(params.pos, params.angle, TREE_SHAPE[i]);
    }
    return fig;
}

// Compute AABB for a figure
[[nodiscard]] inline AABB compute_aabb(const Figure& fig) {
    AABB aabb;
    aabb.expand(fig);
    return aabb;
}

// Compute max abs coordinate from an AABB
[[nodiscard]] inline float compute_aabb_max_abs(const AABB& aabb) {
    float max_abs = std::abs(aabb.min.x);
    max_abs = std::max(max_abs, std::abs(aabb.min.y));
    max_abs = std::max(max_abs, std::abs(aabb.max.x));
    max_abs = std::max(max_abs, std::abs(aabb.max.y));
    return max_abs;
}

// Get tree center from params (using small-radius proxy)
[[nodiscard]] inline Vec2 get_tree_center(const TreeParams& params) {
    float c = std::cos(params.angle);
    float s = std::sin(params.angle);
    return Vec2{
        params.pos.x - s * CENTER_Y,
        params.pos.y + c * CENTER_Y
    };
}

// Get tree center from SoA params
[[nodiscard]] inline Vec2 get_tree_center(float x, float y, float angle) {
    float c = std::cos(angle);
    float s = std::sin(angle);
    return Vec2{x - s * CENTER_Y, y + c * CENTER_Y};
}

// Get rotated edge normals for a tree given angle
[[nodiscard]] inline void get_tree_normals(
    float angle,
    std::array<std::array<Vec2, 3>, TREE_NUM_TRIANGLES>& out
) {
    for (size_t i = 0; i < TREE_NUM_TRIANGLES; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            out[i][j] = rotate_point(TREE_NORMALS[i][j], angle);
        }
    }
}

// Batch transform params to figures
void params_to_figures(const TreeParamsSoA& params, std::vector<Figure>& figures);

// Batch get tree centers
void get_tree_centers(const TreeParamsSoA& params, std::vector<Vec2>& centers);

}  // namespace tree_packing
