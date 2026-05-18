#include "PackingPass.hpp"
#include <Eigen/Core>
#include <cassert>
#include <cmath>
#include <iostream>

// Build a simple box mesh (Nv×3 float matrix, face indices unused by packing).
// Box spans [x0,x1] × [y0,y1] × [z0,z1].
static Eigen::MatrixXf makeBox(float x0, float x1,
                                float y0, float y1,
                                float z0, float z1)
{
    Eigen::MatrixXf V(8, 3);
    V << x0,y0,z0,  x1,y0,z0,  x1,y1,z0,  x0,y1,z0,
         x0,y0,z1,  x1,y0,z1,  x1,y1,z1,  x0,y1,z1;
    return V;
}

// ─── test_keepPosition ────────────────────────────────────────────────────────
// With keepPosition=true, each instance's Z should be lifted so its AABB floor
// sits exactly at bedOffset.  XY should be unchanged (tx=ty=0).
static void test_keepPosition() {
    Eigen::MatrixXf V = makeBox(-5.f, 5.f, -5.f, 5.f, -3.f, 7.f);  // z in [-3,7]
    Eigen::MatrixXi F; // unused by packing

    PackingPass::Params p;
    p.keepPosition = true;
    p.bedOffset    = 2.0f;
    p.plateX       = 100.f;
    p.plateY       = 100.f;

    std::vector<Eigen::MatrixXf> rotV  = {V};
    std::vector<Eigen::MatrixXi> faces = {F};
    std::vector<uint32_t>        idx   = {0};
    // AABB min z = -3, AABB max z = 7
    std::vector<Eigen::Vector3f> aMin  = {{-5.f, -5.f, -3.f}};
    std::vector<Eigen::Vector3f> aMax  = {{ 5.f,  5.f,  7.f}};

    PackingPass packer;
    auto result = packer.pack(rotV, faces, idx, aMin, aMax, p);

    assert(result.size() == 1);
    assert(result[0].tx == 0.f);
    assert(result[0].ty == 0.f);
    // tz should lift the z-floor from -3 to bedOffset=2 → tz = 5
    assert(std::abs(result[0].tz - 5.f) < 1e-4f);

    std::cout << "  keepPosition:    PASSED  tz=" << result[0].tz << "\n";
}

// ─── test_single_part_sits_at_clearance ──────────────────────────────────────
// A single part on an empty plate (bedOffset=0) should sit with its bottom
// at exactly clearance (not floating higher due to Z-normalization).
static void test_single_part_sits_at_clearance() {
    // Box: x ∈ [0,10], y ∈ [0,10], z ∈ [-4, 6]  (z-min is not at 0)
    Eigen::MatrixXf V = makeBox(0.f, 10.f, 0.f, 10.f, -4.f, 6.f);
    Eigen::MatrixXi F;

    PackingPass::Params p;
    p.plateX      = 200.f;
    p.plateY      = 200.f;
    p.plateZ      = 200.f;
    p.resolution  = 1.0f;
    p.clearance   = 0.5f;
    p.bedOffset   = 0.f;

    std::vector<Eigen::MatrixXf> rotV  = {V};
    std::vector<Eigen::MatrixXi> faces = {F};
    std::vector<uint32_t>        idx   = {0};
    std::vector<Eigen::Vector3f> aMin  = {{0.f, 0.f, -4.f}};
    std::vector<Eigen::Vector3f> aMax  = {{10.f, 10.f, 6.f}};

    PackingPass packer;
    auto result = packer.pack(rotV, faces, idx, aMin, aMax, p);

    assert(result.size() == 1);
    // After placement, AABB z-min should be at clearance = 0.5
    float newZMin = aMin[0].z() + result[0].tz; // -4 + tz = 0.5 → tz = 4.5
    assert(std::abs(newZMin - p.clearance) < 1e-4f);

    std::cout << "  single_part_sits_at_clearance: PASSED"
              << "  tz=" << result[0].tz
              << "  newZMin=" << newZMin << "\n";
}

// ─── test_two_parts_no_overlap ────────────────────────────────────────────────
// Two identical parts placed on the same plate must not overlap in XY.
static void test_two_parts_no_overlap() {
    // 20×20mm footprint, 10mm tall, z ∈ [0,10]
    Eigen::MatrixXf V = makeBox(0.f, 20.f, 0.f, 20.f, 0.f, 10.f);
    Eigen::MatrixXi F;

    PackingPass::Params p;
    p.plateX     = 100.f;
    p.plateY     = 100.f;
    p.plateZ     = 100.f;
    p.resolution = 1.0f;
    p.clearance  = 0.f;
    p.bedOffset  = 0.f;

    std::vector<Eigen::MatrixXf> rotV  = {V};
    std::vector<Eigen::MatrixXi> faces = {F};
    std::vector<uint32_t>        idx   = {0, 0};  // both use mesh 0
    std::vector<Eigen::Vector3f> aMin  = {{0.f,0.f,0.f}, {0.f,0.f,0.f}};
    std::vector<Eigen::Vector3f> aMax  = {{20.f,20.f,10.f}, {20.f,20.f,10.f}};

    PackingPass packer;
    auto result = packer.pack(rotV, faces, idx, aMin, aMax, p);

    assert(result.size() == 2);

    // Compute placed AABB X/Y intervals for both parts
    float ax0 = aMin[0].x() + result[0].tx, ax1 = aMax[0].x() + result[0].tx;
    float ay0 = aMin[0].y() + result[0].ty, ay1 = aMax[0].y() + result[0].ty;
    float bx0 = aMin[1].x() + result[1].tx, bx1 = aMax[1].x() + result[1].tx;
    float by0 = aMin[1].y() + result[1].ty, by1 = aMax[1].y() + result[1].ty;

    bool xOverlap = ax0 < bx1 && bx0 < ax1;
    bool yOverlap = ay0 < by1 && by0 < ay1;
    bool zSeparate = false;
    // Parts that do XY-overlap must be Z-separated
    float az0 = aMin[0].z() + result[0].tz, az1 = aMax[0].z() + result[0].tz;
    float bz0 = aMin[1].z() + result[1].tz, bz1 = aMax[1].z() + result[1].tz;
    if (az1 <= bz0 || bz1 <= az0) zSeparate = true;

    bool noCollision = !(xOverlap && yOverlap) || zSeparate;
    assert(noCollision);

    std::cout << "  two_parts_no_overlap:          PASSED\n"
              << "    part0 X=[" << ax0 << "," << ax1 << "] Y=[" << ay0 << "," << ay1 << "]\n"
              << "    part1 X=[" << bx0 << "," << bx1 << "] Y=[" << by0 << "," << by1 << "]\n";
}

// ─── test_bedOffset_respected ─────────────────────────────────────────────────
// bedOffset=5 → all placed parts must have their Z-floor >= 5.
static void test_bedOffset_respected() {
    Eigen::MatrixXf V = makeBox(0.f, 10.f, 0.f, 10.f, 0.f, 5.f);
    Eigen::MatrixXi F;

    PackingPass::Params p;
    p.plateX    = 100.f;
    p.plateY    = 100.f;
    p.resolution = 1.0f;
    p.clearance  = 0.f;
    p.bedOffset  = 5.0f;

    std::vector<Eigen::MatrixXf> rotV  = {V};
    std::vector<Eigen::MatrixXi> faces = {F};
    std::vector<uint32_t>        idx   = {0};
    std::vector<Eigen::Vector3f> aMin  = {{0.f,0.f,0.f}};
    std::vector<Eigen::Vector3f> aMax  = {{10.f,10.f,5.f}};

    PackingPass packer;
    auto result = packer.pack(rotV, faces, idx, aMin, aMax, p);

    float newZMin = aMin[0].z() + result[0].tz;
    assert(newZMin >= p.bedOffset - 1e-4f);

    std::cout << "  bedOffset_respected:           PASSED  newZMin=" << newZMin << "\n";
}

int main() {
    std::cout << "[test_packing]\n";
    test_keepPosition();
    test_single_part_sits_at_clearance();
    test_two_parts_no_overlap();
    test_bedOffset_respected();
    std::cout << "All packing tests PASSED\n";
    return 0;
}
