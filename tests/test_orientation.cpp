#include "OrientationOptimizer.hpp"
#include <MeshPreprocess.hpp>
#include <Eigen/Core>
#include <cassert>
#include <cmath>
#include <iostream>

// Build a tall thin box aligned with Z (0.1 × 0.1 × 10 mm).
// The side-face normals dominate C, so v_min ≈ Z.
// findBestRotation should return near-identity (keep it vertical).
static std::vector<Triangle> makeTallNeedle() {
    // Box corners: X ∈ [-0.05, 0.05], Y ∈ [-0.05, 0.05], Z ∈ [0, 10]
    const float x0=-0.05f, x1=0.05f, y0=-0.05f, y1=0.05f, z0=0.f, z1=10.f;

    // Helper to push a quad as two CCW triangles (seen from outside)
    std::vector<Triangle> tris;
    auto quad = [&](Vec3f a, Vec3f b, Vec3f c, Vec3f d) {
        tris.push_back({ a, b, c });
        tris.push_back({ a, c, d });
    };

    // +X face (normal +X)
    quad({x1,y0,z0},{x1,y1,z0},{x1,y1,z1},{x1,y0,z1});
    // -X face (normal -X)
    quad({x0,y1,z0},{x0,y0,z0},{x0,y0,z1},{x0,y1,z1});
    // +Y face (normal +Y)
    quad({x1,y1,z0},{x0,y1,z0},{x0,y1,z1},{x1,y1,z1});
    // -Y face (normal -Y)
    quad({x0,y0,z0},{x1,y0,z0},{x1,y0,z1},{x0,y0,z1});
    // +Z face (normal +Z, small)
    quad({x0,y0,z1},{x1,y0,z1},{x1,y1,z1},{x0,y1,z1});
    // -Z face (normal -Z, small)
    quad({x1,y0,z0},{x0,y0,z0},{x0,y1,z0},{x1,y1,z0});

    return tris;
}

int main() {
    // ── weldSoup ──────────────────────────────────────────────────────────────
    auto tris = makeTallNeedle();
    Eigen::MatrixXf V;
    Eigen::MatrixXi F;
    weldSoup(tris, V, F);

    // 12 triangles → 8 unique corner vertices
    assert(V.rows() == 8);
    assert(F.rows() == 12);

    // ── HalfEdge one-ring ─────────────────────────────────────────────────────
    HalfEdgeMesh he = HalfEdgeMesh::build(V, F);

    for (int v = 0; v < V.rows(); ++v) {
        int cnt = 0;
        he.forEachNeighbor(v, [&](int) { ++cnt; });
        assert(cnt >= 2); // every corner of the box has at least 2 neighbors
    }

    // ── findBestRotation ──────────────────────────────────────────────────────
    // For a tall needle, v_min of the normal tensor ≈ Z.
    // The optimal rotation should keep the needle vertical: R ≈ Identity,
    // so |R * ẑ · ẑ| should be close to 1.
    OrientWeights w;
    Eigen::Matrix3f R = OrientationOptimizer::findBestRotation(V, F, he, w);

    Eigen::Vector3f zOut = R * Eigen::Vector3f(0, 0, 1);
    float absZ = std::abs(zOut(2));
    assert(absZ > 0.9f);

    std::cout << "test_orientation PASSED (|R*Z · Z| = " << absZ << ")\n";
    return 0;
}
