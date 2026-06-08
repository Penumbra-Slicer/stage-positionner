#include "OrientationOptimizer.hpp"

#include <igl/per_face_normals.h>
#include <igl/doublearea.h>
#include <igl/AABB.h>

#include <Eigen/Geometry>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <unordered_map>

using Clock = std::chrono::steady_clock;
static double msec(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

namespace OrientationOptimizer {

// ─── calculateNormalTensor ───────────────────────────────────────────────────

Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f>
calculateNormalTensor(const Eigen::MatrixXf& V, const Eigen::MatrixXi& F)
{
    Eigen::MatrixXf N;
    igl::per_face_normals(V, F, N);

    Eigen::VectorXd dA;
    igl::doublearea(V, F, dA); // twice the area per face

    Eigen::Matrix3f C = Eigen::Matrix3f::Zero();
    for (int i = 0; i < F.rows(); ++i) {
        Eigen::Vector3f n = N.row(i).transpose();
        C += static_cast<float>(dA(i) * 0.5) * (n * n.transpose());
    }

    return Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f>(C);
}

// ─── helpers ─────────────────────────────────────────────────────────────────

static Eigen::MatrixXf rotateV(const Eigen::MatrixXf& V, const Eigen::Matrix3f& R) {
    return (R * V.transpose()).transpose();
}

// Builds a rotation matrix that aligns axis `a` with +Z.
static Eigen::Matrix3f alignToZ(const Eigen::Vector3f& a) {
    Eigen::Vector3f z(0, 0, 1);
    Eigen::Vector3f v = a.cross(z);
    float s = v.norm();
    float c = a.dot(z);
    if (s < 1e-8f) {
        // Already aligned or anti-aligned
        return (c > 0.f) ? Eigen::Matrix3f::Identity()
                         : Eigen::AngleAxisf(static_cast<float>(M_PI),
                                             Eigen::Vector3f::UnitX()).toRotationMatrix();
    }
    Eigen::Matrix3f vx;
    vx <<    0, -v(2),  v(1),
           v(2),    0, -v(0),
          -v(1),  v(0),    0;
    return Eigen::Matrix3f::Identity() + vx + vx * vx * ((1.f - c) / (s * s));
}

// ─── countIslands ────────────────────────────────────────────────────────────

int countIslands(const Eigen::MatrixXf& V,
                 const HalfEdgeMesh&    he,
                 const Eigen::Matrix3f& R)
{
    Eigen::MatrixXf Vr = rotateV(V, R);
    const int nv = static_cast<int>(Vr.rows());

    std::vector<uint8_t> is_island(nv, 1);

    for (int i = 0; i < (int)he.halfEdges.size(); ++i) {
        const int twin = he.halfEdges[i].twin;
        if (twin >= 0 && twin < i) continue; // each undirected edge once

        const int v0 = he.halfEdges[i].vertex;
        const int v1 = (twin >= 0)
                       ? he.halfEdges[twin].vertex
                       : he.halfEdges[he.halfEdges[i].prev].vertex;

        const float z0 = Vr(v0, 2), z1 = Vr(v1, 2);
        if (z0 >= z1) is_island[v0] = 0;
        if (z1 >= z0) is_island[v1] = 0;
    }

    return static_cast<int>(std::count(is_island.begin(), is_island.end(), uint8_t(1)));
}

// ─── detectSuction ───────────────────────────────────────────────────────────

int detectSuction(const Eigen::MatrixXf& V,
                  const Eigen::MatrixXi& F,
                  const Eigen::Matrix3f& R)
{
    Eigen::MatrixXf Vr = rotateV(V, R);

    // Cast rays upward (+Z) from a regular grid over the XY footprint.
    // A suction cup is detected when the first face hit by a ray is back-facing.
    igl::AABB<Eigen::MatrixXf, 3> tree;
    tree.init(Vr, F);

    float xMin = Vr.col(0).minCoeff(), xMax = Vr.col(0).maxCoeff();
    float yMin = Vr.col(1).minCoeff(), yMax = Vr.col(1).maxCoeff();
    float zMin = Vr.col(2).minCoeff() - 1.f;

    const int gridN  = 32;
    const float dx   = (xMax - xMin) / gridN;
    const float dy   = (yMax - yMin) / gridN;

    Eigen::MatrixXf N;
    igl::per_face_normals(Vr, F, N);

    const Eigen::RowVector3f dir(0, 0, 1);
    int suctionCount = 0;

    for (int i = 0; i < gridN; ++i) {
        for (int j = 0; j < gridN; ++j) {
            float ox = xMin + (i + 0.5f) * dx;
            float oy = yMin + (j + 0.5f) * dy;

            Eigen::RowVector3f origin(ox, oy, zMin);
            igl::Hit hit;
            if (tree.intersect_ray(Vr, F, origin, dir, hit)) {
                // Back-face: normal points in same direction as ray (+Z)
                if (N.row(hit.id).dot(dir) > 0.f)
                    ++suctionCount;
            }
        }
    }
    return suctionCount;
}

// Golden ratio for Fibonacci sphere sampling.
static constexpr float kGoldenRatio = 1.6180339887f;

// ─── candidateAxes ───────────────────────────────────────────────────────────

CandidateInfo
candidateAxes(const Eigen::MatrixXf& V, const Eigen::MatrixXi& F, int numCandidates)
{
    auto t0 = Clock::now();
    auto solver = calculateNormalTensor(V, F);
    std::cerr << "  normalTensor:    " << std::fixed << std::setprecision(1) << msec(t0) << " ms\n";

    Eigen::Vector3f v0 = solver.eigenvectors().col(0);
    Eigen::Vector3f v1 = solver.eigenvectors().col(1);
    Eigen::Vector3f v2 = solver.eigenvectors().col(2);

    std::vector<Eigen::Matrix3f> rots;
    rots.reserve(std::max(numCandidates, 6));

    // Fixed principal-axis candidates (always included).
    rots.push_back(alignToZ( v0));
    rots.push_back(alignToZ(-v0));
    rots.push_back(alignToZ( v1));
    rots.push_back(alignToZ(-v1));
    rots.push_back(alignToZ( v2));
    rots.push_back(alignToZ(-v2));

    // Fill remainder with Fibonacci sphere samples for uniform coverage.
    const int extra = numCandidates - 6;
    for (int i = 0; i < extra; ++i) {
        float theta = std::acos(1.f - 2.f * (i + 0.5f) / extra);
        float phi   = 2.f * static_cast<float>(M_PI) * i / kGoldenRatio;
        Eigen::Vector3f axis(
            std::sin(theta) * std::cos(phi),
            std::sin(theta) * std::sin(phi),
            std::cos(theta));
        rots.push_back(alignToZ(axis));
    }

    return {
        std::move(rots),
        static_cast<float>(solver.eigenvalues()(0)),
        static_cast<float>(solver.eigenvalues()(2))
    };
}

// ─── findBestRotation (shared scoring helper) ─────────────────────────────────

static Eigen::Matrix3f bestRotationFromScores(
    const Eigen::MatrixXf&  V,
    const Eigen::MatrixXi&  F,
    const HalfEdgeMesh&     he,
    const OrientWeights&    w,
    const std::vector<int>& suctionScores,
    const CandidateInfo&    ci)
{
    const int N = static_cast<int>(ci.rotations.size());

    // Pre-compute face normals and half-areas once — reused for every candidate.
    Eigen::MatrixXf faceN;
    Eigen::VectorXd dA;
    igl::per_face_normals(V, F, faceN);
    igl::doublearea(V, F, dA);

    // ── Pass 1: collect raw objective values for all candidates ──────────────
    std::vector<float> peelRaw(N), islandsRaw(N), suctionRaw(N);

    for (int k = 0; k < N; ++k) {
        const Eigen::Matrix3f& R = ci.rotations[k];

        // Peel: sum of downward-facing projected areas (face_area * max(0, -nz_rotated)).
        Eigen::MatrixXf rotN = (R * faceN.transpose()).transpose();
        float peel = 0.f;
        for (int i = 0; i < rotN.rows(); ++i)
            peel += static_cast<float>(dA(i)) * 0.5f * std::max(0.f, -rotN(i, 2));
        peelRaw[k] = peel;

        auto t0 = Clock::now();
        islandsRaw[k] = static_cast<float>(countIslands(V, he, R));
        std::cerr << "  countIslands[" << k << "]: " << msec(t0) << " ms"
                  << "  (count=" << static_cast<int>(islandsRaw[k]) << ")\n";

        suctionRaw[k] = static_cast<float>(suctionScores[k]);
    }

    // ── Normalize each objective to [0,1] across candidates ──────────────────
    auto maxOf = [&](const std::vector<float>& v) {
        return *std::max_element(v.begin(), v.end());
    };
    const float peelMax    = maxOf(peelRaw);
    const float islandsMax = maxOf(islandsRaw);
    const float suctionMax = maxOf(suctionRaw);

    // ── Pass 2: pick best weighted cost ──────────────────────────────────────
    float bestCost = std::numeric_limits<float>::max();
    Eigen::Matrix3f bestR = Eigen::Matrix3f::Identity();

    for (int k = 0; k < N; ++k) {
        float peelNorm    = (peelMax    > 1e-8f) ? peelRaw[k]    / peelMax    : 0.f;
        float islandsNorm = (islandsMax > 1e-8f) ? islandsRaw[k] / islandsMax : 0.f;
        float suctionNorm = (suctionMax > 1e-8f) ? suctionRaw[k] / suctionMax : 0.f;

        float cost = w.peel * peelNorm + w.islands * islandsNorm + w.suction * suctionNorm;
        std::cerr << "  candidate[" << k << "]: peel=" << peelNorm
                  << " islands=" << islandsNorm << " suction=" << suctionNorm
                  << " cost=" << cost << "\n";
        if (cost < bestCost) { bestCost = cost; bestR = ci.rotations[k]; }
    }
    return bestR;
}

// ─── findBestRotation — CPU path ─────────────────────────────────────────────

Eigen::Matrix3f findBestRotation(const Eigen::MatrixXf& V,
                                  const Eigen::MatrixXi& F,
                                  const HalfEdgeMesh&    he,
                                  const OrientWeights&   w)
{
    CandidateInfo ci = candidateAxes(V, F);

    const int N = static_cast<int>(ci.rotations.size());
    std::vector<int> suctionScores(N);
    for (int k = 0; k < N; ++k)
        suctionScores[k] = detectSuction(V, F, ci.rotations[k]);

    return bestRotationFromScores(V, F, he, w, suctionScores, ci);
}

// ─── findBestRotation — GPU path (pre-computed suction scores) ───────────────

Eigen::Matrix3f findBestRotation(const Eigen::MatrixXf&  V,
                                  const Eigen::MatrixXi&  F,
                                  const HalfEdgeMesh&     he,
                                  const OrientWeights&    w,
                                  const std::vector<int>& suctionScores,
                                  const CandidateInfo&    ci)
{
    return bestRotationFromScores(V, F, he, w, suctionScores, ci);
}

} // namespace OrientationOptimizer
