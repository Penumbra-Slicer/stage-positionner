#pragma once
#include <HalfEdgeMesh.hpp>
#include <MeshTypes.hpp>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <vector>

struct OrientWeights {
    float peel    = 0.6f;
    float suction = 0.25f;
    float islands = 0.15f;
};

namespace OrientationOptimizer {

/// Converts a triangle soup to an indexed face set using vertex welding.
/// Returns V (Nv×3) and F (Nf×3).
void weldSoup(const std::vector<Triangle>& tris,
              Eigen::MatrixXf& V, Eigen::MatrixXi& F);

/// Builds C = Σ Aᵢ (nᵢ nᵢᵀ) and returns its eigensolver.
/// The solver's eigenvectors are the candidate orientation axes.
Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f>
calculateNormalTensor(const Eigen::MatrixXf& V, const Eigen::MatrixXi& F);

/// Counts local Z-minima (islands) when the mesh is rotated by R.
/// Uses the half-edge one-ring for efficient neighbor traversal.
int countIslands(const Eigen::MatrixXf& V,
                 const HalfEdgeMesh&    he,
                 const Eigen::Matrix3f& R);

/// Counts trapped-volume candidates (suction cups) when rotated by R.
/// Casts upward rays via igl::AABB; a hit on a back-face indicates a trap.
int detectSuction(const Eigen::MatrixXf& V,
                  const Eigen::MatrixXi& F,
                  const Eigen::Matrix3f& R);

/// Result of candidateAxes(): rotation matrices + eigenvalues needed for scoring.
/// Passed to findBestRotation (GPU path) to avoid recomputing the normal tensor.
struct CandidateInfo {
    std::array<Eigen::Matrix3f, 4> rotations; ///< alignToZ(±vMin, ±vMid)
    float lambdaMin; ///< eigenvalue(0) — peel proxy
    float lambdaMax; ///< eigenvalue(2) — normalization denominator
};

/// Computes the normal tensor once and returns the four candidate rotations
/// (±v_min, ±v_mid aligned to +Z) together with the eigenvalues needed for
/// cost normalisation.  Index matches suctionScores[4] in findBestRotation.
CandidateInfo
candidateAxes(const Eigen::MatrixXf& V, const Eigen::MatrixXi& F);

/// CPU path — evaluates all four candidates, calling detectSuction internally.
Eigen::Matrix3f findBestRotation(const Eigen::MatrixXf& V,
                                  const Eigen::MatrixXi& F,
                                  const HalfEdgeMesh&    he,
                                  const OrientWeights&   w);

/// GPU path — accepts pre-computed suction pixel counts and the CandidateInfo
/// returned by candidateAxes(), so the normal tensor is not recomputed.
Eigen::Matrix3f findBestRotation(const Eigen::MatrixXf& V,
                                  const Eigen::MatrixXi& F,
                                  const HalfEdgeMesh&    he,
                                  const OrientWeights&   w,
                                  const int              suctionScores[4],
                                  const CandidateInfo&   ci);

} // namespace OrientationOptimizer
