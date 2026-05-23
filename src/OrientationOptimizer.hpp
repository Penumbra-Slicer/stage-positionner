#pragma once
#include <HalfEdgeMesh.hpp>
#include <MeshTypes.hpp>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <array>
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
    std::vector<Eigen::Matrix3f> rotations; ///< numCandidates directions aligned to +Z
    float lambdaMin; ///< eigenvalue(0) — peel proxy
    float lambdaMax; ///< eigenvalue(2) — normalization denominator
};

/// Computes the normal tensor once and returns numCandidates candidate rotations
/// aligned to +Z.  The first 6 are always ±v0/±v1/±v2 (principal axes);
/// the remainder are Fibonacci sphere samples.
/// Index matches suctionScores in findBestRotation.
CandidateInfo
candidateAxes(const Eigen::MatrixXf& V, const Eigen::MatrixXi& F, int numCandidates = 32);

/// CPU path — evaluates all four candidates, calling detectSuction internally.
Eigen::Matrix3f findBestRotation(const Eigen::MatrixXf& V,
                                  const Eigen::MatrixXi& F,
                                  const HalfEdgeMesh&    he,
                                  const OrientWeights&   w);

/// GPU path — accepts pre-computed suction pixel counts and the CandidateInfo
/// returned by candidateAxes(), so the normal tensor is not recomputed.
Eigen::Matrix3f findBestRotation(const Eigen::MatrixXf&    V,
                                  const Eigen::MatrixXi&    F,
                                  const HalfEdgeMesh&       he,
                                  const OrientWeights&      w,
                                  const std::vector<int>&   suctionScores,
                                  const CandidateInfo&      ci);

} // namespace OrientationOptimizer
