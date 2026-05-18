#include <HalfEdgeMesh.hpp>
#include <Eigen/Core>
#include <cassert>
#include <iostream>

// Build a tetrahedron and verify half-edge twin/next linkage.
int main() {
    // 4 vertices
    Eigen::MatrixXf V(4, 3);
    V << 0, 0, 0,
         1, 0, 0,
         0, 1, 0,
         0, 0, 1;

    // 4 faces (CCW from outside)
    Eigen::MatrixXi F(4, 3);
    F << 0, 2, 1,
         0, 1, 3,
         0, 3, 2,
         1, 2, 3;

    HalfEdgeMesh he = HalfEdgeMesh::build(V, F);

    // Each face has 3 half-edges → 12 total
    assert(he.halfEdges.size() == 12);

    // Every half-edge's twin should have it as its twin in return
    for (int i = 0; i < (int)he.halfEdges.size(); ++i) {
        int t = he.halfEdges[i].twin;
        if (t >= 0) {
            assert(he.halfEdges[t].twin == i);
        }
    }

    // next/prev consistency: he.halfEdges[next].prev == current
    for (int i = 0; i < (int)he.halfEdges.size(); ++i) {
        int nx = he.halfEdges[i].next;
        assert(he.halfEdges[nx].prev == i);
    }

    // Each vertex's outgoing half-edge must originate from it
    // (destination of previous half-edge around the face)
    for (int v = 0; v < 4; ++v) {
        int h = he.vertexToHalfEdge[v];
        assert(h >= 0);
        // The prev half-edge's destination is v (current he leaves v)
        int pr = he.halfEdges[h].prev;
        assert(he.halfEdges[pr].vertex == v);

        // Collect one-ring and verify it's non-empty
        int neighborCount = 0;
        he.forEachNeighbor(v, [&](int) { ++neighborCount; });
        assert(neighborCount > 0);
    }

    std::cout << "test_half_edge PASSED\n";
    return 0;
}
