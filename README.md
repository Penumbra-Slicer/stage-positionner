# stage-positionner

Orients and packs mesh instances on the build plate before slicing. Uses eigenvalue analysis to score candidate orientations, GPU depth-peeling to detect suction cups, and a greedy heightmap algorithm to pack multiple objects without overlap.

---

## DataBuffer contract

| Direction | Format / Key | Description |
|---|---|---|
| reads cpuData | `mesh/scene_graph_v1` | Scene graph loaded by STL-Loader |
| reads meta | `mesh.half_edge` | Half-edge mesh (pre-populated by the runner) |
| writes cpuData | `mesh/scene_graph_v1` | Same scene graph with updated instance transforms |

---

## Parameters

| Key | Type | Default | Range | Description |
|---|---|---|---|---|
| `orientEnable` | bool | true | — | Run orientation optimization |
| `weightPeel` | float | 1.0 | 0–∞ | Weight for minimizing peel force |
| `weightSuction` | float | 1.0 | 0–∞ | Weight for avoiding suction surfaces |
| `weightIslands` | float | 1.0 | 0–∞ | Weight for minimizing floating islands |
| `numCandidates` | int | 50 | 6–∞ | Number of orientation candidates to evaluate |
| `packEnable` | bool | true | — | Run packing after orientation |
| `packResolution` | float | 1.0 | 0.1–∞ | Heightmap resolution in mm |
| `packClearance` | float | 2.0 | 0–∞ | Minimum gap between instances in mm |
| `bedOffset` | float | 0.0 | 0–∞ | Z offset from the build plate in mm |

---

## Algorithm

**Orientation**
1. Compute the normal covariance tensor from all mesh faces (Eigen3).
2. Generate candidates: 6 principal axes + Fibonacci sphere samples.
3. Score each candidate: peel force (projected normals), suction detection (GPU depth-peeling), floating island count.
4. Select the rotation that minimises the weighted sum.

**Packing**
1. Build a 2D heightmap of the build plate at `packResolution`.
2. For each instance, find the lowest collision-free position using a greedy scan.
3. Place and stamp the instance footprint onto the heightmap.

---

## Build

```bash
cmake -B build/stage-positionner -S stage-positionner
cmake --build build/stage-positionner   # → stage-positionner.so

# With tests
cmake -B build/stage-positionner -S stage-positionner -DBUILD_TESTING=ON
cmake --build build/stage-positionner
ctest --test-dir build/stage-positionner --output-on-failure
```

---

## Tests

| Test | What it covers |
|---|---|
| `test_half_edge` | HalfEdgeMesh construction from a triangle soup |
| `test_orientation` | Orientation scoring on a known mesh |
| `test_packing` | Greedy packing without GPU |
| `test_packing_gpu` | GPU-accelerated packing path |

---

## Dependencies

| Library | Version | Purpose |
|---|---|---|
| `pipeline-core` | — | Stage ABI and DataBuffer types |
| `Eigen3` | 3.4.0 | Eigenvalue analysis for orientation |
| `libigl` | 2.5.0 | Mesh utilities |
| `LavaCake` | fefb35e | Vulkan helper |
| Vulkan SDK | — | GPU depth-peeling |
