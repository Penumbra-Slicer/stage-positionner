#include "PackingPass.hpp"
#include "GpuRuntime.hpp"
#include <Eigen/Core>
#include <cassert>
#include <cmath>
#include <iostream>

static Eigen::MatrixXf makeBox(float x0, float x1,
                                float y0, float y1,
                                float z0, float z1)
{
    Eigen::MatrixXf V(8, 3);
    V << x0,y0,z0,  x1,y0,z0,  x1,y1,z0,  x0,y1,z0,
         x0,y0,z1,  x1,y0,z1,  x1,y1,z1,  x0,y1,z1;
    return V;
}

// Returns a GpuContext, skipping the test with a message if Vulkan is unavailable.
static GpuContext requireGpu() {
    GpuContext ctx = createVulkanContext();
    if (!ctx.isAvailable()) {
        std::cerr << "  [SKIP] No Vulkan device available\n";
        // Return the unavailable context; callers check isAvailable().
    }
    return ctx;
}

// ─── H1: init() succeeds ──────────────────────────────────────────────────────
static bool test_gpu_init(GpuContext& ctx) {
    PackingPass packer;
    bool threw = false;
    try {
        packer.init(ctx);
    } catch (const std::exception& e) {
        std::cerr << "  init threw: " << e.what() << "\n";
        threw = true;
    }
    assert(!threw);
    std::cout << "  gpu_init:                  PASSED\n";
    return true;
}

// ─── H2 + H6: GPU drop-test gives same result as CPU ────────────────────────
// We run the same pack() call twice: once without init (CPU) and once with init (GPU).
static void test_gpu_cpu_equivalence(GpuContext& ctx) {
    Eigen::MatrixXf V = makeBox(0.f, 10.f, 0.f, 10.f, 0.f, 10.f);
    Eigen::MatrixXi F;

    PackingPass::Params p;
    p.plateX     = 200.f;
    p.plateY     = 200.f;
    p.plateZ     = 200.f;
    p.resolution = 1.0f;
    p.clearance  = 0.5f;
    p.bedOffset  = 0.f;

    std::vector<Eigen::MatrixXf> rotV  = {V};
    std::vector<Eigen::MatrixXi> faces = {F};
    std::vector<uint32_t>        idx   = {0};
    std::vector<Eigen::Vector3f> aMin  = {{0.f, 0.f, 0.f}};
    std::vector<Eigen::Vector3f> aMax  = {{10.f, 10.f, 10.f}};

    // CPU path
    PackingPass cpuPacker;
    auto cpuResult = cpuPacker.pack(rotV, faces, idx, aMin, aMax, p);

    // GPU path
    PackingPass gpuPacker;
    gpuPacker.init(ctx);
    auto gpuResult = gpuPacker.pack(rotV, faces, idx, aMin, aMax, p);

    assert(cpuResult.size() == 1 && gpuResult.size() == 1);
    assert(std::abs(gpuResult[0].tx - cpuResult[0].tx) < 1e-3f);
    assert(std::abs(gpuResult[0].ty - cpuResult[0].ty) < 1e-3f);
    assert(std::abs(gpuResult[0].tz - cpuResult[0].tz) < 1e-3f);

    std::cout << "  gpu_cpu_equivalence:       PASSED"
              << "  cpu=(" << cpuResult[0].tx << "," << cpuResult[0].ty << "," << cpuResult[0].tz << ")"
              << "  gpu=(" << gpuResult[0].tx << "," << gpuResult[0].ty << "," << gpuResult[0].tz << ")\n";
}

// ─── H3: push constants are received correctly ───────────────────────────────
// A 1×1 part (single pixel) on a 4×4 plate, G seeded at bedOffset=0.
// zMin[0]=0 → reqZ = G[0]-0 + clearance = 0 + 1 = 1 at every valid position.
// The shader must return exactly 1.0 (clearance=1) for all valid candidate pixels.
static void test_gpu_push_constants(GpuContext& ctx) {
    // 1-pixel part (1 mm × 1 mm × 5 mm box, resolution=1)
    Eigen::MatrixXf V = makeBox(0.f, 1.f, 0.f, 1.f, 0.f, 5.f);
    Eigen::MatrixXi F;

    PackingPass::Params p;
    p.plateX     = 4.f;
    p.plateY     = 4.f;
    p.plateZ     = 100.f;
    p.resolution = 1.0f;
    p.clearance  = 1.0f;
    p.bedOffset  = 0.f;

    std::vector<Eigen::MatrixXf> rotV  = {V};
    std::vector<Eigen::MatrixXi> faces = {F};
    std::vector<uint32_t>        idx   = {0};
    std::vector<Eigen::Vector3f> aMin  = {{0.f, 0.f, 0.f}};
    std::vector<Eigen::Vector3f> aMax  = {{1.f, 1.f, 5.f}};

    PackingPass gpuPacker;
    gpuPacker.init(ctx);
    auto result = gpuPacker.pack(rotV, faces, idx, aMin, aMax, p);

    assert(result.size() == 1);
    // The part's bottom should land at exactly clearance = 1.0
    float newZMin = aMin[0].z() + result[0].tz;
    assert(std::abs(newZMin - p.clearance) < 1e-4f);

    std::cout << "  gpu_push_constants:        PASSED  newZMin=" << newZMin << "\n";
}

// ─── H4: FLT_MAX sentinel is skipped by the shader ───────────────────────────
// A sparse part (only corners covered) — most pixels are FLT_MAX.
// The drop-test must not count sentinel pixels as real geometry.
static void test_gpu_sentinel_skipped(GpuContext& ctx) {
    // A thin ring: 8 vertices at corners of 10×10 box, leaving the centre empty.
    // buildHeightmaps will see FLT_MAX in interior pixels.
    Eigen::MatrixXf V = makeBox(0.f, 10.f, 0.f, 10.f, 0.f, 2.f);
    Eigen::MatrixXi F;

    PackingPass::Params p;
    p.plateX     = 200.f;
    p.plateY     = 200.f;
    p.plateZ     = 200.f;
    p.resolution = 1.0f;
    p.clearance  = 0.5f;
    p.bedOffset  = 0.f;

    std::vector<Eigen::MatrixXf> rotV  = {V};
    std::vector<Eigen::MatrixXi> faces = {F};
    std::vector<uint32_t>        idx   = {0};
    std::vector<Eigen::Vector3f> aMin  = {{0.f, 0.f, 0.f}};
    std::vector<Eigen::Vector3f> aMax  = {{10.f, 10.f, 2.f}};

    PackingPass gpuPacker;
    gpuPacker.init(ctx);
    auto result = gpuPacker.pack(rotV, faces, idx, aMin, aMax, p);

    assert(result.size() == 1);
    // The part's bottom should land at clearance (not at some crazy height from sentinel confusion).
    float newZMin = aMin[0].z() + result[0].tz;
    assert(newZMin < 10.f);  // sentinel confusion would produce absurdly large tz

    std::cout << "  gpu_sentinel_skipped:      PASSED  newZMin=" << newZMin << "\n";
}

// ─── H5: CPU stamp is visible to subsequent GPU dispatches ───────────────────
// Pack two identical 20×20×10 boxes. The second must sit ON TOP of the first
// (not beside it on an empty plate), proving the GPU saw the updated G after stamping.
// We force them to stack by using a plate just wide enough for one footprint.
static void test_gpu_stamp_visible(GpuContext& ctx) {
    // 20×20×10 box
    Eigen::MatrixXf V = makeBox(0.f, 20.f, 0.f, 20.f, 0.f, 10.f);
    Eigen::MatrixXi F;

    PackingPass::Params p;
    // Plate exactly 20mm wide → both parts must stack in Z, no side-by-side room.
    p.plateX     = 20.f;
    p.plateY     = 20.f;
    p.plateZ     = 200.f;
    p.resolution = 1.0f;
    p.clearance  = 0.f;
    p.bedOffset  = 0.f;

    std::vector<Eigen::MatrixXf> rotV  = {V};
    std::vector<Eigen::MatrixXi> faces = {F};
    std::vector<uint32_t>        idx   = {0, 0};
    std::vector<Eigen::Vector3f> aMin  = {{0.f,0.f,0.f}, {0.f,0.f,0.f}};
    std::vector<Eigen::Vector3f> aMax  = {{20.f,20.f,10.f}, {20.f,20.f,10.f}};

    PackingPass gpuPacker;
    gpuPacker.init(ctx);
    auto result = gpuPacker.pack(rotV, faces, idx, aMin, aMax, p);

    assert(result.size() == 2);

    const float tz0 = result[0].tz;
    const float tz1 = result[1].tz;
    const float partH = 10.f;

    // One part sits at bedOffset=0, the other sits on top of it.
    float lower = std::min(tz0, tz1);
    float upper = std::max(tz0, tz1);

    assert(std::abs(lower) < 1e-3f);              // first part at z=0
    assert(std::abs(upper - partH) < 1e-3f);      // second part directly on top

    std::cout << "  gpu_stamp_visible:         PASSED"
              << "  tz=(" << tz0 << "," << tz1 << ")\n";
}

int main() {
    std::cout << "[test_packing_gpu]\n";

    GpuContext ctx = requireGpu();
    if (!ctx.isAvailable()) {
        std::cout << "All GPU tests SKIPPED (no Vulkan device)\n";
        return 0;
    }

    test_gpu_init(ctx);
    test_gpu_cpu_equivalence(ctx);
    test_gpu_push_constants(ctx);
    test_gpu_sentinel_skipped(ctx);
    test_gpu_stamp_visible(ctx);

    destroyVulkanContext(ctx);
    std::cout << "All GPU packing tests PASSED\n";
    return 0;
}
