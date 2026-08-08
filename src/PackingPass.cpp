#define LAVACAKE_NO_VMA_IMPLEMENTATION
#include "PackingPass.hpp"

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>

// ── Embedded GLSL source ──────────────────────────────────────────────────────

static const char kPackDroptestCompGlsl[] = R"glsl(
#version 450

// Each invocation handles one (px, py) candidate position on the build plate.
// It loops over the part footprint and computes the minimum Z lift needed to
// clear the current global heightmap without collision.
//
// Binding 0: G        — global heightmap [plateW * plateH], read-write
// Binding 1: partZMin — part bottom profile [partW * partH], read-only
// Binding 2: cand     — output: required Z per plate position [plateW * plateH]

layout(local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0) buffer GlobalHM  { float g[];    };
layout(set = 0, binding = 1) buffer PartZMin  { float zmin[]; };
layout(set = 0, binding = 2) buffer Cands     { float cand[]; };

layout(push_constant) uniform PC {
    int   plateW;
    int   plateH;
    int   partW;
    int   partH;
} pc;

const float SENTINEL = 1.0e38;

void main() {
    int px = int(gl_GlobalInvocationID.x);
    int py = int(gl_GlobalInvocationID.y);

    if (px >= pc.plateW || py >= pc.plateH) return;

    int outIdx = py * pc.plateW + px;

    // Positions where any part pixel would land outside the plate: invalid.
    if (px + pc.partW > pc.plateW || py + pc.partH > pc.plateH) {
        cand[outIdx] = SENTINEL;
        return;
    }

    float maxInterference = 0.0;
    for (int j = 0; j < pc.partH; ++j) {
        for (int i = 0; i < pc.partW; ++i) {
            float zm = zmin[j * pc.partW + i];
            if (zm >= SENTINEL * 0.5) continue; // sentinel: no vertex in pixel
            float gv = g[(py + j) * pc.plateW + (px + i)];
            maxInterference = max(maxInterference, gv - zm);
        }
    }

    cand[outIdx] = maxInterference;
}
)glsl";

namespace {
using Clock = std::chrono::steady_clock;
double msec(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// Stamp a footprint's Z-max profile onto the global heightmap at
// (bestPx,bestPy), dilated by `clearance` — shared by the main placement
// loop (a part it just placed) and the fixed-obstacle pre-stamp (a
// position-locked instance's existing footprint). Returns the footprint's
// own peak height (zTop) so callers can log/derive absolute top height
// without recomputing it.
float stampHeightmap(std::vector<float>& G_cpu, float* globalHMPtr, bool gpuReady,
                      int plateW, int plateH,
                      const std::vector<float>& zMax, int w, int h,
                      int bestPx, int bestPy, float bestZ,
                      float clearance, float resolution) {
    float zTop = 0.f;
    for (int j = 0; j < h; ++j)
        for (int i = 0; i < w; ++i)
            zTop = std::max(zTop, zMax[j * w + i]);

    const int cp = (clearance > 0.f) ? static_cast<int>(std::ceil(clearance / resolution)) : 0;
    for (int j = -cp; j < h + cp; ++j) {
        for (int i = -cp; i < w + cp; ++i) {
            const int gx = bestPx + i;
            const int gy = bestPy + j;
            if (gx < 0 || gx >= plateW || gy < 0 || gy >= plateH) continue;
            const float zx = (j >= 0 && j < h && i >= 0 && i < w) ? zMax[j * w + i] : zTop;
            const int   gIdx = gy * plateW + gx;
            const float newZ = bestZ + zx;
            G_cpu[gIdx] = std::max(G_cpu[gIdx], newZ);
            if (gpuReady) globalHMPtr[gIdx] = G_cpu[gIdx];
        }
    }
    return zTop;
}
} // namespace

// ─── initGpu ─────────────────────────────────────────────────────────────────

void PackingPass::init(const GpuContext& ctx) {
    device_    = ctx.device;
    queue_     = ctx.queue;
    cmdPool_   = ctx.cmdPool;
    allocator_ = ctx.allocator;

    // 1. Shader module — compiled from embedded GLSL string
    compMod_ = LavaCake::ShaderModule(ctx.device, kPackDroptestCompGlsl,
        LavaCake::ShadingLanguage::eGLSL, vk::ShaderStageFlagBits::eCompute, false);

    // 2. Descriptor set layout: bindings 0,1,2 = storage buffers in compute stage
    {
        std::array<vk::DescriptorSetLayoutBinding, 3> b = {{
            {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
            {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
            {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        }};
        vk::DescriptorSetLayoutCreateInfo li{};
        li.bindingCount = static_cast<uint32_t>(b.size());
        li.pBindings    = b.data();
        dsl_ = device_.createDescriptorSetLayoutUnique(li);
    }

    // 3. Pipeline layout: one descriptor set + push constants (4 × int32 = 16 bytes)
    {
        vk::PushConstantRange pcr{vk::ShaderStageFlagBits::eCompute, 0, 16};
        vk::PipelineLayoutCreateInfo li{};
        li.setLayoutCount         = 1;
        li.pSetLayouts            = &*dsl_;
        li.pushConstantRangeCount = 1;
        li.pPushConstantRanges    = &pcr;
        pipeLayout_ = device_.createPipelineLayoutUnique(li);
    }

    // 4. Compute pipeline
    {
        vk::PipelineShaderStageCreateInfo ss{};
        ss.stage  = vk::ShaderStageFlagBits::eCompute;
        ss.module = compMod_.getShaderModule();
        ss.pName  = "main";

        vk::ComputePipelineCreateInfo ci{};
        ci.stage  = ss;
        ci.layout = *pipeLayout_;

        auto [res, pipe] = device_.createComputePipelineUnique(nullptr, ci);
        if (res != vk::Result::eSuccess)
            throw std::runtime_error("PackingPass: compute pipeline creation failed");
        pipeline_ = std::move(pipe);
    }

    // 5. Descriptor pool (3 storage buffers, 1 set)
    {
        vk::DescriptorPoolSize ps{vk::DescriptorType::eStorageBuffer, 3};
        vk::DescriptorPoolCreateInfo pi{};
        pi.maxSets       = 1;
        pi.poolSizeCount = 1;
        pi.pPoolSizes    = &ps;
        descPool_ = device_.createDescriptorPoolUnique(pi);
    }

    // 6. Allocate descriptor set
    {
        vk::DescriptorSetAllocateInfo ai{};
        ai.descriptorPool     = *descPool_;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts        = &*dsl_;
        descSet_ = device_.allocateDescriptorSets(ai)[0];
    }

    gpuReady_ = true;
}

// ─── GPU helpers ─────────────────────────────────────────────────────────────

void PackingPass::resizeBuf(GpuBuf& b, vk::DeviceSize bytes) {
    if (b.size == bytes) return;
    auto alloc = reinterpret_cast<VmaAllocator>(allocator_);
    b.buf  = LavaCake::Buffer(device_, alloc, bytes,
                              vk::BufferUsageFlagBits::eStorageBuffer,
                              vk::AllocationCreateFlagBits::eCreateHostAccessSequentialWrite);
    b.ptr  = static_cast<float*>(b.buf.map());
    b.size = bytes;
}

void PackingPass::updateDescSet() {
    std::array<vk::DescriptorBufferInfo, 3> infos = {{
        {globalHM_.buf,    0, vk::WholeSize},
        {partZMinBuf_.buf, 0, vk::WholeSize},
        {candidates_.buf,  0, vk::WholeSize},
    }};
    std::array<vk::WriteDescriptorSet, 3> writes = {{
        {descSet_, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &infos[0]},
        {descSet_, 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &infos[1]},
        {descSet_, 2, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &infos[2]},
    }};
    device_.updateDescriptorSets(writes, {});
}

// ─── buildHeightmaps ─────────────────────────────────────────────────────────

void PackingPass::buildHeightmaps(
    const Eigen::MatrixXf& Vr, float resolution,
    std::vector<float>& zMin, std::vector<float>& zMax,
    int& partW, int& partH, float& xOff, float& yOff)
{
    xOff = Vr.col(0).minCoeff();
    yOff = Vr.col(1).minCoeff();
    // Normalize Z: subtract part Z-min so the lowest vertex maps to 0.
    // Without this, the drop-test sees a non-zero "interference" even on an
    // empty plate, causing the part to float above the bed.
    const float zOff = Vr.col(2).minCoeff();

    partW = std::max(1, static_cast<int>(std::ceil(
        (Vr.col(0).maxCoeff() - xOff) / resolution)));
    partH = std::max(1, static_cast<int>(std::ceil(
        (Vr.col(1).maxCoeff() - yOff) / resolution)));

    zMin.assign(partW * partH,  FLT_MAX);
    zMax.assign(partW * partH, -FLT_MAX);

    for (int v = 0; v < Vr.rows(); ++v) {
        int px = std::min(static_cast<int>((Vr(v, 0) - xOff) / resolution), partW - 1);
        int py = std::min(static_cast<int>((Vr(v, 1) - yOff) / resolution), partH - 1);
        float z = Vr(v, 2) - zOff; // relative to part bottom
        int idx  = py * partW + px;
        zMin[idx] = std::min(zMin[idx], z);
        zMax[idx] = std::max(zMax[idx], z);
    }

    // Vertex splatting leaves pixels with no vertex sample at FLT_MAX / -FLT_MAX.
    // Those holes cause the drop-test to skip interior cells, allowing a second
    // part to slip into a position that overlaps the first.  Fill them
    // conservatively: zMin → 0 (part bottom, worst-case collision height),
    // zMax → observed global top (worst-case stamp height).
    float topFill = 0.f;
    for (float v : zMax) if (v != -FLT_MAX) topFill = std::max(topFill, v);
    for (int k = 0; k < partW * partH; ++k) {
        if (zMin[k] ==  FLT_MAX) zMin[k] = 0.f;
        if (zMax[k] == -FLT_MAX) zMax[k] = topFill;
    }
}

// ─── pack ─────────────────────────────────────────────────────────────────────

std::vector<PackingPass::PlacedInstance> PackingPass::pack(
    const std::vector<Eigen::MatrixXf>& rotatedV,
    const std::vector<Eigen::MatrixXi>& /*faces*/,
    const std::vector<uint32_t>&         instMeshIdx,
    const std::vector<Eigen::Vector3f>&  instAABBMin,
    const std::vector<Eigen::Vector3f>&  instAABBMax,
    const Params&                        p,
    const std::vector<Eigen::MatrixXf>&  fixedWorldVerts)
{
    const uint32_t N = static_cast<uint32_t>(instMeshIdx.size());
    std::vector<PlacedInstance> result(N);
    for (uint32_t i = 0; i < N; ++i) result[i] = {i, 0.f, 0.f, 0.f};

    std::cerr << std::fixed << std::setprecision(1);
    std::cerr << "[packing] " << N << " instance(s)"
              << (gpuReady_ ? "  [GPU]" : "  [CPU fallback]") << "\n";

    if (p.plateX <= 0.f || p.plateY <= 0.f) {
        std::cerr << "[packing] plate dimensions not set — skipping\n";
        return result;
    }

    const int plateW = static_cast<int>(std::ceil(p.plateX / p.resolution));
    const int plateH = static_cast<int>(std::ceil(p.plateY / p.resolution));

    // ── Per-mesh heightmap cache ──────────────────────────────────────────────
    struct MeshMaps {
        std::vector<float> zMin, zMax;
        int   w = 0, h = 0;
        float xOff = 0.f, yOff = 0.f;
        float footprintArea = 0.f;
    };
    const uint32_t M = static_cast<uint32_t>(rotatedV.size());
    std::vector<MeshMaps> mm(M);
    for (uint32_t m = 0; m < M; ++m) {
        auto t0 = Clock::now();
        buildHeightmaps(rotatedV[m], p.resolution,
                        mm[m].zMin, mm[m].zMax,
                        mm[m].w, mm[m].h, mm[m].xOff, mm[m].yOff);
        mm[m].footprintArea = static_cast<float>(mm[m].w * mm[m].h);
        std::cerr << "  heightmap[" << m << "]:   " << msec(t0) << " ms"
                  << "  (" << mm[m].w << "×" << mm[m].h << " px)\n";
    }

    // ── Sort instances by footprint area descending ───────────────────────────
    auto t0 = Clock::now();
    std::vector<uint32_t> order(N);
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        return mm[instMeshIdx[a]].footprintArea > mm[instMeshIdx[b]].footprintArea;
    });
    std::cerr << "  sort:            " << msec(t0) << " ms\n";

    // ── Initialise global heightmap ───────────────────────────────────────────
    if (gpuReady_) {
        resizeBuf(globalHM_,   static_cast<vk::DeviceSize>(plateW) * plateH * sizeof(float));
        resizeBuf(candidates_, static_cast<vk::DeviceSize>(plateW) * plateH * sizeof(float));
        gmW_ = plateW; gmH_ = plateH;
    }
    // CPU heightmap — always kept for CPU fallback and for the stamp step.
    std::vector<float> G_cpu(plateW * plateH, p.bedOffset);
    if (gpuReady_)
        std::fill(globalHM_.ptr, globalHM_.ptr + plateW * plateH, p.bedOffset);

    // ── Stamp position-locked instances as fixed obstacles ───────────────────
    // Before anything gets placed, so the placement loop below naturally
    // avoids them (same stamp mechanism a freshly-placed part uses on itself).
    for (const auto& Vw : fixedWorldVerts) {
        std::vector<float> zMinF, zMaxF;
        int wF = 0, hF = 0;
        float xOffF = 0.f, yOffF = 0.f;
        buildHeightmaps(Vw, p.resolution, zMinF, zMaxF, wF, hF, xOffF, yOffF);
        // Vw is already in world space, so buildHeightmaps' own xOff/yOff
        // (= min X/Y of whatever vertices it's given) directly IS the plate
        // offset — no relative-to-absolute translation needed, unlike the
        // drop-test below which searches for a placement.
        const int   bestPxF = static_cast<int>(std::round(xOffF / p.resolution));
        const int   bestPyF = static_cast<int>(std::round(yOffF / p.resolution));
        const float bestZF  = Vw.col(2).minCoeff();
        stampHeightmap(G_cpu, gpuReady_ ? globalHM_.ptr : nullptr, gpuReady_,
                       plateW, plateH, zMaxF, wF, hF, bestPxF, bestPyF, bestZF,
                       p.clearance, p.resolution);
    }

    // ── Main packing loop ─────────────────────────────────────────────────────
    for (uint32_t ii = 0; ii < N; ++ii) {
        const uint32_t     instIdx = order[ii];
        const MeshMaps&    msh     = mm[instMeshIdx[instIdx]];

        if (msh.w > plateW || msh.h > plateH) {
            std::cerr << "  instance " << instIdx
                      << " footprint (" << msh.w << "×" << msh.h
                      << " px) exceeds build plate — skipped\n";
            continue;
        }

        // ── Drop test ─────────────────────────────────────────────────────────
        int   bestPx = 0, bestPy = 0;
        float bestZ  = FLT_MAX;

        if (gpuReady_) {
            // Upload zMin to GPU buffer.
            t0 = Clock::now();
            const vk::DeviceSize partBytes =
                static_cast<vk::DeviceSize>(msh.w) * msh.h * sizeof(float);
            resizeBuf(partZMinBuf_, partBytes);
            std::memcpy(partZMinBuf_.ptr, msh.zMin.data(),
                        static_cast<size_t>(partBytes));
            updateDescSet();
            std::cerr << "  upload[" << instIdx << "]:     " << msec(t0) << " ms\n";

            // Dispatch compute.
            t0 = Clock::now();
            vk::CommandBufferAllocateInfo cbai{};
            cbai.commandPool        = cmdPool_;
            cbai.level              = vk::CommandBufferLevel::ePrimary;
            cbai.commandBufferCount = 1;
            vk::CommandBuffer cmd = device_.allocateCommandBuffers(cbai)[0];

            vk::CommandBufferBeginInfo cbbi{vk::CommandBufferUsageFlagBits::eOneTimeSubmit};
            cmd.begin(cbbi);

            // Ensure CPU stamp writes to globalHM_ are visible to the compute shader.
            vk::MemoryBarrier mb{vk::AccessFlagBits::eHostWrite,
                                 vk::AccessFlagBits::eShaderRead |
                                 vk::AccessFlagBits::eShaderWrite};
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eHost,
                                vk::PipelineStageFlagBits::eComputeShader,
                                {}, mb, {}, {});

            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline_);
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                   *pipeLayout_, 0, descSet_, {});

            struct PC { int32_t plateW, plateH, partW, partH; } pc{
                plateW, plateH, msh.w, msh.h
            };
            cmd.pushConstants(*pipeLayout_, vk::ShaderStageFlagBits::eCompute,
                              0, sizeof(pc), &pc);

            const uint32_t gx = (static_cast<uint32_t>(plateW) + 15u) / 16u;
            const uint32_t gy = (static_cast<uint32_t>(plateH) + 15u) / 16u;
            cmd.dispatch(gx, gy, 1);

            // Ensure compute writes to candidates_ are visible to the host.
            vk::MemoryBarrier mb2{vk::AccessFlagBits::eShaderWrite,
                                  vk::AccessFlagBits::eHostRead};
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                vk::PipelineStageFlagBits::eHost,
                                {}, mb2, {}, {});
            cmd.end();

            vk::FenceCreateInfo fci{};
            auto fence = device_.createFenceUnique(fci);
            vk::SubmitInfo si{};
            si.commandBufferCount = 1;
            si.pCommandBuffers    = &cmd;
            queue_.submit(si, *fence);
            static_cast<void>(
                device_.waitForFences(*fence, VK_TRUE,
                                      std::numeric_limits<uint64_t>::max()));
            device_.freeCommandBuffers(cmdPool_, cmd);

            std::cerr << "  drop-test[" << instIdx << "]: " << msec(t0) << " ms"
                      << "  (GPU " << plateW << "×" << plateH
                      << " × " << msh.w << "×" << msh.h << " px)\n";

            // Find minimum in candidates (CPU, O(plateW*plateH) sequential).
            t0 = Clock::now();
            const float* cptr = candidates_.ptr;
            const int    total = plateW * plateH;
            for (int k = 0; k < total; ++k) {
                if (cptr[k] < bestZ) {
                    bestZ  = cptr[k];
                    bestPx = k % plateW;
                    bestPy = k / plateW;
                }
            }
            std::cerr << "  min-find[" << instIdx << "]:  " << msec(t0) << " ms\n";

        } else {
            // ── CPU fallback (slow for large plates/parts) ─────────────────────
            t0 = Clock::now();
            const int maxPx = plateW - msh.w;
            const int maxPy = plateH - msh.h;
            for (int py = 0; py <= maxPy; ++py) {
                for (int px = 0; px <= maxPx; ++px) {
                    float reqZ = 0.f;
                    for (int j = 0; j < msh.h; ++j)
                        for (int i = 0; i < msh.w; ++i) {
                            float zm = msh.zMin[j * msh.w + i];
                            if (zm == FLT_MAX) continue;
                            reqZ = std::max(reqZ, G_cpu[(py + j) * plateW + (px + i)] - zm);
                        }
                    if (reqZ < bestZ) { bestZ = reqZ; bestPx = px; bestPy = py; }
                }
            }
            std::cerr << "  drop-test[" << instIdx << "]: " << msec(t0) << " ms"
                      << "  (CPU — consider enabling GPU)\n";
        }

        if (bestZ == FLT_MAX) {
            std::cerr << "  instance " << instIdx << " could not be placed\n";
            continue;
        }

        // ── Stamp G with XY clearance expansion ───────────────────────────────
        t0 = Clock::now();
        const float zTop = stampHeightmap(G_cpu, gpuReady_ ? globalHM_.ptr : nullptr, gpuReady_,
                                          plateW, plateH, msh.zMax, msh.w, msh.h,
                                          bestPx, bestPy, bestZ, p.clearance, p.resolution);
        const int cp = (p.clearance > 0.f)
            ? static_cast<int>(std::ceil(p.clearance / p.resolution)) : 0;
        const float absTop = bestZ + zTop;
        std::cerr << "  stamp[" << instIdx << "]:      " << msec(t0) << " ms"
                  << "  z=" << bestZ << " mm  zTop=" << absTop << " mm"
                  << "  cp=" << cp << " px\n";

        if (p.plateZ > 0.f && absTop > p.plateZ)
            std::cerr << "  WARNING: instance " << instIdx
                      << " exceeds build height (" << absTop
                      << " mm > " << p.plateZ << " mm)\n";

        // ── Translation delta ──────────────────────────────────────────────────
        const float tx = static_cast<float>(bestPx) * p.resolution - msh.xOff;
        const float ty = static_cast<float>(bestPy) * p.resolution - msh.yOff;
        const float tz = bestZ - instAABBMin[instIdx].z();
        result[instIdx] = {instIdx, tx, ty, tz};

        std::cerr << "  placed[" << instIdx << "]:     ("
                  << bestPx << "," << bestPy << ") px"
                  << "  Δ=(" << tx << "," << ty << "," << tz << ") mm\n";
    }

    // ── Center the packed scene on the build plate ───────────────────────────
    {
        float sxMin = FLT_MAX, sxMax = -FLT_MAX;
        float syMin = FLT_MAX, syMax = -FLT_MAX;
        for (uint32_t i = 0; i < N; ++i) {
            sxMin = std::min(sxMin, instAABBMin[i].x() + result[i].tx);
            sxMax = std::max(sxMax, instAABBMax[i].x() + result[i].tx);
            syMin = std::min(syMin, instAABBMin[i].y() + result[i].ty);
            syMax = std::max(syMax, instAABBMax[i].y() + result[i].ty);
        }
        if (sxMin < sxMax && syMin < syMax) {
            const float dx = p.plateX * 0.5f - (sxMin + sxMax) * 0.5f;
            const float dy = p.plateY * 0.5f - (syMin + syMax) * 0.5f;
            for (uint32_t i = 0; i < N; ++i) {
                result[i].tx += dx;
                result[i].ty += dy;
            }
            std::cerr << "  center:          Δ=(" << dx << "," << dy << ") mm\n";
        }
    }

    return result;
}
