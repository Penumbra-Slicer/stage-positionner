#pragma once
#include <GpuContext.hpp>
#include <Eigen/Core>
#include <cstdint>
#include <vector>
#include <vulkan/vulkan.hpp>

/// Greedy heightmap packer.
///
/// Algorithm (plan-packing.md):
///   1. Sort instances by XY footprint area (descending).
///   2. Maintain a global heightmap G over the build plate.
///   3. For each instance, run a drop-test: find the (x,y) position with the
///      lowest required Z lift.  GPU path: one compute dispatch per part.
///   4. Stamp G with the placed part's Z-Max profile.
///   5. Emit per-instance translation deltas.
///
/// keepPosition mode: preserves user XY, only lifts each part in Z so its
/// AABB floor sits at bedOffset.  No heightmap, no sorting.
class PackingPass {
public:
    struct Params {
        float plateX      = 0.0f;   ///< from data.printer.buildXMm
        float plateY      = 0.0f;   ///< from data.printer.buildYMm
        float plateZ      = 0.0f;   ///< from data.printer.buildZMm
        float resolution  = 0.1f;   ///< heightmap pixel size (mm/pixel)
        float clearance   = 0.5f;   ///< minimum gap between parts (mm)
        float bedOffset   = 0.0f;   ///< minimum Z floor for all parts (mm)
        bool  keepPosition = false; ///< skip XY packing; only lift Z to bedOffset
    };

    struct PlacedInstance {
        uint32_t instanceIdx;
        float tx, ty, tz; ///< world-space translation delta
    };

    /// One-time GPU setup.  Call before pack() to enable the compute path.
    /// Compiles the embedded GLSL compute shader at runtime via shaderc.
    void init(const GpuContext& ctx);

    ~PackingPass();

    /// Place all instances on the build plate.
    /// rotatedV[m] = already-rotated vertex matrix (Nv×3) for unique mesh m.
    std::vector<PlacedInstance> pack(
        const std::vector<Eigen::MatrixXf>& rotatedV,
        const std::vector<Eigen::MatrixXi>& faces,
        const std::vector<uint32_t>&         instMeshIdx,
        const std::vector<Eigen::Vector3f>&  instAABBMin,
        const std::vector<Eigen::Vector3f>&  instAABBMax,
        const Params&                        params);

private:
    // ── GPU state ─────────────────────────────────────────────────────────────
    vk::Device         device_   = {};
    vk::PhysicalDevice physDev_  = {};
    vk::Queue          queue_    = {};
    vk::CommandPool    cmdPool_  = {};
    bool               gpuReady_ = false;

    vk::UniqueShaderModule        compMod_;
    vk::UniqueDescriptorSetLayout dsl_;
    vk::UniquePipelineLayout      pipeLayout_;
    vk::UniquePipeline            pipeline_;
    vk::UniqueDescriptorPool      descPool_;
    vk::DescriptorSet             descSet_ = {};

    // Persistent host-visible+coherent SSBOs.
    // Persistently mapped: ptr stays valid until the buffer is resized.
    struct GpuBuf {
        vk::UniqueBuffer       buf;
        vk::UniqueDeviceMemory mem;
        vk::DeviceSize         size = 0;
        float*                 ptr  = nullptr;
    };
    GpuBuf globalHM_;    ///< G[plateW * plateH]  — persists across parts
    GpuBuf partZMinBuf_; ///< zMin[partW * partH] — updated per part
    GpuBuf candidates_;  ///< reqZ[plateW * plateH] — GPU-written, CPU-read

    int gmW_ = 0, gmH_ = 0; ///< current global heightmap dimensions in pixels

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Allocate or reallocate a host-visible+coherent SSBO, keeping it mapped.
    void resizeBuf(GpuBuf& b, vk::DeviceSize bytes);

    /// Rebind all three SSBOs into descSet_ after any buffer is resized.
    void updateDescSet();

    uint32_t findMemType(uint32_t typeBits, vk::MemoryPropertyFlags props) const;

    /// Build Z-Min and Z-Max heightmaps from rotated vertices via vertex
    /// bucketing.  Z values are normalized so the part's absolute Z-min → 0,
    /// eliminating the "floating part" bias in the drop-test formula.
    ///
    /// @param xOff  Output: world X of pixel (0,0) = part AABB X-min.
    /// @param yOff  Output: world Y of pixel (0,0) = part AABB Y-min.
    static void buildHeightmaps(
        const Eigen::MatrixXf& Vr, float resolution,
        std::vector<float>& zMin, std::vector<float>& zMax,
        int& partW, int& partH,
        float& xOff, float& yOff);
};
