#pragma once
#include <GpuContext.hpp>
#include <Eigen/Core>
#include <string>

/// Offscreen Vulkan pass that counts suction-cup pixels for a single candidate
/// rotation.  Uses dynamic rendering (no VkRenderPass / VkFramebuffer).
/// Completely LavaCake-free — uses vulkan-hpp directly with the handles from
/// GpuContext.
class SuctionPass {
public:
    /// One-time setup: builds the graphics pipeline from the pre-compiled SPIR-V
    /// shaders in shaderDir.  shaderDir must end with '/'.
    void init(const GpuContext& ctx, const std::string& shaderDir);

    ~SuctionPass();

    /// Rasterises the already-rotated mesh (Vr = R*V, F unchanged) from below
    /// (+Z direction) and counts back-facing pixels.  Returns the pixel count,
    /// or 0 if init() was not called.
    int render(const GpuContext& ctx,
               const Eigen::MatrixXf& Vr,
               const Eigen::MatrixXi& F);

private:
    vk::Device             device_  = {};
    vk::UniquePipeline     pipeline_;
    vk::UniquePipelineLayout layout_;
    vk::UniqueShaderModule vertMod_, fragMod_;

    // Per-frame resources, reallocated when image size changes.
    uint32_t imgW_ = 0, imgH_ = 0;
    vk::UniqueImage        colorImg_, depthImg_;
    vk::UniqueDeviceMemory colorMem_, depthMem_;
    vk::UniqueImageView    colorView_, depthView_;
    // Host-visible readback buffer (w*h*sizeof(float) bytes).
    vk::UniqueBuffer       readbackBuf_;
    vk::UniqueDeviceMemory readbackMem_;

    void resizeImages(const GpuContext& ctx, uint32_t w, uint32_t h);
    uint32_t findMemType(vk::PhysicalDevice pd,
                         uint32_t           typeBits,
                         vk::MemoryPropertyFlags props) const;
};
