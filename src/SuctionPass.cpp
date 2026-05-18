#include "SuctionPass.hpp"

#include <Eigen/Geometry>
#include <shaderc/shaderc.hpp>

#include <array>
#include <cstring>
#include <stdexcept>
#include <vector>

// ── Embedded GLSL sources ─────────────────────────────────────────────────────

static const char kSuctionVertGlsl[] = R"glsl(
#version 450
layout(location = 0) in vec3 inPos;
layout(push_constant) uniform PC { mat4 mvp; } pc;
layout(location = 0) out vec3 fragWorldPos;
void main() {
    fragWorldPos = inPos;
    gl_Position  = pc.mvp * vec4(inPos, 1.0);
}
)glsl";

static const char kSuctionFragGlsl[] = R"glsl(
#version 450
layout(location = 0) in  vec3  fragWorldPos;
layout(location = 0) out float outSuction;
void main() {
    vec3 N = normalize(cross(dFdx(fragWorldPos), dFdy(fragWorldPos)));
    outSuction = (N.z > 0.0) ? 1.0 : 0.0;
}
)glsl";

// ── Helpers ───────────────────────────────────────────────────────────────────

static vk::UniqueShaderModule compileGlsl(vk::Device device, const char* src,
                                          shaderc_shader_kind kind, const char* name) {
    shaderc::Compiler compiler;
    shaderc::CompileOptions opts;
    opts.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_3);
    opts.SetOptimizationLevel(shaderc_optimization_level_performance);
    auto result = compiler.CompileGlslToSpv(src, kind, name, opts);
    if (result.GetCompilationStatus() != shaderc_compilation_status_success)
        throw std::runtime_error(std::string("SuctionPass: shader compile error (") +
                                 name + "): " + result.GetErrorMessage());
    std::vector<uint32_t> spv(result.cbegin(), result.cend());
    vk::ShaderModuleCreateInfo ci{};
    ci.codeSize = spv.size() * sizeof(uint32_t);
    ci.pCode    = spv.data();
    return device.createShaderModuleUnique(ci);
}

uint32_t SuctionPass::findMemType(vk::PhysicalDevice pd,
                                  uint32_t           typeBits,
                                  vk::MemoryPropertyFlags props) const
{
    auto mem = pd.getMemoryProperties();
    for (uint32_t i = 0; i < mem.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) &&
            (mem.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    throw std::runtime_error("SuctionPass: no suitable memory type");
}

// ── init ──────────────────────────────────────────────────────────────────────

void SuctionPass::init(const GpuContext& ctx) {
    device_ = ctx.device;

    // 1. Shader modules
    {
        vertMod_ = compileGlsl(ctx.device, kSuctionVertGlsl, shaderc_vertex_shader,   "suction.vert");
        fragMod_ = compileGlsl(ctx.device, kSuctionFragGlsl, shaderc_fragment_shader, "suction.frag");
    }

    // 2. Pipeline layout — one push constant: mat4 (64 bytes) in vertex stage
    {
        vk::PushConstantRange pcr{vk::ShaderStageFlagBits::eVertex, 0, 64};
        vk::PipelineLayoutCreateInfo li{};
        li.pushConstantRangeCount = 1;
        li.pPushConstantRanges    = &pcr;
        layout_ = ctx.device.createPipelineLayoutUnique(li);
    }

    // 3. Graphics pipeline (dynamic rendering — no VkRenderPass)
    {
        // Shader stages
        std::array<vk::PipelineShaderStageCreateInfo, 2> stages{};
        stages[0].stage  = vk::ShaderStageFlagBits::eVertex;
        stages[0].module = *vertMod_;
        stages[0].pName  = "main";
        stages[1].stage  = vk::ShaderStageFlagBits::eFragment;
        stages[1].module = *fragMod_;
        stages[1].pName  = "main";

        // Vertex input: binding 0, stride 12, vec3 at location 0
        vk::VertexInputBindingDescription vbd{0, 12, vk::VertexInputRate::eVertex};
        vk::VertexInputAttributeDescription vad{0, 0, vk::Format::eR32G32B32Sfloat, 0};
        vk::PipelineVertexInputStateCreateInfo vis{};
        vis.vertexBindingDescriptionCount   = 1;
        vis.pVertexBindingDescriptions      = &vbd;
        vis.vertexAttributeDescriptionCount = 1;
        vis.pVertexAttributeDescriptions    = &vad;

        vk::PipelineInputAssemblyStateCreateInfo ias{};
        ias.topology = vk::PrimitiveTopology::eTriangleList;

        // Dynamic viewport + scissor
        vk::PipelineViewportStateCreateInfo vs{};
        vs.viewportCount = 1;
        vs.scissorCount  = 1;
        std::array<vk::DynamicState, 2> dynStates{vk::DynamicState::eViewport,
                                                   vk::DynamicState::eScissor};
        vk::PipelineDynamicStateCreateInfo dyn{};
        dyn.dynamicStateCount = static_cast<uint32_t>(dynStates.size());
        dyn.pDynamicStates    = dynStates.data();

        vk::PipelineRasterizationStateCreateInfo rast{};
        rast.polygonMode = vk::PolygonMode::eFill;
        rast.cullMode    = vk::CullModeFlagBits::eNone; // see both faces
        rast.frontFace   = vk::FrontFace::eCounterClockwise;
        rast.lineWidth   = 1.0f;

        vk::PipelineMultisampleStateCreateInfo ms{};
        ms.rasterizationSamples = vk::SampleCountFlagBits::e1;

        vk::PipelineDepthStencilStateCreateInfo ds{};
        ds.depthTestEnable  = VK_TRUE;
        ds.depthWriteEnable = VK_TRUE;
        ds.depthCompareOp   = vk::CompareOp::eLess;

        // Color blend: single R32_SFLOAT attachment, no blending
        vk::PipelineColorBlendAttachmentState cba{};
        cba.colorWriteMask = vk::ColorComponentFlagBits::eR;
        vk::PipelineColorBlendStateCreateInfo cbs{};
        cbs.attachmentCount = 1;
        cbs.pAttachments    = &cba;

        // Dynamic rendering formats chained via pNext
        constexpr vk::Format kColorFmt = vk::Format::eR32Sfloat;
        constexpr vk::Format kDepthFmt = vk::Format::eD32Sfloat;
        vk::PipelineRenderingCreateInfo prc{};
        prc.colorAttachmentCount    = 1;
        prc.pColorAttachmentFormats = &kColorFmt;
        prc.depthAttachmentFormat   = kDepthFmt;

        vk::GraphicsPipelineCreateInfo pci{};
        pci.pNext               = &prc;
        pci.stageCount          = static_cast<uint32_t>(stages.size());
        pci.pStages             = stages.data();
        pci.pVertexInputState   = &vis;
        pci.pInputAssemblyState = &ias;
        pci.pViewportState      = &vs;
        pci.pRasterizationState = &rast;
        pci.pMultisampleState   = &ms;
        pci.pDepthStencilState  = &ds;
        pci.pColorBlendState    = &cbs;
        pci.pDynamicState       = &dyn;
        pci.layout              = *layout_;
        pci.renderPass          = nullptr; // dynamic rendering

        auto result = ctx.device.createGraphicsPipelineUnique(nullptr, pci);
        if (result.result != vk::Result::eSuccess)
            throw std::runtime_error("SuctionPass: failed to create pipeline");
        pipeline_ = std::move(result.value);
    }
}

SuctionPass::~SuctionPass() = default;

// ── resizeImages ─────────────────────────────────────────────────────────────

void SuctionPass::resizeImages(const GpuContext& ctx, uint32_t w, uint32_t h) {
    // Color image: R32_SFLOAT, eColorAttachment | eTransferSrc
    {
        vk::ImageCreateInfo ici{};
        ici.imageType   = vk::ImageType::e2D;
        ici.format      = vk::Format::eR32Sfloat;
        ici.extent      = vk::Extent3D{w, h, 1};
        ici.mipLevels   = 1;
        ici.arrayLayers = 1;
        ici.samples     = vk::SampleCountFlagBits::e1;
        ici.tiling      = vk::ImageTiling::eOptimal;
        ici.usage       = vk::ImageUsageFlagBits::eColorAttachment |
                          vk::ImageUsageFlagBits::eTransferSrc;
        ici.initialLayout = vk::ImageLayout::eUndefined;
        colorImg_ = ctx.device.createImageUnique(ici);

        auto req = ctx.device.getImageMemoryRequirements(*colorImg_);
        vk::MemoryAllocateInfo ai{req.size,
            findMemType(ctx.physDevice, req.memoryTypeBits,
                        vk::MemoryPropertyFlagBits::eDeviceLocal)};
        colorMem_ = ctx.device.allocateMemoryUnique(ai);
        ctx.device.bindImageMemory(*colorImg_, *colorMem_, 0);

        vk::ImageViewCreateInfo vci{};
        vci.image    = *colorImg_;
        vci.viewType = vk::ImageViewType::e2D;
        vci.format   = vk::Format::eR32Sfloat;
        vci.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
        colorView_ = ctx.device.createImageViewUnique(vci);
    }

    // Depth image: D32_SFLOAT
    {
        vk::ImageCreateInfo ici{};
        ici.imageType   = vk::ImageType::e2D;
        ici.format      = vk::Format::eD32Sfloat;
        ici.extent      = vk::Extent3D{w, h, 1};
        ici.mipLevels   = 1;
        ici.arrayLayers = 1;
        ici.samples     = vk::SampleCountFlagBits::e1;
        ici.tiling      = vk::ImageTiling::eOptimal;
        ici.usage       = vk::ImageUsageFlagBits::eDepthStencilAttachment;
        ici.initialLayout = vk::ImageLayout::eUndefined;
        depthImg_ = ctx.device.createImageUnique(ici);

        auto req = ctx.device.getImageMemoryRequirements(*depthImg_);
        vk::MemoryAllocateInfo ai{req.size,
            findMemType(ctx.physDevice, req.memoryTypeBits,
                        vk::MemoryPropertyFlagBits::eDeviceLocal)};
        depthMem_ = ctx.device.allocateMemoryUnique(ai);
        ctx.device.bindImageMemory(*depthImg_, *depthMem_, 0);

        vk::ImageViewCreateInfo vci{};
        vci.image    = *depthImg_;
        vci.viewType = vk::ImageViewType::e2D;
        vci.format   = vk::Format::eD32Sfloat;
        vci.subresourceRange = {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1};
        depthView_ = ctx.device.createImageViewUnique(vci);
    }

    // Readback buffer: host-visible, w*h*sizeof(float)
    {
        vk::BufferCreateInfo bci{};
        bci.size  = static_cast<vk::DeviceSize>(w) * h * sizeof(float);
        bci.usage = vk::BufferUsageFlagBits::eTransferDst;
        readbackBuf_ = ctx.device.createBufferUnique(bci);

        auto req = ctx.device.getBufferMemoryRequirements(*readbackBuf_);
        vk::MemoryAllocateInfo ai{req.size,
            findMemType(ctx.physDevice, req.memoryTypeBits,
                        vk::MemoryPropertyFlagBits::eHostVisible |
                        vk::MemoryPropertyFlagBits::eHostCoherent)};
        readbackMem_ = ctx.device.allocateMemoryUnique(ai);
        ctx.device.bindBufferMemory(*readbackBuf_, *readbackMem_, 0);
    }

    imgW_ = w;
    imgH_ = h;
}

// ── render ────────────────────────────────────────────────────────────────────

int SuctionPass::render(const GpuContext& ctx,
                        const Eigen::MatrixXf& Vr,
                        const Eigen::MatrixXi& F)
{
    if (!pipeline_) return 0;

    const int nv = static_cast<int>(Vr.rows());
    const int nf = static_cast<int>(F.rows());
    if (nv == 0 || nf == 0) return 0;

    // Mesh footprint → framebuffer size (1 mm/pixel)
    float xMin = Vr.col(0).minCoeff(), xMax = Vr.col(0).maxCoeff();
    float yMin = Vr.col(1).minCoeff(), yMax = Vr.col(1).maxCoeff();
    float zMin = Vr.col(2).minCoeff(), zMax = Vr.col(2).maxCoeff();

    auto w = static_cast<uint32_t>(std::max(1.f, std::ceil(xMax - xMin)));
    auto h = static_cast<uint32_t>(std::max(1.f, std::ceil(yMax - yMin)));
    w = std::min(w, 4096u);
    h = std::min(h, 4096u);

    if (w != imgW_ || h != imgH_)
        resizeImages(ctx, w, h);

    // ── Upload vertex + index buffers (host-visible, one-shot) ───────────────

    // Row-major Nv*3 float array for the vertex buffer
    std::vector<float> vdata(nv * 3);
    for (int i = 0; i < nv; ++i) {
        vdata[i * 3 + 0] = Vr(i, 0);
        vdata[i * 3 + 1] = Vr(i, 1);
        vdata[i * 3 + 2] = Vr(i, 2);
    }
    std::vector<uint32_t> idata(nf * 3);
    for (int i = 0; i < nf; ++i) {
        idata[i * 3 + 0] = static_cast<uint32_t>(F(i, 0));
        idata[i * 3 + 1] = static_cast<uint32_t>(F(i, 1));
        idata[i * 3 + 2] = static_cast<uint32_t>(F(i, 2));
    }

    auto makeHostBuf = [&](const void* src, vk::DeviceSize sz,
                           vk::BufferUsageFlags usage)
        -> std::pair<vk::UniqueBuffer, vk::UniqueDeviceMemory>
    {
        vk::BufferCreateInfo bci{};
        bci.size  = sz;
        bci.usage = usage;
        auto buf = ctx.device.createBufferUnique(bci);
        auto req = ctx.device.getBufferMemoryRequirements(*buf);
        vk::MemoryAllocateInfo ai{req.size,
            findMemType(ctx.physDevice, req.memoryTypeBits,
                        vk::MemoryPropertyFlagBits::eHostVisible |
                        vk::MemoryPropertyFlagBits::eHostCoherent)};
        auto mem = ctx.device.allocateMemoryUnique(ai);
        ctx.device.bindBufferMemory(*buf, *mem, 0);
        void* mapped = ctx.device.mapMemory(*mem, 0, sz);
        std::memcpy(mapped, src, sz);
        ctx.device.unmapMemory(*mem);
        return {std::move(buf), std::move(mem)};
    };

    auto [vbuf, vmem] = makeHostBuf(vdata.data(),
                                    vdata.size() * sizeof(float),
                                    vk::BufferUsageFlagBits::eVertexBuffer);
    auto [ibuf, imem] = makeHostBuf(idata.data(),
                                    idata.size() * sizeof(uint32_t),
                                    vk::BufferUsageFlagBits::eIndexBuffer);

    // ── Orthographic MVP ─────────────────────────────────────────────────────
    // Maps [xMin,xMax]×[yMin,yMax]×[zMin,zMax] → NDC [-1,1]×[-1,1]×[0,1].
    // Column-major Eigen::Matrix4f matches Vulkan's column-major push constant.
    float sx = 2.f / (xMax - xMin);
    float sy = 2.f / (yMax - yMin);
    float sz2 = 1.f / (std::max(zMax - zMin, 1e-6f));
    float tx = -(xMax + xMin) / (xMax - xMin);
    float ty = -(yMax + yMin) / (yMax - yMin);
    float tz = -zMin / (std::max(zMax - zMin, 1e-6f));

    // Column-major: mvp[col][row]
    float mvp[16] = {
        sx, 0,  0,  0,
        0,  sy, 0,  0,
        0,  0,  sz2,0,
        tx, ty, tz, 1
    };

    // ── Record command buffer ────────────────────────────────────────────────
    vk::CommandBufferAllocateInfo cbai{};
    cbai.commandPool        = ctx.cmdPool;
    cbai.level              = vk::CommandBufferLevel::ePrimary;
    cbai.commandBufferCount = 1;
    auto cmds = ctx.device.allocateCommandBuffers(cbai);
    vk::CommandBuffer cmd = cmds[0];

    vk::CommandBufferBeginInfo cbbi{vk::CommandBufferUsageFlagBits::eOneTimeSubmit};
    cmd.begin(cbbi);

    // Transition color → eColorAttachmentOptimal
    {
        vk::ImageMemoryBarrier b{};
        b.oldLayout        = vk::ImageLayout::eUndefined;
        b.newLayout        = vk::ImageLayout::eColorAttachmentOptimal;
        b.image            = *colorImg_;
        b.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
        b.dstAccessMask    = vk::AccessFlagBits::eColorAttachmentWrite;
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                            vk::PipelineStageFlagBits::eColorAttachmentOutput,
                            {}, {}, {}, b);
    }
    // Transition depth → eDepthStencilAttachmentOptimal
    {
        vk::ImageMemoryBarrier b{};
        b.oldLayout        = vk::ImageLayout::eUndefined;
        b.newLayout        = vk::ImageLayout::eDepthStencilAttachmentOptimal;
        b.image            = *depthImg_;
        b.subresourceRange = {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1};
        b.dstAccessMask    = vk::AccessFlagBits::eDepthStencilAttachmentWrite;
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                            vk::PipelineStageFlagBits::eEarlyFragmentTests,
                            {}, {}, {}, b);
    }

    // Begin dynamic rendering
    vk::RenderingAttachmentInfo colorAtt{};
    colorAtt.imageView   = *colorView_;
    colorAtt.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
    colorAtt.loadOp      = vk::AttachmentLoadOp::eClear;
    colorAtt.storeOp     = vk::AttachmentStoreOp::eStore;
    colorAtt.clearValue.color = vk::ClearColorValue{std::array<float,4>{0,0,0,0}};

    vk::RenderingAttachmentInfo depthAtt{};
    depthAtt.imageView   = *depthView_;
    depthAtt.imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
    depthAtt.loadOp      = vk::AttachmentLoadOp::eClear;
    depthAtt.storeOp     = vk::AttachmentStoreOp::eDontCare;
    depthAtt.clearValue.depthStencil = vk::ClearDepthStencilValue{1.0f, 0};

    vk::RenderingInfo ri{};
    ri.renderArea           = vk::Rect2D{{0, 0}, {w, h}};
    ri.layerCount           = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments    = &colorAtt;
    ri.pDepthAttachment     = &depthAtt;
    cmd.beginRendering(ri);

    vk::Viewport vp{0, 0, static_cast<float>(w), static_cast<float>(h), 0, 1};
    cmd.setViewport(0, vp);
    vk::Rect2D sc{{0, 0}, {w, h}};
    cmd.setScissor(0, sc);

    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline_);
    vk::DeviceSize zero = 0;
    cmd.bindVertexBuffers(0, *vbuf, zero);
    cmd.bindIndexBuffer(*ibuf, 0, vk::IndexType::eUint32);
    cmd.pushConstants(*layout_, vk::ShaderStageFlagBits::eVertex, 0, 64, mvp);
    cmd.drawIndexed(static_cast<uint32_t>(nf * 3), 1, 0, 0, 0);

    cmd.endRendering();

    // Transition color → eTransferSrcOptimal, then copy to readback buffer
    {
        vk::ImageMemoryBarrier b{};
        b.oldLayout     = vk::ImageLayout::eColorAttachmentOptimal;
        b.newLayout     = vk::ImageLayout::eTransferSrcOptimal;
        b.image         = *colorImg_;
        b.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
        b.srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
        b.dstAccessMask = vk::AccessFlagBits::eTransferRead;
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
                            vk::PipelineStageFlagBits::eTransfer,
                            {}, {}, {}, b);
    }
    {
        vk::BufferImageCopy region{};
        region.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
        region.imageExtent      = vk::Extent3D{w, h, 1};
        cmd.copyImageToBuffer(*colorImg_,
                              vk::ImageLayout::eTransferSrcOptimal,
                              *readbackBuf_, region);
    }
    // Ensure copy is visible to host
    {
        vk::BufferMemoryBarrier b{};
        b.buffer      = *readbackBuf_;
        b.offset      = 0;
        b.size        = VK_WHOLE_SIZE;
        b.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        b.dstAccessMask = vk::AccessFlagBits::eHostRead;
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                            vk::PipelineStageFlagBits::eHost,
                            {}, {}, b, {});
    }

    cmd.end();

    // Submit and wait
    vk::FenceCreateInfo fci{};
    auto fence = ctx.device.createFenceUnique(fci);
    vk::SubmitInfo si{};
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cmd;
    ctx.queue.submit(si, *fence);
    static_cast<void>(ctx.device.waitForFences(*fence, VK_TRUE,
                                               std::numeric_limits<uint64_t>::max()));

    ctx.device.freeCommandBuffers(ctx.cmdPool, cmd);

    // ── Count back-facing (suction) pixels ───────────────────────────────────
    const auto* pixels = static_cast<const float*>(
        ctx.device.mapMemory(*readbackMem_, 0, VK_WHOLE_SIZE));
    int count = 0;
    const int total = static_cast<int>(w * h);
    for (int i = 0; i < total; ++i)
        if (pixels[i] > 0.5f) ++count;
    ctx.device.unmapMemory(*readbackMem_);

    return count;
}
