#include "CubemapOffscreenRender.hpp"
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <fmt/core.h>
#include "stb_image_write.h"
#include <memory>

constexpr vk::VertexInputBindingDescription OffscreenVertex::getBindingDescription() {
    return vk::VertexInputBindingDescription{0, sizeof(OffscreenVertex), vk::VertexInputRate::eVertex};
}

constexpr std::array<vk::VertexInputAttributeDescription, 2> OffscreenVertex::getAttributeDescriptions() {
    return std::array<vk::VertexInputAttributeDescription, 2>{
        vk::VertexInputAttributeDescription{0, 0, vk::Format::eR32G32B32Sfloat, offsetof(OffscreenVertex, pos)},
        vk::VertexInputAttributeDescription{1, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(OffscreenVertex, color)}};
}

CubemapOffscreenRender::CubemapOffscreenRender(VulkanCore& core, std::shared_ptr<SharedRenderData> data)
    : vulkanCore(core),
      sharedData(data),
      offscreenVertexBuffer(nullptr),
      offscreenVertexBufferMemory(nullptr),
      offscreenIndexBuffer(nullptr),
      offscreenIndexBufferMemory(nullptr),
      textureImage(nullptr),
      textureImageMemory(nullptr),
      offscreenRenderPass(nullptr),
      offscreenPipeline(nullptr),
      offscreenPipelineLayout(nullptr) {}

CubemapOffscreenRender::~CubemapOffscreenRender() = default;

void CubemapOffscreenRender::execute() {
    createEmptyCubemapTexture();
    createOffscreenRenderPass();
    createOffscreenPipeline();
    createOffscreenVertexBuffer();
    createOffscreenIndexBuffer();
    createOffscreenFramebuffers();
    renderToCubemap();
    saveCubemapToPNG();
    if (sharedData) {
        sharedData->cubemapTexture = &textureImage;
    }
}

void CubemapOffscreenRender::createEmptyCubemapTexture() {
    const uint32_t cubemapSize = std::max(vulkanCore.swapChainExtent.width, vulkanCore.swapChainExtent.height);
    vulkanCore.createImage(cubemapSize, cubemapSize, vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal,
                           vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled |
                               vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst,
                           vk::MemoryPropertyFlagBits::eDeviceLocal, textureImage, textureImageMemory, 6,
                           vk::ImageCreateFlagBits::eCubeCompatible);
}

void CubemapOffscreenRender::createOffscreenVertexBuffer() {
    constexpr vk::DeviceSize bufferSize = sizeof(offscreenVertices[0]) * offscreenVertices.size();
    vk::raii::Buffer stagingBuffer = nullptr;
    vk::raii::DeviceMemory stagingBufferMemory = nullptr;
    vulkanCore.createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                            stagingBuffer, stagingBufferMemory);
    const auto data = stagingBufferMemory.mapMemory(0, bufferSize);
    std::memcpy(data, offscreenVertices.data(), bufferSize);
    stagingBufferMemory.unmapMemory();
    vulkanCore.createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
                            vk::MemoryPropertyFlagBits::eDeviceLocal, offscreenVertexBuffer,
                            offscreenVertexBufferMemory);
    vulkanCore.copyBuffer(*stagingBuffer, *offscreenVertexBuffer, bufferSize);
}

void CubemapOffscreenRender::createOffscreenIndexBuffer() {
    constexpr vk::DeviceSize bufferSize = sizeof(offscreenIndices[0]) * offscreenIndices.size();
    vk::raii::Buffer stagingBuffer = nullptr;
    vk::raii::DeviceMemory stagingBufferMemory = nullptr;
    vulkanCore.createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                            stagingBuffer, stagingBufferMemory);
    const auto data = stagingBufferMemory.mapMemory(0, bufferSize);
    std::memcpy(data, offscreenIndices.data(), bufferSize);
    stagingBufferMemory.unmapMemory();
    vulkanCore.createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
                            vk::MemoryPropertyFlagBits::eDeviceLocal, offscreenIndexBuffer, offscreenIndexBufferMemory);
    vulkanCore.copyBuffer(*stagingBuffer, *offscreenIndexBuffer, bufferSize);
}

void CubemapOffscreenRender::createOffscreenRenderPass() {
    vk::AttachmentDescription colorAttachment = {};
    colorAttachment.format = vk::Format::eR8G8B8A8Srgb;
    colorAttachment.samples = vk::SampleCountFlagBits::e1;
    colorAttachment.loadOp = vk::AttachmentLoadOp::eClear;
    colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;
    colorAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
    colorAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
    colorAttachment.initialLayout = vk::ImageLayout::eUndefined;
    colorAttachment.finalLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    vk::AttachmentReference colorAttachmentRef{0, vk::ImageLayout::eColorAttachmentOptimal};
    vk::SubpassDescription subpass{};
    subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;
    vk::SubpassDependency dependency{};
    dependency.srcSubpass = vk::SubpassExternal;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    dependency.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    dependency.srcAccessMask = {};
    dependency.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
    vk::RenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = vk::StructureType::eRenderPassCreateInfo;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &colorAttachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;
    offscreenRenderPass = vulkanCore.device.createRenderPass(renderPassInfo);
}

void CubemapOffscreenRender::createOffscreenPipeline() {
    auto vertShaderModule = vulkanCore.createShaderModule(
        vulkanCore.readShaderFile("shaders/offscreen_vert.spv"));
    auto fragShaderModule = vulkanCore.createShaderModule(
        vulkanCore.readShaderFile("shaders/offscreen_frag.spv"));
    vk::PipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.sType = vk::StructureType::ePipelineShaderStageCreateInfo;
    vertShaderStageInfo.stage = vk::ShaderStageFlagBits::eVertex;
    vertShaderStageInfo.module = vertShaderModule;
    vertShaderStageInfo.pName = "main";
    vk::PipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.sType = vk::StructureType::ePipelineShaderStageCreateInfo;
    fragShaderStageInfo.stage = vk::ShaderStageFlagBits::eFragment;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";
    vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};
    auto bindingDescription = OffscreenVertex::getBindingDescription();
    auto attributeDescriptions = OffscreenVertex::getAttributeDescriptions();
    vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = vk::StructureType::ePipelineVertexInputStateCreateInfo;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();
    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = vk::StructureType::ePipelineInputAssemblyStateCreateInfo;
    inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;
    inputAssembly.primitiveRestartEnable = VK_FALSE;
    const uint32_t cubemapSize = std::max(vulkanCore.swapChainExtent.width, vulkanCore.swapChainExtent.height);
    vk::Viewport viewport{0.0f, 0.0f, static_cast<float>(cubemapSize), static_cast<float>(cubemapSize), 0.0f, 1.0f};
    vk::Rect2D scissor{{0, 0}, {cubemapSize, cubemapSize}};
    vk::PipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = vk::StructureType::ePipelineViewportStateCreateInfo;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;
    vk::PipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = vk::StructureType::ePipelineRasterizationStateCreateInfo;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = vk::PolygonMode::eFill;
    rasterizer.cullMode = vk::CullModeFlagBits::eNone;
    rasterizer.frontFace = vk::FrontFace::eCounterClockwise;
    rasterizer.depthBiasEnable = VK_FALSE;
    rasterizer.lineWidth = 1.0f;
    vk::PipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = vk::StructureType::ePipelineMultisampleStateCreateInfo;
    multisampling.rasterizationSamples = vk::SampleCountFlagBits::e1;
    multisampling.sampleShadingEnable = VK_FALSE;
    vk::PipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.blendEnable = VK_FALSE;
    colorBlendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
    vk::PipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = vk::StructureType::ePipelineColorBlendStateCreateInfo;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = vk::StructureType::ePipelineLayoutCreateInfo;
    pipelineLayoutInfo.setLayoutCount = 0;
    pipelineLayoutInfo.pushConstantRangeCount = 0;
    pipelineLayoutInfo.pPushConstantRanges = nullptr;
    offscreenPipelineLayout = vulkanCore.device.createPipelineLayout(pipelineLayoutInfo);
    vk::GraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = vk::StructureType::eGraphicsPipelineCreateInfo;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.layout = offscreenPipelineLayout;
    pipelineInfo.renderPass = offscreenRenderPass;
    pipelineInfo.subpass = 0;
    offscreenPipeline = vk::raii::Pipeline(vulkanCore.device, nullptr, pipelineInfo);
}

void CubemapOffscreenRender::createOffscreenFramebuffers() {
    const uint32_t cubemapSize = std::max(vulkanCore.swapChainExtent.width, vulkanCore.swapChainExtent.height);
    offscreenFramebuffers.clear();
    offscreenImageViews.clear();
    offscreenFramebuffers.reserve(6);
    offscreenImageViews.reserve(6);
    for (uint32_t i = 0; i < 6; ++i) {
        offscreenImageViews.emplace_back(vulkanCore.createImageView(
            textureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageAspectFlagBits::eColor, vk::ImageViewType::e2D, 1, i));
        vk::ImageView imageViewHandle = *offscreenImageViews[i];
        vk::FramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = vk::StructureType::eFramebufferCreateInfo;
        framebufferInfo.renderPass = *offscreenRenderPass;
        framebufferInfo.attachmentCount = 1;
        framebufferInfo.pAttachments = &imageViewHandle;
        framebufferInfo.width = cubemapSize;
        framebufferInfo.height = cubemapSize;
        framebufferInfo.layers = 1;
        offscreenFramebuffers.emplace_back(vulkanCore.device, framebufferInfo);
    }
}

void CubemapOffscreenRender::renderToCubemap() {
    const auto offscreenCommandBuffer =
        ScopedOneTimeCommandBuffer(vulkanCore.device, vulkanCore.commandPool, vulkanCore.graphicsQueue);
    vulkanCore.transitionImageLayout(offscreenCommandBuffer.commandBuffer, *textureImage, vk::ImageLayout::eUndefined,
                                     vk::ImageLayout::eColorAttachmentOptimal, vk::AccessFlagBits::eNone,
                                     vk::AccessFlagBits::eColorAttachmentWrite, vk::PipelineStageFlagBits::eTopOfPipe,
                                     vk::PipelineStageFlagBits::eColorAttachmentOutput, 6, 0);
    for (uint32_t face = 0; face < 6; ++face) {
        vk::ClearValue clearColor{vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f})};
        const uint32_t cubemapSize = std::max(vulkanCore.swapChainExtent.width, vulkanCore.swapChainExtent.height);
        const vk::RenderPassBeginInfo renderPassInfo{
            vk::StructureType::eRenderPassBeginInfo, nullptr, *offscreenRenderPass, *offscreenFramebuffers[face],
            {{0, 0}, {cubemapSize, cubemapSize}},    1,       &clearColor};
        offscreenCommandBuffer.commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
        offscreenCommandBuffer.commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *offscreenPipeline);
        const vk::Buffer vertexBufferHandle = *offscreenVertexBuffer;
        const std::array<vk::Buffer, 1> vertexBuffers = {vertexBufferHandle};
        constexpr std::array<vk::DeviceSize, 1> offsets{};
        offscreenCommandBuffer.commandBuffer.bindVertexBuffers(0, vertexBuffers, offsets);
        offscreenCommandBuffer.commandBuffer.bindIndexBuffer(*offscreenIndexBuffer, 0, vk::IndexType::eUint32);
        offscreenCommandBuffer.commandBuffer.drawIndexed(6, 1, face * 6, 0, 0);
        offscreenCommandBuffer.commandBuffer.endRenderPass();
    }
}

void CubemapOffscreenRender::saveCubemapToPNG() {
    const uint32_t cubemapSize = std::max(vulkanCore.swapChainExtent.width, vulkanCore.swapChainExtent.height);
    const vk::DeviceSize imageSize = cubemapSize * cubemapSize * 4;
    vk::raii::Buffer stagingBuffer = nullptr;
    vk::raii::DeviceMemory stagingBufferMemory = nullptr;
    vulkanCore.createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferDst,
                            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                            stagingBuffer, stagingBufferMemory);
    std::vector<char> pixelData;
    for (uint32_t face = 0; face < 6; ++face) {
        {
            auto commandBuffer =
                ScopedOneTimeCommandBuffer(vulkanCore.device, vulkanCore.commandPool, vulkanCore.graphicsQueue);
            vulkanCore.transitionImageLayout(
                commandBuffer.commandBuffer, *textureImage, vk::ImageLayout::eShaderReadOnlyOptimal,
                vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits::eShaderRead,
                vk::AccessFlagBits::eTransferRead, vk::PipelineStageFlagBits::eFragmentShader,
                vk::PipelineStageFlagBits::eTransfer, 1, face);
            vk::BufferImageCopy region{};
            region.bufferOffset = 0;
            region.bufferRowLength = 0;
            region.bufferImageHeight = 0;
            region.imageSubresource = vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, face, 1};
            region.imageOffset = {0, 0, 0};
            region.imageExtent = {cubemapSize, cubemapSize, 1};
            commandBuffer.commandBuffer.copyImageToBuffer(*textureImage, vk::ImageLayout::eTransferSrcOptimal,
                                                          *stagingBuffer, region);
            vulkanCore.transitionImageLayout(commandBuffer.commandBuffer, *textureImage,
                                             vk::ImageLayout::eTransferSrcOptimal,
                                             vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eTransferRead,
                                             vk::AccessFlagBits::eShaderRead, vk::PipelineStageFlagBits::eTransfer,
                                             vk::PipelineStageFlagBits::eFragmentShader, 1, face);
        }
        void* data = stagingBufferMemory.mapMemory(0, imageSize);
        if (!data) {
            throw std::runtime_error("Failed to map memory for staging buffer.");
        }
        pixelData.reserve(pixelData.size() + imageSize);
        pixelData.insert(pixelData.end(), static_cast<char*>(data), static_cast<char*>(data) + imageSize);
        stagingBufferMemory.unmapMemory();
    }
    const std::string filename = "texture_output.png";
    stbi_write_png(filename.c_str(), cubemapSize, cubemapSize * 6, 4, pixelData.data(), cubemapSize * 4);
    fmt::print("Saved cubemap faces to {}\n", filename);
}
