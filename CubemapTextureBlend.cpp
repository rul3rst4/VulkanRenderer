#include "VulkanCore.hpp"
#include <memory>

#ifndef SHARED_RENDER_DATA_DEFINED
#define SHARED_RENDER_DATA_DEFINED
// Shared data structure for passing textures between render processes
struct SharedRenderData {
    vk::raii::Image* cubemapTexture = nullptr;
    vk::raii::ImageView* cubemapImageView = nullptr;
    vk::raii::Sampler* cubemapSampler = nullptr;
};
#endif

struct BlendVertex {
    glm::vec2 pos;
    glm::vec2 texCoord;

    constexpr static vk::VertexInputBindingDescription getBindingDescription() {
        constexpr vk::VertexInputBindingDescription bindingDescription{
            .binding = 0, .stride = sizeof(BlendVertex), .inputRate = vk::VertexInputRate::eVertex};

        return bindingDescription;
    }

    constexpr static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions() {
        constexpr std::array<vk::VertexInputAttributeDescription, 2> attributeDescriptions{{
            {.location = 0, .binding = 0, .format = vk::Format::eR32G32Sfloat, .offset = offsetof(BlendVertex, pos)},
            {.location = 1,
             .binding = 0,
             .format = vk::Format::eR32G32Sfloat,
             .offset = offsetof(BlendVertex, texCoord)},
        }};

        return attributeDescriptions;
    }
};

class CubemapTextureBlend : public IRenderProcess {
   public:
    VulkanCore& vulkanCore;
    std::shared_ptr<SharedRenderData> sharedData;

    // Screen rendering resources
    vk::raii::RenderPass screenRenderPass;
    vk::raii::Pipeline screenPipeline;
    vk::raii::PipelineLayout screenPipelineLayout;
    vk::raii::DescriptorSetLayout descriptorSetLayout;
    vk::raii::DescriptorSet descriptorSet;
    vk::raii::Sampler textureSampler;
    vk::raii::ImageView screenCubemapView;

    std::vector<vk::raii::Framebuffer> screenFramebuffers;
    std::vector<vk::raii::ImageView> swapChainImageViews;

    vk::raii::Buffer vertexBuffer;
    vk::raii::DeviceMemory vertexBufferMemory;
    vk::raii::Buffer indexBuffer;
    vk::raii::DeviceMemory indexBufferMemory;

    // Fullscreen quad vertices
    static constexpr std::array<BlendVertex, 4> vertices = {{{{-1.0f, -1.0f}, {0.0f, 0.0f}},
                                                             {{1.0f, -1.0f}, {1.0f, 0.0f}},
                                                             {{1.0f, 1.0f}, {1.0f, 1.0f}},
                                                             {{-1.0f, 1.0f}, {0.0f, 1.0f}}}};

    static constexpr std::array<uint32_t, 6> indices = {0, 1, 2, 2, 3, 0};

    explicit CubemapTextureBlend(VulkanCore& core, std::shared_ptr<SharedRenderData> data)
        : vulkanCore(core),
          sharedData(data),
          screenRenderPass(nullptr),
          screenPipeline(nullptr),
          screenPipelineLayout(nullptr),
          descriptorSetLayout(nullptr),
          descriptorSet(nullptr),
          textureSampler(nullptr),
          screenCubemapView(nullptr),
          vertexBuffer(nullptr),
          vertexBufferMemory(nullptr),
          indexBuffer(nullptr),
          indexBufferMemory(nullptr) {}

    virtual ~CubemapTextureBlend() = default;

    void execute() override {
        if (!sharedData || !sharedData->cubemapTexture) {
            throw std::runtime_error("No cubemap texture available for blending!");
        }

        // Only setup resources, don't render
        createSwapChainImageViews();
        createTextureSampler();
        createCubemapImageView();
        createDescriptorSetLayout();
        createScreenRenderPass();
        createScreenPipeline();
        createVertexBuffer();
        createIndexBuffer();
        createDescriptorSet();
        createScreenFramebuffers();

        fmt::print("CubemapTextureBlend setup complete\n");
    }

    bool supportsContinuousRendering() const override { return true; }

    void render() override { renderFrame(); }

   private:
    void createSwapChainImageViews() {
        swapChainImageViews.clear();
        swapChainImageViews.reserve(vulkanCore.swapChainImages.size());

        for (const auto& swapChainImage : vulkanCore.swapChainImages) {
            swapChainImageViews.emplace_back(vulkanCore.createImageView(swapChainImage, vulkanCore.swapChainImageFormat,
                                                                        vk::ImageAspectFlagBits::eColor,
                                                                        vk::ImageViewType::e2D, 1));
        }
    }

    void createTextureSampler() {
        vk::SamplerCreateInfo samplerInfo{.sType = vk::StructureType::eSamplerCreateInfo,
                                          .magFilter = vk::Filter::eLinear,
                                          .minFilter = vk::Filter::eLinear,
                                          .mipmapMode = vk::SamplerMipmapMode::eLinear,
                                          .addressModeU = vk::SamplerAddressMode::eClampToEdge,
                                          .addressModeV = vk::SamplerAddressMode::eClampToEdge,
                                          .addressModeW = vk::SamplerAddressMode::eClampToEdge,
                                          .anisotropyEnable = vk::True,
                                          .maxAnisotropy = 16.0f,
                                          .compareEnable = vk::False,
                                          .compareOp = vk::CompareOp::eAlways,
                                          .minLod = 0.0f,
                                          .maxLod = 0.0f,
                                          .borderColor = vk::BorderColor::eIntOpaqueBlack,
                                          .unnormalizedCoordinates = vk::False};

        textureSampler = vulkanCore.device.createSampler(samplerInfo);
    }

    void createCubemapImageView() {
        if (!sharedData->cubemapTexture) {
            throw std::runtime_error("Shared cubemap texture is null!");
        }

        screenCubemapView = vulkanCore.createImageView(*sharedData->cubemapTexture, vk::Format::eR8G8B8A8Srgb,
                                                       vk::ImageAspectFlagBits::eColor, vk::ImageViewType::eCube, 6);

        fmt::print("Created cubemap image view for sampling\n");
    }

    void createDescriptorSetLayout() {
        vk::DescriptorSetLayoutBinding samplerLayoutBinding{.binding = 0,
                                                            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                                            .descriptorCount = 1,
                                                            .stageFlags = vk::ShaderStageFlagBits::eFragment,
                                                            .pImmutableSamplers = nullptr};

        vk::DescriptorSetLayoutCreateInfo layoutInfo{.sType = vk::StructureType::eDescriptorSetLayoutCreateInfo,
                                                     .bindingCount = 1,
                                                     .pBindings = &samplerLayoutBinding};

        descriptorSetLayout = vulkanCore.device.createDescriptorSetLayout(layoutInfo);
    }

    void createScreenRenderPass() {
        vk::AttachmentDescription colorAttachment{.format = vulkanCore.swapChainImageFormat,
                                                  .samples = vk::SampleCountFlagBits::e1,
                                                  .loadOp = vk::AttachmentLoadOp::eClear,
                                                  .storeOp = vk::AttachmentStoreOp::eStore,
                                                  .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
                                                  .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
                                                  .initialLayout = vk::ImageLayout::eUndefined,
                                                  .finalLayout = vk::ImageLayout::ePresentSrcKHR};

        vk::AttachmentReference colorAttachmentRef{.attachment = 0, .layout = vk::ImageLayout::eColorAttachmentOptimal};

        vk::SubpassDescription subpass{.pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
                                       .colorAttachmentCount = 1,
                                       .pColorAttachments = &colorAttachmentRef};

        vk::SubpassDependency dependency{.srcSubpass = vk::SubpassExternal,
                                         .dstSubpass = 0,
                                         .srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
                                         .dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
                                         .srcAccessMask = vk::AccessFlags{},
                                         .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite};

        vk::RenderPassCreateInfo renderPassInfo{.sType = vk::StructureType::eRenderPassCreateInfo,
                                                .attachmentCount = 1,
                                                .pAttachments = &colorAttachment,
                                                .subpassCount = 1,
                                                .pSubpasses = &subpass,
                                                .dependencyCount = 1,
                                                .pDependencies = &dependency};

        screenRenderPass = vulkanCore.device.createRenderPass(renderPassInfo);
    }

    void createScreenPipeline() {
        auto vertShaderModule = vulkanCore.createShaderModule(vulkanCore.readShaderFile("shaders/blend_vert.spv"));
        auto fragShaderModule = vulkanCore.createShaderModule(vulkanCore.readShaderFile("shaders/blend_frag.spv"));

        vk::PipelineShaderStageCreateInfo vertShaderStageInfo{
            .sType = vk::StructureType::ePipelineShaderStageCreateInfo,
            .stage = vk::ShaderStageFlagBits::eVertex,
            .module = vertShaderModule,
            .pName = "main"};

        vk::PipelineShaderStageCreateInfo fragShaderStageInfo{
            .sType = vk::StructureType::ePipelineShaderStageCreateInfo,
            .stage = vk::ShaderStageFlagBits::eFragment,
            .module = fragShaderModule,
            .pName = "main"};

        vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

        auto bindingDescription = BlendVertex::getBindingDescription();
        auto attributeDescriptions = BlendVertex::getAttributeDescriptions();

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
            .sType = vk::StructureType::ePipelineVertexInputStateCreateInfo,
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &bindingDescription,
            .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
            .pVertexAttributeDescriptions = attributeDescriptions.data()};

        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
            .sType = vk::StructureType::ePipelineInputAssemblyStateCreateInfo,
            .topology = vk::PrimitiveTopology::eTriangleList,
            .primitiveRestartEnable = vk::False};

        vk::Viewport viewport{.x = 0.0f,
                              .y = 0.0f,
                              .width = static_cast<float>(vulkanCore.swapChainExtent.width),
                              .height = static_cast<float>(vulkanCore.swapChainExtent.height),
                              .minDepth = 0.0f,
                              .maxDepth = 1.0f};

        vk::Rect2D scissor{.offset = {0, 0}, .extent = vulkanCore.swapChainExtent};

        vk::PipelineViewportStateCreateInfo viewportState{.sType = vk::StructureType::ePipelineViewportStateCreateInfo,
                                                          .viewportCount = 1,
                                                          .pViewports = &viewport,
                                                          .scissorCount = 1,
                                                          .pScissors = &scissor};

        vk::PipelineRasterizationStateCreateInfo rasterizer{
            .sType = vk::StructureType::ePipelineRasterizationStateCreateInfo,
            .depthClampEnable = vk::False,
            .rasterizerDiscardEnable = vk::False,
            .polygonMode = vk::PolygonMode::eFill,
            .cullMode = vk::CullModeFlagBits::eBack,
            .frontFace = vk::FrontFace::eCounterClockwise,
            .depthBiasEnable = vk::False,
            .lineWidth = 1.0f};

        vk::PipelineMultisampleStateCreateInfo multisampling{
            .sType = vk::StructureType::ePipelineMultisampleStateCreateInfo,
            .rasterizationSamples = vk::SampleCountFlagBits::e1,
            .sampleShadingEnable = vk::False};

        vk::PipelineColorBlendAttachmentState colorBlendAttachment{
            .blendEnable = vk::False,
            .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                              vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA};

        vk::PipelineColorBlendStateCreateInfo colorBlending{
            .sType = vk::StructureType::ePipelineColorBlendStateCreateInfo,
            .logicOpEnable = vk::False,
            .attachmentCount = 1,
            .pAttachments = &colorBlendAttachment};

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo{.sType = vk::StructureType::ePipelineLayoutCreateInfo,
                                                        .setLayoutCount = 1,
                                                        .pSetLayouts = &*descriptorSetLayout,
                                                        .pushConstantRangeCount = 0,
                                                        .pPushConstantRanges = nullptr};

        screenPipelineLayout = vulkanCore.device.createPipelineLayout(pipelineLayoutInfo);

        vk::GraphicsPipelineCreateInfo pipelineInfo{.sType = vk::StructureType::eGraphicsPipelineCreateInfo,
                                                    .stageCount = 2,
                                                    .pStages = shaderStages,
                                                    .pVertexInputState = &vertexInputInfo,
                                                    .pInputAssemblyState = &inputAssembly,
                                                    .pViewportState = &viewportState,
                                                    .pRasterizationState = &rasterizer,
                                                    .pMultisampleState = &multisampling,
                                                    .pColorBlendState = &colorBlending,
                                                    .layout = screenPipelineLayout,
                                                    .renderPass = screenRenderPass,
                                                    .subpass = 0};

        screenPipeline = vk::raii::Pipeline(vulkanCore.device, nullptr, pipelineInfo);
    }

    void createVertexBuffer() {
        constexpr vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

        vk::raii::Buffer stagingBuffer = nullptr;
        vk::raii::DeviceMemory stagingBufferMemory = nullptr;

        vulkanCore.createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                stagingBuffer, stagingBufferMemory);

        const auto data = stagingBufferMemory.mapMemory(0, bufferSize);
        memcpy(data, vertices.data(), bufferSize);
        stagingBufferMemory.unmapMemory();

        vulkanCore.createBuffer(bufferSize,
                                vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
                                vk::MemoryPropertyFlagBits::eDeviceLocal, vertexBuffer, vertexBufferMemory);

        vulkanCore.copyBuffer(*stagingBuffer, *vertexBuffer, bufferSize);
    }

    void createIndexBuffer() {
        constexpr vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();

        vk::raii::Buffer stagingBuffer = nullptr;
        vk::raii::DeviceMemory stagingBufferMemory = nullptr;

        vulkanCore.createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                stagingBuffer, stagingBufferMemory);

        const auto data = stagingBufferMemory.mapMemory(0, bufferSize);
        memcpy(data, indices.data(), bufferSize);
        stagingBufferMemory.unmapMemory();

        vulkanCore.createBuffer(bufferSize,
                                vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
                                vk::MemoryPropertyFlagBits::eDeviceLocal, indexBuffer, indexBufferMemory);

        vulkanCore.copyBuffer(*stagingBuffer, *indexBuffer, bufferSize);
    }

    void createDescriptorSet() {
        vk::DescriptorSetAllocateInfo allocInfo{.sType = vk::StructureType::eDescriptorSetAllocateInfo,
                                                .descriptorPool = vulkanCore.descriptorPool,
                                                .descriptorSetCount = 1,
                                                .pSetLayouts = &*descriptorSetLayout};

        auto descriptorSets = vulkanCore.device.allocateDescriptorSets(allocInfo);
        descriptorSet = std::move(descriptorSets[0]);

        vk::DescriptorImageInfo imageInfo{.sampler = *textureSampler,
                                          .imageView = *screenCubemapView,
                                          .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};

        vk::WriteDescriptorSet descriptorWrite{.sType = vk::StructureType::eWriteDescriptorSet,
                                               .dstSet = *descriptorSet,
                                               .dstBinding = 0,
                                               .dstArrayElement = 0,
                                               .descriptorCount = 1,
                                               .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                               .pImageInfo = &imageInfo};

        vulkanCore.device.updateDescriptorSets(descriptorWrite, nullptr);
    }

    void createScreenFramebuffers() {
        screenFramebuffers.clear();
        screenFramebuffers.reserve(swapChainImageViews.size());

        for (const auto& imageView : swapChainImageViews) {
            vk::ImageView attachments[] = {*imageView};

            vk::FramebufferCreateInfo framebufferInfo{.sType = vk::StructureType::eFramebufferCreateInfo,
                                                      .renderPass = *screenRenderPass,
                                                      .attachmentCount = 1,
                                                      .pAttachments = attachments,
                                                      .width = vulkanCore.swapChainExtent.width,
                                                      .height = vulkanCore.swapChainExtent.height,
                                                      .layers = 1};

            screenFramebuffers.emplace_back(vulkanCore.device, framebufferInfo);
        }
    }

    void renderFrame() {
        // Wait for the previous frame to complete
        const auto result =
            vulkanCore.device.waitForFences(*vulkanCore.inFlightFences[vulkanCore.currentFrame], vk::True, UINT64_MAX);
        if (result != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to wait for fence!");
        }

        // Acquire next swap chain image
        uint32_t imageIndex;
        try {
            const auto [result, index] = vulkanCore.swapChain.acquireNextImage(
                UINT64_MAX, *vulkanCore.imageAvailableSemaphores[vulkanCore.currentFrame], nullptr);
            imageIndex = index;

            if (result == vk::Result::eErrorOutOfDateKHR) {
                vulkanCore.recreateSwapChain();
                return;
            } else if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
                throw std::runtime_error("Failed to acquire swap chain image!");
            }
        } catch (vk::OutOfDateKHRError&) {
            vulkanCore.recreateSwapChain();
            return;
        }

        // Reset fence after acquiring image
        vulkanCore.device.resetFences(*vulkanCore.inFlightFences[vulkanCore.currentFrame]);

        // Record command buffer
        auto& commandBuffer = vulkanCore.commandBuffers[vulkanCore.currentFrame];
        commandBuffer.reset();

        const vk::CommandBufferBeginInfo beginInfo{.sType = vk::StructureType::eCommandBufferBeginInfo};
        commandBuffer.begin(beginInfo);

        // Begin render pass
        vk::ClearValue clearColor{.color = vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f})};
        const vk::RenderPassBeginInfo renderPassInfo{.sType = vk::StructureType::eRenderPassBeginInfo,
                                                     .renderPass = *screenRenderPass,
                                                     .framebuffer = *screenFramebuffers[imageIndex],
                                                     .renderArea = {{0, 0}, vulkanCore.swapChainExtent},
                                                     .clearValueCount = 1,
                                                     .pClearValues = &clearColor};

        commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *screenPipeline);

        // Bind vertex buffer
        const vk::Buffer vertexBufferHandle = *vertexBuffer;
        const std::array<vk::Buffer, 1> vertexBuffers = {vertexBufferHandle};
        constexpr std::array<vk::DeviceSize, 1> offsets{};
        commandBuffer.bindVertexBuffers(0, vertexBuffers, offsets);

        // Bind index buffer
        commandBuffer.bindIndexBuffer(*indexBuffer, 0, vk::IndexType::eUint32);

        // Bind descriptor set
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *screenPipelineLayout, 0, *descriptorSet,
                                         nullptr);

        // Draw
        commandBuffer.drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
        commandBuffer.endRenderPass();
        commandBuffer.end();

        // Submit command buffer
        const vk::Semaphore waitSemaphores[] = {*vulkanCore.imageAvailableSemaphores[vulkanCore.currentFrame]};
        const vk::PipelineStageFlags waitStages[] = {vk::PipelineStageFlagBits::eColorAttachmentOutput};
        const vk::Semaphore signalSemaphores[] = {*vulkanCore.renderFinishedSemaphores[vulkanCore.currentFrame]};

        const vk::SubmitInfo submitInfo{.sType = vk::StructureType::eSubmitInfo,
                                        .waitSemaphoreCount = 1,
                                        .pWaitSemaphores = waitSemaphores,
                                        .pWaitDstStageMask = waitStages,
                                        .commandBufferCount = 1,
                                        .pCommandBuffers = &*commandBuffer,
                                        .signalSemaphoreCount = 1,
                                        .pSignalSemaphores = signalSemaphores};

        vulkanCore.graphicsQueue.submit(submitInfo, *vulkanCore.inFlightFences[vulkanCore.currentFrame]);

        // Present result
        const vk::SwapchainKHR swapChains[] = {*vulkanCore.swapChain};
        const vk::PresentInfoKHR presentInfo{.sType = vk::StructureType::ePresentInfoKHR,
                                             .waitSemaphoreCount = 1,
                                             .pWaitSemaphores = signalSemaphores,
                                             .swapchainCount = 1,
                                             .pSwapchains = swapChains,
                                             .pImageIndices = &imageIndex};

        try {
            const auto result = vulkanCore.presenteQueue.presentKHR(presentInfo);
            if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR ||
                vulkanCore.framebufferResized) {
                vulkanCore.framebufferResized = false;
                vulkanCore.recreateSwapChain();
            } else if (result != vk::Result::eSuccess) {
                throw std::runtime_error("Failed to present swap chain image!");
            }
        } catch (vk::OutOfDateKHRError&) {
            vulkanCore.framebufferResized = false;
            vulkanCore.recreateSwapChain();
        }

        vulkanCore.currentFrame = (vulkanCore.currentFrame + 1) % vulkanCore.maxFramesInFlight;
    }
};
