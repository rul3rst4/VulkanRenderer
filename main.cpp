#include <exception>
#include <iostream>
#include "VulkanCore.hpp"

struct OffscreenVertex {
    glm::vec3 pos;
    glm::vec4 color;  // r, g, b, face_id

    constexpr static vk::VertexInputBindingDescription getBindingDescription() {
        constexpr vk::VertexInputBindingDescription bindingDescription{
            .binding = 0, .stride = sizeof(OffscreenVertex), .inputRate = vk::VertexInputRate::eVertex};

        return bindingDescription;
    }

    constexpr static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions() {
        constexpr std::array<vk::VertexInputAttributeDescription, 2> attributeDescriptions{{
            {.location = 0,
             .binding = 0,
             .format = vk::Format::eR32G32B32Sfloat,
             .offset = offsetof(OffscreenVertex, pos)},
            {.location = 1,
             .binding = 0,
             .format = vk::Format::eR32G32B32A32Sfloat,
             .offset = offsetof(OffscreenVertex, color)},
        }};

        return attributeDescriptions;
    }
};

class CubemapOffcreenRender : public VulkanCore {
   private:
    vk::raii::Buffer offscreenVertexBuffer;
    vk::raii::DeviceMemory offscreenVertexBufferMemory;
    vk::raii::Buffer offscreenIndexBuffer;
    vk::raii::DeviceMemory offscreenIndexBufferMemory;

    vk::raii::Image textureImage;
    vk::raii::DeviceMemory textureImageMemory;

    // Offscreen rendering resources
    vk::raii::RenderPass offscreenRenderPass;
    vk::raii::Pipeline offscreenPipeline;
    vk::raii::PipelineLayout offscreenPipelineLayout;
    std::vector<vk::raii::Framebuffer> offscreenFramebuffers;
    std::vector<vk::raii::ImageView> offscreenImageViews;

    // Offscreen rendering vertex data for cubemap faces
    // x, y, z, r, g, b, face_id
    static constexpr std::array<OffscreenVertex, 24> offscreenVertices = {{
        // quad 0 with red color
        {{-1.0f, -1.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f}},
        {{1.0f, -1.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f}},
        {{-1.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f}},
        {{1.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f}},
        // quad 1 with green color
        {{-1.0f, -1.0f, 0.0f}, {0.0f, 1.0f, 0.0f, 1.0f}},
        {{1.0f, -1.0f, 0.0f}, {0.0f, 1.0f, 0.0f, 1.0f}},
        {{-1.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f, 1.0f}},
        {{1.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f, 1.0f}},
        // quad 2 with blue color
        {{-1.0f, -1.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 2.0f}},
        {{1.0f, -1.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 2.0f}},
        {{-1.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 2.0f}},
        {{1.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 2.0f}},
        // quad 3 with yellow color
        {{-1.0f, -1.0f, 0.0f}, {1.0f, 1.0f, 0.0f, 3.0f}},
        {{1.0f, -1.0f, 0.0f}, {1.0f, 1.0f, 0.0f, 3.0f}},
        {{-1.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 0.0f, 3.0f}},
        {{1.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 0.0f, 3.0f}},
        // quad 4 with cyan color
        {{-1.0f, -1.0f, 0.0f}, {0.0f, 1.0f, 1.0f, 4.0f}},
        {{1.0f, -1.0f, 0.0f}, {0.0f, 1.0f, 1.0f, 4.0f}},
        {{-1.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 1.0f, 4.0f}},
        {{1.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 1.0f, 4.0f}},
        // quad 5 with white color
        {{-1.0f, -1.0f, 0.0f}, {1.0f, 1.0f, 1.0f, 5.0f}},
        {{1.0f, -1.0f, 0.0f}, {1.0f, 1.0f, 1.0f, 5.0f}},
        {{-1.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 1.0f, 5.0f}},
        {{1.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 1.0f, 5.0f}},
    }};

    static constexpr std::array<uint32_t, 36> offscreenIndices = {0,  1,  2,  1,  3,  2,  4,  5,  6,  5,  7,  6,
                                                                  8,  9,  10, 9,  11, 10, 12, 13, 14, 13, 15, 14,
                                                                  16, 17, 18, 17, 19, 18, 20, 21, 22, 21, 23, 22};

    void initVulkan() {
        createEmptyCubemapTexture();
        createOffscreenRenderPass();
        createOffscreenPipeline();
        createOffscreenVertexBuffer();
        createOffscreenIndexBuffer();
        createOffscreenFramebuffers();
        renderToCubemap();
        saveCubemapToPNG();
    }

    void createEmptyCubemapTexture() {
        // Create empty cubemap with viewport dimensions
        const uint32_t cubemapSize = std::max(swapChainExtent.width, swapChainExtent.height);

        createImage(cubemapSize, cubemapSize, vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal,
                    vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled |
                        vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst,
                    vk::MemoryPropertyFlagBits::eDeviceLocal, textureImage, textureImageMemory, 6,
                    vk::ImageCreateFlagBits::eCubeCompatible);
    }

    void createOffscreenVertexBuffer() {
        constexpr vk::DeviceSize bufferSize = sizeof(offscreenVertices[0]) * offscreenVertices.size();

        vk::raii::Buffer stagingBuffer = nullptr;
        vk::raii::DeviceMemory stagingBufferMemory = nullptr;

        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                     vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                     stagingBuffer, stagingBufferMemory);

        const auto data = stagingBufferMemory.mapMemory(0, bufferSize);
        memcpy(data, offscreenVertices.data(), bufferSize);
        stagingBufferMemory.unmapMemory();

        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
                     vk::MemoryPropertyFlagBits::eDeviceLocal, offscreenVertexBuffer, offscreenVertexBufferMemory);

        copyBuffer(*stagingBuffer, *offscreenVertexBuffer, bufferSize);
    }

    void createOffscreenIndexBuffer() {
        constexpr vk::DeviceSize bufferSize = sizeof(offscreenIndices[0]) * offscreenIndices.size();

        vk::raii::Buffer stagingBuffer = nullptr;
        vk::raii::DeviceMemory stagingBufferMemory = nullptr;

        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                     vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                     stagingBuffer, stagingBufferMemory);

        const auto data = stagingBufferMemory.mapMemory(0, bufferSize);
        memcpy(data, offscreenIndices.data(), bufferSize);
        stagingBufferMemory.unmapMemory();

        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
                     vk::MemoryPropertyFlagBits::eDeviceLocal, offscreenIndexBuffer, offscreenIndexBufferMemory);

        copyBuffer(*stagingBuffer, *offscreenIndexBuffer, bufferSize);
    }

    void createOffscreenRenderPass() {
        vk::AttachmentDescription colorAttachment{.format = vk::Format::eR8G8B8A8Srgb,
                                                  .samples = vk::SampleCountFlagBits::e1,
                                                  .loadOp = vk::AttachmentLoadOp::eClear,
                                                  .storeOp = vk::AttachmentStoreOp::eStore,
                                                  .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
                                                  .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
                                                  .initialLayout = vk::ImageLayout::eUndefined,
                                                  .finalLayout = vk::ImageLayout::eShaderReadOnlyOptimal};

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

        offscreenRenderPass = device.createRenderPass(renderPassInfo);
    }

    void createOffscreenPipeline() {
        auto vertShaderModule = createShaderModule(readShaderFile("shaders/offscreen_vert.spv"));
        auto fragShaderModule = createShaderModule(readShaderFile("shaders/offscreen_frag.spv"));

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

        auto bindingDescription = OffscreenVertex::getBindingDescription();
        auto attributeDescriptions = OffscreenVertex::getAttributeDescriptions();

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

        const uint32_t cubemapSize = std::max(swapChainExtent.width, swapChainExtent.height);
        vk::Viewport viewport{.x = 0.0f,
                              .y = 0.0f,
                              .width = static_cast<float>(cubemapSize),
                              .height = static_cast<float>(cubemapSize),
                              .minDepth = 0.0f,
                              .maxDepth = 1.0f};

        vk::Rect2D scissor{.offset = {0, 0}, .extent = {cubemapSize, cubemapSize}};

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
            .cullMode = vk::CullModeFlagBits::eNone,
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
                                                        .setLayoutCount = 0,
                                                        .pushConstantRangeCount = 0,
                                                        .pPushConstantRanges = nullptr};

        offscreenPipelineLayout = device.createPipelineLayout(pipelineLayoutInfo);

        vk::GraphicsPipelineCreateInfo pipelineInfo{.sType = vk::StructureType::eGraphicsPipelineCreateInfo,
                                                    .stageCount = 2,
                                                    .pStages = shaderStages,
                                                    .pVertexInputState = &vertexInputInfo,
                                                    .pInputAssemblyState = &inputAssembly,
                                                    .pViewportState = &viewportState,
                                                    .pRasterizationState = &rasterizer,
                                                    .pMultisampleState = &multisampling,
                                                    .pColorBlendState = &colorBlending,
                                                    .layout = offscreenPipelineLayout,
                                                    .renderPass = offscreenRenderPass,
                                                    .subpass = 0};

        offscreenPipeline = vk::raii::Pipeline(device, nullptr, pipelineInfo);
    }

    void createOffscreenFramebuffers() {
        const uint32_t cubemapSize = std::max(swapChainExtent.width, swapChainExtent.height);
        offscreenFramebuffers.clear();
        offscreenImageViews.clear();
        offscreenFramebuffers.reserve(6);
        offscreenImageViews.reserve(6);

        for (uint32_t i = 0; i < 6; ++i) {
            offscreenImageViews.emplace_back(createImageView(textureImage, vk::Format::eR8G8B8A8Srgb,
                                                             vk::ImageAspectFlagBits::eColor, vk::ImageViewType::e2D, 1,
                                                             i));

            vk::ImageView imageViewHandle = *offscreenImageViews[i];
            vk::FramebufferCreateInfo framebufferInfo{.sType = vk::StructureType::eFramebufferCreateInfo,
                                                      .renderPass = *offscreenRenderPass,
                                                      .attachmentCount = 1,
                                                      .pAttachments = &imageViewHandle,
                                                      .width = cubemapSize,
                                                      .height = cubemapSize,
                                                      .layers = 1};

            offscreenFramebuffers.emplace_back(device, framebufferInfo);
        }
    }

    void renderToCubemap() {
        const auto offscreenCommandBuffer = ScopedOneTimeCommandBuffer(device, commandPool, graphicsQueue);

        for (uint32_t face = 0; face < 6; ++face) {
            transitionImageLayout(offscreenCommandBuffer.commandBuffer, *textureImage, vk::ImageLayout::eUndefined,
                                  vk::ImageLayout::eColorAttachmentOptimal, vk::AccessFlagBits::eShaderRead,
                                  vk::AccessFlagBits::eColorAttachmentWrite, vk::PipelineStageFlagBits::eFragmentShader,
                                  vk::PipelineStageFlagBits::eColorAttachmentOutput, 1, face);

            vk::ClearValue clearColor{.color = vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f})};

            const uint32_t cubemapSize = std::max(swapChainExtent.width, swapChainExtent.height);
            const vk::RenderPassBeginInfo renderPassInfo{.sType = vk::StructureType::eRenderPassBeginInfo,
                                                         .renderPass = *offscreenRenderPass,
                                                         .framebuffer = *offscreenFramebuffers[face],
                                                         .renderArea = {{0, 0}, {cubemapSize, cubemapSize}},
                                                         .clearValueCount = 1,
                                                         .pClearValues = &clearColor};

            offscreenCommandBuffer.commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
            offscreenCommandBuffer.commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *offscreenPipeline);

            // Bind vertex buffer
            const vk::Buffer vertexBufferHandle = *offscreenVertexBuffer;
            const std::array<vk::Buffer, 1> vertexBuffers = {vertexBufferHandle};
            constexpr std::array<vk::DeviceSize, 1> offsets{};
            offscreenCommandBuffer.commandBuffer.bindVertexBuffers(0, vertexBuffers, offsets);

            // Bind index buffer
            offscreenCommandBuffer.commandBuffer.bindIndexBuffer(*offscreenIndexBuffer, 0, vk::IndexType::eUint32);

            // Draw the quad for this face (6 indices per face)
            offscreenCommandBuffer.commandBuffer.drawIndexed(6, 1, face * 6, 0, 0);
            offscreenCommandBuffer.commandBuffer.endRenderPass();

            transitionImageLayout(offscreenCommandBuffer.commandBuffer, *textureImage, vk::ImageLayout::eUndefined,
                                  vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eColorAttachmentWrite,
                                  vk::AccessFlagBits::eShaderRead, vk::PipelineStageFlagBits::eColorAttachmentOutput,
                                  vk::PipelineStageFlagBits::eFragmentShader, 1);
        }
    }

    void saveCubemapToPNG() {
        const uint32_t cubemapSize = std::max(swapChainExtent.width, swapChainExtent.height);
        const vk::DeviceSize imageSize = cubemapSize * cubemapSize * 4;

        // Create staging buffer using RAII
        vk::raii::Buffer stagingBuffer = nullptr;
        vk::raii::DeviceMemory stagingBufferMemory = nullptr;
        createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferDst,
                     vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                     stagingBuffer, stagingBufferMemory);

        std::vector<char> pixelData;

        // Save each face individually
        for (uint32_t face = 0; face < 6; ++face) {
            // Copy each face to staging buffer and save
            {
                auto commandBuffer = ScopedOneTimeCommandBuffer(device, commandPool, graphicsQueue);

                transitionImageLayout(commandBuffer.commandBuffer, *textureImage, vk::ImageLayout::eUndefined,
                                      vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits::eShaderRead,
                                      vk::AccessFlagBits::eTransferRead, vk::PipelineStageFlagBits::eFragmentShader,
                                      vk::PipelineStageFlagBits::eTransfer, 1, face);

                // Copy image to buffer
                vk::BufferImageCopy region{
                    .bufferOffset = 0,
                    .bufferRowLength = 0,
                    .bufferImageHeight = 0,
                    .imageSubresource = vk::ImageSubresourceLayers{.aspectMask = vk::ImageAspectFlagBits::eColor,
                                                                   .mipLevel = 0,
                                                                   .baseArrayLayer = face,
                                                                   .layerCount = 1},
                    .imageOffset = {0, 0, 0},
                    .imageExtent = {cubemapSize, cubemapSize, 1}};

                commandBuffer.commandBuffer.copyImageToBuffer(*textureImage, vk::ImageLayout::eTransferSrcOptimal,
                                                              *stagingBuffer, region);

                transitionImageLayout(commandBuffer.commandBuffer, *textureImage, vk::ImageLayout::eUndefined,
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

   public:
    explicit CubemapOffcreenRender()
        : offscreenVertexBuffer(nullptr),
          offscreenVertexBufferMemory(nullptr),
          offscreenIndexBuffer(nullptr),
          offscreenIndexBufferMemory(nullptr),
          textureImage(nullptr),
          textureImageMemory(nullptr),
          offscreenRenderPass(nullptr),
          offscreenPipeline(nullptr),
          offscreenPipelineLayout(nullptr) {}
    virtual ~CubemapOffcreenRender() = default;

    void run() { initVulkan(); }
};

int main() {
    try {
        CubemapOffcreenRender app;
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
    }

    return EXIT_SUCCESS;
}
