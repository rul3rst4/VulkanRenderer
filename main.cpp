#include <exception>
#include <initializer_list>
#include <iterator>
#include <ranges>
#define VULKAN_HPP_NO_CONSTRUCTORS  // Permite usar Designated Initializers pra construir os objetos.
// #define VULKAN_HPP_NO_EXCEPTIONS // Retorna um result type pra ser tratado.
// #define GLFW_INCLUDE_VULKAN
#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <chrono>
#include <iostream>
#include <memory>
#include <fmt/format.h>
#include <algorithm>
#include <optional>
#include <set>
#include <limits>
#include <fstream>

struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
};

struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;
    glm::vec3 texCoord;

    constexpr static vk::VertexInputBindingDescription getBindingDescription() {
        constexpr vk::VertexInputBindingDescription bindingDescription{
            .binding = 0, .stride = sizeof(Vertex), .inputRate = vk::VertexInputRate::eVertex};

        return bindingDescription;
    }

    constexpr static std::array<vk::VertexInputAttributeDescription, 3> getAttributeDescriptions() {
        constexpr std::array<vk::VertexInputAttributeDescription, 3> attributeDescriptions{{
            {.binding = 0, .location = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(Vertex, pos)},
            {.binding = 0, .location = 1, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(Vertex, color)},
            {.binding = 0, .location = 2, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(Vertex, texCoord)},
        }};

        return attributeDescriptions;
    }
};

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presenteFamily;

    [[nodiscard]] bool isComplete() const { return graphicsFamily.has_value() && presenteFamily.has_value(); }
};

struct SwapChainSupportDetails {
    vk::SurfaceCapabilitiesKHR capabilities;
    std::vector<vk::SurfaceFormatKHR> formats;
    std::vector<vk::PresentModeKHR> presenteModes;
};

static std::vector<char> readFile(const std::string& fileName) {
    std::ifstream file(fileName, std::ios::ate | std::ios::binary);  // le o arquivo começando pelo final.

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file!");
    }

    const int64_t fileSize = file.tellg();

    if (fileSize < 0) {
        throw std::runtime_error("file size is negative or error occurred!");
    }

    std::vector<char> buffer(fileSize);
    file.seekg(0);  // volta para o começo do arquivo.
    file.read(buffer.data(), fileSize);

    return buffer;
}

class ScopedOneTimeCommandBuffer {
   public:
    explicit ScopedOneTimeCommandBuffer(const vk::Device& device,
                                        const vk::CommandPool& commandPool,
                                        const vk::Queue& graphicsQueue)
        : device(device), commandPool(commandPool), graphicsQueue(graphicsQueue) {
        const vk::CommandBufferAllocateInfo allocInfo{.sType = vk::StructureType::eCommandBufferAllocateInfo,
                                                      .commandPool = commandPool,
                                                      .level = vk::CommandBufferLevel::ePrimary,
                                                      .commandBufferCount = 1};

        commandBuffer = device.allocateCommandBuffers(allocInfo)[0];

        constexpr vk::CommandBufferBeginInfo beginInfo{.sType = vk::StructureType::eCommandBufferBeginInfo,
                                                       .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit};

        commandBuffer.begin(beginInfo);
    }

    ~ScopedOneTimeCommandBuffer() {
        commandBuffer.end();

        const vk::SubmitInfo submitInfo{
            .sType = vk::StructureType::eSubmitInfo, .commandBufferCount = 1, .pCommandBuffers = &commandBuffer};

        vk::detail::resultCheck(graphicsQueue.submit(1, &submitInfo, nullptr),
                                "Error submiting CopyBuffer commandBuffer");
        graphicsQueue.waitIdle();
        device.freeCommandBuffers(commandPool, 1, &commandBuffer);
    }

   public:
    vk::CommandBuffer commandBuffer;

   private:
    const vk::Device& device;
    const vk::CommandPool& commandPool;
    const vk::Queue& graphicsQueue;
};

class HelloTriangle {
   private:
    std::unique_ptr<GLFWwindow, decltype(&::glfwDestroyWindow)> window;
    static constexpr int width = 800;
    static constexpr int height = 600;
    const std::vector<const char*> validationLayers = {"VK_LAYER_KHRONOS_validation"};
    const std::vector<const char*> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        "VK_KHR_portability_subset"};  // Ativei VK_KHR_portability_subset a pedido da camada de validação e
                                       // documentação, que diz que quando o dispositivo fisico suporta essa extensão,
                                       // ela sempre deve ser ativada.
    static constexpr int maxFramesInFlight = 2;
    static constexpr bool enableValidationLayers = [] {
#ifdef NDEBUG
        return false;
#else
        return true;
#endif
    }();

    /// vulkan_hpp não tem destrutores para essas variaveis, apenas no vulkan_raii.hpp.
    vk::Instance instance;
    vk::PhysicalDevice physicalDevice;
    vk::Device device;
    vk::Queue graphicsQueue;
    vk::Queue presenteQueue;
    vk::SurfaceKHR surface;
    vk::SwapchainKHR swapChain;
    vk::Format swapChainImageFormat;
    vk::Extent2D swapChainExtent;
    vk::RenderPass renderPass;
    vk::DescriptorSetLayout descriptorSetLayout;
    vk::PipelineLayout pipelineLayout;
    vk::Pipeline graphicsPipeline;
    vk::CommandPool commandPool;

    // Para mais performance, podemos ter vertexBuffer e indexBuffer no mesmo Buffer.
    // https://developer.nvidia.com/vulkan-memory-management
    vk::Buffer vertexBuffer;
    vk::DeviceMemory vertexBufferMemory;
    vk::Buffer indexBuffer;
    vk::DeviceMemory indexBufferMemory;

    vk::Image textureImage;
    vk::DeviceMemory textureImageMemory;
    vk::ImageView textureImageView;
    vk::Sampler textureSampler;

    vk::Image depthImage;
    vk::DeviceMemory depthImageMemory;
    vk::ImageView depthImageView;

    vk::DescriptorPool descriptorPool;
    std::vector<vk::DescriptorSet> descriptorSets;

    std::vector<vk::Fence> inFlightFences;
    std::vector<vk::Semaphore> imageAvailableSemaphores;
    std::vector<vk::Semaphore> renderFinishedSemaphores;
    std::vector<vk::CommandBuffer> commandBuffers;
    std::vector<vk::Image> swapChainImages;
    std::vector<vk::ImageView> swapChainImageViews;
    std::vector<vk::Framebuffer> swapChainFramebuffers;

    std::vector<vk::Buffer> uniformBufers;
    std::vector<vk::DeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;

    // Offscreen rendering resources
    vk::RenderPass offscreenRenderPass;
    vk::Pipeline offscreenPipeline;
    vk::PipelineLayout offscreenPipelineLayout;
    std::vector<vk::Framebuffer> offscreenFramebuffers;
    std::vector<vk::ImageView> offscreenImageViews;
    vk::CommandBuffer offscreenCommandBuffer;

    // TODO: Destruir tudo. Ou criando unique_ptrs ou usando vk_raii

    bool framebufferResized = false;

    uint32_t currentFrame{};

    static constexpr std::array<Vertex, 8> vertices = {{
        {{-0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {-1.0f, -1.0f, -1.0f}},  // 0
        {{0.5f, -0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, -1.0f, -1.0f}},    // 1
        {{0.5f, 0.5f, -0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f, -1.0f}},      // 2
        {{-0.5f, 0.5f, -0.5f}, {1.0f, 1.0f, 1.0f}, {-1.0f, 1.0f, -1.0f}},    // 3
        {{-0.5f, -0.5f, 0.5f}, {1.0f, 0.0f, 1.0f}, {-1.0f, -1.0f, 1.0f}},    // 4
        {{0.5f, -0.5f, 0.5f}, {1.0f, 1.0f, 0.0f}, {1.0f, -1.0f, 1.0f}},      // 5
        {{0.5f, 0.5f, 0.5f}, {0.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}},        // 6
        {{-0.5f, 0.5f, 0.5f}, {0.5f, 0.5f, 0.5f}, {-1.0f, 1.0f, 1.0f}},      // 7
    }};

    static constexpr std::array<uint32_t, 36> indices = {
        // Front face (z = 0.5)
        4, 5, 6, 4, 6, 7,
        // Back face (z = -0.5)
        1, 0, 3, 1, 3, 2,
        // Left face (x = -0.5)
        0, 4, 7, 0, 7, 3,
        // Right face (x = 0.5)
        5, 1, 2, 5, 2, 6,
        // Bottom face (y = -0.5)
        0, 1, 5, 0, 5, 4,
        // Top face (y = 0.5)
        3, 7, 6, 3, 6, 2};

    void initVulkan() {
        createVkInstance();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createDescriptorSetLayout();
        createGraphicsPipeline();
        createCommandPool();
        createDepthResources();
        createFramebuffers();

        createEmptyCubemapTexture();
        createOffscreenRenderPass();
        createOffscreenPipeline();
        createOffscreenFramebuffers();
        renderToCubemap();
        saveCubemapToPNG();

        createTextureImageView();
        createTextureSampler();
        createVertexBuffer();
        createIndexBuffer();
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();
        createSyncObjects();
    }

    void createDepthResources() {
        auto depthFormat = findDepthFormat();

        createImage(swapChainExtent.width, swapChainExtent.height, depthFormat, vk::ImageTiling::eOptimal,
                    vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::MemoryPropertyFlagBits::eDeviceLocal,
                    depthImage, depthImageMemory, 1);
        depthImageView =
            createImageView(depthImage, depthFormat, vk::ImageAspectFlagBits::eDepth, vk::ImageViewType::e2D, 1);
    }

    vk::Format findDepthFormat() const {
        return findSupportedFormat({vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint},
                                   vk::ImageTiling::eOptimal, vk::FormatFeatureFlagBits::eDepthStencilAttachment);
    }

    bool hasStencilComponent(vk::Format format) const {
        return format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eD24UnormS8Uint;
    }

    vk::Format findSupportedFormat(const std::vector<vk::Format>& candidates,
                                   vk::ImageTiling tiling,
                                   vk::FormatFeatureFlags features) const {
        for (const auto& format : candidates) {
            const auto props = physicalDevice.getFormatProperties(format);

            if (tiling == vk::ImageTiling::eLinear && (props.linearTilingFeatures & features) == features) {
                return format;
            } else if (tiling == vk::ImageTiling::eOptimal && (props.optimalTilingFeatures & features) == features) {
                return format;
            }
        }

        throw std::runtime_error("failed to find supported format!");
    }

    void createTextureSampler() {
        const auto properties = physicalDevice.getProperties();

        const vk::SamplerCreateInfo samplerInfo{.sType = vk::StructureType::eSamplerCreateInfo,
                                                .magFilter = vk::Filter::eLinear,
                                                .minFilter = vk::Filter::eLinear,
                                                .mipmapMode = vk::SamplerMipmapMode::eLinear,
                                                .addressModeU = vk::SamplerAddressMode::eClampToEdge,
                                                .addressModeV = vk::SamplerAddressMode::eClampToEdge,
                                                .addressModeW = vk::SamplerAddressMode::eClampToEdge,
                                                .anisotropyEnable = vk::True,
                                                .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
                                                .borderColor = vk::BorderColor::eIntOpaqueBlack,
                                                .unnormalizedCoordinates = vk::False,
                                                .compareEnable = vk::False,
                                                .compareOp = vk::CompareOp::eAlways,
                                                .mipLodBias = 0.0f,
                                                .minLod = 0.0f,
                                                .maxLod = 0.0f};

        textureSampler = device.createSampler(samplerInfo);
    }

    void createTextureImageView() {
        textureImageView = createImageView(textureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageAspectFlagBits::eColor,
                                           vk::ImageViewType::eCube, 6);
    }

    void copyBufferToImage(const vk::Buffer buffer,
                           const vk::Image image,
                           const uint32_t width,
                           const uint32_t height,
                           const uint32_t layerCount) const {
        auto scopedCommandBuffer = ScopedOneTimeCommandBuffer(device, commandPool, graphicsQueue);

        const vk::BufferImageCopy region{
            .bufferOffset = 0,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource = vk::ImageSubresourceLayers{.mipLevel = 0,
                                                           .baseArrayLayer = 0,
                                                           .layerCount = layerCount,
                                                           .aspectMask = vk::ImageAspectFlagBits::eColor},
            .imageOffset = {0, 0, 0},
            .imageExtent = {width, height, 1},
        };

        scopedCommandBuffer.commandBuffer.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, 1,
                                                            &region);
    }

    void transitionImageLayout(const vk::Image image,
                               vk::Format /* format */,
                               const vk::ImageLayout oldLayout,
                               const vk::ImageLayout newLayout,
                               const uint32_t layerCount) const {
        auto scopedCommandBuffer = ScopedOneTimeCommandBuffer(device, commandPool, graphicsQueue);

        vk::PipelineStageFlagBits srcStage;
        vk::PipelineStageFlagBits dstStage;

        vk::ImageMemoryBarrier barrier{
            .sType = vk::StructureType::eImageMemoryBarrier,
            .oldLayout = oldLayout,
            .newLayout = newLayout,
            .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
            .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
            .image = image,
            .subresourceRange = vk::ImageSubresourceRange{.aspectMask = vk::ImageAspectFlagBits::eColor,
                                                          .baseMipLevel = 0,
                                                          .levelCount = 1,
                                                          .baseArrayLayer = 0,
                                                          .layerCount = layerCount},
        };

        if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) {
            barrier.srcAccessMask = vk::AccessFlags{};
            barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

            srcStage = vk::PipelineStageFlagBits::eTopOfPipe;
            dstStage = vk::PipelineStageFlagBits::eTransfer;
        } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal &&
                   newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
            barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

            srcStage = vk::PipelineStageFlagBits::eTransfer;
            dstStage = vk::PipelineStageFlagBits::eFragmentShader;
        } else {
            throw std::invalid_argument("unsupported layout transition!");
        }

        scopedCommandBuffer.commandBuffer.pipelineBarrier(srcStage, dstStage, vk::DependencyFlags{}, 0, nullptr, 0,
                                                          nullptr, 1, &barrier);
    }

    void createImage(const uint32_t width,
                     const uint32_t height,
                     const vk::Format format,
                     const vk::ImageTiling tiling,
                     const vk::Flags<vk::ImageUsageFlagBits> usage,
                     const vk::Flags<vk::MemoryPropertyFlagBits> properties,
                     vk::Image& image,
                     vk::DeviceMemory& imageMemory,
                     const uint32_t arrayLayers,
                     const vk::ImageCreateFlags flags = {}) const {
        const vk::ImageCreateInfo imageInfo{
            .sType = vk::StructureType::eImageCreateInfo,
            .flags = flags,
            .imageType = vk::ImageType::e2D,
            .format = format,
            .extent =
                vk::Extent3D{
                    .width = static_cast<uint32_t>(width), .height = static_cast<uint32_t>(height), .depth = 1},
            .mipLevels = 1,
            .arrayLayers = arrayLayers,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = tiling,
            .usage = usage,
            .sharingMode = vk::SharingMode::eExclusive,
            .initialLayout = vk::ImageLayout::eUndefined,
        };

        image = device.createImage(imageInfo);
        const auto memRequirements = device.getImageMemoryRequirements(image);

        const vk::MemoryAllocateInfo allocInfo{
            .sType = vk::StructureType::eMemoryAllocateInfo,
            .allocationSize = memRequirements.size,
            .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)};

        imageMemory = device.allocateMemory(allocInfo);
        device.bindImageMemory(image, imageMemory, 0);
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

    void createDescriptorSets() {
        std::vector<vk::DescriptorSetLayout> layouts(maxFramesInFlight, descriptorSetLayout);

        const vk::DescriptorSetAllocateInfo allocInfo{.sType = vk::StructureType::eDescriptorSetAllocateInfo,
                                                      .descriptorPool = descriptorPool,
                                                      .descriptorSetCount = maxFramesInFlight,
                                                      .pSetLayouts = layouts.data()};

        descriptorSets = device.allocateDescriptorSets(allocInfo);

        for (size_t i = 0; i < descriptorSets.size(); i++) {
            vk::DescriptorBufferInfo bufferInfo{
                .buffer = uniformBufers[i], .offset = 0, .range = sizeof(UniformBufferObject)};

            vk::DescriptorImageInfo imageInfo{.sampler = textureSampler,
                                              .imageView = textureImageView,
                                              .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};

            const vk::WriteDescriptorSet bufferDescriptorWrite{.sType = vk::StructureType::eWriteDescriptorSet,
                                                               .dstSet = descriptorSets[i],
                                                               .dstBinding = 0,
                                                               .dstArrayElement = 0,
                                                               .descriptorType = vk::DescriptorType::eUniformBuffer,
                                                               .descriptorCount = 1,
                                                               .pBufferInfo = &bufferInfo,
                                                               .pImageInfo = nullptr,
                                                               .pTexelBufferView = nullptr};

            const vk::WriteDescriptorSet samplerDescriptorWrite{
                .sType = vk::StructureType::eWriteDescriptorSet,
                .dstSet = descriptorSets[i],
                .dstBinding = 1,
                .dstArrayElement = 0,
                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                .descriptorCount = 1,
                .pImageInfo = &imageInfo};

            std::array descriptorWrites = {bufferDescriptorWrite, samplerDescriptorWrite};

            device.updateDescriptorSets(descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
        }
    }

    void createDescriptorPool() {
        static constexpr vk::DescriptorPoolSize uboPoolSize{.type = vk::DescriptorType::eUniformBuffer,
                                                            .descriptorCount = maxFramesInFlight};

        static constexpr vk::DescriptorPoolSize samplerPoolSize{.type = vk::DescriptorType::eCombinedImageSampler,
                                                                .descriptorCount = maxFramesInFlight};

        static constexpr std::array<vk::DescriptorPoolSize, 2> poolSizes = {uboPoolSize, samplerPoolSize};

        static constexpr vk::DescriptorPoolCreateInfo poolInfo{.sType = vk::StructureType::eDescriptorPoolCreateInfo,
                                                               .poolSizeCount = poolSizes.size(),
                                                               .pPoolSizes = poolSizes.data(),
                                                               .maxSets = maxFramesInFlight};

        descriptorPool = device.createDescriptorPool(poolInfo);
    }

    void createUniformBuffers() {
        uniformBufers.resize(maxFramesInFlight);
        uniformBuffersMemory.resize(maxFramesInFlight);
        uniformBuffersMapped.resize(maxFramesInFlight);

        for (size_t i = 0; i < maxFramesInFlight; i++) {
            constexpr vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
            createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer,
                         vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                         uniformBufers[i], uniformBuffersMemory[i]);

            uniformBuffersMapped[i] = device.mapMemory(uniformBuffersMemory[i], 0, bufferSize);
        }
    }

    void createDescriptorSetLayout() {
        static constexpr vk::DescriptorSetLayoutBinding uboLayoutBinding{
            .binding = 0,
            .descriptorType = vk::DescriptorType::eUniformBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eVertex,
            .pImmutableSamplers = nullptr};

        static constexpr vk::DescriptorSetLayoutBinding samplerLayoutBinding{
            .binding = 1,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .pImmutableSamplers = nullptr,
            .stageFlags = vk::ShaderStageFlagBits::eFragment};

        static constexpr std::array<vk::DescriptorSetLayoutBinding, 2> bindings = {uboLayoutBinding,
                                                                                   samplerLayoutBinding};

        static constexpr vk::DescriptorSetLayoutCreateInfo layoutInfo{
            .sType = vk::StructureType::eDescriptorSetLayoutCreateInfo,
            .bindingCount = bindings.size(),
            .pBindings = bindings.data()};

        descriptorSetLayout = device.createDescriptorSetLayout(layoutInfo);
    }

    void createIndexBuffer() {
        constexpr vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();

        vk::Buffer stagingBuffer;
        vk::DeviceMemory statingBufferMemory;

        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                     vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                     stagingBuffer, statingBufferMemory);

        const auto data = device.mapMemory(statingBufferMemory, 0, bufferSize);
        memcpy(data, indices.data(), bufferSize);
        device.unmapMemory(statingBufferMemory);

        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
                     vk::MemoryPropertyFlagBits::eDeviceLocal, indexBuffer, indexBufferMemory);

        copyBuffer(stagingBuffer, indexBuffer, bufferSize);
        device.destroyBuffer(stagingBuffer);
        device.freeMemory(statingBufferMemory);
    }

    void createBuffer(const vk::DeviceSize size,
                      const vk::Flags<vk::BufferUsageFlagBits> usage,
                      const vk::Flags<vk::MemoryPropertyFlagBits> properties,
                      vk::Buffer& buffer,
                      vk::DeviceMemory& bufferMemory) const {
        const vk::BufferCreateInfo bufferInfo{.sType = vk::StructureType::eBufferCreateInfo,
                                              .size = size,
                                              .usage = usage,
                                              .sharingMode = vk::SharingMode::eExclusive};

        buffer = device.createBuffer(bufferInfo);

        const auto memRequirements = device.getBufferMemoryRequirements(buffer);

        const vk::MemoryAllocateInfo allocInfo{
            .sType = vk::StructureType::eMemoryAllocateInfo,
            .allocationSize = memRequirements.size,
            .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)};

        bufferMemory = device.allocateMemory(allocInfo);
        device.bindBufferMemory(buffer, bufferMemory, 0);
    }

    void createVertexBuffer() {
        constexpr vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

        vk::Buffer stagingBuffer;
        vk::DeviceMemory statingBufferMemory;

        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                     vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                     stagingBuffer, statingBufferMemory);

        const auto data = device.mapMemory(statingBufferMemory, 0, bufferSize);
        memcpy(data, vertices.data(), bufferSize);
        device.unmapMemory(statingBufferMemory);

        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
                     vk::MemoryPropertyFlagBits::eDeviceLocal, vertexBuffer, vertexBufferMemory);

        copyBuffer(stagingBuffer, vertexBuffer, bufferSize);
        device.destroyBuffer(stagingBuffer);
        device.freeMemory(statingBufferMemory);
    }

    void copyBuffer(const vk::Buffer srcBuffer, const vk::Buffer dstBuffer, const vk::DeviceSize size) const {
        auto scopedCommandBuffer = ScopedOneTimeCommandBuffer(device, commandPool, graphicsQueue);

        const vk::BufferCopy copyRegion{.srcOffset = 0, .dstOffset = 0, .size = size};

        scopedCommandBuffer.commandBuffer.copyBuffer(srcBuffer, dstBuffer, 1, &copyRegion);
    }

    [[nodiscard]] uint32_t findMemoryType(const uint32_t typeFilter,
                                          const vk::Flags<vk::MemoryPropertyFlagBits> properties) const {
        const auto memProperties = physicalDevice.getMemoryProperties();

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }

    void cleanupSwapChain() const {
        for (auto& framebuffer : swapChainFramebuffers) {
            device.destroyFramebuffer(framebuffer);
        }

        for (auto& imageView : swapChainImageViews) {
            device.destroyImageView(imageView);
        }

        device.destroySwapchainKHR(swapChain);
    }

    void recreateSwapChain() {
        int width{}, height{};
        glfwGetFramebufferSize(window.get(), &width, &height);

        while (width == 0 && height == 0) {
            glfwGetFramebufferSize(window.get(), &width, &height);
            glfwWaitEvents();
        }

        device.waitIdle();

        cleanupSwapChain();

        createSwapChain();
        createImageViews();
        createDepthResources();
        createFramebuffers();
    }

    void createSyncObjects() {
        imageAvailableSemaphores.resize(maxFramesInFlight);
        renderFinishedSemaphores.resize(maxFramesInFlight);
        inFlightFences.resize(maxFramesInFlight);

        constexpr vk::SemaphoreCreateInfo semaphoreInfo{.sType = vk::StructureType::eSemaphoreCreateInfo};

        constexpr vk::FenceCreateInfo fenceInfo{.sType = vk::StructureType::eFenceCreateInfo,
                                                .flags = vk::FenceCreateFlagBits::eSignaled};

        for (size_t i = 0; i < maxFramesInFlight; i++) {
            imageAvailableSemaphores[i] = device.createSemaphore(semaphoreInfo);
            renderFinishedSemaphores[i] = device.createSemaphore(semaphoreInfo);
            inFlightFences[i] = device.createFence(fenceInfo);
        }
    }

    void recordCommandBuffer(const vk::CommandBuffer& commandBuffer, const uint32_t imageIndex) const {
        constexpr vk::CommandBufferBeginInfo beginInfo{.sType = vk::StructureType::eCommandBufferBeginInfo,
                                                       // .flags = 0,
                                                       .pInheritanceInfo = nullptr};

        commandBuffer.begin(beginInfo);

        vk::ClearValue clearColor = {.color.float32 = std::array<float, 4>{1.0f, 1.0f, 1.0f, 1.0f}};
        vk::ClearValue clearDepth = {.depthStencil = {.depth = 1.0f, .stencil = 0}};
        std::array clearValues = {clearColor, clearDepth};

        const vk::RenderPassBeginInfo renderPassInfo{.sType = vk::StructureType::eRenderPassBeginInfo,
                                                     .renderPass = renderPass,
                                                     .framebuffer = swapChainFramebuffers[imageIndex],
                                                     .renderArea.offset = {0, 0},
                                                     .renderArea.extent = swapChainExtent,
                                                     .clearValueCount = clearValues.size(),
                                                     .pClearValues = clearValues.data()};

        commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);

        const auto vertexBuffers = std::array{vertexBuffer};
        constexpr std::array<vk::DeviceSize, 1> offsets{};

        commandBuffer.bindVertexBuffers(0, 1, vertexBuffers.data(), offsets.data());
        commandBuffer.bindIndexBuffer(indexBuffer, 0, vk::IndexType::eUint32);

        const vk::Viewport viewport{.x = 0.0f,
                                    .y = 0.0f,
                                    .width = static_cast<float>(swapChainExtent.width),
                                    .height = static_cast<float>(swapChainExtent.height),
                                    .minDepth = 0.0f,
                                    .maxDepth = 1.0f};

        commandBuffer.setViewport(0, 1, &viewport);

        const vk::Rect2D scissor{.offset = {0, 0}, .extent = swapChainExtent};
        commandBuffer.setScissor(0, 1, &scissor);

        // commandBuffer.draw(vertices.size(), 1, 0, 0); draw with vertex array only
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, 1,
                                         &descriptorSets[currentFrame], 0, nullptr);
        commandBuffer.drawIndexed(indices.size(), 1, 0, 0, 0);
        commandBuffer.endRenderPass();
        commandBuffer.end();
    }

    void createCommandBuffers() {
        const vk::CommandBufferAllocateInfo allocInfo{.sType = vk::StructureType::eCommandBufferAllocateInfo,
                                                      .commandPool = commandPool,
                                                      .level = vk::CommandBufferLevel::ePrimary,
                                                      .commandBufferCount = static_cast<uint32_t>(maxFramesInFlight)};

        commandBuffers = device.allocateCommandBuffers(allocInfo);
    }

    void createCommandPool() {
        const auto queueFamilyIndices = findQueueFamilies(physicalDevice);

        vk::CommandPoolCreateInfo poolInfo{.sType = vk::StructureType::eCommandPoolCreateInfo,
                                           .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                           .queueFamilyIndex = queueFamilyIndices.graphicsFamily.value()};

        commandPool = device.createCommandPool(poolInfo);
    }

    void createFramebuffers() {
        swapChainFramebuffers.resize(swapChainImageViews.size());

        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            std::array attachments = {swapChainImageViews[i], depthImageView};

            vk::FramebufferCreateInfo framebufferInfo{.sType = vk::StructureType::eFramebufferCreateInfo,
                                                      .renderPass = renderPass,
                                                      .attachmentCount = attachments.size(),
                                                      .pAttachments = attachments.data(),
                                                      .width = swapChainExtent.width,
                                                      .height = swapChainExtent.height,
                                                      .layers = 1};

            swapChainFramebuffers[i] = device.createFramebuffer(framebufferInfo);
        }
    }

    void createRenderPass() {
        vk::AttachmentDescription colorAttachment{.format = swapChainImageFormat,
                                                  .samples = vk::SampleCountFlagBits::e1,
                                                  .loadOp = vk::AttachmentLoadOp::eClear,
                                                  .storeOp = vk::AttachmentStoreOp::eStore,
                                                  .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
                                                  .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
                                                  .initialLayout = vk::ImageLayout::eUndefined,
                                                  .finalLayout = vk::ImageLayout::ePresentSrcKHR};

        vk::AttachmentReference colorAttachmentRef{.attachment = 0, .layout = vk::ImageLayout::eColorAttachmentOptimal};

        vk::AttachmentDescription depthAttachment{.format = findDepthFormat(),
                                                  .samples = vk::SampleCountFlagBits::e1,
                                                  .loadOp = vk::AttachmentLoadOp::eClear,
                                                  .storeOp = vk::AttachmentStoreOp::eDontCare,
                                                  .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
                                                  .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
                                                  .initialLayout = vk::ImageLayout::eUndefined,
                                                  .finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal};

        vk::AttachmentReference depthAttachmentRef{.attachment = 1,
                                                   .layout = vk::ImageLayout::eDepthStencilAttachmentOptimal};

        vk::SubpassDescription subpass{.pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
                                       .colorAttachmentCount = 1,
                                       .pColorAttachments = &colorAttachmentRef,
                                       .pDepthStencilAttachment = &depthAttachmentRef};

        vk::SubpassDependency dependency{
            .srcSubpass = vk::SubpassExternal,
            .dstSubpass = 0,
            .srcStageMask =
                vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eLateFragmentTests,
            .dstStageMask =
                vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests,
            .srcAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentWrite,
            .dstAccessMask =
                vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eDepthStencilAttachmentWrite,
        };

        std::array<vk::AttachmentDescription, 2> attachments = {colorAttachment, depthAttachment};

        const vk::RenderPassCreateInfo renderPassInfo{.sType = vk::StructureType::eRenderPassCreateInfo,
                                                      .attachmentCount = attachments.size(),
                                                      .pAttachments = attachments.data(),
                                                      .subpassCount = 1,
                                                      .pSubpasses = &subpass,
                                                      .dependencyCount = 1,
                                                      .pDependencies = &dependency};

        renderPass = device.createRenderPass(renderPassInfo);
    }

    void createGraphicsPipeline() {
        auto vertexShaderCode =
            readFile("/Users/andersonkulitch/Documents/dev/vulkan/shaders/vert.spv");  // esses shaders foram compilados
                                                                                       // usando glslc
        auto fragmentShaderCode =
            readFile("/Users/andersonkulitch/Documents/dev/vulkan/shaders/frag.spv");  // esses shaders foram compilados
                                                                                       // usando glslc

        auto vertexShaderModule = createShaderModule(vertexShaderCode);
        auto fragmentShaderModule = createShaderModule(fragmentShaderCode);

        vk::PipelineShaderStageCreateInfo vertexShaderStageInfo{
            .sType = vk::StructureType::ePipelineShaderStageCreateInfo,
            .stage = vk::ShaderStageFlagBits::eVertex,
            .module = vertexShaderModule,
            .pName = "main",
            // .pSpecializationInfo // Permite configurar constantes que podem ser usadas no shader para otimizar o
            // código em tempo de criação do pipeline.
        };

        vk::PipelineShaderStageCreateInfo fragmentShaderStageInfo{
            .sType = vk::StructureType::ePipelineShaderStageCreateInfo,
            .stage = vk::ShaderStageFlagBits::eFragment,
            .module = fragmentShaderModule,
            .pName = "main",
            // .pSpecializationInfo // Permite configurar constantes que podem ser usadas no shader para otimizar o
            // código em tempo de criação do pipeline.
        };

        vk::PipelineShaderStageCreateInfo shaderStages[] = {vertexShaderStageInfo, fragmentShaderStageInfo};

        constexpr auto bindingDescription = Vertex::getBindingDescription();
        constexpr auto attributeDescription = Vertex::getAttributeDescriptions();

        vk::PipelineVertexInputStateCreateInfo vertexInputCreateInfo{
            .sType = vk::StructureType::ePipelineVertexInputStateCreateInfo,
            .vertexBindingDescriptionCount = 1,
            .vertexAttributeDescriptionCount = attributeDescription.size(),
            .pVertexBindingDescriptions = &bindingDescription,
            .pVertexAttributeDescriptions = attributeDescription.data()};

        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
            .sType = vk::StructureType::ePipelineInputAssemblyStateCreateInfo,
            .topology = vk::PrimitiveTopology::eTriangleList,
            .primitiveRestartEnable = VK_FALSE};

        vk::Viewport viewport{.x = 0.0f,
                              .y = 0.0f,
                              .width = static_cast<float>(swapChainExtent.width),
                              .height = static_cast<float>(swapChainExtent.height),
                              .minDepth = 0.0f,
                              .maxDepth = 1.0f};

        vk::Rect2D scissor{.offset = {0, 0}, .extent = swapChainExtent};

        vk::PipelineViewportStateCreateInfo viewportState{.sType = vk::StructureType::ePipelineViewportStateCreateInfo,
                                                          .viewportCount = 1,
                                                          .pViewports = &viewport,
                                                          .scissorCount = 1,
                                                          .pScissors = &scissor};

        vk::PipelineRasterizationStateCreateInfo rasterizer{
            .sType = vk::StructureType::ePipelineRasterizationStateCreateInfo,
            .depthClampEnable = VK_FALSE,
            .rasterizerDiscardEnable = VK_FALSE,
            .polygonMode = vk::PolygonMode::eFill,
            .lineWidth = 1.0f,
            // .cullMode = vk::CullModeFlagBits::eBack,
            // .frontFace = vk::FrontFace::eClockwise,
            .cullMode = vk::CullModeFlagBits::eNone,
            .frontFace = vk::FrontFace::eCounterClockwise,
            .depthBiasEnable = VK_FALSE,
            .depthBiasConstantFactor = 0.0f,
            .depthBiasClamp = 0.0f,
            .depthBiasSlopeFactor = 0.0f,
        };

        vk::PipelineMultisampleStateCreateInfo multisampling{
            .sType = vk::StructureType::ePipelineMultisampleStateCreateInfo,
            .sampleShadingEnable = VK_FALSE,
            .rasterizationSamples = vk::SampleCountFlagBits::e1,
            .minSampleShading = 1.0f,
            .pSampleMask = nullptr,
            .alphaToCoverageEnable = VK_FALSE,
            .alphaToOneEnable = VK_FALSE};

        vk::PipelineColorBlendAttachmentState colorBlendAttachment{
            .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                              vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
            .blendEnable = VK_FALSE,
            .srcColorBlendFactor = vk::BlendFactor::eOne,
            .dstColorBlendFactor = vk::BlendFactor::eZero,
            .colorBlendOp = vk::BlendOp::eAdd,
            .srcAlphaBlendFactor = vk::BlendFactor::eOne,
            .dstAlphaBlendFactor = vk::BlendFactor::eZero,
            .alphaBlendOp = vk::BlendOp::eAdd,
        };

        vk::PipelineColorBlendStateCreateInfo colorBlending{
            .sType = vk::StructureType::ePipelineColorBlendStateCreateInfo,
            .logicOpEnable = VK_FALSE,
            .logicOp = vk::LogicOp::eCopy,
            .attachmentCount = 1,
            .pAttachments = &colorBlendAttachment,
            //.blendConstants[0] = 0.0f
        };

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo{.sType = vk::StructureType::ePipelineLayoutCreateInfo,
                                                        .setLayoutCount = 1,
                                                        .pSetLayouts = &descriptorSetLayout,
                                                        .pushConstantRangeCount = 0,
                                                        .pPushConstantRanges = nullptr};

        pipelineLayout = device.createPipelineLayout(pipelineLayoutInfo);

        vk::PipelineDepthStencilStateCreateInfo depthStencil{
            .sType = vk::StructureType::ePipelineDepthStencilStateCreateInfo,
            .depthTestEnable = VK_TRUE,
            .depthWriteEnable = VK_TRUE,
            .depthCompareOp = vk::CompareOp::eLess,
            .depthBoundsTestEnable = VK_FALSE,
            .stencilTestEnable = VK_FALSE,
            .front = {},
            .back = {},
            .minDepthBounds = 0.0f,
            .maxDepthBounds = 1.0f};

        std::vector<vk::DynamicState> dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};

        vk::PipelineDynamicStateCreateInfo dynamicState{
            .sType = vk::StructureType::ePipelineDynamicStateCreateInfo,
            .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
            .pDynamicStates = dynamicStates.data()};

        vk::GraphicsPipelineCreateInfo pipelineInfo{
            .sType = vk::StructureType::eGraphicsPipelineCreateInfo,
            .stageCount = 2,
            .pStages = shaderStages,
            .pVertexInputState = &vertexInputCreateInfo,
            .pInputAssemblyState = &inputAssembly,
            .pViewportState = &viewportState,
            .pRasterizationState = &rasterizer,
            .pMultisampleState = &multisampling,
            .pDepthStencilState = &depthStencil,
            .pColorBlendState = &colorBlending,
            .pDynamicState = &dynamicState,
            .layout = pipelineLayout,
            .renderPass = renderPass,
            .subpass = 0,
            .basePipelineHandle = nullptr,
            .basePipelineIndex = -1,
        };

        auto pipelineCreationResult = device.createGraphicsPipeline(nullptr, pipelineInfo);
        if (pipelineCreationResult.result != vk::Result::eSuccess) {
            throw std::runtime_error("failed to create graphics pipeline!");
        }

        graphicsPipeline = pipelineCreationResult.value;

        device.destroyShaderModule(vertexShaderModule);
        device.destroyShaderModule(fragmentShaderModule);
    }

    [[nodiscard]] vk::ShaderModule createShaderModule(const std::vector<char>& code) const {
        const vk::ShaderModuleCreateInfo createInfo{.sType = vk::StructureType::eShaderModuleCreateInfo,
                                                    .codeSize = code.size(),
                                                    .pCode = reinterpret_cast<const uint32_t*>(code.data())};

        return device.createShaderModule(createInfo);
    }

    [[nodiscard]] vk::ImageView createImageView(const vk::Image& image,
                                                const vk::Format format,
                                                const vk::ImageAspectFlagBits aspectFlags,
                                                const vk::ImageViewType viewType,
                                                const uint32_t layerCount,
                                                const uint32_t baseArrayLayer = 0) const {
        const vk::ImageViewCreateInfo createInfo{
            .sType = vk::StructureType::eImageViewCreateInfo,
            .image = image,
            .viewType = viewType,
            .format = format,
            .components.r = vk::ComponentSwizzle::eIdentity,
            .components.g = vk::ComponentSwizzle::eIdentity,
            .components.b = vk::ComponentSwizzle::eIdentity,
            .components.a = vk::ComponentSwizzle::eIdentity,
            .subresourceRange.aspectMask = aspectFlags,
            .subresourceRange.baseMipLevel = 0,
            .subresourceRange.levelCount = 1,
            .subresourceRange.baseArrayLayer = baseArrayLayer,
            .subresourceRange.layerCount = layerCount,
        };

        return device.createImageView(createInfo);
    }

    void createImageViews() {
        swapChainImageViews.resize(swapChainImages.size());

        for (size_t i = 0; i < swapChainImages.size(); i++) {
            swapChainImageViews[i] = createImageView(swapChainImages[i], swapChainImageFormat,
                                                     vk::ImageAspectFlagBits::eColor, vk::ImageViewType::e2D, 1);
        }
    }

    void createSurface() {
        VkSurfaceKHR surfaceHandle;
        if (glfwCreateWindowSurface(instance, window.get(), nullptr, &surfaceHandle) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }

        surface = surfaceHandle;
    }

    void createLogicalDevice() {
        auto [graphicsFamily, presenteFamily] = findQueueFamilies(physicalDevice);

        std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = {graphicsFamily.value(), presenteFamily.value()};

        float queuePriority = 1.0f;

        for (auto queueFamily : uniqueQueueFamilies) {
            vk::DeviceQueueCreateInfo queueCreateInfo{.sType = vk::StructureType::eDeviceQueueCreateInfo,
                                                      .queueFamilyIndex = queueFamily,
                                                      .queueCount = 1,
                                                      .pQueuePriorities = &queuePriority};

            queueCreateInfos.emplace_back(queueCreateInfo);
        }

        vk::PhysicalDeviceFeatures deviceFeatures{.samplerAnisotropy = vk::True};

        vk::DeviceCreateInfo createInfo{.sType = vk::StructureType::eDeviceCreateInfo,
                                        .pQueueCreateInfos = queueCreateInfos.data(),
                                        .queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size()),
                                        .pEnabledFeatures = &deviceFeatures,
                                        .enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
                                        .ppEnabledExtensionNames = deviceExtensions.data()};

        device = physicalDevice.createDevice(createInfo);
        graphicsQueue = device.getQueue(graphicsFamily.value(), 0);
        presenteQueue = device.getQueue(presenteFamily.value(), 0);
    }

    void createSwapChain() {
        const auto [capabilities, formats, presenteModes] = querySwapChainSupport(physicalDevice);

        const auto [format, colorSpace] = chooseSwapSurfaceFormat(formats);
        const auto presenteMode = chooseSwapPresentMode(presenteModes);
        const auto extent = chooseSwapExtent(capabilities);

        swapChainImageFormat = format;
        swapChainExtent = extent;

        uint32_t imageCount = capabilities.minImageCount + 1;

        if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount) {
            imageCount = capabilities.maxImageCount;
        }

        vk::SwapchainCreateInfoKHR createInfo{.sType = vk::StructureType::eSwapchainCreateInfoKHR,
                                              .surface = surface,
                                              .minImageCount = imageCount,
                                              .imageFormat = format,
                                              .imageColorSpace = colorSpace,
                                              .imageExtent = extent,
                                              .imageArrayLayers = 1,
                                              .imageUsage = vk::ImageUsageFlagBits::eColorAttachment};

        const auto [graphicsFamily, presenteFamily] = findQueueFamilies(physicalDevice);
        const uint32_t queueFamilyIndices[] = {graphicsFamily.value(), presenteFamily.value()};

        if (graphicsFamily != presenteFamily) {
            createInfo.imageSharingMode = vk::SharingMode::eConcurrent;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        } else {
            createInfo.imageSharingMode = vk::SharingMode::eExclusive;
            createInfo.queueFamilyIndexCount = 0;
            createInfo.pQueueFamilyIndices = nullptr;
        }

        createInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
        createInfo.presentMode = presenteMode;
        createInfo.clipped = VK_TRUE;
        createInfo.oldSwapchain = nullptr;

        swapChain = device.createSwapchainKHR(createInfo);
        swapChainImages = device.getSwapchainImagesKHR(swapChain);
    }

    void pickPhysicalDevice() {
        const auto devices = instance.enumeratePhysicalDevices();

        if (devices.empty()) {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }

        for (const auto& device : devices) {
            if (isDeviceSuitable(device)) {
                physicalDevice = device;
                return;
            }
        }

        throw std::runtime_error("failed to find a suitable GPU!");
    }

    [[nodiscard]] bool isDeviceSuitable(const vk::PhysicalDevice& device) const {
        // auto properties = device.getProperties();
        // auto features = device.getFeatures();
        const auto indices = findQueueFamilies(device);

        const auto supportedFeatures = device.getFeatures();
        const auto extensionsSupported = checkDeviceExtensionSupport(device);

        const auto swapChainAdequate = [&]() {
            if (extensionsSupported) {
                const auto swapChainSupport = querySwapChainSupport(device);
                return !swapChainSupport.formats.empty() && !swapChainSupport.presenteModes.empty();
            }

            return false;
        }();

        return indices.isComplete() && extensionsSupported && swapChainAdequate && supportedFeatures.samplerAnisotropy;
    }

    static vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats) {
        for (const auto& availableFormat : availableFormats) {
            if (availableFormat.format == vk::Format::eB8G8R8A8Srgb) {
                return availableFormat;
            }
        }

        return availableFormats[0];
    }

    static vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresenteModes) {
        for (const auto& availablePresentMode : availablePresenteModes) {
            if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
                return availablePresentMode;
            }
        }

        return vk::PresentModeKHR::eFifo;
    }

    [[nodiscard]] vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities) const {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        }

        int width, height;
        glfwGetFramebufferSize(window.get(), &width, &height);

        vk::Extent2D actualExtent{
            .width = static_cast<uint32_t>(width),
            .height = static_cast<uint32_t>(height),
        };

        actualExtent.width =
            std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        actualExtent.height =
            std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

        return actualExtent;
    }

    [[nodiscard]] SwapChainSupportDetails querySwapChainSupport(const vk::PhysicalDevice& device) const {
        return SwapChainSupportDetails{.capabilities = device.getSurfaceCapabilitiesKHR(surface),
                                       .formats = device.getSurfaceFormatsKHR(surface),
                                       .presenteModes = device.getSurfacePresentModesKHR(surface)};
    }

    [[nodiscard]] bool checkDeviceExtensionSupport(const vk::PhysicalDevice& device) const {
        const auto availableExtensions = device.enumerateDeviceExtensionProperties();

        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        for (const auto& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName.data());
        }

        for (const auto& requiredExtension : requiredExtensions) {
            fmt::println("Required extension {} not available.", requiredExtension);
        }

        return requiredExtensions.empty();
    }

    [[nodiscard]] QueueFamilyIndices findQueueFamilies(const vk::PhysicalDevice& device) const {
        const auto queueFamilies = device.getQueueFamilyProperties();
        QueueFamilyIndices indices;

        uint32_t i = 0;
        for (const auto& queueFamily : queueFamilies) {
            if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
                indices.graphicsFamily = i;
            }

            const auto presentSupport =
                device.getSurfaceSupportKHR(i, surface);  // Teoricamente podem ser filas diferentes, mas o ideal e mais
                                                          // comum, é que sejam as mesmas filas.

            if (presentSupport) {
                indices.presenteFamily = i;
            }

            if (indices.isComplete()) {
                break;
            }

            i++;
        }

        return indices;
    }

    void createVkInstance() {
        if (enableValidationLayers && !checkValidationLayersSupport()) {
            throw std::runtime_error("validation layers requested, but not available!");
        }

        vk::ApplicationInfo appInfo{.sType = vk::StructureType::eApplicationInfo,
                                    .pApplicationName = "Hello Triangle",
                                    .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
                                    .pEngineName = "No Engine",
                                    .engineVersion = VK_MAKE_VERSION(1, 0, 0),
                                    .apiVersion = VK_API_VERSION_1_4};

        vk::InstanceCreateInfo createInfo{
            .sType = vk::StructureType::eInstanceCreateInfo,
            .flags = vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR,  // Essa flag é necessaria para o MacOS
            .pApplicationInfo = &appInfo,
        };

        uint32_t glfwExtensionCount{};
        const char** p_glfwExtensions;  // TODO: revisar essa alocação, talvez seja um leak

        p_glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char*> glfwExtensions(p_glfwExtensions, p_glfwExtensions + glfwExtensionCount);

#ifdef __APPLE__
        addMacSpecificExtensions(glfwExtensions);
#endif

        createInfo.enabledExtensionCount = static_cast<uint32_t>(glfwExtensions.size());
        createInfo.ppEnabledExtensionNames = glfwExtensions.data();

        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        } else {
            createInfo.enabledLayerCount = 0;
        }

        instance = vk::createInstance(createInfo, nullptr);
        printGlfwRequiredExtensions(glfwExtensions);
    }

    static void printGlfwRequiredExtensions(const std::vector<const char*>& glfwExtensions) {
        const auto extensions = vk::enumerateInstanceExtensionProperties();

        fmt::println("Available extensions:");

        for (const auto& [extensionRawName, specVersion] : extensions) {
            const char* extensionName{extensionRawName.data()};
            fmt::print("\t {}", extensionName);

            auto glfwRequiresExtension = std::ranges::any_of(
                glfwExtensions, [&extensionName](const char* value) { return std::strcmp(extensionName, value) == 0; });

            if (glfwRequiresExtension) {
                fmt::print("*** \t -> GLFW Required\n");
            } else {
                fmt::print("\n");
            }
        }
    }

    [[nodiscard]] bool checkValidationLayersSupport() const {
        auto availableLayers = vk::enumerateInstanceLayerProperties();

        for (const auto& layerName : validationLayers) {
            auto layerFound = std::ranges::any_of(availableLayers, [&layerName](const vk::LayerProperties& layer) {
                return std::strcmp(layerName, layer.layerName.data()) == 0;
            });

            if (!layerFound) {
                return false;
            }
        }

        return true;
    }

    static void addMacSpecificExtensions(std::vector<const char*>& glfwExtensions) {
        // glfwExtensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME); Só é necessario no
        // Vulkan 1.0
        glfwExtensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window.get())) {
            glfwPollEvents();
            drawFrame();
        }

        device.waitIdle();
    }

    void drawFrame() {
        vk::detail::resultCheck(device.waitForFences(1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX),
                                "failed to wait inflightFences.");

        const auto nextImageResult =
            device.acquireNextImageKHR(swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], nullptr);

        if (nextImageResult.result == vk::Result::eErrorOutOfDateKHR) {
            recreateSwapChain();
            return;
        } else if (nextImageResult.result != vk::Result::eSuccess &&
                   nextImageResult.result != vk::Result::eSuboptimalKHR) {
            vk::detail::resultCheck(nextImageResult.result, "Failed to acquire swap chain image!");
        }

        // Only reset the fence if we are submitting work
        vk::detail::resultCheck(device.resetFences(1, &inFlightFences[currentFrame]),
                                "failed to reset inflightFences.");

        uint32_t imageIndex = nextImageResult.value;

        commandBuffers[currentFrame].reset();
        recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

        updateUniformBuffer(currentFrame);

        vk::Semaphore waitSemaphores[] = {imageAvailableSemaphores[currentFrame]};
        vk::PipelineStageFlags waitStages[] = {vk::PipelineStageFlagBits::eColorAttachmentOutput};
        vk::Semaphore signalSemaphores[] = {renderFinishedSemaphores[currentFrame]};
        vk::SubmitInfo submitInfo{.sType = vk::StructureType::eSubmitInfo,
                                  .waitSemaphoreCount = 1,
                                  .pWaitSemaphores = waitSemaphores,
                                  .pWaitDstStageMask = waitStages,
                                  .commandBufferCount = 1,
                                  .pCommandBuffers = &commandBuffers[currentFrame],
                                  .signalSemaphoreCount = 1,
                                  .pSignalSemaphores = signalSemaphores};

        const auto submitResult = graphicsQueue.submit(1, &submitInfo, inFlightFences[currentFrame]);
        vk::detail::resultCheck(submitResult, "failed to submit command to graphicsQueue.");

        vk::SwapchainKHR swapChains[] = {swapChain};

        vk::PresentInfoKHR presentInfo{.sType = vk::StructureType::ePresentInfoKHR,
                                       .waitSemaphoreCount = 1,
                                       .pWaitSemaphores = signalSemaphores,
                                       .swapchainCount = 1,
                                       .pSwapchains = swapChains,
                                       .pImageIndices = &imageIndex,
                                       .pResults = nullptr};

        const auto presentResult = presenteQueue.presentKHR(&presentInfo);

        if (presentResult == vk::Result::eErrorOutOfDateKHR || presentResult == vk::Result::eSuboptimalKHR ||
            framebufferResized) {
            framebufferResized = false;
            recreateSwapChain();
        } else {
            vk::detail::resultCheck(presentResult, "Failed to present swap chain image!");
        }

        currentFrame = (currentFrame + 1) % maxFramesInFlight;
    }

    void updateUniformBuffer(uint32_t currentImage) {
        static auto startTime = std::chrono::high_resolution_clock::now();

        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();
        UniformBufferObject ubo{
            .model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
            .proj = glm::perspective(glm::radians(45.0F), swapChainExtent.width / (float)swapChainExtent.height, 0.1f,
                                     10.0f),
            .view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f))};

        ubo.proj[1][1] *= -1.0f;
        std::memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
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
        auto vertShaderModule = createShaderModule(readFile("shaders/offscreen_vert.spv"));
        auto fragShaderModule = createShaderModule(readFile("shaders/offscreen_frag.spv"));

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

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
            .sType = vk::StructureType::ePipelineVertexInputStateCreateInfo,
            .vertexBindingDescriptionCount = 0,
            .vertexAttributeDescriptionCount = 0};

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

        vk::PushConstantRange pushConstantRange{
            .stageFlags = vk::ShaderStageFlagBits::eVertex, .offset = 0, .size = sizeof(glm::vec3)};

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo{.sType = vk::StructureType::ePipelineLayoutCreateInfo,
                                                        .setLayoutCount = 0,
                                                        .pushConstantRangeCount = 1,
                                                        .pPushConstantRanges = &pushConstantRange};

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

        auto result = device.createGraphicsPipeline(nullptr, pipelineInfo);
        offscreenPipeline = result.value;

        device.destroyShaderModule(fragShaderModule);
        device.destroyShaderModule(vertShaderModule);
    }

    void createOffscreenFramebuffers() {
        const uint32_t cubemapSize = std::max(swapChainExtent.width, swapChainExtent.height);
        offscreenFramebuffers.resize(6);
        offscreenImageViews.resize(6);

        for (uint32_t i = 0; i < 6; ++i) {
            offscreenImageViews[i] = createImageView(textureImage, vk::Format::eR8G8B8A8Srgb,
                                                     vk::ImageAspectFlagBits::eColor, vk::ImageViewType::e2D, 1, i);

            vk::FramebufferCreateInfo framebufferInfo{.sType = vk::StructureType::eFramebufferCreateInfo,
                                                      .renderPass = offscreenRenderPass,
                                                      .attachmentCount = 1,
                                                      .pAttachments = &offscreenImageViews[i],
                                                      .width = cubemapSize,
                                                      .height = cubemapSize,
                                                      .layers = 1};

            offscreenFramebuffers[i] = device.createFramebuffer(framebufferInfo);
        }
    }

    void renderToCubemap() {
        // Colors for each face: Red, Green, Blue, Magenta, Cyan, White
        const std::array<glm::vec3, 6> faceColors = {{
            {1.0f, 0.0f, 0.0f},  // Face 0: Red
            {0.0f, 1.0f, 0.0f},  // Face 1: Green
            {0.0f, 0.0f, 1.0f},  // Face 2: Blue
            {1.0f, 0.0f, 1.0f},  // Face 3: Magenta
            {0.0f, 1.0f, 1.0f},  // Face 4: Cyan
            {1.0f, 1.0f, 1.0f}   // Face 5: White
        }};

        // Allocate command buffer for offscreen rendering
        vk::CommandBufferAllocateInfo allocInfo{.sType = vk::StructureType::eCommandBufferAllocateInfo,
                                                .commandPool = commandPool,
                                                .level = vk::CommandBufferLevel::ePrimary,
                                                .commandBufferCount = 1};

        offscreenCommandBuffer = device.allocateCommandBuffers(allocInfo)[0];

        vk::CommandBufferBeginInfo beginInfo{.sType = vk::StructureType::eCommandBufferBeginInfo,
                                             .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit};

        offscreenCommandBuffer.begin(beginInfo);

        for (uint32_t face = 0; face < 6; ++face) {
            // Transition this face from shader read only to color attachment optimal
            vk::ImageMemoryBarrier barrier{
                .sType = vk::StructureType::eImageMemoryBarrier,
                .srcAccessMask = vk::AccessFlagBits::eShaderRead,
                .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
                .oldLayout = vk::ImageLayout::eUndefined,
                .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
                .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
                .image = textureImage,
                .subresourceRange = vk::ImageSubresourceRange{.aspectMask = vk::ImageAspectFlagBits::eColor,
                                                              .baseMipLevel = 0,
                                                              .levelCount = 1,
                                                              .baseArrayLayer = face,
                                                              .layerCount = 1}};

            offscreenCommandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eFragmentShader,
                                                   vk::PipelineStageFlagBits::eColorAttachmentOutput,
                                                   vk::DependencyFlags{}, 0, nullptr, 0, nullptr, 1, &barrier);

            vk::ClearValue clearColor{};
            clearColor.color.float32[0] = 0.0f;
            clearColor.color.float32[1] = 0.0f;
            clearColor.color.float32[2] = 0.0f;
            clearColor.color.float32[3] = 1.0f;

            const uint32_t cubemapSize = std::max(swapChainExtent.width, swapChainExtent.height);
            vk::RenderPassBeginInfo renderPassInfo{.sType = vk::StructureType::eRenderPassBeginInfo,
                                                   .renderPass = offscreenRenderPass,
                                                   .framebuffer = offscreenFramebuffers[face],
                                                   .renderArea = {{0, 0}, {cubemapSize, cubemapSize}},
                                                   .clearValueCount = 1,
                                                   .pClearValues = &clearColor};

            offscreenCommandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
            offscreenCommandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, offscreenPipeline);

            // Push the color for this face
            offscreenCommandBuffer.pushConstants(offscreenPipelineLayout, vk::ShaderStageFlagBits::eVertex, 0,
                                                 sizeof(glm::vec3), &faceColors[face]);

            offscreenCommandBuffer.draw(6, 1, 0, 0);
            offscreenCommandBuffer.endRenderPass();

            // Transition this face back to shader read only optimal
            barrier.srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
            barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
            barrier.oldLayout = vk::ImageLayout::eUndefined;
            barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

            offscreenCommandBuffer.pipelineBarrier(
                vk::PipelineStageFlagBits::eColorAttachmentOutput,
                vk::PipelineStageFlagBits::eFragmentShader,
                vk::DependencyFlags{}, 0, nullptr, 0, nullptr, 1, &barrier);
        }

        offscreenCommandBuffer.end();

        // Submit the command buffer
        vk::SubmitInfo submitInfo{.sType = vk::StructureType::eSubmitInfo,
                                  .commandBufferCount = 1,
                                  .pCommandBuffers = &offscreenCommandBuffer};

        vk::detail::resultCheck(graphicsQueue.submit(1, &submitInfo, nullptr),
                                "Failed to submit offscreen rendering commands");
        graphicsQueue.waitIdle();

        device.freeCommandBuffers(commandPool, 1, &offscreenCommandBuffer);
    }

    void saveCubemapToPNG() {
        const uint32_t cubemapSize = std::max(swapChainExtent.width, swapChainExtent.height);
        const vk::DeviceSize imageSize = cubemapSize * cubemapSize * 4;

        // Create staging buffer
        vk::Buffer stagingBuffer;
        vk::DeviceMemory stagingBufferMemory;
        createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferDst,
                     vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                     stagingBuffer, stagingBufferMemory);

        const std::array<std::string, 6> faceNames = {"face0_red",     "face1_green", "face2_blue",
                                                      "face3_magenta", "face4_cyan",  "face5_white"};

        // Save each face individually
        for (uint32_t face = 0; face < 6; ++face) {
            // Copy each face to staging buffer and save
            {
                auto commandBuffer = ScopedOneTimeCommandBuffer(device, commandPool, graphicsQueue);

                // // Transition image layout for transfer
                vk::ImageMemoryBarrier barrier{
                    .sType = vk::StructureType::eImageMemoryBarrier,
                    .srcAccessMask = vk::AccessFlagBits::eShaderRead,
                    .dstAccessMask = vk::AccessFlagBits::eTransferRead,
                    .oldLayout = vk::ImageLayout::eUndefined,
                    .newLayout = vk::ImageLayout::eTransferSrcOptimal,
                    .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
                    .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
                    .image = textureImage,
                    .subresourceRange = vk::ImageSubresourceRange{.aspectMask = vk::ImageAspectFlagBits::eColor,
                                                                  .baseMipLevel = 0,
                                                                  .levelCount = 1,
                                                                  .baseArrayLayer = face,
                                                                  .layerCount = 1}};

                commandBuffer.commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eFragmentShader,
                                                            vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlags{},
                                                            0, nullptr, 0, nullptr, 1, &barrier);

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

                commandBuffer.commandBuffer.copyImageToBuffer(textureImage, vk::ImageLayout::eTransferSrcOptimal,
                                                              stagingBuffer, 1, &region);

                // // Transition back to shader read only
                barrier.oldLayout = vk::ImageLayout::eUndefined;
                barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
                barrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
                barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

                commandBuffer.commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                                            vk::PipelineStageFlagBits::eFragmentShader,
                                                            vk::DependencyFlags{}, 0, nullptr, 0, nullptr, 1, &barrier);
            }

            // Map memory and save this face to PNG
            void* data = device.mapMemory(stagingBufferMemory, 0, imageSize);
            const std::string filename = faceNames[face] + ".png";
            stbi_write_png(filename.c_str(), cubemapSize, cubemapSize, 4, static_cast<char*>(data), cubemapSize * 4);
            device.unmapMemory(stagingBufferMemory);

            fmt::print("Saved cubemap face {} to {}\n", face, filename);
        }

        device.destroyBuffer(stagingBuffer);
        device.freeMemory(stagingBufferMemory);
    }

    void cleanup() const {
        glfwTerminate();

        // Clean up offscreen rendering resources
        for (auto& framebuffer : offscreenFramebuffers) {
            device.destroyFramebuffer(framebuffer);
        }
        for (auto& imageView : offscreenImageViews) {
            device.destroyImageView(imageView);
        }
        device.destroyPipeline(offscreenPipeline);
        device.destroyPipelineLayout(offscreenPipelineLayout);
        device.destroyRenderPass(offscreenRenderPass);

        device.destroyBuffer(vertexBuffer);
        device.freeMemory(vertexBufferMemory);

        device.destroyBuffer(indexBuffer);
        device.freeMemory(indexBufferMemory);

        for (size_t i = 0; i < maxFramesInFlight; i++) {
            device.destroyBuffer(uniformBufers[i]);
            device.freeMemory(uniformBuffersMemory[i]);
        }

        device.destroyDescriptorPool(descriptorPool);
        device.destroyDescriptorSetLayout(descriptorSetLayout);

        device.destroySampler(textureSampler);
        device.destroyImageView(textureImageView);
        device.destroyImage(textureImage);
        device.freeMemory(textureImageMemory);

        device.destroyImageView(depthImageView);
        device.destroyImage(depthImage);
        device.freeMemory(depthImageMemory);
    }

    void initWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
        window.reset(glfwCreateWindow(width, height, "Vulkan", nullptr, nullptr));
        glfwSetWindowUserPointer(window.get(), this);
        glfwSetFramebufferSizeCallback(window.get(), framebufferResizeCallback);
    }

    static void framebufferResizeCallback(GLFWwindow* window, int /* width */, int /* height */) {
        const auto app = static_cast<HelloTriangle*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

   public:
    explicit HelloTriangle()
        : window(std::unique_ptr<GLFWwindow, decltype(&::glfwDestroyWindow)>(nullptr, &::glfwDestroyWindow)) {}
    ~HelloTriangle() = default;

    void run() {
        initWindow();
        initVulkan();
        // mainLoop();
        // cleanup();
    }
};

int main() {
    try {
        HelloTriangle app;
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
    }

    return EXIT_SUCCESS;
}
