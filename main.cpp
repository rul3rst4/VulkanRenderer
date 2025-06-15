#include <exception>
#include <initializer_list>
#define VULKAN_HPP_NO_CONSTRUCTORS  // Permite usar Designated Initializers pra construir os objetos.
// #define VULKAN_HPP_NO_EXCEPTIONS // Retorna um result type pra ser tratado.
// #define GLFW_INCLUDE_VULKAN
#include <vulkan/vulkan_raii.hpp>
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

struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;

    constexpr static vk::VertexInputBindingDescription getBindingDescription() {
        constexpr vk::VertexInputBindingDescription bindingDescription{
            .binding = 0, .stride = sizeof(Vertex), .inputRate = vk::VertexInputRate::eVertex};

        return bindingDescription;
    }

    constexpr static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions() {
        constexpr std::array<vk::VertexInputAttributeDescription, 2> attributeDescriptions{{
            {.location = 0, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(Vertex, pos)},
            {.location = 1, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(Vertex, color)},
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
    explicit ScopedOneTimeCommandBuffer(const vk::raii::Device& device,
                                        const vk::raii::CommandPool& commandPool,
                                        const vk::raii::Queue& graphicsQueue)
        : commandBuffer(nullptr), graphicsQueue(graphicsQueue) {
        const vk::CommandBufferAllocateInfo allocInfo{.sType = vk::StructureType::eCommandBufferAllocateInfo,
                                                      .commandPool = *commandPool,
                                                      .level = vk::CommandBufferLevel::ePrimary,
                                                      .commandBufferCount = 1};

        auto commandBuffers = device.allocateCommandBuffers(allocInfo);
        commandBuffer = std::move(commandBuffers[0]);

        constexpr vk::CommandBufferBeginInfo beginInfo{.sType = vk::StructureType::eCommandBufferBeginInfo,
                                                       .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit};

        commandBuffer.begin(beginInfo);
    }

    ~ScopedOneTimeCommandBuffer() {
        commandBuffer.end();

        const vk::SubmitInfo submitInfo{
            .sType = vk::StructureType::eSubmitInfo, .commandBufferCount = 1, .pCommandBuffers = &*commandBuffer};

        graphicsQueue.submit(submitInfo);
        graphicsQueue.waitIdle();
    }

    vk::raii::CommandBuffer commandBuffer;

   private:
    const vk::raii::Queue& graphicsQueue;
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

    /// vulkan_raii provides automatic destruction for these variables.
    vk::raii::Context context;
    vk::raii::Instance instance;
    vk::raii::PhysicalDevice physicalDevice;
    vk::raii::Device device;
    vk::raii::Queue graphicsQueue;
    vk::raii::Queue presenteQueue;
    vk::raii::SurfaceKHR surface;
    vk::raii::SwapchainKHR swapChain;
    vk::Format swapChainImageFormat;
    vk::Extent2D swapChainExtent;
    vk::raii::RenderPass renderPass;
    vk::raii::DescriptorSetLayout descriptorSetLayout;
    vk::raii::PipelineLayout pipelineLayout;
    vk::raii::Pipeline graphicsPipeline;
    vk::raii::CommandPool commandPool;

    // Para mais performance, podemos ter vertexBuffer e indexBuffer no mesmo Buffer.
    // https://developer.nvidia.com/vulkan-memory-management
    vk::raii::Buffer vertexBuffer;
    vk::raii::DeviceMemory vertexBufferMemory;
    vk::raii::Buffer indexBuffer;
    vk::raii::DeviceMemory indexBufferMemory;

    // Offscreen rendering buffers
    vk::raii::Buffer offscreenVertexBuffer;
    vk::raii::DeviceMemory offscreenVertexBufferMemory;
    vk::raii::Buffer offscreenIndexBuffer;
    vk::raii::DeviceMemory offscreenIndexBufferMemory;

    vk::raii::Image textureImage;
    vk::raii::DeviceMemory textureImageMemory;
    vk::raii::ImageView textureImageView;
    vk::raii::Sampler textureSampler;

    vk::raii::Image depthImage;
    vk::raii::DeviceMemory depthImageMemory;
    vk::raii::ImageView depthImageView;

    vk::raii::DescriptorPool descriptorPool;
    std::vector<vk::DescriptorSet> descriptorSets;

    std::vector<vk::raii::Fence> inFlightFences;
    std::vector<vk::raii::Semaphore> imageAvailableSemaphores;
    std::vector<vk::raii::Semaphore> renderFinishedSemaphores;
    std::vector<vk::raii::CommandBuffer> commandBuffers;
    std::vector<vk::Image> swapChainImages;
    std::vector<vk::raii::ImageView> swapChainImageViews;
    std::vector<vk::raii::Framebuffer> swapChainFramebuffers;

    std::vector<vk::raii::Buffer> uniformBufers;
    std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;

    // Offscreen rendering resources
    vk::raii::RenderPass offscreenRenderPass;
    vk::raii::Pipeline offscreenPipeline;
    vk::raii::PipelineLayout offscreenPipelineLayout;
    std::vector<vk::raii::Framebuffer> offscreenFramebuffers;
    std::vector<vk::raii::ImageView> offscreenImageViews;
    vk::raii::CommandBuffer offscreenCommandBuffer;

    // TODO: Destruir tudo. Ou criando unique_ptrs ou usando vk_raii

    bool framebufferResized = false;

    uint32_t currentFrame{};

    static constexpr std::array<Vertex, 8> vertices = {{
        {{-0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},  // 0
        {{0.5f, -0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},   // 1
        {{0.5f, 0.5f, -0.5f}, {0.0f, 0.0f, 1.0f}},    // 2
        {{-0.5f, 0.5f, -0.5f}, {1.0f, 1.0f, 1.0f}},   // 3
        {{-0.5f, -0.5f, 0.5f}, {1.0f, 0.0f, 1.0f}},   // 4
        {{0.5f, -0.5f, 0.5f}, {1.0f, 1.0f, 0.0f}},    // 5
        {{0.5f, 0.5f, 0.5f}, {0.0f, 1.0f, 1.0f}},     // 6
        {{-0.5f, 0.5f, 0.5f}, {0.5f, 0.5f, 0.5f}},    // 7
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
        createVkInstance();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createDescriptorSetLayout();
        // createGraphicsPipeline();
        createCommandPool();
        // createDepthResources();
        // createFramebuffers();

        createEmptyCubemapTexture();
        createOffscreenRenderPass();
        createOffscreenPipeline();
        createOffscreenVertexBuffer();
        createOffscreenIndexBuffer();
        createOffscreenFramebuffers();
        renderToCubemap();
        saveCubemapToPNG();

        // createTextureImageView();
        // createTextureSampler();
        // createVertexBuffer();
        // createIndexBuffer();
        // createUniformBuffers();
        // createDescriptorPool();
        // createDescriptorSets();
        // createCommandBuffers();
        // createSyncObjects();
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

        const vk::SamplerCreateInfo samplerInfo{
            .sType = vk::StructureType::eSamplerCreateInfo,
            .magFilter = vk::Filter::eLinear,
            .minFilter = vk::Filter::eLinear,
            .mipmapMode = vk::SamplerMipmapMode::eLinear,
            .addressModeU = vk::SamplerAddressMode::eClampToEdge,
            .addressModeV = vk::SamplerAddressMode::eClampToEdge,
            .addressModeW = vk::SamplerAddressMode::eClampToEdge,
            .mipLodBias = 0.0f,
            .anisotropyEnable = vk::True,
            .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
            .compareEnable = vk::False,
            .compareOp = vk::CompareOp::eAlways,
            .minLod = 0.0f,
            .maxLod = 0.0f,
            .borderColor = vk::BorderColor::eIntOpaqueBlack,
            .unnormalizedCoordinates = vk::False,
        };

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
            .imageSubresource =
                vk::ImageSubresourceLayers{
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .mipLevel = 0,
                    .baseArrayLayer = 0,
                    .layerCount = layerCount,
                },
            .imageOffset = {0, 0, 0},
            .imageExtent = {width, height, 1},
        };

        scopedCommandBuffer.commandBuffer.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal,
                                                            region);
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

        scopedCommandBuffer.commandBuffer.pipelineBarrier(srcStage, dstStage, vk::DependencyFlags{}, nullptr, nullptr,
                                                          barrier);
    }

    void createImage(const uint32_t width,
                     const uint32_t height,
                     const vk::Format format,
                     const vk::ImageTiling tiling,
                     const vk::Flags<vk::ImageUsageFlagBits> usage,
                     const vk::Flags<vk::MemoryPropertyFlagBits> properties,
                     vk::raii::Image& image,
                     vk::raii::DeviceMemory& imageMemory,
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

        image = vk::raii::Image(device, imageInfo);
        const auto memRequirements = image.getMemoryRequirements();

        const vk::MemoryAllocateInfo allocInfo{
            .sType = vk::StructureType::eMemoryAllocateInfo,
            .allocationSize = memRequirements.size,
            .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)};

        imageMemory = vk::raii::DeviceMemory(device, allocInfo);
        image.bindMemory(*imageMemory, 0);
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

        auto allocatedSets = device.allocateDescriptorSets(allocInfo);
        descriptorSets.clear();
        descriptorSets.reserve(allocatedSets.size());
        for (auto& set : allocatedSets) {
            descriptorSets.push_back(*set);
        }

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
                                                               .descriptorCount = 1,
                                                               .descriptorType = vk::DescriptorType::eUniformBuffer,
                                                               .pImageInfo = nullptr,
                                                               .pBufferInfo = &bufferInfo,
                                                               .pTexelBufferView = nullptr};

            const vk::WriteDescriptorSet samplerDescriptorWrite{
                .sType = vk::StructureType::eWriteDescriptorSet,
                .dstSet = descriptorSets[i],
                .dstBinding = 1,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                .pImageInfo = &imageInfo};

            std::array descriptorWrites = {bufferDescriptorWrite, samplerDescriptorWrite};

            device.updateDescriptorSets(descriptorWrites, nullptr);
        }
    }

    void createDescriptorPool() {
        static constexpr vk::DescriptorPoolSize uboPoolSize{.type = vk::DescriptorType::eUniformBuffer,
                                                            .descriptorCount = maxFramesInFlight};

        static constexpr vk::DescriptorPoolSize samplerPoolSize{.type = vk::DescriptorType::eCombinedImageSampler,
                                                                .descriptorCount = maxFramesInFlight};

        static constexpr std::array<vk::DescriptorPoolSize, 2> poolSizes = {uboPoolSize, samplerPoolSize};

        static constexpr vk::DescriptorPoolCreateInfo poolInfo{
            .sType = vk::StructureType::eDescriptorPoolCreateInfo,
            .maxSets = maxFramesInFlight,
            .poolSizeCount = poolSizes.size(),
            .pPoolSizes = poolSizes.data(),
        };

        descriptorPool = device.createDescriptorPool(poolInfo);
    }

    void createUniformBuffers() {
        uniformBufers.clear();
        uniformBuffersMemory.clear();
        uniformBufers.reserve(maxFramesInFlight);
        uniformBuffersMemory.reserve(maxFramesInFlight);
        uniformBuffersMapped.resize(maxFramesInFlight);

        for (size_t i = 0; i < maxFramesInFlight; i++) {
            constexpr vk::DeviceSize bufferSize = sizeof(UniformBufferObject);

            vk::raii::Buffer buffer = nullptr;
            vk::raii::DeviceMemory memory = nullptr;
            createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer,
                         vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer,
                         memory);

            uniformBufers.emplace_back(std::move(buffer));
            uniformBuffersMemory.emplace_back(std::move(memory));
            uniformBuffersMapped[i] = uniformBuffersMemory[i].mapMemory(0, bufferSize);
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
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eFragment,
            .pImmutableSamplers = nullptr,
        };

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

        vk::raii::Buffer stagingBuffer = nullptr;
        vk::raii::DeviceMemory stagingBufferMemory = nullptr;

        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                     vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                     stagingBuffer, stagingBufferMemory);

        const auto data = stagingBufferMemory.mapMemory(0, bufferSize);
        memcpy(data, indices.data(), bufferSize);
        stagingBufferMemory.unmapMemory();

        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
                     vk::MemoryPropertyFlagBits::eDeviceLocal, indexBuffer, indexBufferMemory);

        copyBuffer(*stagingBuffer, *indexBuffer, bufferSize);
    }

    void createBuffer(const vk::DeviceSize size,
                      const vk::Flags<vk::BufferUsageFlagBits> usage,
                      const vk::Flags<vk::MemoryPropertyFlagBits> properties,
                      vk::raii::Buffer& buffer,
                      vk::raii::DeviceMemory& bufferMemory) const {
        const vk::BufferCreateInfo bufferInfo{.sType = vk::StructureType::eBufferCreateInfo,
                                              .size = size,
                                              .usage = usage,
                                              .sharingMode = vk::SharingMode::eExclusive};

        buffer = vk::raii::Buffer(device, bufferInfo);

        const auto memRequirements = buffer.getMemoryRequirements();

        const vk::MemoryAllocateInfo allocInfo{
            .sType = vk::StructureType::eMemoryAllocateInfo,
            .allocationSize = memRequirements.size,
            .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)};

        bufferMemory = vk::raii::DeviceMemory(device, allocInfo);
        buffer.bindMemory(*bufferMemory, 0);
    }

    void createVertexBuffer() {
        constexpr vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

        vk::raii::Buffer stagingBuffer = nullptr;
        vk::raii::DeviceMemory stagingBufferMemory = nullptr;

        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                     vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                     stagingBuffer, stagingBufferMemory);

        const auto data = stagingBufferMemory.mapMemory(0, bufferSize);
        memcpy(data, vertices.data(), bufferSize);
        stagingBufferMemory.unmapMemory();

        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
                     vk::MemoryPropertyFlagBits::eDeviceLocal, vertexBuffer, vertexBufferMemory);

        copyBuffer(*stagingBuffer, *vertexBuffer, bufferSize);
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

    void copyBuffer(const vk::Buffer srcBuffer, const vk::Buffer dstBuffer, const vk::DeviceSize size) const {
        auto scopedCommandBuffer = ScopedOneTimeCommandBuffer(device, commandPool, graphicsQueue);

        const vk::BufferCopy copyRegion{.srcOffset = 0, .dstOffset = 0, .size = size};

        scopedCommandBuffer.commandBuffer.copyBuffer(srcBuffer, dstBuffer, copyRegion);
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

    void recreateSwapChain() {
        int width{}, height{};
        glfwGetFramebufferSize(window.get(), &width, &height);

        while (width == 0 && height == 0) {
            glfwGetFramebufferSize(window.get(), &width, &height);
            glfwWaitEvents();
        }

        device.waitIdle();

        createSwapChain();
        createImageViews();
        createDepthResources();
        createFramebuffers();
    }

    void createSyncObjects() {
        imageAvailableSemaphores.clear();
        renderFinishedSemaphores.clear();
        inFlightFences.clear();

        imageAvailableSemaphores.reserve(maxFramesInFlight);
        renderFinishedSemaphores.reserve(maxFramesInFlight);
        inFlightFences.reserve(maxFramesInFlight);

        constexpr vk::SemaphoreCreateInfo semaphoreInfo{.sType = vk::StructureType::eSemaphoreCreateInfo};

        constexpr vk::FenceCreateInfo fenceInfo{.sType = vk::StructureType::eFenceCreateInfo,
                                                .flags = vk::FenceCreateFlagBits::eSignaled};

        for (size_t i = 0; i < maxFramesInFlight; i++) {
            imageAvailableSemaphores.emplace_back(device, semaphoreInfo);
            renderFinishedSemaphores.emplace_back(device, semaphoreInfo);
            inFlightFences.emplace_back(device, fenceInfo);
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
                                                     .renderPass = *renderPass,
                                                     .framebuffer = *swapChainFramebuffers[imageIndex],
                                                     .renderArea.offset = {0, 0},
                                                     .renderArea.extent = swapChainExtent,
                                                     .clearValueCount = clearValues.size(),
                                                     .pClearValues = clearValues.data()};

        commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);

        const vk::Buffer vertexBufferHandle = *vertexBuffer;
        const std::array<vk::Buffer, 1> vertexBuffers = {vertexBufferHandle};
        constexpr std::array<vk::DeviceSize, 1> offsets{};

        commandBuffer.bindVertexBuffers(0, 1, vertexBuffers.data(), offsets.data());
        commandBuffer.bindIndexBuffer(*indexBuffer, 0, vk::IndexType::eUint32);

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
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipelineLayout, 0, 1,
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
        swapChainFramebuffers.clear();
        swapChainFramebuffers.reserve(swapChainImageViews.size());

        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            const std::array<vk::ImageView, 2> attachments = {*swapChainImageViews[i], *depthImageView};

            vk::FramebufferCreateInfo framebufferInfo{.sType = vk::StructureType::eFramebufferCreateInfo,
                                                      .renderPass = *renderPass,
                                                      .attachmentCount = attachments.size(),
                                                      .pAttachments = attachments.data(),
                                                      .width = swapChainExtent.width,
                                                      .height = swapChainExtent.height,
                                                      .layers = 1};

            swapChainFramebuffers.emplace_back(device, framebufferInfo);
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
            .pVertexBindingDescriptions = &bindingDescription,
            .vertexAttributeDescriptionCount = attributeDescription.size(),
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
            .cullMode = vk::CullModeFlagBits::eNone,
            .frontFace = vk::FrontFace::eCounterClockwise,
            .lineWidth = 1.0f,
            .depthBiasEnable = VK_FALSE,
            .depthBiasConstantFactor = 0.0f,
            .depthBiasClamp = 0.0f,
            .depthBiasSlopeFactor = 0.0f,
        };

        vk::PipelineMultisampleStateCreateInfo multisampling{
            .sType = vk::StructureType::ePipelineMultisampleStateCreateInfo,
            .rasterizationSamples = vk::SampleCountFlagBits::e1,
            .sampleShadingEnable = VK_FALSE,
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
                                                        .pSetLayouts = &*descriptorSetLayout,
                                                        .pushConstantRangeCount = 0,
                                                        .pPushConstantRanges = nullptr};

        pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

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
            .layout = *pipelineLayout,
            .renderPass = *renderPass,
            .subpass = 0,
            .basePipelineHandle = nullptr,
            .basePipelineIndex = -1,
        };

        graphicsPipeline = vk::raii::Pipeline(device, nullptr, pipelineInfo);
    }

    [[nodiscard]] vk::raii::ShaderModule createShaderModule(const std::vector<char>& code) const {
        const vk::ShaderModuleCreateInfo createInfo{.sType = vk::StructureType::eShaderModuleCreateInfo,
                                                    .codeSize = code.size(),
                                                    .pCode = reinterpret_cast<const uint32_t*>(code.data())};

        return vk::raii::ShaderModule(device, createInfo);
    }

    [[nodiscard]] vk::raii::ImageView createImageView(const vk::raii::Image& image,
                                                      const vk::Format format,
                                                      const vk::ImageAspectFlagBits aspectFlags,
                                                      const vk::ImageViewType viewType,
                                                      const uint32_t layerCount,
                                                      const uint32_t baseArrayLayer = 0) const {
        const vk::ImageViewCreateInfo createInfo{
            .sType = vk::StructureType::eImageViewCreateInfo,
            .image = *image,
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

        return vk::raii::ImageView(device, createInfo);
    }

    [[nodiscard]] vk::raii::ImageView createImageView(const vk::Image& image,
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

        return vk::raii::ImageView(device, createInfo);
    }

    void createImageViews() {
        swapChainImageViews.clear();
        swapChainImageViews.reserve(swapChainImages.size());

        for (size_t i = 0; i < swapChainImages.size(); i++) {
            swapChainImageViews.emplace_back(createImageView(
                swapChainImages[i], swapChainImageFormat, vk::ImageAspectFlagBits::eColor, vk::ImageViewType::e2D, 1));
        }
    }

    void createSurface() {
        VkSurfaceKHR surfaceHandle;
        if (glfwCreateWindowSurface(*instance, window.get(), nullptr, &surfaceHandle) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }

        surface = vk::raii::SurfaceKHR(instance, surfaceHandle);
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

        vk::DeviceCreateInfo createInfo{
            .sType = vk::StructureType::eDeviceCreateInfo,
            .queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size()),
            .pQueueCreateInfos = queueCreateInfos.data(),
            .enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
            .ppEnabledExtensionNames = deviceExtensions.data(),
            .pEnabledFeatures = &deviceFeatures,
        };

        device = vk::raii::Device(physicalDevice, createInfo);
        graphicsQueue = vk::raii::Queue(device, graphicsFamily.value(), 0);
        presenteQueue = vk::raii::Queue(device, presenteFamily.value(), 0);
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

        swapChain = vk::raii::SwapchainKHR(device, createInfo);
        swapChainImages = swapChain.getImages();
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
                                    .pApplicationName = "Cubemap Renderer",
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

        instance = vk::raii::Instance(context, createInfo);
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
        (void)device.waitForFences({*inFlightFences[currentFrame]}, VK_TRUE, UINT64_MAX);

        const auto nextImageResult = swapChain.acquireNextImage(UINT64_MAX, *imageAvailableSemaphores[currentFrame]);

        if (nextImageResult.first == vk::Result::eErrorOutOfDateKHR) {
            recreateSwapChain();
            return;
        } else if (nextImageResult.first != vk::Result::eSuccess &&
                   nextImageResult.first != vk::Result::eSuboptimalKHR) {
            throw std::runtime_error("failed to acquire swap chain image!");
        }

        // Only reset the fence if we are submitting work
        device.resetFences({*inFlightFences[currentFrame]});

        uint32_t imageIndex = nextImageResult.second;

        commandBuffers[currentFrame].reset();
        recordCommandBuffer(*commandBuffers[currentFrame], imageIndex);

        updateUniformBuffer(currentFrame);

        vk::Semaphore waitSemaphores[] = {*imageAvailableSemaphores[currentFrame]};
        vk::PipelineStageFlags waitStages[] = {vk::PipelineStageFlagBits::eColorAttachmentOutput};
        vk::Semaphore signalSemaphores[] = {*renderFinishedSemaphores[currentFrame]};
        vk::CommandBuffer commandBufferHandle = *commandBuffers[currentFrame];
        vk::SubmitInfo submitInfo{.sType = vk::StructureType::eSubmitInfo,
                                  .waitSemaphoreCount = 1,
                                  .pWaitSemaphores = waitSemaphores,
                                  .pWaitDstStageMask = waitStages,
                                  .commandBufferCount = 1,
                                  .pCommandBuffers = &commandBufferHandle,
                                  .signalSemaphoreCount = 1,
                                  .pSignalSemaphores = signalSemaphores};

        graphicsQueue.submit(submitInfo, *inFlightFences[currentFrame]);

        vk::SwapchainKHR swapChains[] = {*swapChain};

        vk::PresentInfoKHR presentInfo{.sType = vk::StructureType::ePresentInfoKHR,
                                       .waitSemaphoreCount = 1,
                                       .pWaitSemaphores = signalSemaphores,
                                       .swapchainCount = 1,
                                       .pSwapchains = swapChains,
                                       .pImageIndices = &imageIndex,
                                       .pResults = nullptr};

        const auto presentResult = presenteQueue.presentKHR(presentInfo);

        if (presentResult == vk::Result::eErrorOutOfDateKHR || presentResult == vk::Result::eSuboptimalKHR ||
            framebufferResized) {
            framebufferResized = false;
            recreateSwapChain();
        } else if (presentResult != vk::Result::eSuccess) {
            throw std::runtime_error("failed to present swap chain image!");
        }

        currentFrame = (currentFrame + 1) % maxFramesInFlight;
    }

    void updateUniformBuffer(uint32_t currentImage) {
        static auto startTime = std::chrono::high_resolution_clock::now();

        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();
        UniformBufferObject ubo{
            .model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
            .view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
            .proj = glm::perspective(glm::radians(45.0F), swapChainExtent.width / (float)swapChainExtent.height, 0.1f,
                                     10.0f)};

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
        // Allocate command buffer for offscreen rendering
        const vk::CommandBufferAllocateInfo allocInfo{.sType = vk::StructureType::eCommandBufferAllocateInfo,
                                                      .commandPool = *commandPool,
                                                      .level = vk::CommandBufferLevel::ePrimary,
                                                      .commandBufferCount = 1};

        auto commandBuffers = device.allocateCommandBuffers(allocInfo);
        offscreenCommandBuffer = std::move(commandBuffers[0]);

        const vk::CommandBufferBeginInfo beginInfo{.sType = vk::StructureType::eCommandBufferBeginInfo,
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
                .image = *textureImage,
                .subresourceRange = vk::ImageSubresourceRange{.aspectMask = vk::ImageAspectFlagBits::eColor,
                                                              .baseMipLevel = 0,
                                                              .levelCount = 1,
                                                              .baseArrayLayer = face,
                                                              .layerCount = 1}};

            offscreenCommandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eFragmentShader,
                                                   vk::PipelineStageFlagBits::eColorAttachmentOutput,
                                                   vk::DependencyFlags{}, nullptr, nullptr, barrier);

            vk::ClearValue clearColor{};
            clearColor.color.float32[0] = 0.0f;
            clearColor.color.float32[1] = 0.0f;
            clearColor.color.float32[2] = 0.0f;
            clearColor.color.float32[3] = 1.0f;

            const uint32_t cubemapSize = std::max(swapChainExtent.width, swapChainExtent.height);
            const vk::RenderPassBeginInfo renderPassInfo{.sType = vk::StructureType::eRenderPassBeginInfo,
                                                         .renderPass = *offscreenRenderPass,
                                                         .framebuffer = *offscreenFramebuffers[face],
                                                         .renderArea = {{0, 0}, {cubemapSize, cubemapSize}},
                                                         .clearValueCount = 1,
                                                         .pClearValues = &clearColor};

            offscreenCommandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
            offscreenCommandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *offscreenPipeline);

            // Bind vertex buffer
            const vk::Buffer vertexBufferHandle = *offscreenVertexBuffer;
            const std::array<vk::Buffer, 1> vertexBuffers = {vertexBufferHandle};
            constexpr std::array<vk::DeviceSize, 1> offsets{};
            offscreenCommandBuffer.bindVertexBuffers(0, vertexBuffers, offsets);

            // Bind index buffer
            offscreenCommandBuffer.bindIndexBuffer(*offscreenIndexBuffer, 0, vk::IndexType::eUint32);

            // Draw the quad for this face (6 indices per face)
            offscreenCommandBuffer.drawIndexed(6, 1, face * 6, 0, 0);
            offscreenCommandBuffer.endRenderPass();

            // Transition this face back to shader read only optimal
            barrier.srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
            barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
            barrier.oldLayout = vk::ImageLayout::eUndefined;
            barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

            offscreenCommandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
                                                   vk::PipelineStageFlagBits::eFragmentShader, vk::DependencyFlags{},
                                                   nullptr, nullptr, barrier);
        }

        offscreenCommandBuffer.end();

        // Submit the command buffer
        const vk::SubmitInfo submitInfo{.sType = vk::StructureType::eSubmitInfo,
                                        .commandBufferCount = 1,
                                        .pCommandBuffers = &*offscreenCommandBuffer};

        graphicsQueue.submit(submitInfo);
        graphicsQueue.waitIdle();
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

                // // Transition image layout for transfer
                vk::ImageMemoryBarrier barrier{
                    .sType = vk::StructureType::eImageMemoryBarrier,
                    .srcAccessMask = vk::AccessFlagBits::eShaderRead,
                    .dstAccessMask = vk::AccessFlagBits::eTransferRead,
                    .oldLayout = vk::ImageLayout::eUndefined,
                    .newLayout = vk::ImageLayout::eTransferSrcOptimal,
                    .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
                    .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
                    .image = *textureImage,
                    .subresourceRange = vk::ImageSubresourceRange{.aspectMask = vk::ImageAspectFlagBits::eColor,
                                                                  .baseMipLevel = 0,
                                                                  .levelCount = 1,
                                                                  .baseArrayLayer = face,
                                                                  .layerCount = 1}};

                commandBuffer.commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eFragmentShader,
                                                            vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlags{},
                                                            nullptr, nullptr, barrier);

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

                // // Transition back to shader read only
                barrier.oldLayout = vk::ImageLayout::eUndefined;
                barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
                barrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
                barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

                commandBuffer.commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                                            vk::PipelineStageFlagBits::eFragmentShader,
                                                            vk::DependencyFlags{}, nullptr, nullptr, barrier);
            }

            // Map memory and save this face to PNG
            void* data = stagingBufferMemory.mapMemory(0, imageSize);
            if (!data) {
                throw std::runtime_error("Failed to map memory for staging buffer.");
            }
            pixelData.reserve(pixelData.size() + imageSize);
            // Copy the data from the staging buffer to pixelData
            pixelData.insert(pixelData.end(), static_cast<char*>(data), static_cast<char*>(data) + imageSize);
            stagingBufferMemory.unmapMemory();
        }

        const std::string filename = "texture_output.png";
        stbi_write_png(filename.c_str(), cubemapSize, cubemapSize * 6, 4, pixelData.data(), cubemapSize * 4);
        fmt::print("Saved cubemap faces to {}\n", filename);
    }

    void cleanup() const {
        glfwTerminate();
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
        : window(std::unique_ptr<GLFWwindow, decltype(&::glfwDestroyWindow)>(nullptr, &::glfwDestroyWindow)),
          context(),
          instance(nullptr),
          physicalDevice(nullptr),
          device(nullptr),
          graphicsQueue(nullptr),
          presenteQueue(nullptr),
          surface(nullptr),
          swapChain(nullptr),
          renderPass(nullptr),
          descriptorSetLayout(nullptr),
          pipelineLayout(nullptr),
          graphicsPipeline(nullptr),
          commandPool(nullptr),
          vertexBuffer(nullptr),
          vertexBufferMemory(nullptr),
          indexBuffer(nullptr),
          indexBufferMemory(nullptr),
          offscreenVertexBuffer(nullptr),
          offscreenVertexBufferMemory(nullptr),
          offscreenIndexBuffer(nullptr),
          offscreenIndexBufferMemory(nullptr),
          textureImage(nullptr),
          textureImageMemory(nullptr),
          textureImageView(nullptr),
          textureSampler(nullptr),
          depthImage(nullptr),
          depthImageMemory(nullptr),
          depthImageView(nullptr),
          descriptorPool(nullptr),
          offscreenRenderPass(nullptr),
          offscreenPipeline(nullptr),
          offscreenPipelineLayout(nullptr),
          offscreenCommandBuffer(nullptr) {}
    ~HelloTriangle() = default;

    void run() {
        initWindow();
        initVulkan();
        // mainLoop();
        cleanup();
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
