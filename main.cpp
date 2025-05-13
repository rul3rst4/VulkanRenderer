#include <exception>
#define VULKAN_HPP_NO_CONSTRUCTORS  // Permite usar Designated Initializers pra construir os objetos.
// #define VULKAN_HPP_NO_EXCEPTIONS // Retorna um result type pra ser tratado.
// #define GLFW_INCLUDE_VULKAN
#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include <iostream>
#include <memory>
#include <fmt/format.h>
#include <algorithm>
#include <optional>
#include <set>
#include <limits>
#include <fstream>

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
    vk::PipelineLayout pipelineLayout;
    vk::Pipeline graphicsPipeline;
    vk::CommandPool commandPool;

    std::vector<vk::Fence> inFlightFences;
    std::vector<vk::Semaphore> imageAvailableSemaphores;
    std::vector<vk::Semaphore> renderFinishedSemaphores;
    std::vector<vk::CommandBuffer> commandBuffers;
    std::vector<vk::Image> swapChainImages;
    std::vector<vk::ImageView> swapChainImageViews;
    std::vector<vk::Framebuffer> swapChainFramebuffers;
    // TODO: Destruir tudo. Ou criando unique_ptrs ou usando vk_raii

    bool framebufferResized = false;

    uint32_t currentFrame{};

    void initVulkan() {
        createVkInstance();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createGraphicsPipeline();
        createFramebuffers();
        createCommandPool();
        createCommandBuffers();
        createSyncObjects();
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

        const vk::RenderPassBeginInfo renderPassInfo{.sType = vk::StructureType::eRenderPassBeginInfo,
                                                     .renderPass = renderPass,
                                                     .framebuffer = swapChainFramebuffers[imageIndex],
                                                     .renderArea.offset = {0, 0},
                                                     .renderArea.extent = swapChainExtent,
                                                     .clearValueCount = 1,
                                                     .pClearValues = &clearColor};

        commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);

        const vk::Viewport viewport{.x = 0.0f,
                                    .y = 0.0f,
                                    .width = static_cast<float>(swapChainExtent.width),
                                    .height = static_cast<float>(swapChainExtent.height),
                                    .minDepth = 0.0f,
                                    .maxDepth = 1.0f};

        commandBuffer.setViewport(0, 1, &viewport);

        const vk::Rect2D scissor{.offset = {0, 0}, .extent = swapChainExtent};
        commandBuffer.setScissor(0, 1, &scissor);

        commandBuffer.draw(3, 1, 0, 0);
        commandBuffer.endRenderPass();
        commandBuffer.end();
    }

    void createCommandBuffers() {
        vk::CommandBufferAllocateInfo allocInfo{.sType = vk::StructureType::eCommandBufferAllocateInfo,
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
            vk::ImageView attachments[] = {swapChainImageViews[i]};

            vk::FramebufferCreateInfo framebufferInfo{.sType = vk::StructureType::eFramebufferCreateInfo,
                                                      .renderPass = renderPass,
                                                      .attachmentCount = 1,
                                                      .pAttachments = attachments,
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

        vk::SubpassDescription subpass{.pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
                                       .colorAttachmentCount = 1,
                                       .pColorAttachments = &colorAttachmentRef};

        vk::SubpassDependency dependency{
            .srcSubpass = vk::SubpassExternal,
            .dstSubpass = 0,
            .srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
            .srcAccessMask = vk::AccessFlagBits::eNone,
            .dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
            .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
        };

        const vk::RenderPassCreateInfo renderPassInfo{.sType = vk::StructureType::eRenderPassCreateInfo,
                                                      .attachmentCount = 1,
                                                      .pAttachments = &colorAttachment,
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

        vk::PipelineVertexInputStateCreateInfo vertexInputCreateInfo{
            .sType = vk::StructureType::ePipelineVertexInputStateCreateInfo,
            .vertexBindingDescriptionCount = 0,
            .pVertexBindingDescriptions = nullptr,
            .vertexAttributeDescriptionCount = 0,
            .pVertexAttributeDescriptions = nullptr};

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
            .cullMode = vk::CullModeFlagBits::eBack,
            .frontFace = vk::FrontFace::eClockwise,
            .depthBiasEnable = VK_FALSE,
            .depthBiasConstantFactor = 0.0f,
            .depthBiasClamp = 0.0f,
            .depthBiasSlopeFactor = 0.0f};

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
                                                        .setLayoutCount = 0,
                                                        .pSetLayouts = nullptr,
                                                        .pushConstantRangeCount = 0,
                                                        .pPushConstantRanges = nullptr};

        pipelineLayout = device.createPipelineLayout(pipelineLayoutInfo);

        std::vector<vk::DynamicState> dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};

        vk::PipelineDynamicStateCreateInfo dynamicState{
            .sType = vk::StructureType::ePipelineDynamicStateCreateInfo,
            .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
            .pDynamicStates = dynamicStates.data()};

        vk::GraphicsPipelineCreateInfo pipelineInfo{.sType = vk::StructureType::eGraphicsPipelineCreateInfo,
                                                    .stageCount = 2,
                                                    .pStages = shaderStages,
                                                    .pVertexInputState = &vertexInputCreateInfo,
                                                    .pInputAssemblyState = &inputAssembly,
                                                    .pViewportState = &viewportState,
                                                    .pRasterizationState = &rasterizer,
                                                    .pMultisampleState = &multisampling,
                                                    .pDepthStencilState = nullptr,
                                                    .pColorBlendState = &colorBlending,
                                                    .pDynamicState = &dynamicState,
                                                    .layout = pipelineLayout,
                                                    .renderPass = renderPass,
                                                    .subpass = 0,
                                                    .basePipelineHandle = nullptr,
                                                    .basePipelineIndex = -1};

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

    void createImageViews() {
        swapChainImageViews.resize(swapChainImages.size());

        for (size_t i = 0; i < swapChainImages.size(); i++) {
            vk::ImageViewCreateInfo createInfo{
                .sType = vk::StructureType::eImageViewCreateInfo,
                .image = swapChainImages[i],
                .viewType = vk::ImageViewType::e2D,
                .format = swapChainImageFormat,
                .components.r = vk::ComponentSwizzle::eIdentity,
                .components.g = vk::ComponentSwizzle::eIdentity,
                .components.b = vk::ComponentSwizzle::eIdentity,
                .components.a = vk::ComponentSwizzle::eIdentity,
                .subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor,
                .subresourceRange.baseMipLevel = 0,
                .subresourceRange.levelCount = 1,
                .subresourceRange.baseArrayLayer = 0,
                .subresourceRange.layerCount = 1,
            };

            swapChainImageViews[i] = device.createImageView(createInfo);
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

        vk::PhysicalDeviceFeatures deviceFeatures{};

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

    bool isDeviceSuitable(const vk::PhysicalDevice& device) const {
        // auto properties = device.getProperties();
        // auto features = device.getFeatures();
        const auto indices = findQueueFamilies(device);

        const auto extensionsSupported = checkDeviceExtensionSupport(device);

        const auto swapChainAdequate = [&]() {
            if (extensionsSupported) {
                const auto swapChainSupport = querySwapChainSupport(device);
                return !swapChainSupport.formats.empty() && !swapChainSupport.presenteModes.empty();
            }

            return false;
        }();

        return indices.isComplete() && extensionsSupported && swapChainAdequate;
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

    void cleanup() { glfwTerminate(); }

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
        mainLoop();
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
