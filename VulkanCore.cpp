#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "VulkanCore.hpp"
#include <iostream>
#include <set>
#include <limits>
#include <fstream>
#include <algorithm>

void VulkanCore::initCore() {
    createVkInstance();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapChain();
    createCommandPool();
    createCommandBuffers();
    createSyncObjects();
    createDescriptorPool();
}

vk::Format VulkanCore::findDepthFormat() const {
    return findSupportedFormat({vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint},
                               vk::ImageTiling::eOptimal, vk::FormatFeatureFlagBits::eDepthStencilAttachment);
}

bool VulkanCore::hasStencilComponent(const vk::Format format) {
    return format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eD24UnormS8Uint;
}

vk::Format VulkanCore::findSupportedFormat(const std::vector<vk::Format>& candidates,
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

void VulkanCore::copyBufferToImage(const vk::Buffer buffer,
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

    scopedCommandBuffer.commandBuffer.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, region);
}

void VulkanCore::transitionImageLayout(const vk::CommandBuffer commandBuffer,
                                       const vk::Image image,
                                       const vk::ImageLayout oldLayout,
                                       const vk::ImageLayout newLayout,
                                       const vk::AccessFlagBits srcAccessMask,
                                       const vk::AccessFlagBits dstAccessMask,
                                       const vk::PipelineStageFlagBits srcStage,
                                       const vk::PipelineStageFlagBits dstStage,
                                       const uint32_t layerCount,
                                       const uint32_t baseArrayLayer) const {
    vk::ImageMemoryBarrier barrier{
        .sType = vk::StructureType::eImageMemoryBarrier,
        .srcAccessMask = srcAccessMask,
        .dstAccessMask = dstAccessMask,
        .oldLayout = oldLayout,
        .newLayout = newLayout,
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .image = image,
        .subresourceRange = vk::ImageSubresourceRange{.aspectMask = vk::ImageAspectFlagBits::eColor,
                                                      .baseMipLevel = 0,
                                                      .levelCount = 1,
                                                      .baseArrayLayer = baseArrayLayer,
                                                      .layerCount = layerCount},
    };

    commandBuffer.pipelineBarrier(srcStage, dstStage, vk::DependencyFlags{}, nullptr, nullptr, barrier);
}

void VulkanCore::createImage(const uint32_t width,
                             const uint32_t height,
                             const vk::Format format,
                             const vk::ImageTiling tiling,
                             const vk::Flags<vk::ImageUsageFlagBits> usage,
                             const vk::Flags<vk::MemoryPropertyFlagBits> properties,
                             vk::raii::Image& image,
                             vk::raii::DeviceMemory& imageMemory,
                             const uint32_t arrayLayers,
                             const vk::ImageCreateFlags flags) const {
    const vk::ImageCreateInfo imageInfo{
        .sType = vk::StructureType::eImageCreateInfo,
        .flags = flags,
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent =
            vk::Extent3D{.width = static_cast<uint32_t>(width), .height = static_cast<uint32_t>(height), .depth = 1},
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

void VulkanCore::createDescriptorPool() {
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

void VulkanCore::createBuffer(const vk::DeviceSize size,
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

void VulkanCore::copyBuffer(const vk::Buffer srcBuffer, const vk::Buffer dstBuffer, const vk::DeviceSize size) const {
    const auto scopedCommandBuffer = ScopedOneTimeCommandBuffer(device, commandPool, graphicsQueue);

    const vk::BufferCopy copyRegion{.srcOffset = 0, .dstOffset = 0, .size = size};

    scopedCommandBuffer.commandBuffer.copyBuffer(srcBuffer, dstBuffer, copyRegion);
}

uint32_t VulkanCore::findMemoryType(const uint32_t typeFilter,
                                    const vk::Flags<vk::MemoryPropertyFlagBits> properties) const {
    const auto memProperties = physicalDevice.getMemoryProperties();

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type!");
}

void VulkanCore::baseRecreateSwapChain() {
    auto windowSize = windowManager.getFrameBufferSize();

    while (windowSize.x == 0 && windowSize.y == 0) {
        windowSize = windowManager.getFrameBufferSize();
        glfwWaitEvents();
    }

    device.waitIdle();

    createSwapChain();
    recreateSwapChain();
}

void VulkanCore::recreateSwapChain() {
    // This method should be overridden in derived classes to handle specific swap chain recreation logic.
    // The base implementation is empty to allow derived classes to implement their own logic.
    // Derived classes should call baseRecreateSwapChain() if they want to keep the base functionality.
    baseRecreateSwapChain();
}

void VulkanCore::createSyncObjects() {
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

void VulkanCore::createCommandBuffers() {
    const vk::CommandBufferAllocateInfo allocInfo{.sType = vk::StructureType::eCommandBufferAllocateInfo,
                                                  .commandPool = commandPool,
                                                  .level = vk::CommandBufferLevel::ePrimary,
                                                  .commandBufferCount = static_cast<uint32_t>(maxFramesInFlight)};

    commandBuffers = device.allocateCommandBuffers(allocInfo);
}

void VulkanCore::createCommandPool() {
    const auto queueFamilyIndices = findQueueFamilies(physicalDevice);

    vk::CommandPoolCreateInfo poolInfo{.sType = vk::StructureType::eCommandPoolCreateInfo,
                                       .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                       .queueFamilyIndex = queueFamilyIndices.graphicsFamily.value()};

    commandPool = device.createCommandPool(poolInfo);
}

std::vector<char> VulkanCore::readShaderFile(const std::string& fileName) {
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

vk::raii::ShaderModule VulkanCore::createShaderModule(const std::vector<char>& code) const {
    const vk::ShaderModuleCreateInfo createInfo{.sType = vk::StructureType::eShaderModuleCreateInfo,
                                                .codeSize = code.size(),
                                                .pCode = reinterpret_cast<const uint32_t*>(code.data())};

    return vk::raii::ShaderModule(device, createInfo);
}

vk::raii::ImageView VulkanCore::createImageView(const vk::raii::Image& image,
                                                const vk::Format format,
                                                const vk::ImageAspectFlagBits aspectFlags,
                                                const vk::ImageViewType viewType,
                                                const uint32_t layerCount,
                                                const uint32_t baseArrayLayer) const {
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

vk::raii::ImageView VulkanCore::createImageView(const vk::Image& image,
                                                const vk::Format format,
                                                const vk::ImageAspectFlagBits aspectFlags,
                                                const vk::ImageViewType viewType,
                                                const uint32_t layerCount,
                                                const uint32_t baseArrayLayer) const {
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

void VulkanCore::createSurface() {
    surface = vk::raii::SurfaceKHR(instance, windowManager.createWindowSurface(*instance));
}

void VulkanCore::createLogicalDevice() {
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

void VulkanCore::createSwapChain() {
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

void VulkanCore::pickPhysicalDevice() {
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

bool VulkanCore::isDeviceSuitable(const vk::PhysicalDevice& device) const {
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

vk::SurfaceFormatKHR VulkanCore::chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats) {
    for (const auto& availableFormat : availableFormats) {
        if (availableFormat.format == vk::Format::eB8G8R8A8Srgb) {
            return availableFormat;
        }
    }

    return availableFormats[0];
}

vk::PresentModeKHR VulkanCore::chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresenteModes) {
    for (const auto& availablePresentMode : availablePresenteModes) {
        if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
            return availablePresentMode;
        }
    }

    return vk::PresentModeKHR::eFifo;
}

vk::Extent2D VulkanCore::chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities) const {
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        return capabilities.currentExtent;
    }

    const auto windowSize = windowManager.getFrameBufferSize();

    vk::Extent2D actualExtent{
        .width = static_cast<uint32_t>(windowSize.x),
        .height = static_cast<uint32_t>(windowSize.y),
    };

    actualExtent.width =
        std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
    actualExtent.height =
        std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

    return actualExtent;
}

SwapChainSupportDetails VulkanCore::querySwapChainSupport(const vk::PhysicalDevice& device) const {
    return SwapChainSupportDetails{.capabilities = device.getSurfaceCapabilitiesKHR(surface),
                                   .formats = device.getSurfaceFormatsKHR(surface),
                                   .presenteModes = device.getSurfacePresentModesKHR(surface)};
}

bool VulkanCore::checkDeviceExtensionSupport(const vk::PhysicalDevice& device) const {
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

QueueFamilyIndices VulkanCore::findQueueFamilies(const vk::PhysicalDevice& device) const {
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

void VulkanCore::createVkInstance() {
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

    addOsSpecificExtensions(glfwExtensions);

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

void VulkanCore::printGlfwRequiredExtensions(const std::vector<const char*>& glfwExtensions) {
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

bool VulkanCore::checkValidationLayersSupport() const {
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

void VulkanCore::addOsSpecificExtensions(std::vector<const char*>& glfwExtensions) {
    // glfwExtensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME); Só é necessario no
    // Vulkan 1.0
#ifdef __APPLE__
    glfwExtensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
#endif
}


void VulkanCore::mainLoop() {
    while (!windowManager.shouldClose()) {
        windowManager.pollEvents();
        drawFrame();
    }

    device.waitIdle();
}

void VulkanCore::drawFrame() {
    // Call render method on all processes that support continuous rendering
    for (const auto& process : renderProcesses) {
        if (process->supportsContinuousRendering()) {
            process->render();
        }
    }
}

void VulkanCore::framebufferResizeCallback(GLFWwindow* window, int /* width */, int /* height */) {
    const auto app = static_cast<VulkanCore*>(glfwGetWindowUserPointer(window));
    app->framebufferResized = true;
}

VulkanCore::VulkanCore()
    : windowManager(windowWidth, windowHeight, "Vulkan"),
      instance(nullptr),
      physicalDevice(nullptr),
      device(nullptr),
      graphicsQueue(nullptr),
      presenteQueue(nullptr),
      surface(nullptr),
      swapChain(nullptr),
      commandPool(nullptr),
      descriptorPool(nullptr) {
    windowManager.setWindowUserPointer(this);
    windowManager.registerResizeCallback(framebufferResizeCallback);
    initCore();
}

void VulkanCore::runRenderProcesses() {
    for (const auto& process : renderProcesses) {
        process->execute();
    }
}
