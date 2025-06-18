#pragma once
#include <initializer_list>
#define VULKAN_HPP_NO_CONSTRUCTORS  // Permite usar Designated Initializers pra construir os objetos.
// #define VULKAN_HPP_NO_EXCEPTIONS // Retorna um result type pra ser tratado.
// #define GLFW_INCLUDE_VULKAN
#include <vulkan/vulkan_raii.hpp>
#include <GLFW/glfw3.h>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include "stb_image.h"
#include "stb_image_write.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <fmt/format.h>
#include <optional>
#include "ScopedOneTimeCommandBuffer.hpp"
#include "WindowManager.hpp"

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

class IRenderProcess {
   public:
    virtual ~IRenderProcess() = default;
    virtual void execute() = 0;
};

class VulkanCore {
   public:
    static constexpr int windowWidth = 800;
    static constexpr int windowHeight = 600;
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
    WindowManager windowManager;

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
    vk::raii::CommandPool commandPool;

    vk::raii::DescriptorPool descriptorPool;

    std::vector<vk::raii::Fence> inFlightFences;
    std::vector<vk::raii::Semaphore> imageAvailableSemaphores;
    std::vector<vk::raii::Semaphore> renderFinishedSemaphores;
    std::vector<vk::raii::CommandBuffer> commandBuffers;
    std::vector<vk::Image> swapChainImages;
    std::vector<std::unique_ptr<IRenderProcess>> renderProcesses;

    bool framebufferResized = false;

    uint32_t currentFrame{};

    void initCore();

    vk::Format findDepthFormat() const;

    static bool hasStencilComponent(const vk::Format format);

    vk::Format findSupportedFormat(const std::vector<vk::Format>& candidates,
                                   vk::ImageTiling tiling,
                                   vk::FormatFeatureFlags features) const;

    void copyBufferToImage(const vk::Buffer buffer,
                           const vk::Image image,
                           const uint32_t width,
                           const uint32_t height,
                           const uint32_t layerCount) const;

    void transitionImageLayout(const vk::CommandBuffer commandBuffer,
                               const vk::Image image,
                               const vk::ImageLayout oldLayout,
                               const vk::ImageLayout newLayout,
                               const vk::AccessFlagBits srcAccessMask,
                               const vk::AccessFlagBits dstAccessMask,
                               const vk::PipelineStageFlagBits srcStage,
                               const vk::PipelineStageFlagBits dstStage,
                               const uint32_t layerCount,
                               const uint32_t baseArrayLayer = 0) const;

    void createImage(const uint32_t width,
                     const uint32_t height,
                     const vk::Format format,
                     const vk::ImageTiling tiling,
                     const vk::Flags<vk::ImageUsageFlagBits> usage,
                     const vk::Flags<vk::MemoryPropertyFlagBits> properties,
                     vk::raii::Image& image,
                     vk::raii::DeviceMemory& imageMemory,
                     const uint32_t arrayLayers,
                     const vk::ImageCreateFlags flags = {}) const;

    void createDescriptorPool();

    void createBuffer(const vk::DeviceSize size,
                      const vk::Flags<vk::BufferUsageFlagBits> usage,
                      const vk::Flags<vk::MemoryPropertyFlagBits> properties,
                      vk::raii::Buffer& buffer,
                      vk::raii::DeviceMemory& bufferMemory) const;

    void copyBuffer(const vk::Buffer srcBuffer, const vk::Buffer dstBuffer, const vk::DeviceSize size) const;

    [[nodiscard]] uint32_t findMemoryType(const uint32_t typeFilter,
                                          const vk::Flags<vk::MemoryPropertyFlagBits> properties) const;

    void baseRecreateSwapChain();
    virtual void recreateSwapChain();

    void createSyncObjects();

    void createCommandBuffers();

    void createCommandPool();

    void createRenderPass();

    static std::vector<char> readShaderFile(const std::string& fileName);

    [[nodiscard]] vk::raii::ShaderModule createShaderModule(const std::vector<char>& code) const;

    [[nodiscard]] vk::raii::ImageView createImageView(const vk::raii::Image& image,
                                                      const vk::Format format,
                                                      const vk::ImageAspectFlagBits aspectFlags,
                                                      const vk::ImageViewType viewType,
                                                      const uint32_t layerCount,
                                                      const uint32_t baseArrayLayer = 0) const;

    [[nodiscard]] vk::raii::ImageView createImageView(const vk::Image& image,
                                                      const vk::Format format,
                                                      const vk::ImageAspectFlagBits aspectFlags,
                                                      const vk::ImageViewType viewType,
                                                      const uint32_t layerCount,
                                                      const uint32_t baseArrayLayer = 0) const;

    void createSurface();

    void createLogicalDevice();

    void createSwapChain();

    void pickPhysicalDevice();

    [[nodiscard]] bool isDeviceSuitable(const vk::PhysicalDevice& device) const;

    static vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats);

    static vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresenteModes);

    [[nodiscard]] vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities) const;

    [[nodiscard]] SwapChainSupportDetails querySwapChainSupport(const vk::PhysicalDevice& device) const;

    [[nodiscard]] bool checkDeviceExtensionSupport(const vk::PhysicalDevice& device) const;

    [[nodiscard]] QueueFamilyIndices findQueueFamilies(const vk::PhysicalDevice& device) const;

    void createVkInstance();

    static void printGlfwRequiredExtensions(const std::vector<const char*>& glfwExtensions);

    [[nodiscard]] bool checkValidationLayersSupport() const;

    static void addOsSpecificExtensions(std::vector<const char*>& glfwExtensions);

    //
    // void mainLoop() {
    //     while (!windowManager.shouldClose()) {
    //         windowManager.pollEvents();
    //         drawFrame();
    //     }
    //
    //     device.waitIdle();
    // }

    static void framebufferResizeCallback(GLFWwindow* window, int /* width */, int /* height */);

    template <typename T>
    void registerRenderProcess() {
        static_assert(std::is_base_of_v<IRenderProcess, T>, "T must derive from IRenderProcess");
        renderProcesses.push_back(std::make_unique<T>(*this));
    }

    void runRenderProcesses();

    [[nodiscard]] vk::raii::CommandBuffer beginSingleTimeCommands() const;

   public:
    explicit VulkanCore();
    virtual ~VulkanCore() = default;
};
