#define VULKAN_HPP_NO_CONSTRUCTORS  // Permite usar Designated Initializers pra construir os objetos.
// #define VULKAN_HPP_NO_EXCEPTIONS // Retorna um result type pra ser tratado.
// #define GLFW_INCLUDE_VULKAN
#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>

#include <iostream>
#include <memory>
#include <format>
#include <fmt/format.h>
#include <algorithm>
#include <optional>
#include <set>

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presenteFamily;

    bool isComplete() { return graphicsFamily.has_value() && presenteFamily.has_value(); }
};

class HelloTiangle {
   private:
    std::unique_ptr<GLFWwindow, decltype(&::glfwDestroyWindow)> window;
    static inline constexpr int width = 800;
    static inline constexpr int height = 600;
    const std::vector<const char*> validationLayers = {"VK_LAYER_KHRONOS_validation"};
    static inline constexpr bool enableValidationLayers = [] {
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
    // TODO: Destruir tudo. Ou criando unique_ptrs ou usando vk_raii

    void initVulkan() {
        createVkInstance();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
    }

    void createSurface() {
        VkSurfaceKHR surfaceHandle;
        if (glfwCreateWindowSurface(instance, window.get(), nullptr, &surfaceHandle) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }

        surface = surfaceHandle;
    }

    void createLogicalDevice() {
        auto indices = findQueueFamilies(physicalDevice);

        std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(), indices.presenteFamily.value()};

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
                                        .enabledExtensionCount = 0};

        device = physicalDevice.createDevice(createInfo);
        graphicsQueue = device.getQueue(indices.graphicsFamily.value(), 0);
        presenteQueue = device.getQueue(indices.presenteFamily.value(), 0);
    }

    void pickPhysicalDevice() {
        auto devices = instance.enumeratePhysicalDevices();

        if (devices.size() == 0) {
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

    bool isDeviceSuitable(const vk::PhysicalDevice& device) {
        // auto properties = device.getProperties();
        // auto features = device.getFeatures();
        auto indices = findQueueFamilies(device);

        return indices.isComplete();
    }

    QueueFamilyIndices findQueueFamilies(const vk::PhysicalDevice& device) {
        auto queueFamilies = device.getQueueFamilyProperties();
        QueueFamilyIndices indices;

        int i = 0;
        for (const auto& queueFamily : queueFamilies) {
            if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
                indices.graphicsFamily = i;
            }

            auto presentSupport =
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

    void printGlfwRequiredExtensions(const std::vector<const char*> glfwExtensions) {
        auto extensions = vk::enumerateInstanceExtensionProperties();

        fmt::println("Available extensions:");

        for (const auto& extension : extensions) {
            const char* extensionName{extension.extensionName.data()};
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

    bool checkValidationLayersSupport() {
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

    void addMacSpecificExtensions(std::vector<const char*>& glfwExtensions) {
        // glfwExtensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME); Só é necessario no
        // Vulkan 1.0
        glfwExtensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window.get())) {
            glfwPollEvents();
        }
    }

    void cleanup() { glfwTerminate(); }

    void initWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        window.reset(glfwCreateWindow(width, height, "Vulkan", nullptr, nullptr));
    }

   public:
    explicit HelloTiangle()
        : window(std::unique_ptr<GLFWwindow, decltype(&::glfwDestroyWindow)>(nullptr, &::glfwDestroyWindow)) {}
    ~HelloTiangle() {}

    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }
};

int main() {
    HelloTiangle app;

    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
    }

    return EXIT_SUCCESS;
}
