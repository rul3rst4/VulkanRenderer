#pragma once
#include <vulkan/vulkan_raii.hpp>
#include <GLFW/glfw3.h>
#include <memory>
#include <glm/glm.hpp>

class IApp {
public:
    bool framebufferResized{};
};

class WindowManager {
public:
    WindowManager(const int width, const int height, const char* title);
    ~WindowManager();
    void registerResizeCallback(const GLFWframebuffersizefun resizeCallback) const;
    void setWindowUserPointer(void* pointer) const;
    [[nodiscard]] glm::i32vec2 getFrameBufferSize() const;
#if defined(VK_VERSION_1_0)
    [[nodiscard]] vk::SurfaceKHR createWindowSurface(const vk::Instance instance) const;
#endif
    [[nodiscard]] bool shouldClose() const;
    void pollEvents();

private:
    std::unique_ptr<GLFWwindow, decltype(&::glfwDestroyWindow)> window;
};
