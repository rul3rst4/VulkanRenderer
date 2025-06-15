#include "WindowManager.hpp"

WindowManager::WindowManager(const int width, const int height, const char* title)
    : window(std::unique_ptr<GLFWwindow, decltype(&::glfwDestroyWindow)>(nullptr, &::glfwDestroyWindow)) {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    window.reset(glfwCreateWindow(width, height, title, nullptr, nullptr));
}

WindowManager::~WindowManager() {
    glfwTerminate();
}

void WindowManager::registerResizeCallback(const GLFWframebuffersizefun resizeCallback) const {
    glfwSetFramebufferSizeCallback(window.get(), resizeCallback);
}

void WindowManager::setWindowUserPointer(void* pointer) const {
    glfwSetWindowUserPointer(window.get(), pointer);
}

glm::i32vec2 WindowManager::getFrameBufferSize() const {
    glm::i32vec2 windowSize;
    glfwGetFramebufferSize(window.get(), &windowSize.x, &windowSize.y);
    return windowSize;
}

vk::SurfaceKHR WindowManager::createWindowSurface(const vk::Instance instance) const {
    VkSurfaceKHR surfaceHandle;
    if (glfwCreateWindowSurface(instance, window.get(), nullptr, &surfaceHandle) != VK_SUCCESS) {
        throw std::runtime_error("failed to create window surface!");
    }

    return surfaceHandle;
}

bool WindowManager::shouldClose() const {
    return glfwWindowShouldClose(window.get());
}

void WindowManager::pollEvents() {
    glfwPollEvents();
}