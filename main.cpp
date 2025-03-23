#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>

#include <iostream>
#include <vulkan/vulkan.hpp>

class HelloTiangle
{
private:
    void initVulkan() {

    }

    void mainLoop() {

    }

    void cleanup() {

    }

public:
    HelloTiangle(/* args */) {

    }
    ~HelloTiangle() {

    }

    void run() {
        initVulkan();
        mainLoop();
        cleanup();
    }
};



int main() {
    HelloTiangle app;

    try
    {
        app.run();
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }

    return EXIT_SUCCESS;
}
