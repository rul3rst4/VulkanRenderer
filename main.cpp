#include <exception>
#include <iostream>
#include "VulkanCore.hpp"
#include "CubemapOffscreenRender.hpp"

int main() {
    try {
        VulkanCore app;
        app.registerRenderProcess<CubemapOffscreenRender>();
        app.runRenderProcesses();
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
    }

    return EXIT_SUCCESS;
}
