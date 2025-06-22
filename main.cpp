#include <exception>
#include <iostream>
#include <memory>
#include "VulkanCore.hpp"
#include "CubemapOffscreenRender.hpp"
#include "CubemapTextureBlend.cpp"

int main() {
    try {
        VulkanCore app;

        // Create shared data for communication between render processes
        auto sharedData = std::make_shared<SharedRenderData>();

        // Register both render processes with shared data
        app.registerRenderProcess<CubemapOffscreenRender>(sharedData);
        // app.registerRenderProcess<CubemapTextureBlend>(sharedData);

        // Run one-time setup processes (like cubemap generation)
        app.runRenderProcesses();

        // // Enter main loop for continuous rendering
        // app.mainLoop();
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
    }

    return EXIT_SUCCESS;
}
