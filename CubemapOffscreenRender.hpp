#pragma once
#include "VulkanCore.hpp"
#include <memory>
#include <array>
#include <vector>
#include <glm/glm.hpp>

#ifndef SHARED_RENDER_DATA_DEFINED
#define SHARED_RENDER_DATA_DEFINED
// Shared data structure for passing textures between render processes
struct SharedRenderData {
    vk::raii::Image* cubemapTexture = nullptr;
    vk::raii::ImageView* cubemapImageView = nullptr;
    vk::raii::Sampler* cubemapSampler = nullptr;
};
#endif

struct OffscreenVertex {
    glm::vec3 pos;
    glm::vec4 color;  // r, g, b, face_id

    constexpr static vk::VertexInputBindingDescription getBindingDescription();
    constexpr static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions();
};

class CubemapOffscreenRender : public IRenderProcess {
   public:
    VulkanCore& vulkanCore;
    std::shared_ptr<SharedRenderData> sharedData;
    vk::raii::Buffer offscreenVertexBuffer;
    vk::raii::DeviceMemory offscreenVertexBufferMemory;
    vk::raii::Buffer offscreenIndexBuffer;
    vk::raii::DeviceMemory offscreenIndexBufferMemory;
    vk::raii::Image textureImage;
    vk::raii::DeviceMemory textureImageMemory;
    vk::raii::RenderPass offscreenRenderPass;
    vk::raii::Pipeline offscreenPipeline;
    vk::raii::PipelineLayout offscreenPipelineLayout;
    std::vector<vk::raii::Framebuffer> offscreenFramebuffers;
    std::vector<vk::raii::ImageView> offscreenImageViews;

    explicit CubemapOffscreenRender(VulkanCore& core, std::shared_ptr<SharedRenderData> data);
    virtual ~CubemapOffscreenRender();

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

    void execute() override;
    void createEmptyCubemapTexture();
    void createOffscreenVertexBuffer();
    void createOffscreenIndexBuffer();
    void createOffscreenRenderPass();
    void createOffscreenPipeline();
    void createOffscreenFramebuffers();
    void renderToCubemap();
    void saveCubemapToPNG();
};
