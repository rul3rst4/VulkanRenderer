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

struct BlendVertex {
    glm::vec2 pos;
    glm::vec2 texCoord;

    constexpr static vk::VertexInputBindingDescription getBindingDescription();
    constexpr static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions();
};

class CubemapTextureBlend : public IRenderProcess {
public:
    VulkanCore& vulkanCore;
    std::shared_ptr<SharedRenderData> sharedData;

    // Screen rendering resources
    vk::raii::RenderPass screenRenderPass;
    vk::raii::Pipeline screenPipeline;
    vk::raii::PipelineLayout screenPipelineLayout;
    vk::raii::DescriptorSetLayout descriptorSetLayout;
    vk::raii::DescriptorSet descriptorSet;
    vk::raii::Sampler textureSampler;
    vk::raii::ImageView screenCubemapView;

    std::vector<vk::raii::Framebuffer> screenFramebuffers;
    std::vector<vk::raii::ImageView> swapChainImageViews;

    vk::raii::Buffer vertexBuffer;
    vk::raii::DeviceMemory vertexBufferMemory;
    vk::raii::Buffer indexBuffer;
    vk::raii::DeviceMemory indexBufferMemory;

    static constexpr std::array<BlendVertex, 4> vertices = {{
        {{-1.0f, -1.0f}, {0.0f, 0.0f}},
        {{1.0f, -1.0f}, {1.0f, 0.0f}},
        {{1.0f, 1.0f}, {1.0f, 1.0f}},
        {{-1.0f, 1.0f}, {0.0f, 1.0f}}
    }};
    static constexpr std::array<uint32_t, 6> indices = {0, 1, 2, 2, 3, 0};

    explicit CubemapTextureBlend(VulkanCore& core, std::shared_ptr<SharedRenderData> data);
    virtual ~CubemapTextureBlend();

    void execute() override;
    bool supportsContinuousRendering() const override;
    void render() override;

private:
    void createSwapChainImageViews();
    void createTextureSampler();
    void createCubemapImageView();
    void createDescriptorSetLayout();
    void createScreenRenderPass();
    void createScreenPipeline();
    void createVertexBuffer();
    void createIndexBuffer();
    void createDescriptorSet();
    void createScreenFramebuffers();
    void renderFrame();
};
