#define VULKAN_HPP_NO_CONSTRUCTORS

#include <vulkan/vulkan_raii.hpp>

class ScopedOneTimeCommandBuffer {
   public:
    explicit ScopedOneTimeCommandBuffer(const vk::raii::Device& device,
                                        const vk::raii::CommandPool& commandPool,
                                        const vk::raii::Queue& graphicsQueue);

    ~ScopedOneTimeCommandBuffer();
    vk::raii::CommandBuffer commandBuffer;

   private:
    const vk::raii::Queue& graphicsQueue;
};
