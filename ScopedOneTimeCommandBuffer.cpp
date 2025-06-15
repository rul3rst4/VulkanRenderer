#include "ScopedOneTimeCommandBuffer.hpp"

ScopedOneTimeCommandBuffer::ScopedOneTimeCommandBuffer(const vk::raii::Device& device,
                                    const vk::raii::CommandPool& commandPool,
                                    const vk::raii::Queue& graphicsQueue)
    : commandBuffer(nullptr), graphicsQueue(graphicsQueue) {
    const vk::CommandBufferAllocateInfo allocInfo{.sType = vk::StructureType::eCommandBufferAllocateInfo,
                                                    .commandPool = *commandPool,
                                                    .level = vk::CommandBufferLevel::ePrimary,
                                                    .commandBufferCount = 1};

    auto commandBuffers = device.allocateCommandBuffers(allocInfo);
    commandBuffer = std::move(commandBuffers[0]);

    constexpr vk::CommandBufferBeginInfo beginInfo{.sType = vk::StructureType::eCommandBufferBeginInfo,
                                                    .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit};

    commandBuffer.begin(beginInfo);
}

ScopedOneTimeCommandBuffer::~ScopedOneTimeCommandBuffer() {


    commandBuffer.end();

    const vk::SubmitInfo submitInfo{
        .sType = vk::StructureType::eSubmitInfo, .commandBufferCount = 1, .pCommandBuffers = &*commandBuffer};

    graphicsQueue.submit(submitInfo);
    graphicsQueue.waitIdle();
}
