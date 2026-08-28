#pragma once

#include <vulkan/vulkan.h>
#include <cstdint>
#include <cstring>
#include <vector>

struct StagingBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void* mapped = nullptr;
    bool coherent = false;
};

class VertexBufferManager {
public:
    bool init(VkDevice device, VkPhysicalDevice phys);
    void shutdown(VkDevice device);

    StagingBuffer createStaging(VkDeviceSize size);
    StagingBuffer createDeviceLocal(VkDeviceSize size, VkBufferUsageFlags extraUsage = 0);
    void destroyBuffer(VkDevice device, StagingBuffer& buf);

    void map(VkDevice device, StagingBuffer& buf);
    void unmap(VkDevice device, StagingBuffer& buf);
    void upload(VkDevice device, StagingBuffer& buf, const void* data, VkDeviceSize size);

    void copyToDevice(VkCommandBuffer cmd, VkBuffer src, VkBuffer dst,
                      VkDeviceSize size);

    struct GpuBuffer {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize size = 0;
    };

    GpuBuffer createGpuBuffer(VkDeviceSize size, VkBufferUsageFlags usage);
    void destroyGpuBuffer(VkDevice device, GpuBuffer& buf);

private:
    uint32_t findMemType(uint32_t typeBits, VkMemoryPropertyFlags props);

    VkDevice device_ = VK_NULL_HANDLE;
    VkPhysicalDevice phys_ = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memProps_{};
};
