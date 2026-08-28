#include "vertex_buffer_manager.h"
#include <stdexcept>

bool VertexBufferManager::init(VkDevice device, VkPhysicalDevice phys) {
    device_ = device;
    phys_ = phys;
    vkGetPhysicalDeviceMemoryProperties(phys, &memProps_);
    return true;
}

void VertexBufferManager::shutdown(VkDevice device) {
}

uint32_t VertexBufferManager::findMemType(uint32_t typeBits,
                                            VkMemoryPropertyFlags props) {
    for (uint32_t i = 0; i < memProps_.memoryTypeCount; i++) {
        if ((typeBits & (1 << i)) &&
            (memProps_.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    throw std::runtime_error("No suitable memory type");
}

StagingBuffer VertexBufferManager::createStaging(VkDeviceSize size) {
    StagingBuffer sb{};
    sb.size = size;
    sb.coherent = true;

    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = size;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(device_, &bi, nullptr, &sb.buffer);

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device_, sb.buffer, &req);

    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = findMemType(req.memoryTypeBits,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkAllocateMemory(device_, &ai, nullptr, &sb.memory);
    vkBindBufferMemory(device_, sb.buffer, sb.memory, 0);

    vkMapMemory(device_, sb.memory, 0, size, 0, &sb.mapped);
    return sb;
}

StagingBuffer VertexBufferManager::createDeviceLocal(VkDeviceSize size,
                                                       VkBufferUsageFlags extraUsage) {
    StagingBuffer sb{};
    sb.size = size;
    sb.coherent = false;

    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = size;
    bi.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
               VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
               VK_BUFFER_USAGE_TRANSFER_DST_BIT | extraUsage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(device_, &bi, nullptr, &sb.buffer);

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device_, sb.buffer, &req);

    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = findMemType(req.memoryTypeBits,
                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkAllocateMemory(device_, &ai, nullptr, &sb.memory);
    vkBindBufferMemory(device_, sb.buffer, sb.memory, 0);
    return sb;
}

void VertexBufferManager::destroyBuffer(VkDevice device, StagingBuffer& buf) {
    if (buf.mapped) {
        vkUnmapMemory(device, buf.memory);
        buf.mapped = nullptr;
    }
    if (buf.buffer) vkDestroyBuffer(device, buf.buffer, nullptr);
    if (buf.memory) vkFreeMemory(device, buf.memory, nullptr);
    buf = {};
}

void VertexBufferManager::map(VkDevice device, StagingBuffer& buf) {
    if (!buf.mapped)
        vkMapMemory(device, buf.memory, 0, buf.size, 0, &buf.mapped);
}

void VertexBufferManager::unmap(VkDevice device, StagingBuffer& buf) {
    if (buf.mapped) {
        vkUnmapMemory(device, buf.memory);
        buf.mapped = nullptr;
    }
}

void VertexBufferManager::upload(VkDevice device, StagingBuffer& buf,
                                   const void* data, VkDeviceSize size) {
    if (!buf.mapped) map(device, buf);
    memcpy(buf.mapped, data, size);
}

void VertexBufferManager::copyToDevice(VkCommandBuffer cmd, VkBuffer src,
                                         VkBuffer dst, VkDeviceSize size) {
    VkBufferCopy copy{};
    copy.srcOffset = 0;
    copy.dstOffset = 0;
    copy.size = size;
    vkCmdCopyBuffer(cmd, src, dst, 1, &copy);
}

VertexBufferManager::GpuBuffer VertexBufferManager::createGpuBuffer(
    VkDeviceSize size, VkBufferUsageFlags usage) {
    GpuBuffer gb{};
    gb.size = size;

    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(device_, &bi, nullptr, &gb.buffer);

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device_, gb.buffer, &req);

    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = findMemType(req.memoryTypeBits,
                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkAllocateMemory(device_, &ai, nullptr, &gb.memory);
    vkBindBufferMemory(device_, gb.buffer, gb.memory, 0);
    return gb;
}

void VertexBufferManager::destroyGpuBuffer(VkDevice device, GpuBuffer& buf) {
    if (buf.buffer) vkDestroyBuffer(device, buf.buffer, nullptr);
    if (buf.memory) vkFreeMemory(device, buf.memory, nullptr);
    buf = {};
}
