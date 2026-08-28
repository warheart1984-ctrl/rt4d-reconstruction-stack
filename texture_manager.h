#pragma once

#include <vulkan/vulkan.h>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

struct Texture {
    std::string path;
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
    int width = 0, height = 0;
    VkFormat format = VK_FORMAT_R8G8B8A8_SRGB;
};

class TextureManager {
public:
    TextureManager(VkDevice device, VkPhysicalDevice phys);
    ~TextureManager();

    bool load(const std::string& path);
    void cleanup(VkDevice device);

    VkImageView view(uint32_t idx) const { return views_[idx]; }
    VkSampler sampler(uint32_t idx) const { return samplers_[idx]; }
    VkImage image(uint32_t idx) const { return images_[idx]; }
    uint32_t count() const { return (uint32_t)views_.size(); }

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkPhysicalDevice phys_ = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memProps_{};

    std::vector<VkImage> images_;
    std::vector<VkDeviceMemory> memories_;
    std::vector<VkImageView> views_;
    std::vector<VkSampler> samplers_;

    int findMemType(uint32_t typeBits, VkMemoryPropertyFlags flags);
};