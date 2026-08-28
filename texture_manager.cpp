#include "texture_manager.h"
#include <cstring>
#include <iostream>

TextureManager::TextureManager(VkDevice device, VkPhysicalDevice phys)
    : device_(device), phys_(phys) {
    vkGetPhysicalDeviceMemoryProperties(phys_, &memProps_);
}

TextureManager::~TextureManager() {
    cleanup(device_);
}

int TextureManager::findMemType(uint32_t typeBits, VkMemoryPropertyFlags flags) {
    for (uint32_t i = 0; i < memProps_.memoryTypeCount; i++) {
        if ((typeBits & (1u << i)) &&
            (memProps_.memoryTypes[i].propertyFlags & flags) == flags)
            return i;
    }
    return 0;
}

bool TextureManager::load(const std::string& path) {
    int w, h, c;
    stbi_set_flip_vertically_on_load(true);
    unsigned char* p = stbi_load(path.c_str(), &w, &h, &c, 4);
    if (!p) { std::cerr << "stb load fail: " << path << "\n"; return false; }

    VkDeviceSize sz = (VkDeviceSize)w * h * 4;

    // Create image
    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.extent = { (uint32_t)w, (uint32_t)h, 1 };
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.format = VK_FORMAT_R8G8B8A8_SRGB;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ici.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                VK_IMAGE_USAGE_SAMPLED_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.flags = 0;

    VkImage img;
    if (vkCreateImage(device_, &ici, nullptr, &img) != VK_SUCCESS) {
        std::cerr << "vkCreateImage fail\n"; stbi_image_free(p); return false;
    }

    // Memory
    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(device_, img, &mr);
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = findMemType(mr.memoryTypeBits,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    VkDeviceMemory mem;
    if (vkAllocateMemory(device_, &mai, nullptr, &mem) != VK_SUCCESS) {
        std::cerr << "vkAllocateMemory fail\n"; vkDestroyImage(device_, img, nullptr);
        stbi_image_free(p); return false;
    }
    if (vkBindImageMemory(device_, img, mem, 0) != VK_SUCCESS) {
        std::cerr << "vkBindImageMemory fail\n"; vkDestroyImage(device_, img, nullptr);
        vkFreeMemory(device_, mem, nullptr); stbi_image_free(p); return false;
    }

    // Map & copy pixels
    void* dst;
    if (vkMapMemory(device_, mem, 0, sz, 0, &dst) != VK_SUCCESS) {
        std::cerr << "vkMapMemory fail\n"; vkDestroyImage(device_, img, nullptr);
        vkFreeMemory(device_, mem, nullptr); stbi_image_free(p); return false;
    }
    memcpy(dst, p, (size_t)sz);
    vkUnmapMemory(device_, mem);
    stbi_image_free(p);

    // Image view
    VkImageViewCreateInfo vci{};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = img;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = VK_FORMAT_R8G8B8A8_SRGB;
    vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vci.subresourceRange.levelCount = 1;
    vci.subresourceRange.layerCount = 1;
    VkImageView iv;
    if (vkCreateImageView(device_, &vci, nullptr, &iv) != VK_SUCCESS) {
        std::cerr << "vkCreateImageView fail\n"; vkDestroyImage(device_, img, nullptr);
        vkFreeMemory(device_, mem, nullptr); stbi_image_free(p); return false;
    }

    // Sampler -- every member via sci
    VkSamplerCreateInfo sci{};
    sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter = VK_FILTER_LINEAR;
    sci.minFilter = VK_FILTER_LINEAR;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.anisotropyEnable = VK_FALSE;
    sci.maxAnisotropy = 1.0f;
    sci.compareEnable = VK_FALSE;
    sci.compareOp = VK_COMPARE_OP_ALWAYS;
    sci.minLod = 0.0f;
    sci.maxLod = 1.0f;
    sci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    sci.unnormalizedCoordinates = VK_FALSE;

    VkSampler smp;
    if (vkCreateSampler(device_, &sci, nullptr, &smp) != VK_SUCCESS) {
        std::cerr << "vkCreateSampler fail\n"; vkDestroyImageView(device_, iv, nullptr);
        vkDestroyImage(device_, img, nullptr); vkFreeMemory(device_, mem, nullptr);
        stbi_image_free(p); return false;
    }

    images_.push_back(img);
    memories_.push_back(mem);
    views_.push_back(iv);
    samplers_.push_back(smp);

    return true;
}

void TextureManager::cleanup(VkDevice device) {
    for (size_t i = 0; i < images_.size(); i++) {
        vkDestroyImageView(device, views_[i], nullptr);
        vkDestroyImage(device, images_[i], nullptr);
        vkFreeMemory(device, memories_[i], nullptr);
        vkDestroySampler(device, samplers_[i], nullptr);
    }
    images_.clear(); memories_.clear(); views_.clear(); samplers_.clear();
}