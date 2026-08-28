#pragma once

#include <vulkan/vulkan.h>

// Temporal reprojection compute pass. Reads the current G-buffer + history, writes
// the reprojected color and a per-pixel reprojection confidence.

class ReprojectionPipeline {
public:
    struct Internal {
        VkImage reprojectedColor = VK_NULL_HANDLE;
        VkImageView reprojectedColorView = VK_NULL_HANDLE;
        VkImage reprojectionConfidence = VK_NULL_HANDLE;
        VkImageView reprojectionConfidenceView = VK_NULL_HANDLE;
    };

    struct Params {
        float sigmaD = 0.02f;
        float mMax = 32.0f;
        float historyValid = 0.f;
        float _pad1 = 0.f;
    };

    // dsLayout: [0]=current color, [1]=current depth, [2]=motion,
    //           [3]=history color, [4]=history depth, [5]=write reproj color,
    //           [6]=write confidence.
    bool init(VkDevice device, VkDescriptorSetLayout dsLayout);
    void shutdown(VkDevice device);

    void record(VkCommandBuffer cmd, VkDescriptorSet ds, uint32_t w, uint32_t h,
                bool historyValid);

    VkPipelineLayout layout() const { return layout_; }

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
};
