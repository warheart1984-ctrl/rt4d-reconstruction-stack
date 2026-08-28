#pragma once

#include <vulkan/vulkan.h>

// Super-resolution compute pass (classic edge-aware baseline for Phase 1; ML path for
// Phase 2 via sr_ml_conv.comp.glsl + MLWeightsLoader). Writes the HR output images.

class SRPipeline {
public:
    struct Params {
        uint32_t scale = 2;
        uint32_t lrWidth = 0;
        uint32_t lrHeight = 0;
        float edgeTau = 1.0f;
    };

    // dsLayout bindings 0..4 samplers, 5..7 write HR images.
    bool init(VkDevice device, VkDescriptorSetLayout dsLayout);
    void shutdown(VkDevice device);
    void record(VkCommandBuffer cmd, VkDescriptorSet ds, const Params& p);
    VkPipelineLayout layout() const { return layout_; }

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
};
