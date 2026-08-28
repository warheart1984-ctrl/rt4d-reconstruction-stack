#pragma once

#include <vulkan/vulkan.h>

// Fullscreen tone-map composite pass: reads ColorSR_HR and writes to a target view
// (the swapchain image). Reuses fullscreen_quad.vert.glsl + tone_map.frag.glsl.

class ToneMapPipeline {
public:
    struct Params {
        uint32_t useACES = 1;
        float exposure = 1.0f;
        float _pad0 = 0.f;
        float _pad1 = 0.f;
    };

    bool init(VkDevice device, VkDescriptorSetLayout dsLayout, VkRenderPass renderPass);
    void shutdown(VkDevice device);
    void record(VkCommandBuffer cmd, VkDescriptorSet ds, VkFramebuffer fb,
                uint32_t w, uint32_t h);
    VkPipelineLayout layout() const { return layout_; }

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkRenderPass renderPass_ = VK_NULL_HANDLE;
};
