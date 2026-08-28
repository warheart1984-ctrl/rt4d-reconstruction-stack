#pragma once

#include <vulkan/vulkan.h>

// Denoiser / Ray Reconstruction compute pass with the confidence heuristic blend:
//   c_hist = clamp(c_d^wD * c_n^wN * c_m^wM * c_v^wV, 0, 1)
//   C = w_curr*C_t + w_hist*C_prev   (w_hist = c_hist, w_curr = 1 - c_hist)
// Special cases: disocclusion w=0, specular/emissive w*=0.5, edge w*=0.7.

class DenoiserPipeline {
public:
    struct Params {
        float wD = 1.0f, wN = 1.0f, wM = 1.0f, wV = 0.5f;
        float sigmaD = 0.02f, mMax = 32.0f, specThreshold = 0.3f, depthGradSlope = 0.05f;
    };

    // dsLayout bindings 0..8 samplers, 9 write color denoised, 10 write confidence.
    bool init(VkDevice device, VkDescriptorSetLayout dsLayout);
    void shutdown(VkDevice device);
    void record(VkCommandBuffer cmd, VkDescriptorSet ds, uint32_t w, uint32_t h);
    VkPipelineLayout layout() const { return layout_; }

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
};
