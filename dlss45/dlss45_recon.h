#pragma once

#include <vulkan/vulkan.h>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "gbuffer_contract.h"
#include "dlss45_pass_graph.h"
#include "rt4d_observability.h"
#include "reprojection_pipeline.h"
#include "denoiser_pipeline.h"
#include "sr_pipeline.h"
#include "tone_map_pipeline.h"
#include "../gltf_mesh.h"

// RT4D Reconstruction Stack coordinator. The DLSS45* symbols below are
// internal compatibility names, not NVIDIA DLSS integration. It owns everything
// necessary for the custom reconstruction pass graph
// that is NOT already owned by the renderer:
//   * the low-res G-buffer render pass + framebuffer
//   * the LR/HR internal images (G-buffer, history, reconstruction, upscale)
//   * the dedicated LR graphics pipeline (mesh -> 5 G-buffer attachments + depth)
//   * the three compute pipelines (reprojection, denoiser, super-resolution)
//   * descriptors
// The tone-map composite reuses the renderer's swapchain render pass + framebuffer.

struct DLSS45CameraUBO {
    float viewProj[16];
    float prevViewProj[16];
    float camPos[4];
    float resolution[4];
    float sunDir[4];
};

struct DLSS45SceneUBO {
    float skyColor[4];
    float lampColor[4];
    float lampPos[4];
};

class DLSS45Recon {
public:
    // lrScaleFactor: hr = lr * scale (e.g. 2 => render LR at half resolution).
    bool init(VkDevice device, VkPhysicalDevice phys,
              VkRenderPass rendererRenderPass,
              const GLTFMeshScene& mesh,
              VkCommandPool uploadPool, VkQueue uploadQueue,
              uint32_t displayW, uint32_t displayH,
              uint32_t lrScale = 2);
    void shutdown(VkDevice device);

    void render(VkCommandBuffer cmd,
                const GLTFMeshScene& mesh, VkBuffer vertexBuffer, VkBuffer indexBuffer,
                const float viewMatrix[16], const float projMatrix[16],
                const float camPos[3],
                uint32_t swapIndex, VkFramebuffer swapFramebuffer,
                uint32_t displayW, uint32_t displayH,
                uint64_t frameIndex);

    void requestHistoryReset(RT4DHistoryResetReason reason);
    bool exportObservability(VkCommandPool commandPool, VkQueue queue,
                             const std::string& directory,
                             uint32_t frameIndex, bool writeImages,
                             RT4DFrameMetrics& metrics);

    uint32_t lrWidth() const { return lrW_; }
    uint32_t lrHeight() const { return lrH_; }
    uint32_t deviceLocalTextureCount() const { return deviceLocalTextureCount_; }
    uint32_t sourceTextureUploadCount() const { return sourceTextureUploadCount_; }
    uint64_t historyResetCount() const { return historyResetCount_; }

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkPhysicalDevice phys_ = VK_NULL_HANDLE;

    uint32_t lrW_ = 0, lrH_ = 0;
    uint32_t hrW_ = 0, hrH_ = 0;
    uint32_t scale_ = 2;

    // --- G-buffer / history / recon / HR images ---
    struct ImageObj {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
    };
    ImageObj colorLDR_, depth_, normals_, motion_, material_, noise_;
    ImageObj colorHistory_, depthHistory_, normalHistory_;
    ImageObj reprojColor_, reprojConf_, denoised_, denoiseConf_;
    ImageObj colorSR_, depthHR_, motionHR_;

    VkSampler sampler_ = VK_NULL_HANDLE;
    VkSampler materialSampler_ = VK_NULL_HANDLE;

    ImageObj fallbackWhiteTexture_;
    std::vector<ImageObj> baseColorTextures_;
    std::vector<VkDescriptorSet> materialSets_;
    uint32_t deviceLocalTextureCount_ = 0;
    uint32_t sourceTextureUploadCount_ = 0;

    // --- descriptor sets / layouts / pool ---
    VkDescriptorSetLayout reprojSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout denoiserSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout srSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout toneMapSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    VkDescriptorSet reprojSet_ = VK_NULL_HANDLE;
    VkDescriptorSet denoiserSet_ = VK_NULL_HANDLE;
    VkDescriptorSet srSet_ = VK_NULL_HANDLE;
    VkDescriptorSet toneMapSet_ = VK_NULL_HANDLE;

    // --- UBO buffers for LR mesh pipeline ---
    VkBuffer camUBO_ = VK_NULL_HANDLE;   VkDeviceMemory camUBOMem_ = VK_NULL_HANDLE;
    VkBuffer sceneUBO_ = VK_NULL_HANDLE; VkDeviceMemory sceneUBOMem_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout meshSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout materialSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout meshSceneLayout_ = VK_NULL_HANDLE;
    VkDescriptorSet meshSet_ = VK_NULL_HANDLE;

    // --- render pass + framebuffer (LR) ---
    VkRenderPass gbufferPass_ = VK_NULL_HANDLE;
    VkFramebuffer gbufferFramebuffer_ = VK_NULL_HANDLE;
    VkPipelineLayout meshLayout_ = VK_NULL_HANDLE;
    VkPipeline meshPipeline_ = VK_NULL_HANDLE;

    // Composite render pass (renderer's) + tone map pipeline.
    VkRenderPass compositePass_ = VK_NULL_HANDLE;

    // --- pipelines ---
    ReprojectionPipeline reproj_;
    DenoiserPipeline denoiser_;
    SRPipeline sr_;
    ToneMapPipeline toneMap_;

    float prevViewProj_[16] = {};
    bool firstFrame_ = true;
    RT4DHistoryResetReason pendingHistoryReset_ = RT4DHistoryResetReason::None;
    RT4DHistoryResetReason lastFrameResetReason_ =
        RT4DHistoryResetReason::InitialFrame;
    bool lastFrameHistoryValid_ = false;
    uint64_t historyResetCount_ = 0;

    bool createImage(ImageObj& img, uint32_t w, uint32_t h, VkFormat fmt,
                     VkImageUsageFlags usage, VkImageAspectFlags aspect);
    bool createRenderPass();
    bool createMeshResources();
    bool createMaterialResources(const GLTFMeshScene& mesh,
                                 VkCommandPool uploadPool, VkQueue uploadQueue);
    bool uploadBaseColorTexture(ImageObj& image, const uint8_t* rgba,
                                uint32_t width, uint32_t height,
                                VkCommandPool uploadPool, VkQueue uploadQueue);
    bool createDescriptors();
    void clearHistory(VkCommandBuffer cmd, bool resourcesFirstFrame);
    void updateHistory(VkCommandBuffer cmd);

public:
    struct ReadbackImage {
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t channels = 0;
        std::vector<float> values;
    };

private:
    bool readbackImage(const ImageObj& image, VkFormat format,
                       VkImageAspectFlags aspect, uint32_t width, uint32_t height,
                       VkCommandPool commandPool, VkQueue queue,
                       ReadbackImage& output);
    void matMul(float* out, const float* a, const float* b); // out = a*b (4x4)
};
