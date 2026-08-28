#pragma once

#include "raster_pipeline.h"
#include "swapchain_manager.h"
#include "vertex_buffer_manager.h"
#include "scene_data.h"
#include "gltf_mesh.h"
#include "dlss45/dlss45_recon.h"
#include <vulkan/vulkan.h>
#include <vector>
#include <array>
#include <fstream>
#include <string>

struct ProvenanceFrame {
    std::string intentId;
    std::string timelineId;
    std::string worldId;
    double timeSeconds;
    std::string parameters;
    std::string frameId;
};

enum class RenderScene : int {
    LIVING_MAP = 0,
    TACO = 1,
    BATTLE = 2,
    DRAGON_HATCH = 3,
    SENTINEL = 4,
    RECON = 5           // RT4D Reconstruction Stack around the GLTF mesh scene
};

struct RenderConfig {
    uint32_t width = 1280;
    uint32_t height = 720;
    float fovDegrees = 60.0f;
    RenderScene scene = RenderScene::LIVING_MAP;
};

struct FrameResources {
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkSemaphore imageAcquired = VK_NULL_HANDLE;
    VkSemaphore renderFinished = VK_NULL_HANDLE;
    bool inFlight = false;
};

struct GpuTimer {
    VkQueryPool pools[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDevice device = VK_NULL_HANDLE;
    float timestampPeriod = 1.0f;
    uint32_t count = 0;
    int currentPool = 0;
    bool firstFrame = true;

    void init(VkDevice dev, VkPhysicalDevice phys, uint32_t maxQueries = 64) {
        device = dev;
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(phys, &props);
        timestampPeriod = props.limits.timestampPeriod;

        for (int i = 0; i < 2; i++) {
            VkQueryPoolCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
            info.queryType = VK_QUERY_TYPE_TIMESTAMP;
            info.queryCount = maxQueries;
            count = maxQueries;
            vkCreateQueryPool(device, &info, nullptr, &pools[i]);
        }
    }

    void shutdown() {
        for (int i = 0; i < 2; i++) {
            if (pools[i]) vkDestroyQueryPool(device, pools[i], nullptr);
            pools[i] = VK_NULL_HANDLE;
        }
    }

    void write(VkCommandBuffer cmd, uint32_t idx) {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, pools[currentPool], idx);
    }

    void nextFrame() {
        currentPool = 1 - currentPool;
        firstFrame = false;
    }

    void resetPoolInCmd(VkCommandBuffer cmd) {
        vkCmdResetQueryPool(cmd, pools[currentPool], 0, count);
    }

double getMs(uint32_t start, uint32_t end, int poolIndex) {
        uint64_t timestamps[2] = {0, 0};
        VkResult res = vkGetQueryPoolResults(device, pools[poolIndex], start, 2, sizeof(timestamps),
                              timestamps, sizeof(uint64_t),
                              VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        if (res != VK_SUCCESS) {
            return 0.0;
        }
        return double(timestamps[end] - timestamps[start]) * timestampPeriod / 1e6;
    }
};

class MandalaRasterRenderer {
public:
    bool init(VkInstance instance, VkPhysicalDevice phys,
              VkDevice device, VkSurfaceKHR surface,
              const RenderConfig& cfg);
    void shutdown();
    void resize(uint32_t w, uint32_t h);
    bool renderFrame(float deltaTime);
    void setScene(RenderScene scene);
    bool captureScreenshot(const std::string& path);

    // Debug / profiling
    void setDebugFlags(bool gpuTimer, bool dumpGBuffer,
                       int visualizer, const std::string& captureSeqDir,
                       const std::string& encodePath);
    void dumpGBuffer(const std::string& dir);
    void dumpImage(VkImage image, VkFormat format, uint32_t w, uint32_t h,
                   const std::string& path);

    LivingMapScene livingMap;
    TacoScene taco;
    BattleScene battle;
    DragonHatchScene dragonHatch;
    GLTFMeshScene sentinelMesh;
    DLSS45Recon dlss45;

    VkDevice device() const { return device_; }

private:
    bool createRenderPass();
    bool createFramebuffers();
    bool createFrameResources();
    bool uploadLivingMapBuffers();
    void recordCommandBuffer(FrameResources& frame, uint32_t swapIndex);
    void updateCamera(float aspect);

    VkInstance instance_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkPhysicalDevice phys_ = VK_NULL_HANDLE;
    VkQueue graphicsQueue_ = VK_NULL_HANDLE;
    VkQueue presentQueue_ = VK_NULL_HANDLE;
    uint32_t graphicsFamily_ = 0;
    uint32_t presentFamily_ = 0;

    SwapchainManager swapchain_;
    RasterPipeline pipeline_;
    VertexBufferManager vbm_;

    VkCommandPool cmdPool_ = VK_NULL_HANDLE;
    VkRenderPass renderPass_ = VK_NULL_HANDLE;
    VkFormat depthFormat_ = VK_FORMAT_D32_SFLOAT;

    std::vector<VkImage> depthImages_;
    std::vector<VkDeviceMemory> depthMemorys_;
    std::vector<VkImageView> depthViews_;
    std::vector<VkFramebuffer> framebuffers_;

    AllocatedBuffer nodeBuffer_{};
    AllocatedBuffer edgeBuffer_{};

    AllocatedBuffer sentinelVertexBuffer_{};
    AllocatedBuffer sentinelIndexBuffer_{};
    bool sentinelLoaded_ = false;

    std::array<FrameResources, 2> frames_{};
    int currentFrame_ = 0;
    uint32_t lastSwapIndex_ = 0;

    // Debug / profiling
    GpuTimer gpuTimer_;
    bool debugGpuTimer_ = false;
    bool debugDumpGBuffer_ = false;
    int debugVisualizer_ = 0;  // 0=none, 1=motion, 2=confidence, 3=history, 4=stability
    std::string debugCaptureSeqDir_;
    int debugFrameIndex_ = 0;

    RenderConfig config_;
    float time_ = 0.0f;
};
