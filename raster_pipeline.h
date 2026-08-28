#pragma once

#include <vulkan/vulkan.h>
#include <array>
#include <cstdint>

enum class RasterMode : int {
    LIVING_MAP_POINTS  = 0,
    LIVING_MAP_EDGES   = 1,
    TACO_SCENE         = 2,
    BATTLE_CROWD       = 3,
    BATTLE_ATMOSPHERE  = 4,
    DRAGON_HATCH       = 5,
    COUNT              = 6
};

struct CameraUBO {
    float view[16];
    float proj[16];
    float camPos[3];
    float camForward[3];
    float camRight[3];
    float camUp[3];
    float resolution[2];
    float time;
    float fovX;   // radians
    float fovY;
    float fovZ;
    float fovW;
};

struct SceneUBO {
    float sunDir[3];
    float _pad0;
    float lampPos[3];
    float _pad1;
    float skyColor[3];
    float _pad2;
    float lampColor[3];
    float _pad3;
    float model[16];
};

struct BattleUBO {
    float sunDir[3];
    float _pad0;
    float fogColor[3];
    float fogDensity;
    float model[16];
};

struct AtmosphereUBO {
    float time;
    float _pad0;
    float _pad1;
    float _pad2;
    float fogColor[3];
    float density;
    float windDir[3];
    float turbulence;
};

struct HatchUBO {
    float lightPos[3];
    float _pad0;
    float viewPos[3];
    float _pad1;
    float baseColor[3];
    float _pad2;
    float sssColor[3];
    float hatchProgress;
    float model[16];
};

static_assert(sizeof(BattleUBO) <= sizeof(SceneUBO),
              "battle uniforms must fit the shared mesh scene allocation");
static_assert(sizeof(HatchUBO) <= sizeof(SceneUBO),
              "hatch uniforms must fit the shared mesh scene allocation");

struct LivingMapNode {
    float position[3];
    float color[3];
    float magnitude;
    float confidence;
    float normal[3];
    float viewDir[3];
    float lightDir[3];
};

struct CelParams {
    float baseColor[3];
    float _p0;
    float rimColor[3];
    float _p1;
    float shadowSteps;
    float shadowSmoothness;
    float sssStrength;
    float rimPower;
    float blushStrength;
    float _p2;
    float blushColor[3];
    float specularIntensity;
    float specularSize;
};

struct LivingMapEdge {
    float posA[3];
    float posB[3];
    float strength;
    float _pad;
};

struct AllocatedBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void* mapped = nullptr;
};

class RasterPipeline {
public:
    bool init(VkDevice device, VkPhysicalDevice phys,
              VkRenderPass renderPass, uint32_t width, uint32_t height);
    void shutdown(VkDevice device);

    AllocatedBuffer allocBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                 VkMemoryPropertyFlags props);
    void freeBuffer(VkDevice device, AllocatedBuffer& buf);
    void uploadToBuffer(AllocatedBuffer& buf, const void* data, VkDeviceSize size);

    void beginRenderPass(VkCommandBuffer cmd, VkFramebuffer fb,
                          uint32_t w, uint32_t h);
    void endRenderPass(VkCommandBuffer cmd);

    AllocatedBuffer& cameraUBO() { return cameraUBOs_[frameIndex_]; }
    AllocatedBuffer& celUBO() { return celUBOs_[frameIndex_]; }
    AllocatedBuffer& meshSceneUBO() { return meshSceneUBOs_[frameIndex_]; }
    VkPipeline pipeline(RasterMode m) { return pipelines_[static_cast<int>(m)]; }
    void advanceFrame() { frameIndex_ = (frameIndex_ + 1) % 2; }

    void bindCamera(VkCommandBuffer cmd);
    void drawLivingMapPoints(VkCommandBuffer cmd, uint32_t count);
    void drawLivingMapEdges(VkCommandBuffer cmd, uint32_t count);
    void drawMeshScene(VkCommandBuffer cmd, AllocatedBuffer& sceneUBO,
                       const SceneUBO& scene,
                       VkBuffer vertexBuffer, VkBuffer indexBuffer, uint32_t indexCount);
    void drawTacoScene(VkCommandBuffer cmd, VkBuffer vertexBuffer,
                       VkBuffer indexBuffer, uint32_t indexCount);
    void drawBattleCrowd(VkCommandBuffer cmd, VkBuffer vertexBuffer,
                         VkBuffer indexBuffer, uint32_t indexCount);
    void drawBattleAtmosphere(VkCommandBuffer cmd);
    void drawDragonHatch(VkCommandBuffer cmd, VkBuffer vertexBuffer,
                         VkBuffer indexBuffer, uint32_t indexCount);

private:
    VkPipeline createPipeline(const char* vertSpv, const char* fragSpv,
                               VkPrimitiveTopology topo,
                               const VkVertexInputBindingDescription* binds,
                               uint32_t bindCount,
                               const VkVertexInputAttributeDescription* attrs,
                               uint32_t attrCount,
                               VkPipelineLayout layout,
                               bool alphaBlend);

    VkDescriptorSetLayout cameraLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout sceneLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout textureLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout celLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;

    VkPipelineLayout cameraOnlyLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout cameraSceneLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout celPipelineLayout_ = VK_NULL_HANDLE;

    std::array<VkPipeline, 6> pipelines_{};
    std::array<VkDescriptorSet, 6> sceneSets_{};

    AllocatedBuffer cameraUBOs_[2]{};
    VkDescriptorSet cameraSets_[2] = {};
    AllocatedBuffer celUBOs_[2]{};
    VkDescriptorSet celSets_[2] = {};
    AllocatedBuffer meshSceneUBOs_[2]{};
    VkDescriptorSet meshSets_[2] = {};
    VkRenderPass renderPass_ = VK_NULL_HANDLE;
    int frameIndex_ = 0;

    VkDevice device_ = VK_NULL_HANDLE;
    VkPhysicalDevice phys_ = VK_NULL_HANDLE;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
};
