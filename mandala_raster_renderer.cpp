#include "mandala_raster_renderer.h"
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <iomanip>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <unistd.h>

static uint32_t findMemType(VkPhysicalDevice phys, uint32_t bits,
                              VkMemoryPropertyFlags flags) {
    VkPhysicalDeviceMemoryProperties props;
    vkGetPhysicalDeviceMemoryProperties(phys, &props);
    for (uint32_t i = 0; i < props.memoryTypeCount; i++) {
        if ((bits & (1 << i)) && (props.memoryTypes[i].propertyFlags & flags) == flags)
            return i;
    }
    throw std::runtime_error("No suitable memory type");
}

static VkFormat findDepthFormat(VkPhysicalDevice phys) {
    VkFormat candidates[] = {
        VK_FORMAT_D32_SFLOAT,
        VK_FORMAT_D32_SFLOAT_S8_UINT,
        VK_FORMAT_D24_UNORM_S8_UINT
    };
    for (auto fmt : candidates) {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(phys, fmt, &props);
        if (props.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)
            return fmt;
    }
    return VK_FORMAT_D32_SFLOAT;
}

bool MandalaRasterRenderer::createRenderPass() {
    VkAttachmentDescription colorAtt{};
    colorAtt.format = swapchain_.colorFormat();
    colorAtt.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAtt.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentDescription depthAtt{};
    depthAtt.format = depthFormat_;
    depthAtt.samples = VK_SAMPLE_COUNT_1_BIT;
    depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAtt.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depthRef{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;
    subpass.pDepthStencilAttachment = &depthRef;

    VkSubpassDependency dep{};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass = 0;
    dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                       VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.dstStageMask = dep.srcStageMask;
    dep.srcAccessMask = 0;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    std::array<VkAttachmentDescription, 2> attachments = {colorAtt, depthAtt};
    VkRenderPassCreateInfo rpCI{};
    rpCI.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpCI.attachmentCount = (uint32_t)attachments.size();
    rpCI.pAttachments = attachments.data();
    rpCI.subpassCount = 1;
    rpCI.pSubpasses = &subpass;
    rpCI.dependencyCount = 1;
    rpCI.pDependencies = &dep;

    return vkCreateRenderPass(device_, &rpCI, nullptr, &renderPass_) == VK_SUCCESS;
}

bool MandalaRasterRenderer::createFramebuffers() {
    uint32_t count = swapchain_.imageCount();
    depthImages_.resize(count);
    depthMemorys_.resize(count);
    depthViews_.resize(count);
    framebuffers_.resize(count);

    auto ext = swapchain_.extent();

    for (uint32_t i = 0; i < count; i++) {
        VkImageCreateInfo di{};
        di.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        di.imageType = VK_IMAGE_TYPE_2D;
        di.format = depthFormat_;
        di.extent = {ext.width, ext.height, 1};
        di.mipLevels = 1;
        di.arrayLayers = 1;
        di.samples = VK_SAMPLE_COUNT_1_BIT;
        di.tiling = VK_IMAGE_TILING_OPTIMAL;
        di.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        vkCreateImage(device_, &di, nullptr, &depthImages_[i]);

        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(device_, depthImages_[i], &req);
        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = findMemType(phys_, req.memoryTypeBits,
                                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        vkAllocateMemory(device_, &ai, nullptr, &depthMemorys_[i]);
        vkBindImageMemory(device_, depthImages_[i], depthMemorys_[i], 0);

        VkImageViewCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vi.image = depthImages_[i];
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = depthFormat_;
        vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        vi.subresourceRange.levelCount = 1;
        vi.subresourceRange.layerCount = 1;
        vkCreateImageView(device_, &vi, nullptr, &depthViews_[i]);

        std::array<VkImageView, 2> attachments = {swapchain_.imageView(i), depthViews_[i]};
        VkFramebufferCreateInfo fbCI{};
        fbCI.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbCI.renderPass = renderPass_;
        fbCI.attachmentCount = (uint32_t)attachments.size();
        fbCI.pAttachments = attachments.data();
        fbCI.width = ext.width;
        fbCI.height = ext.height;
        fbCI.layers = 1;
        vkCreateFramebuffer(device_, &fbCI, nullptr, &framebuffers_[i]);
    }
    return true;
}

bool MandalaRasterRenderer::createFrameResources() {
    VkCommandPoolCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = graphicsFamily_;
    vkCreateCommandPool(device_, &cpci, nullptr, &cmdPool_);

    for (auto& f : frames_) {
        VkCommandBufferAllocateInfo cbai{};
        cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool = cmdPool_;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        vkAllocateCommandBuffers(device_, &cbai, &f.cmd);

        VkFenceCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        vkCreateFence(device_, &fci, nullptr, &f.fence);

        VkSemaphoreCreateInfo sci{};
        sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        vkCreateSemaphore(device_, &sci, nullptr, &f.imageAcquired);
        vkCreateSemaphore(device_, &sci, nullptr, &f.renderFinished);
    }
    return true;
}

bool MandalaRasterRenderer::uploadLivingMapBuffers() {
    livingMap.generateDefault(1000);

    VkDeviceSize nodeSz = livingMap.nodeCount() * sizeof(LivingMapNode);
    nodeBuffer_ = pipeline_.allocBuffer(nodeSz,
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    pipeline_.uploadToBuffer(device_, nodeBuffer_, livingMap.nodes().data(), nodeSz);

    std::vector<LivingMapEdge> gpuEdges;
    gpuEdges.reserve(livingMap.edgeCount());
    for (auto& e : livingMap.edges()) {
        LivingMapEdge ge{};
        auto& na = livingMap.nodes()[e.nodeA];
        auto& nb = livingMap.nodes()[e.nodeB];
        memcpy(ge.posA, na.position, sizeof(float) * 3);
        memcpy(ge.posB, nb.position, sizeof(float) * 3);
        ge.strength = e.strength;
        gpuEdges.push_back(ge);
    }

    VkDeviceSize edgeSz = gpuEdges.size() * sizeof(LivingMapEdge);
    if (edgeSz > 0) {
        edgeBuffer_ = pipeline_.allocBuffer(edgeSz,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        pipeline_.uploadToBuffer(device_, edgeBuffer_, gpuEdges.data(), edgeSz);
    }
    return true;
}

void MandalaRasterRenderer::updateCamera(float aspect) {
    if (config_.scene == RenderScene::LIVING_MAP) {
        livingMap.setCamera(0, 1.5f, 3.0f, 0, 0, 0,
                            config_.fovDegrees, config_.fovDegrees, aspect);
    } else if (config_.scene == RenderScene::SENTINEL ||
               config_.scene == RenderScene::RECON) {
        float ex = sentinelMesh.center[0];
        float ey = sentinelMesh.center[1];
        float ez = sentinelMesh.center[2] + sentinelMesh.radius * 2.5f;
        livingMap.setCamera(ex, ey, ez,
                            sentinelMesh.center[0], sentinelMesh.center[1], sentinelMesh.center[2],
                            config_.fovDegrees, config_.fovDegrees, aspect);
    }
}

void MandalaRasterRenderer::recordCommandBuffer(FrameResources& frame,
                                                  uint32_t swapIndex) {
    VkCommandBuffer cmd = frame.cmd;
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    if (debugGpuTimer_) {
        gpuTimer_.resetPoolInCmd(cmd);
    }

    auto ext = swapchain_.extent();

    if (config_.scene == RenderScene::RECON) {
        // RT4D Reconstruction Stack pass graph (G-buffer -> compute -> composite).
        if (debugGpuTimer_) gpuTimer_.write(cmd, 0);  // frame start
        if (sentinelLoaded_) {
            dlss45.render(cmd, sentinelMesh,
                          sentinelVertexBuffer_.buffer, sentinelIndexBuffer_.buffer,
                          livingMap.viewMatrix, livingMap.projMatrix, livingMap.camPos,
                          swapIndex, framebuffers_[swapIndex],
                          ext.width, ext.height);
        }
        if (debugGpuTimer_) gpuTimer_.write(cmd, 1);  // recon done
        vkEndCommandBuffer(cmd);
        return;
    }

    if (debugGpuTimer_) gpuTimer_.write(cmd, 0);  // frame start

    pipeline_.beginRenderPass(cmd, framebuffers_[swapIndex], ext.width, ext.height);

    if (debugGpuTimer_) gpuTimer_.write(cmd, 2);  // after begin render pass

    // ... rest of non-RECON path ...

    CameraUBO cam{};
    memcpy(cam.view, livingMap.viewMatrix, sizeof(float) * 16);
    memcpy(cam.proj, livingMap.projMatrix, sizeof(float) * 16);
    memcpy(cam.camPos, livingMap.camPos, sizeof(float) * 3);
    cam.resolution[0] = (float)ext.width;
    cam.resolution[1] = (float)ext.height;
    cam.time = time_;
    pipeline_.uploadToBuffer(device_, pipeline_.cameraUBO(), &cam, sizeof(CameraUBO));
    pipeline_.bindCamera(cmd, pipeline_.cameraUBO());

    CelParams cel{};
    cel.baseColor[0] = 0.95f; cel.baseColor[1] = 0.85f; cel.baseColor[2] = 0.78f;
    cel.rimColor[0] = 1.0f; cel.rimColor[1] = 0.35f; cel.rimColor[2] = 0.0f;
    cel.shadowSteps = 3.0f;
    cel.shadowSmoothness = 0.35f;
    cel.sssStrength = 0.6f;
    cel.rimPower = 3.0f;
    cel.blushStrength = 0.5f;
    cel.blushColor[0] = 1.0f; cel.blushColor[1] = 0.3f; cel.blushColor[2] = 0.25f;
    cel.specularIntensity = 0.4f;
    cel.specularSize = 0.15f;
    pipeline_.uploadToBuffer(device_, pipeline_.celUBO(), &cel, sizeof(CelParams));

    switch (config_.scene) {
        case RenderScene::LIVING_MAP:
            if (nodeBuffer_.buffer) {
                VkDeviceSize off = 0;
                vkCmdBindVertexBuffers(cmd, 0, 1, &nodeBuffer_.buffer, &off);
                pipeline_.drawLivingMapPoints(cmd, livingMap.nodeCount());
            }
            if (edgeBuffer_.buffer) {
                VkDeviceSize off = 0;
                vkCmdBindVertexBuffers(cmd, 0, 1, &edgeBuffer_.buffer, &off);
                pipeline_.drawLivingMapEdges(cmd, livingMap.edgeCount());
            }
            break;

        case RenderScene::TACO:
            // Bind the camera+scene descriptor set (inside drawTacoScene) and
            // draw the fox-fixture mesh with the taco_scene pipeline. Previously
            // this passed NULL buffers and left only the camera-only set bound,
            // crashing and producing VUID-08600 layout mismatches.
            if (sentinelLoaded_ && sentinelIndexBuffer_.buffer) {
                SceneUBO sc{};
                for (int i = 0; i < 16; i++) sc.model[i] = 0;
                sc.model[0] = sc.model[5] = sc.model[10] = sc.model[15] = 1.0f;
                sc.sunDir[0] = -0.5f; sc.sunDir[1] = -0.8f; sc.sunDir[2] = -0.3f;
                sc.lampPos[0] = 3.0f; sc.lampPos[1] = 2.0f; sc.lampPos[2] = 1.0f;
                sc.skyColor[0] = 0.4f; sc.skyColor[1] = 0.6f; sc.skyColor[2] = 0.9f;
                sc.lampColor[0] = 1.0f; sc.lampColor[1] = 0.85f; sc.lampColor[2] = 0.6f;
                pipeline_.uploadToBuffer(device_, pipeline_.meshSceneUBO(), &sc, sizeof(SceneUBO));
                pipeline_.drawTacoScene(cmd, sentinelVertexBuffer_.buffer,
                                        sentinelIndexBuffer_.buffer, sentinelMesh.indexCount());
            }
            break;

        case RenderScene::BATTLE:
            pipeline_.drawBattleAtmosphere(cmd);
            break;

        case RenderScene::DRAGON_HATCH:
            // Same fix as TACO: bind camera+scene set and real mesh geometry.
            if (sentinelLoaded_ && sentinelIndexBuffer_.buffer) {
                SceneUBO sc{};
                for (int i = 0; i < 16; i++) sc.model[i] = 0;
                sc.model[0] = sc.model[5] = sc.model[10] = sc.model[15] = 1.0f;
                sc.sunDir[0] = -0.5f; sc.sunDir[1] = -0.8f; sc.sunDir[2] = -0.3f;
                sc.lampPos[0] = 3.0f; sc.lampPos[1] = 2.0f; sc.lampPos[2] = 1.0f;
                sc.skyColor[0] = 0.4f; sc.skyColor[1] = 0.6f; sc.skyColor[2] = 0.9f;
                sc.lampColor[0] = 1.0f; sc.lampColor[1] = 0.85f; sc.lampColor[2] = 0.6f;
                pipeline_.uploadToBuffer(device_, pipeline_.meshSceneUBO(), &sc, sizeof(SceneUBO));
                pipeline_.drawDragonHatch(cmd, sentinelVertexBuffer_.buffer,
                                          sentinelIndexBuffer_.buffer, sentinelMesh.indexCount());
            }
            break;

        case RenderScene::SENTINEL:
            if (sentinelLoaded_ && sentinelIndexBuffer_.buffer) {
                SceneUBO s{};
                for (int i = 0; i < 16; i++) s.model[i] = 0;
                s.model[0] = s.model[5] = s.model[10] = s.model[15] = 1.0f;
                s.sunDir[0] = -0.5f; s.sunDir[1] = -0.8f; s.sunDir[2] = -0.3f;
                s.lampPos[0] = 3.0f; s.lampPos[1] = 2.0f; s.lampPos[2] = 1.0f;
                s.skyColor[0] = 0.4f; s.skyColor[1] = 0.6f; s.skyColor[2] = 0.9f;
                s.lampColor[0] = 1.0f; s.lampColor[1] = 0.85f; s.lampColor[2] = 0.6f;
                pipeline_.drawMeshScene(cmd, pipeline_.cameraUBO(), pipeline_.meshSceneUBO(),
                                        s, sentinelVertexBuffer_.buffer,
                                        sentinelIndexBuffer_.buffer, sentinelMesh.indexCount());
            }
            break;
    }

    // Keep descriptor selection aligned with the UBOs uploaded above for this
    // command buffer. Advancing before the draw selected an uninitialized
    // descriptor slot, making the mesh path silently render black.
    pipeline_.advanceFrame();

    if (debugGpuTimer_) gpuTimer_.write(cmd, 3);  // after scene draws

    pipeline_.endRenderPass(cmd);

    if (debugGpuTimer_) gpuTimer_.write(cmd, 4);  // after end render pass

    vkEndCommandBuffer(cmd);
}

bool MandalaRasterRenderer::init(VkInstance instance, VkPhysicalDevice phys,
                                   VkDevice device, VkSurfaceKHR surface,
                                   const RenderConfig& cfg) {
    instance_ = instance;
    phys_ = phys;
    device_ = device;
    config_ = cfg;
    time_ = 0.0f;

    uint32_t qfCount;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &qfCount, nullptr);
    std::vector<VkQueueFamilyProperties> qfProps(qfCount);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &qfCount, qfProps.data());

    for (uint32_t i = 0; i < qfCount; i++) {
        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(phys, i, surface, &presentSupport);
        if ((qfProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && presentSupport) {
            graphicsFamily_ = i;
            presentFamily_ = i;
            break;
        }
    }
    vkGetDeviceQueue(device, graphicsFamily_, 0, &graphicsQueue_);
    vkGetDeviceQueue(device, presentFamily_, 0, &presentQueue_);

    if (!swapchain_.init(phys, device, surface, cfg.width, cfg.height))
        return false;

    depthFormat_ = findDepthFormat(phys);

    if (!createRenderPass()) return false;
    if (!createFramebuffers()) return false;
    if (!createFrameResources()) return false;

    vbm_.init(device, phys);

    if (!pipeline_.init(device, phys, renderPass_, cfg.width, cfg.height))
        return false;

    if (!uploadLivingMapBuffers()) return false;

    const char* sentinelCandidates[] = {
        "armored-sentinel-v1.glb",
        "../armored-sentinel-v1.glb",
        "../../armored-sentinel-v1.glb",
        "../../armored-sentinel-v1.glb",
        "../../../docs/proofs/rt4d-fox-fixture-smoke/fox-fixture.glb",
        "../../docs/proofs/rt4d-fox-fixture-smoke/fox-fixture.glb",
        "docs/proofs/rt4d-fox-fixture-smoke/fox-fixture.glb"
    };
    for (auto* p : sentinelCandidates) {
        if (sentinelMesh.load(p)) {
            fprintf(stderr, "[SENTINEL] loaded %s (%u verts, %u idx; center %.3f %.3f %.3f; radius %.3f)\n",
                    p, sentinelMesh.vertexCount(), sentinelMesh.indexCount(),
                    sentinelMesh.center[0], sentinelMesh.center[1], sentinelMesh.center[2],
                    sentinelMesh.radius);
            break;
        }
    }
    if (sentinelMesh.vertexCount() == 0) {
        fprintf(stderr, "[SENTINEL] no GLB available; sentinel scene disabled\n");
    } else {
        VkDeviceSize vsz = sentinelMesh.vertexCount() * sizeof(GLTFMeshVertex);
        sentinelVertexBuffer_ = pipeline_.allocBuffer(vsz, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        pipeline_.uploadToBuffer(device_, sentinelVertexBuffer_,
                                 sentinelMesh.vertices().data(), vsz);

        VkDeviceSize isz = sentinelMesh.indexCount() * sizeof(uint32_t);
        if (isz > 0) {
            sentinelIndexBuffer_ = pipeline_.allocBuffer(isz, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            pipeline_.uploadToBuffer(device_, sentinelIndexBuffer_,
                                     sentinelMesh.indices().data(), isz);
        }
        sentinelLoaded_ = true;
    }

    if (!dlss45.init(device_, phys_, renderPass_, swapchain_.colorFormat(),
                     config_.width, config_.height, 2)) {
        fprintf(stderr, "[RT4D Reconstruction Stack] initialization failed; refusing RECON mode\n");
        return false;
    }

    if (debugGpuTimer_) {
        gpuTimer_.init(device_, phys_, 64);
        // One-time command buffer to reset both query pools before first use
        VkCommandBufferAllocateInfo alloc{};
        alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc.commandPool = cmdPool_;
        alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc.commandBufferCount = 1;
        VkCommandBuffer cmd;
        vkAllocateCommandBuffers(device_, &alloc, &cmd);
        VkCommandBufferBeginInfo cbbi{};
        cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &cbbi);
        vkCmdResetQueryPool(cmd, gpuTimer_.pools[0], 0, gpuTimer_.count);
        vkCmdResetQueryPool(cmd, gpuTimer_.pools[1], 0, gpuTimer_.count);
        vkEndCommandBuffer(cmd);
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        vkQueueSubmit(graphicsQueue_, 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue_);
        vkFreeCommandBuffers(device_, cmdPool_, 1, &cmd);
    }

    return true;
}

void MandalaRasterRenderer::shutdown() {
    if (!device_) return;
    vkDeviceWaitIdle(device_);

    pipeline_.freeBuffer(device_, nodeBuffer_);
    pipeline_.freeBuffer(device_, edgeBuffer_);
    pipeline_.freeBuffer(device_, sentinelVertexBuffer_);
    pipeline_.freeBuffer(device_, sentinelIndexBuffer_);

    dlss45.shutdown(device_);
    pipeline_.shutdown(device_);

    if (debugGpuTimer_)
        gpuTimer_.shutdown();

    for (auto& f : frames_) {
        if (f.fence) vkDestroyFence(device_, f.fence, nullptr);
        if (f.imageAcquired) vkDestroySemaphore(device_, f.imageAcquired, nullptr);
        if (f.renderFinished) vkDestroySemaphore(device_, f.renderFinished, nullptr);
    }
    if (cmdPool_) vkDestroyCommandPool(device_, cmdPool_, nullptr);

    for (auto fb : framebuffers_) if (fb) vkDestroyFramebuffer(device_, fb, nullptr);
    for (auto dv : depthViews_) if (dv) vkDestroyImageView(device_, dv, nullptr);
    for (auto dm : depthMemorys_) if (dm) vkFreeMemory(device_, dm, nullptr);
    for (auto di : depthImages_) if (di) vkDestroyImage(device_, di, nullptr);

    if (renderPass_) vkDestroyRenderPass(device_, renderPass_, nullptr);

    swapchain_.shutdown(device_);
}

void MandalaRasterRenderer::resize(uint32_t w, uint32_t h) {
    if (w == 0 || h == 0) return;
    vkDeviceWaitIdle(device_);

    for (auto fb : framebuffers_) if (fb) vkDestroyFramebuffer(device_, fb, nullptr);
    for (auto dv : depthViews_) if (dv) vkDestroyImageView(device_, dv, nullptr);
    for (auto dm : depthMemorys_) if (dm) vkFreeMemory(device_, dm, nullptr);
    for (auto di : depthImages_) if (di) vkDestroyImage(device_, di, nullptr);
    framebuffers_.clear();
    depthViews_.clear();
    depthMemorys_.clear();
    depthImages_.clear();

    if (renderPass_) {
        vkDestroyRenderPass(device_, renderPass_, nullptr);
        renderPass_ = VK_NULL_HANDLE;
    }

    config_.width = w;
    config_.height = h;

    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(phys_, VK_NULL_HANDLE, &caps);
    uint32_t w2 = std::clamp(w, caps.minImageExtent.width, caps.maxImageExtent.width);
    uint32_t h2 = std::clamp(h, caps.minImageExtent.height, caps.maxImageExtent.height);
    swapchain_.recreate(device_, w2, h2);

    createRenderPass();
    createFramebuffers();
}

bool MandalaRasterRenderer::renderFrame(float deltaTime) {
    time_ += deltaTime;
    livingMap.updateTime(deltaTime);
    battle.setTime(time_);

    auto& frame = frames_[currentFrame_];

    // Read GPU timestamps from PREVIOUS frame (uses previous frame's query pool)
    int prevPool = 1 - gpuTimer_.currentPool;
    if (debugGpuTimer_ && currentFrame_ > 0) {
        if (config_.scene == RenderScene::RECON) {
            double frameMs = gpuTimer_.getMs(0, 1, prevPool);
            fprintf(stderr, "[gpu-timer] RECON total: %.3f ms\n", frameMs);
            fflush(stderr);
        } else {
            double totalMs = gpuTimer_.getMs(0, 4, prevPool);
            double renderPassMs = gpuTimer_.getMs(0, 2, prevPool);
            double sceneMs = gpuTimer_.getMs(2, 3, prevPool);
            double endPassMs = gpuTimer_.getMs(3, 4, prevPool);
            fprintf(stderr, "[gpu-timer] total: %.3f ms  (renderPass: %.3f  scene: %.3f  endPass: %.3f)\n",
                    totalMs, renderPassMs, sceneMs, endPassMs);
            fflush(stderr);
        }
    }

    // Skip fence wait for first frame (fence is initially unsignaled)
    if (currentFrame_ > 0) {
        vkWaitForFences(device_, 1, &frame.fence, VK_TRUE, UINT64_MAX);
    }

    uint32_t swapIndex = swapchain_.acquireNextImage(device_, frame.imageAcquired);
    lastSwapIndex_ = swapIndex;

    vkResetFences(device_, 1, &frame.fence);
    vkResetCommandBuffer(frame.cmd, 0);

    // Switch to next frame's query pool
    gpuTimer_.nextFrame();

    updateCamera((float)config_.width / config_.height);
    recordCommandBuffer(frame, swapIndex);

    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.waitSemaphoreCount = 1;
    si.pWaitSemaphores = &frame.imageAcquired;
    si.pWaitDstStageMask = &waitStage;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &frame.cmd;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &frame.renderFinished;
    VkResult subRes = vkQueueSubmit(graphicsQueue_, 1, &si, frame.fence);

    VkResult presRes = swapchain_.present(presentQueue_, swapIndex, frame.renderFinished);

    static int frameCount = 0;
    frameCount++;
    if (frameCount % 60 == 0) {
        fprintf(stderr, "[gpu-timer] frames rendered: %d\n", frameCount);
        fflush(stderr);
    }

    currentFrame_ = (currentFrame_ + 1) % 2;
    return true;
}

bool MandalaRasterRenderer::captureScreenshot(const std::string& path) {
    // Wait for GPU to finish the current frame to ensure swapchain image is ready.
    vkDeviceWaitIdle(device_);

    VkExtent2D ext = swapchain_.extent();
    uint32_t w = ext.width;
    uint32_t h = ext.height;
    uint32_t imgIdx = lastSwapIndex_;
    VkImage srcImage = swapchain_.image(imgIdx);

    // Create a host-visible staging buffer for the readback.
    VkDeviceSize bufferSize = (VkDeviceSize)w * h * 4;
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = bufferSize;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer stagingBuf;
    VkDeviceMemory stagingMem;
    if (vkCreateBuffer(device_, &bi, nullptr, &stagingBuf) != VK_SUCCESS)
        return false;

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device_, stagingBuf, &memReq);
    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = memReq.size;
    ai.memoryTypeIndex = findMemType(phys_, memReq.memoryTypeBits,
                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (vkAllocateMemory(device_, &ai, nullptr, &stagingMem) != VK_SUCCESS) {
        vkDestroyBuffer(device_, stagingBuf, nullptr);
        return false;
    }
    if (vkBindBufferMemory(device_, stagingBuf, stagingMem, 0) != VK_SUCCESS) {
        vkFreeMemory(device_, stagingMem, nullptr);
        vkDestroyBuffer(device_, stagingBuf, nullptr);
        return false;
    }

    // One-time command buffer to copy swapchain image -> staging buffer.
    VkCommandBufferAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc.commandPool = cmdPool_;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 1;
    VkCommandBuffer cmd;
    if (vkAllocateCommandBuffers(device_, &alloc, &cmd) != VK_SUCCESS) {
        vkFreeMemory(device_, stagingMem, nullptr);
        vkDestroyBuffer(device_, stagingBuf, nullptr);
        return false;
    }

    VkCommandBufferBeginInfo cbbi{};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &cbbi);

    // Transition swapchain image from PRESENT_SRC to TRANSFER_SRC.
    VkImageMemoryBarrier barr{};
    barr.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barr.srcAccessMask = 0;
    barr.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barr.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    barr.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barr.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barr.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barr.image = srcImage;
    barr.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barr.subresourceRange.baseMipLevel = 0;
    barr.subresourceRange.levelCount = 1;
    barr.subresourceRange.baseArrayLayer = 0;
    barr.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &barr);

    // Copy image to buffer.
    VkBufferImageCopy bic{};
    bic.bufferOffset = 0;
    bic.bufferRowLength = w;
    bic.bufferImageHeight = h;
    bic.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    bic.imageSubresource.mipLevel = 0;
    bic.imageSubresource.baseArrayLayer = 0;
    bic.imageSubresource.layerCount = 1;
    bic.imageOffset = {0, 0, 0};
    bic.imageExtent = {w, h, 1};
    vkCmdCopyImageToBuffer(cmd, srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           stagingBuf, 1, &bic);

    // Transition buffer for host read.
    VkBufferMemoryBarrier bmb{};
    bmb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bmb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bmb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bmb.buffer = stagingBuf;
    bmb.offset = 0;
    bmb.size = bufferSize;
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT,
                         0, 0, nullptr, 1, &bmb, 0, nullptr);

    vkEndCommandBuffer(cmd);

    // Submit and wait.
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    if (vkQueueSubmit(graphicsQueue_, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS) {
        vkFreeCommandBuffers(device_, cmdPool_, 1, &cmd);
        vkFreeMemory(device_, stagingMem, nullptr);
        vkDestroyBuffer(device_, stagingBuf, nullptr);
        return false;
    }
    vkQueueWaitIdle(graphicsQueue_);

    // Map and write PNG (swapchain is B8G8R8A8_UNORM -> stb wants RGBA).
    void* data;
    vkMapMemory(device_, stagingMem, 0, bufferSize, 0, &data);
    // Allocate RGBA buffer for stb.
    std::vector<unsigned char> rgba(w * h * 4);
    const uint32_t* src = static_cast<const uint32_t*>(data);
    for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
            uint32_t bgra = src[y * w + x];
            rgba[(y * w + x) * 4 + 0] = (bgra >> 0) & 0xFF;  // R
            rgba[(y * w + x) * 4 + 1] = (bgra >> 8) & 0xFF;  // G
            rgba[(y * w + x) * 4 + 2] = (bgra >> 16) & 0xFF; // B
            rgba[(y * w + x) * 4 + 3] = (bgra >> 24) & 0xFF; // A
        }
    }
    vkUnmapMemory(device_, stagingMem);

    int ok = stbi_write_png(path.c_str(), (int)w, (int)h, 4, rgba.data(), (int)(w * 4));
    vkFreeCommandBuffers(device_, cmdPool_, 1, &cmd);
    vkFreeMemory(device_, stagingMem, nullptr);
    vkDestroyBuffer(device_, stagingBuf, nullptr);

    return ok != 0;
}

void MandalaRasterRenderer::setScene(RenderScene scene) {
    config_.scene = scene;
}

void MandalaRasterRenderer::setDebugFlags(bool gpuTimer, bool dumpGBuffer,
                                          int visualizer, const std::string& captureSeqDir,
                                          const std::string& encodePath) {
    debugGpuTimer_ = gpuTimer;
    debugDumpGBuffer_ = dumpGBuffer;
    debugVisualizer_ = visualizer;
    debugCaptureSeqDir_ = captureSeqDir;
    // encodePath is handled in main.cpp via the ffmpeg pipe
    (void)encodePath;
}

void MandalaRasterRenderer::dumpImage(VkImage image, VkFormat format, uint32_t w, uint32_t h,
                                      const std::string& path) {
    // Create staging buffer
    VkDeviceSize bufferSize = (VkDeviceSize)w * h * 4;
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = bufferSize;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer stagingBuf;
    VkDeviceMemory stagingMem;
    if (vkCreateBuffer(device_, &bi, nullptr, &stagingBuf) != VK_SUCCESS) return;

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device_, stagingBuf, &memReq);
    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = memReq.size;
    ai.memoryTypeIndex = [&]() {
        VkPhysicalDeviceMemoryProperties props;
        vkGetPhysicalDeviceMemoryProperties(phys_, &props);
        for (uint32_t i = 0; i < props.memoryTypeCount; i++) {
            if ((memReq.memoryTypeBits & (1 << i)) &&
                (props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
                (props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
                return i;
        }
        return uint32_t(-1);
    }();
    if (vkAllocateMemory(device_, &ai, nullptr, &stagingMem) != VK_SUCCESS) {
        vkDestroyBuffer(device_, stagingBuf, nullptr);
        return;
    }
    if (vkBindBufferMemory(device_, stagingBuf, stagingMem, 0) != VK_SUCCESS) {
        vkFreeMemory(device_, stagingMem, nullptr);
        vkDestroyBuffer(device_, stagingBuf, nullptr);
        return;
    }

    // One-time command buffer
    VkCommandBufferAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc.commandPool = cmdPool_;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 1;
    VkCommandBuffer cmd;
    if (vkAllocateCommandBuffers(device_, &alloc, &cmd) != VK_SUCCESS) {
        vkFreeMemory(device_, stagingMem, nullptr);
        vkDestroyBuffer(device_, stagingBuf, nullptr);
        return;
    }

    VkCommandBufferBeginInfo cbbi{};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &cbbi);

    // Transition image to TRANSFER_SRC
    VkImageMemoryBarrier barr{};
    barr.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barr.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    barr.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barr.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barr.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barr.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barr.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barr.image = image;
    barr.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barr.subresourceRange.baseMipLevel = 0;
    barr.subresourceRange.levelCount = 1;
    barr.subresourceRange.baseArrayLayer = 0;
    barr.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &barr);

    VkBufferImageCopy bic{};
    bic.bufferOffset = 0;
    bic.bufferRowLength = w;
    bic.bufferImageHeight = h;
    bic.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    bic.imageSubresource.mipLevel = 0;
    bic.imageSubresource.baseArrayLayer = 0;
    bic.imageSubresource.layerCount = 1;
    bic.imageOffset = {0, 0, 0};
    bic.imageExtent = {w, h, 1};
    vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           stagingBuf, 1, &bic);

    // Buffer barrier for host read
    VkBufferMemoryBarrier bmb{};
    bmb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bmb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bmb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bmb.buffer = stagingBuf;
    bmb.offset = 0;
    bmb.size = bufferSize;
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT,
                         0, 0, nullptr, 1, &bmb, 0, nullptr);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    if (vkQueueSubmit(graphicsQueue_, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS) {
        vkFreeCommandBuffers(device_, cmdPool_, 1, &cmd);
        vkFreeMemory(device_, stagingMem, nullptr);
        vkDestroyBuffer(device_, stagingBuf, nullptr);
        return;
    }
    vkQueueWaitIdle(graphicsQueue_);

    // Map and write PNG
    void* data;
    vkMapMemory(device_, stagingMem, 0, bufferSize, 0, &data);
    std::vector<unsigned char> rgba(w * h * 4);
    const uint32_t* src = static_cast<const uint32_t*>(data);
    for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
            uint32_t bgra = src[y * w + x];
            rgba[(y * w + x) * 4 + 0] = (bgra >> 0) & 0xFF;
            rgba[(y * w + x) * 4 + 1] = (bgra >> 8) & 0xFF;
            rgba[(y * w + x) * 4 + 2] = (bgra >> 16) & 0xFF;
            rgba[(y * w + x) * 4 + 3] = (bgra >> 24) & 0xFF;
        }
    }
    vkUnmapMemory(device_, stagingMem);

    stbi_write_png(path.c_str(), (int)w, (int)h, 4, rgba.data(), (int)(w * 4));

    vkFreeCommandBuffers(device_, cmdPool_, 1, &cmd);
    vkFreeMemory(device_, stagingMem, nullptr);
    vkDestroyBuffer(device_, stagingBuf, nullptr);
}

void MandalaRasterRenderer::dumpGBuffer(const std::string& dir) {
    if (config_.scene != RenderScene::RECON) return;

    VkExtent2D ext = swapchain_.extent();
    uint32_t w = ext.width / 2;   // LR resolution
    uint32_t h = ext.height / 2;

    // Access RT4D Reconstruction Stack internal images (these are private, so we need accessors)
    // For now dump what we can access via the swapchain extent
    // The actual G-buffer images are in dlss45 - we'd need public accessors
    // Dump the LR G-buffer attachments we can reach
    fprintf(stderr, "[dump-gbuffer] G-buffer dump not fully hooked (RT4D Reconstruction Stack images private)\n");
}
