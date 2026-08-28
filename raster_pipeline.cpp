#include "raster_pipeline.h"
#include "gltf_mesh.h"
#include <fstream>
#include <string>
#include <vector>
#include <cstring>

static std::vector<uint32_t> readSpirv(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    f.seekg(0, std::ios::end);
    size_t sz = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<uint32_t> d(sz / 4);
    f.read(reinterpret_cast<char*>(d.data()), sz);
    return d;
}

static VkShaderModule loadModule(VkDevice dev, const char* spv) {
    auto code = readSpirv(spv);
    if (code.empty()) return VK_NULL_HANDLE;
    VkShaderModuleCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = code.size() * sizeof(uint32_t);
    ci.pCode = code.data();
    VkShaderModule m;
    vkCreateShaderModule(dev, &ci, nullptr, &m);
    return m;
}

static uint32_t findMemType(VkPhysicalDevice phys, uint32_t bits, VkMemoryPropertyFlags flags) {
    VkPhysicalDeviceMemoryProperties props;
    vkGetPhysicalDeviceMemoryProperties(phys, &props);
    for (uint32_t i = 0; i < props.memoryTypeCount; i++) {
        if ((bits & (1 << i)) && (props.memoryTypes[i].propertyFlags & flags) == flags)
            return i;
    }
    return 0;
}

static void setAttr(VkVertexInputAttributeDescription& a,
                     uint32_t loc, uint32_t bind, VkFormat fmt, uint32_t off) {
    a = {loc, bind, fmt, off};
}

bool RasterPipeline::init(VkDevice device, VkPhysicalDevice phys,
                            VkRenderPass renderPass,
                            uint32_t width, uint32_t height) {
    device_ = device;
    phys_ = phys;
    width_ = width;
    renderPass_ = renderPass;
    height_ = height;

    VkDescriptorPoolSize poolSizes[] = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 128},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 32}
    };
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = 64;
    dpci.poolSizeCount = 2;
    dpci.pPoolSizes = poolSizes;
    vkCreateDescriptorPool(device, &dpci, nullptr, &pool_);

    VkDescriptorSetLayoutBinding camBind{};
    camBind.binding = 0;
    camBind.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    camBind.descriptorCount = 1;
    camBind.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo slci{};
    slci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    slci.bindingCount = 1;
    slci.pBindings = &camBind;
    vkCreateDescriptorSetLayout(device, &slci, nullptr, &cameraLayout_);

    VkDescriptorSetLayoutBinding bindings2[2];
    bindings2[0] = camBind;
    VkDescriptorSetLayoutBinding sceneBind{};
    sceneBind.binding = 1;
    sceneBind.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    sceneBind.descriptorCount = 1;
    sceneBind.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings2[1] = sceneBind;
    slci.bindingCount = 2;
    slci.pBindings = bindings2;
    vkCreateDescriptorSetLayout(device, &slci, nullptr, &sceneLayout_);

    VkDescriptorSetLayoutBinding texBind{};
    texBind.binding = 0;
    texBind.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    texBind.descriptorCount = 1;
    texBind.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    slci.bindingCount = 1;
    slci.pBindings = &texBind;
    vkCreateDescriptorSetLayout(device, &slci, nullptr, &textureLayout_);

    VkDescriptorSetLayoutBinding celBind{};
    celBind.binding = 0;
    celBind.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    celBind.descriptorCount = 1;
    celBind.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    slci.bindingCount = 1;
    slci.pBindings = &celBind;
    vkCreateDescriptorSetLayout(device, &slci, nullptr, &celLayout_);

    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &cameraLayout_;
    vkCreatePipelineLayout(device, &plci, nullptr, &cameraOnlyLayout_);

    // cameraSceneLayout_ = { sceneLayout_ } — ONE set. The mesh scene shaders
    // (taco/battle_crowd/dragon/sentinel) declare BOTH camera (binding 0) and
    // scene (binding 1) inside a single descriptor set #0, which matches the
    // 2-binding sceneLayout_. These scene draws MUST bind meshSets_ (allocated
    // from sceneLayout_), NOT the camera-only set, or set 0 is incompatible.
    plci.pSetLayouts = &sceneLayout_;
    vkCreatePipelineLayout(device, &plci, nullptr, &cameraSceneLayout_);

    VkDescriptorSetLayout celSetLayouts[] = {cameraLayout_, celLayout_};
    plci.setLayoutCount = 2;
    plci.pSetLayouts = celSetLayouts;
    vkCreatePipelineLayout(device, &plci, nullptr, &celPipelineLayout_);

    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(phys, &memProps);

    auto allocUBO = [&](VkDeviceSize sz) -> AllocatedBuffer {
        AllocatedBuffer b{};
        b.size = sz;
        VkBufferCreateInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.size = sz;
        bi.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        vkCreateBuffer(device, &bi, nullptr, &b.buffer);
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(device, b.buffer, &req);
        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = findMemType(phys, req.memoryTypeBits,
                                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkAllocateMemory(device, &ai, nullptr, &b.memory);
        vkBindBufferMemory(device, b.buffer, b.memory, 0);
        vkMapMemory(device, b.memory, 0, sz, 0, &b.mapped);
        return b;
    };

    for (int i = 0; i < 2; i++) {
        cameraUBOs_[i] = allocUBO(sizeof(CameraUBO));

        VkDescriptorSetAllocateInfo dai{};
        dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool = pool_;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts = &cameraLayout_;
        vkAllocateDescriptorSets(device, &dai, &cameraSets_[i]);

        VkDescriptorBufferInfo camBI{cameraUBOs_[i].buffer, 0, sizeof(CameraUBO)};
        VkWriteDescriptorSet wcam{};
        wcam.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wcam.dstSet = cameraSets_[i];
        wcam.dstBinding = 0;
        wcam.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        wcam.descriptorCount = 1;
        wcam.pBufferInfo = &camBI;
        vkUpdateDescriptorSets(device, 1, &wcam, 0, nullptr);

        celUBOs_[i] = allocUBO(sizeof(CelParams));

        dai.pSetLayouts = &celLayout_;
        vkAllocateDescriptorSets(device, &dai, &celSets_[i]);

        VkDescriptorBufferInfo celBI{celUBOs_[i].buffer, 0, sizeof(CelParams)};
        VkWriteDescriptorSet wcel{};
        wcel.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wcel.dstSet = celSets_[i];
        wcel.dstBinding = 0;
        wcel.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        wcel.descriptorCount = 1;
        wcel.pBufferInfo = &celBI;
        vkUpdateDescriptorSets(device, 1, &wcel, 0, nullptr);

        meshSceneUBOs_[i] = allocUBO(sizeof(SceneUBO));

        dai.pSetLayouts = &sceneLayout_;
        vkAllocateDescriptorSets(device, &dai, &meshSets_[i]);

        VkDescriptorBufferInfo meshCamBI{cameraUBOs_[i].buffer, 0, sizeof(CameraUBO)};
        VkDescriptorBufferInfo meshSceneBI{meshSceneUBOs_[i].buffer, 0, sizeof(SceneUBO)};
        VkWriteDescriptorSet winds[2]{};
        winds[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        winds[0].dstSet = meshSets_[i];
        winds[0].dstBinding = 0;
        winds[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        winds[0].descriptorCount = 1;
        winds[0].pBufferInfo = &meshCamBI;
        winds[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        winds[1].dstSet = meshSets_[i];
        winds[1].dstBinding = 1;
        winds[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        winds[1].descriptorCount = 1;
        winds[1].pBufferInfo = &meshSceneBI;
        vkUpdateDescriptorSets(device, 2, winds, 0, nullptr);
    }

    auto sp = std::string("spirv_rasterize/");

    VkVertexInputBindingDescription nodeBind{};
    nodeBind.binding = 0;
    nodeBind.stride = sizeof(LivingMapNode);
    nodeBind.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    VkVertexInputAttributeDescription nodeAttrs[7];
    setAttr(nodeAttrs[0], 0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0);
    setAttr(nodeAttrs[1], 1, 0, VK_FORMAT_R32G32B32_SFLOAT, 12);
    setAttr(nodeAttrs[2], 2, 0, VK_FORMAT_R32_SFLOAT, 24);
    setAttr(nodeAttrs[3], 3, 0, VK_FORMAT_R32_SFLOAT, 28);
    setAttr(nodeAttrs[4], 4, 0, VK_FORMAT_R32G32B32_SFLOAT, 32);
    setAttr(nodeAttrs[5], 5, 0, VK_FORMAT_R32G32B32_SFLOAT, 44);
    setAttr(nodeAttrs[6], 6, 0, VK_FORMAT_R32G32B32_SFLOAT, 56);

    pipelines_[0] = createPipeline(
        (sp + "living_map_point.vert.spv").c_str(),
        (sp + "living_map_point.frag.spv").c_str(),
        VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
        &nodeBind, 1, nodeAttrs, 7, celPipelineLayout_, true);

    VkVertexInputBindingDescription edgeBind{};
    edgeBind.binding = 0;
    edgeBind.stride = sizeof(LivingMapEdge);
    edgeBind.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    VkVertexInputAttributeDescription edgeAttrs[2];
    setAttr(edgeAttrs[0], 0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0);
    setAttr(edgeAttrs[1], 1, 0, VK_FORMAT_R32G32B32_SFLOAT, 16);

    pipelines_[1] = createPipeline(
        (sp + "living_map_edge.vert.spv").c_str(),
        (sp + "living_map_edge.frag.spv").c_str(),
        VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
        &edgeBind, 1, edgeAttrs, 2, cameraOnlyLayout_, true);

    VkVertexInputBindingDescription meshBind{};
    meshBind.binding = 0;
    meshBind.stride = sizeof(GLTFMeshVertex);
    meshBind.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    VkVertexInputAttributeDescription meshAttrs[3];
    setAttr(meshAttrs[0], 0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0);
    setAttr(meshAttrs[1], 1, 0, VK_FORMAT_R32G32B32_SFLOAT, 12);
    setAttr(meshAttrs[2], 2, 0, VK_FORMAT_R32G32_SFLOAT, 24);

    pipelines_[2] = createPipeline(
        (sp + "taco_scene.vert.spv").c_str(),
        (sp + "taco_scene.frag.spv").c_str(),
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        &meshBind, 1, meshAttrs, 3, cameraSceneLayout_, false);

    pipelines_[3] = createPipeline(
        (sp + "battle_crowd.vert.spv").c_str(),
        (sp + "battle_crowd.frag.spv").c_str(),
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        &meshBind, 1, meshAttrs, 3, cameraSceneLayout_, false);

    pipelines_[4] = createPipeline(
        (sp + "battle_atmosphere.vert.spv").c_str(),
        (sp + "battle_atmosphere.frag.spv").c_str(),
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        nullptr, 0, nullptr, 0, cameraSceneLayout_, false);

    pipelines_[5] = createPipeline(
        (sp + "dragon_hatch.vert.spv").c_str(),
        (sp + "dragon_hatch.frag.spv").c_str(),
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        &meshBind, 1, meshAttrs, 3, cameraSceneLayout_, false);

    return true;
}

VkPipeline RasterPipeline::createPipeline(
    const char* vertSpv, const char* fragSpv,
    VkPrimitiveTopology topo,
    const VkVertexInputBindingDescription* binds, uint32_t bindCount,
    const VkVertexInputAttributeDescription* attrs, uint32_t attrCount,
    VkPipelineLayout layout, bool alphaBlend) {

    auto vMod = loadModule(device_, vertSpv);
    auto fMod = loadModule(device_, fragSpv);
    if (!vMod || !fMod) return VK_NULL_HANDLE;

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  0, 0, VK_SHADER_STAGE_VERTEX_BIT, vMod, "main", nullptr};
    stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  0, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fMod, "main", nullptr};

    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.vertexBindingDescriptionCount = bindCount;
    vi.pVertexBindingDescriptions = binds;
    vi.vertexAttributeDescriptionCount = attrCount;
    vi.pVertexAttributeDescriptions = attrs;

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = topo;

    VkViewport vp{0, 0, (float)width_, (float)height_, 0.0f, 1.0f};
    VkRect2D sc{{0, 0}, {width_, height_}};
    VkPipelineViewportStateCreateInfo vps{};
    vps.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vps.viewportCount = 1;
    vps.pViewports = &vp;
    vps.scissorCount = 1;
    vps.pScissors = &sc;

    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.lineWidth = 1.0f;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{};
    ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable = VK_TRUE;
    ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState ba{};
    ba.colorWriteMask = 0xF;
    if (alphaBlend) {
        ba.blendEnable = VK_TRUE;
        ba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        ba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        ba.colorBlendOp = VK_BLEND_OP_ADD;
        ba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        ba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        ba.alphaBlendOp = VK_BLEND_OP_ADD;
    }

    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1;
    cb.pAttachments = &ba;

    VkDynamicState dynStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dy{};
    dy.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dy.dynamicStateCount = 2;
    dy.pDynamicStates = dynStates;

    VkGraphicsPipelineCreateInfo gpci{};
    gpci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gpci.stageCount = 2;
    gpci.pStages = stages;
    gpci.pVertexInputState = &vi;
    gpci.pInputAssemblyState = &ia;
    gpci.pViewportState = &vps;
    gpci.pRasterizationState = &rs;
    gpci.pMultisampleState = &ms;
    gpci.pDepthStencilState = &ds;
    gpci.pColorBlendState = &cb;
    gpci.pDynamicState = &dy;
    gpci.layout = layout;
    gpci.renderPass = renderPass_;

    VkPipeline pipe;
    vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &gpci, nullptr, &pipe);

    vkDestroyShaderModule(device_, vMod, nullptr);
    vkDestroyShaderModule(device_, fMod, nullptr);
    return pipe;
}

AllocatedBuffer RasterPipeline::allocBuffer(VkDeviceSize size,
                                              VkBufferUsageFlags usage,
                                              VkMemoryPropertyFlags props) {
    AllocatedBuffer b{};
    b.size = size;
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(device_, &bi, nullptr, &b.buffer);

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device_, b.buffer, &req);

    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(phys_, &memProps);

    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((req.memoryTypeBits & (1 << i)) &&
            (memProps.memoryTypes[i].propertyFlags & props) == props) {
            ai.memoryTypeIndex = i;
            break;
        }
    }
    vkAllocateMemory(device_, &ai, nullptr, &b.memory);
    vkBindBufferMemory(device_, b.buffer, b.memory, 0);

    if (props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
        vkMapMemory(device_, b.memory, 0, size, 0, &b.mapped);

    return b;
}

void RasterPipeline::freeBuffer(VkDevice device, AllocatedBuffer& buf) {
    if (buf.mapped) { vkUnmapMemory(device, buf.memory); buf.mapped = nullptr; }
    if (buf.buffer) vkDestroyBuffer(device, buf.buffer, nullptr);
    if (buf.memory) vkFreeMemory(device, buf.memory, nullptr);
    buf = {};
}

void RasterPipeline::uploadToBuffer(AllocatedBuffer& buf, const void* data,
                                    VkDeviceSize size) {
    if (!buf.mapped) return;
    memcpy(buf.mapped, data, size);
}

void RasterPipeline::beginRenderPass(VkCommandBuffer cmd, VkFramebuffer fb,
                                       uint32_t w, uint32_t h) {
    VkRenderPassBeginInfo rpBI{};
    rpBI.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpBI.renderPass = renderPass_;
    rpBI.framebuffer = fb;
    rpBI.renderArea = {{0, 0}, {w, h}};
    VkClearValue clears[2];
    // Diagnostic-friendly neutral background; geometry must remain visible
    // against it in GPU captures instead of being indistinguishable from an
    // untouched swapchain image.
    clears[0].color = {{0.035f, 0.055f, 0.09f, 1.0f}};
    clears[1].depthStencil = {1.0f, 0};
    rpBI.clearValueCount = 2;
    rpBI.pClearValues = clears;
    vkCmdBeginRenderPass(cmd, &rpBI, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport vp{0, 0, (float)w, (float)h, 0.0f, 1.0f};
    VkRect2D sc{{0, 0}, {w, h}};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &sc);
}

void RasterPipeline::endRenderPass(VkCommandBuffer cmd) {
    vkCmdEndRenderPass(cmd);
}

void RasterPipeline::bindCamera(VkCommandBuffer cmd) {
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            cameraOnlyLayout_, 0, 1, &cameraSets_[frameIndex_], 0, nullptr);
}

void RasterPipeline::drawLivingMapPoints(VkCommandBuffer cmd, uint32_t count) {
    VkDescriptorSet sets[] = {cameraSets_[frameIndex_], celSets_[frameIndex_]};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            celPipelineLayout_, 0, 2, sets, 0, nullptr);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines_[0]);
    vkCmdDraw(cmd, count, 1, 0, 0);
}

void RasterPipeline::drawLivingMapEdges(VkCommandBuffer cmd, uint32_t count) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines_[1]);
    vkCmdDraw(cmd, count * 2, 1, 0, 0);
}

void RasterPipeline::drawMeshScene(VkCommandBuffer cmd, AllocatedBuffer& sceneUBO,
                                    const SceneUBO& scene,
                                    VkBuffer vb, VkBuffer ib, uint32_t indexCount) {
    uploadToBuffer(sceneUBO, &scene, sizeof(SceneUBO));

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            cameraSceneLayout_, 0, 1, &meshSets_[frameIndex_], 0, nullptr);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines_[2]);
    VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &off);
    vkCmdBindIndexBuffer(cmd, ib, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmd, indexCount, 1, 0, 0, 0);
}

void RasterPipeline::drawTacoScene(VkCommandBuffer cmd, VkBuffer vb,
                                     VkBuffer ib, uint32_t indexCount) {
    // Bind the camera+scene descriptor set (meshSets_), matching the taco_scene
    // pipeline's set 0 (camera@0 + scene@1 live together in sceneLayout_).
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            cameraSceneLayout_, 0, 1, &meshSets_[frameIndex_], 0, nullptr);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines_[2]);
    VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &off);
    vkCmdBindIndexBuffer(cmd, ib, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmd, indexCount, 1, 0, 0, 0);
}

void RasterPipeline::drawBattleCrowd(VkCommandBuffer cmd, VkBuffer vb,
                                       VkBuffer ib, uint32_t indexCount,
                                       VkBuffer instBuf, uint32_t instCount) {
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            cameraSceneLayout_, 0, 1, &meshSets_[frameIndex_], 0, nullptr);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines_[3]);
    VkBuffer bufs[] = {vb, instBuf};
    VkDeviceSize offs[] = {0, 0};
    vkCmdBindVertexBuffers(cmd, 0, 2, bufs, offs);
    vkCmdBindIndexBuffer(cmd, ib, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmd, indexCount, instCount, 0, 0, 0);
}

void RasterPipeline::drawBattleAtmosphere(VkCommandBuffer cmd) {
    // The atmosphere shader only uses set 0 binding 0 (camera); bind the mesh
    // scene set so set 0 (camera) is compatible with cameraSceneLayout_.
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            cameraSceneLayout_, 0, 1, &meshSets_[frameIndex_], 0, nullptr);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines_[4]);
    vkCmdDraw(cmd, 3, 1, 0, 0);
}

void RasterPipeline::drawDragonHatch(VkCommandBuffer cmd, VkBuffer vb,
                                       VkBuffer ib, uint32_t indexCount) {
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            cameraSceneLayout_, 0, 1, &meshSets_[frameIndex_], 0, nullptr);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines_[5]);
    VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &off);
    vkCmdBindIndexBuffer(cmd, ib, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmd, indexCount, 1, 0, 0, 0);
}

void RasterPipeline::shutdown(VkDevice device) {
    for (auto& p : pipelines_) { if (p) vkDestroyPipeline(device, p, nullptr); p = VK_NULL_HANDLE; }
    freeBuffer(device, cameraUBOs_[0]);
    freeBuffer(device, cameraUBOs_[1]);
    freeBuffer(device, celUBOs_[0]);
    freeBuffer(device, celUBOs_[1]);
    freeBuffer(device, meshSceneUBOs_[0]);
    freeBuffer(device, meshSceneUBOs_[1]);
    if (cameraOnlyLayout_) vkDestroyPipelineLayout(device, cameraOnlyLayout_, nullptr);
    if (cameraSceneLayout_) vkDestroyPipelineLayout(device, cameraSceneLayout_, nullptr);
    if (celPipelineLayout_) vkDestroyPipelineLayout(device, celPipelineLayout_, nullptr);
    if (cameraLayout_) vkDestroyDescriptorSetLayout(device, cameraLayout_, nullptr);
    if (sceneLayout_) vkDestroyDescriptorSetLayout(device, sceneLayout_, nullptr);
    if (textureLayout_) vkDestroyDescriptorSetLayout(device, textureLayout_, nullptr);
    if (celLayout_) vkDestroyDescriptorSetLayout(device, celLayout_, nullptr);
    if (pool_) vkDestroyDescriptorPool(device, pool_, nullptr);
}
