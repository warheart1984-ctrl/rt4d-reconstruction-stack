#include "dlss45_recon.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

// ---- small local helpers -------------------------------------------------

static uint32_t findMemType(VkPhysicalDevice phys, uint32_t bits,
                            VkMemoryPropertyFlags flags) {
    VkPhysicalDeviceMemoryProperties props;
    vkGetPhysicalDeviceMemoryProperties(phys, &props);
    for (uint32_t i = 0; i < props.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (props.memoryTypes[i].propertyFlags & flags) == flags)
            return i;
    return ~0u;
}

static std::vector<uint32_t> readSpv(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    std::streamsize size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<uint32_t> code((size_t)(size / 4));
    if (!code.empty()) f.read((char*)code.data(), size);
    return code;
}

static VkShaderModule makeModule(VkDevice device, const char* path) {
    auto code = readSpv(path);
    if (code.empty()) return VK_NULL_HANDLE;
    VkShaderModuleCreateInfo sm{};
    sm.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    sm.codeSize = code.size() * sizeof(uint32_t);
    sm.pCode = code.data();
    VkShaderModule mod = VK_NULL_HANDLE;
    vkCreateShaderModule(device, &sm, nullptr, &mod);
    return mod;
}

// ---------------------------------------------------------------------------

bool DLSS45Recon::createImage(ImageObj& img, uint32_t w, uint32_t h, VkFormat fmt,
                              VkImageUsageFlags usage, VkImageAspectFlags aspect) {
    VkImageCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = fmt;
    ci.extent = {w, h, 1};
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = usage;
    if (vkCreateImage(device_, &ci, nullptr, &img.image) != VK_SUCCESS) return false;

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(device_, img.image, &req);
    uint32_t mt = findMemType(phys_, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mt == ~0u) return false;
    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = mt;
    if (vkAllocateMemory(device_, &ai, nullptr, &img.memory) != VK_SUCCESS) return false;
    if (vkBindImageMemory(device_, img.image, img.memory, 0) != VK_SUCCESS) return false;

    VkImageViewCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image = img.image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = fmt;
    vi.subresourceRange.aspectMask = aspect;
    vi.subresourceRange.levelCount = 1;
    vi.subresourceRange.layerCount = 1;
    if (vkCreateImageView(device_, &vi, nullptr, &img.view) != VK_SUCCESS) return false;
    return true;
}

void DLSS45Recon::matMul(float* out, const float* a, const float* b) {
    // Column-major 4x4: out = a * b
    for (int c = 0; c < 4; c++)
        for (int r = 0; r < 4; r++) {
            float v = 0.f;
            for (int k = 0; k < 4; k++) v += a[k * 4 + r] * b[c * 4 + k];
            out[c * 4 + r] = v;
        }
}

bool DLSS45Recon::createRenderPass() {
    std::array<VkAttachmentDescription, 6> atts{};
    VkFormat colorFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
    for (int i = 0; i < 5; i++) {
        atts[i].format = colorFormat;
        atts[i].samples = VK_SAMPLE_COUNT_1_BIT;
        atts[i].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        atts[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        atts[i].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        atts[i].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        atts[i].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        atts[i].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
    atts[5].format = VK_FORMAT_D32_SFLOAT;
    atts[5].samples = VK_SAMPLE_COUNT_1_BIT;
    atts[5].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    atts[5].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    atts[5].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    atts[5].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    atts[5].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    atts[5].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    std::array<VkAttachmentReference, 5> colorRefs;
    for (int i = 0; i < 5; i++)
        colorRefs[i] = {(uint32_t)i, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depthRef{5, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    VkSubpassDescription sub{};
    sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 5;
    sub.pColorAttachments = colorRefs.data();
    sub.pDepthStencilAttachment = &depthRef;

    VkSubpassDependency dep{};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass = 0;
    dep.srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo rp{};
    rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rp.attachmentCount = (uint32_t)atts.size();
    rp.pAttachments = atts.data();
    rp.subpassCount = 1;
    rp.pSubpasses = &sub;
    rp.dependencyCount = 1;
    rp.pDependencies = &dep;
    if (vkCreateRenderPass(device_, &rp, nullptr, &gbufferPass_) != VK_SUCCESS) return false;

    std::array<VkImageView, 6> views = {
        colorLDR_.view, normals_.view, motion_.view, material_.view, noise_.view, depth_.view
    };
    VkFramebufferCreateInfo fb{};
    fb.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fb.renderPass = gbufferPass_;
    fb.attachmentCount = (uint32_t)views.size();
    fb.pAttachments = views.data();
    fb.width = lrW_;
    fb.height = lrH_;
    fb.layers = 1;
    if (vkCreateFramebuffer(device_, &fb, nullptr, &gbufferFramebuffer_) != VK_SUCCESS) return false;

    // NOTE: The composite tone-map pass must reuse the RENDERER's render pass +
    // swapchain framebuffer (set as compositePass_ in init). We do NOT create our own
    // composite pass here, otherwise its single-attachment layout would be
    // incompatible with the swapchain framebuffer's color+depth attachments.
    return true;
}

bool DLSS45Recon::createMeshResources() {
    // Camera + scene UBOs (storage buffers with combined view+proj).
    VkBufferCreateInfo camBC{};
    camBC.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    camBC.size = sizeof(DLSS45CameraUBO);
    camBC.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    camBC.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(device_, &camBC, nullptr, &camUBO_);
    VkBufferCreateInfo scBC{};
    scBC.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    scBC.size = sizeof(DLSS45SceneUBO);
    scBC.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    vkCreateBuffer(device_, &scBC, nullptr, &sceneUBO_);
    auto allocHostBuffer = [&](VkBuffer buf, VkDeviceMemory& mem) {
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(device_, buf, &req);
        uint32_t mt = findMemType(phys_, req.memoryTypeBits,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = mt;
        vkAllocateMemory(device_, &ai, nullptr, &mem);
        vkBindBufferMemory(device_, buf, mem, 0);
    };
    allocHostBuffer(camUBO_, camUBOMem_);
    allocHostBuffer(sceneUBO_, sceneUBOMem_);

    // Mesh descriptor: set0 = camera(U0) + scene(U1). Single combined layout.
    VkDescriptorSetLayoutBinding bnds[2] = {};
    bnds[0].binding = 0; bnds[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bnds[0].descriptorCount = 1; bnds[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    bnds[1].binding = 1; bnds[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bnds[1].descriptorCount = 1; bnds[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo dl{};
    dl.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dl.bindingCount = 2;
    dl.pBindings = bnds;
    if (vkCreateDescriptorSetLayout(device_, &dl, nullptr, &meshSetLayout_) != VK_SUCCESS) return false;

    VkDescriptorSetLayout layouts[1] = {meshSetLayout_};
    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = pool_;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = layouts;
    vkAllocateDescriptorSets(device_, &ai, &meshSet_);

    VkDescriptorBufferInfo camInfo{camUBO_, 0, sizeof(DLSS45CameraUBO)};
    VkDescriptorBufferInfo scInfo{sceneUBO_, 0, sizeof(DLSS45SceneUBO)};
    VkWriteDescriptorSet w[2] = {};
    w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[0].dstSet = meshSet_; w[0].dstBinding = 0; w[0].descriptorCount = 1;
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; w[0].pBufferInfo = &camInfo;
    w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[1].dstSet = meshSet_; w[1].dstBinding = 1; w[1].descriptorCount = 1;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; w[1].pBufferInfo = &scInfo;
    vkUpdateDescriptorSets(device_, 2, w, 0, nullptr);

    // Mesh pipeline layout (single descriptor set).
    VkPipelineLayoutCreateInfo pl{};
    pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl.setLayoutCount = 1;
    pl.pSetLayouts = layouts;
    if (vkCreatePipelineLayout(device_, &pl, nullptr, &meshLayout_) != VK_SUCCESS) return false;

    // Pipeline.
    VkShaderModule vs = makeModule(device_, "spirv_rasterize/gbuffer.vert.spv");
    VkShaderModule fs = makeModule(device_, "spirv_rasterize/gbuffer.frag.spv");
    if (!vs || !fs) { fprintf(stderr, "[RT4D Reconstruction Stack] gbuffer shaders missing\n"); return false; }

    VkPipelineShaderStageCreateInfo st[2] = {};
    st[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    st[0].stage = VK_SHADER_STAGE_VERTEX_BIT; st[0].module = vs; st[0].pName = "main";
    st[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; st[1].module = fs; st[1].pName = "main";

    VkVertexInputBindingDescription vi{};
    vi.binding = 0; vi.stride = sizeof(GLTFMeshVertex); vi.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    std::array<VkVertexInputAttributeDescription, 4> attrs{};
    attrs[0].location = 0; attrs[0].binding = 0; attrs[0].format = VK_FORMAT_R32G32B32_SFLOAT; attrs[0].offset = 0;
    attrs[1].location = 1; attrs[1].binding = 0; attrs[1].format = VK_FORMAT_R32G32B32_SFLOAT; attrs[1].offset = 12;
    attrs[2].location = 2; attrs[2].binding = 0; attrs[2].format = VK_FORMAT_R32G32_SFLOAT; attrs[2].offset = 24;
    attrs[3].location = 3; attrs[3].binding = 0; attrs[3].format = VK_FORMAT_R32_SFLOAT; attrs[3].offset = 32;

    VkPipelineVertexInputStateCreateInfo pvi{};
    pvi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    pvi.vertexBindingDescriptionCount = 1; pvi.pVertexBindingDescriptions = &vi;
    pvi.vertexAttributeDescriptionCount = (uint32_t)attrs.size(); pvi.pVertexAttributeDescriptions = attrs.data();

    VkPipelineInputAssemblyStateCreateInfo pia{};
    pia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    pia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo pvp{};
    pvp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    pvp.viewportCount = 1; pvp.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo prs{};
    prs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    prs.polygonMode = VK_POLYGON_MODE_FILL; prs.cullMode = VK_CULL_MODE_BACK_BIT;
    prs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; prs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo pms{};
    pms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    pms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo pds{};
    pds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    pds.depthTestEnable = VK_TRUE; pds.depthWriteEnable = VK_TRUE;
    pds.depthCompareOp = VK_COMPARE_OP_LESS;

    std::array<VkPipelineColorBlendAttachmentState, 5> cba{};
    for (size_t i = 0; i < cba.size(); i++) {
        cba[i].colorWriteMask = 0xf;
        cba[i].blendEnable = VK_FALSE;
        cba[i].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        cba[i].dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
        cba[i].colorBlendOp = VK_BLEND_OP_ADD;
        cba[i].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        cba[i].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        cba[i].alphaBlendOp = VK_BLEND_OP_ADD;
    }
    VkPipelineColorBlendStateCreateInfo pcb{};
    pcb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    pcb.attachmentCount = (uint32_t)cba.size(); pcb.pAttachments = cba.data();

    VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo pdy{};
    pdy.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    pdy.dynamicStateCount = 2; pdy.pDynamicStates = dyn;

    VkGraphicsPipelineCreateInfo gpi{};
    gpi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gpi.stageCount = 2; gpi.pStages = st;
    gpi.pVertexInputState = &pvi; gpi.pInputAssemblyState = &pia;
    gpi.pViewportState = &pvp; gpi.pRasterizationState = &prs;
    gpi.pMultisampleState = &pms; gpi.pDepthStencilState = &pds;
    gpi.pColorBlendState = &pcb; gpi.pDynamicState = &pdy;
    gpi.layout = meshLayout_;
    gpi.renderPass = gbufferPass_;
    gpi.subpass = 0;
    if (vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &gpi, nullptr, &meshPipeline_) != VK_SUCCESS) {
        vkDestroyShaderModule(device_, vs, nullptr);
        vkDestroyShaderModule(device_, fs, nullptr);
        return false;
    }
    vkDestroyShaderModule(device_, vs, nullptr);
    vkDestroyShaderModule(device_, fs, nullptr);
    return true;
}

bool DLSS45Recon::createDescriptors() {
    // Pool big enough for all sets + descriptors.
    std::array<VkDescriptorPoolSize, 4> sizes{};
    sizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; sizes[0].descriptorCount = 24;
    sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;           sizes[1].descriptorCount = 8;
    sizes[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;          sizes[2].descriptorCount = 4;
    sizes[3].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;          sizes[3].descriptorCount = 4;
    VkDescriptorPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.maxSets = 8;
    pci.poolSizeCount = (uint32_t)sizes.size();
    pci.pPoolSizes = sizes.data();
    if (vkCreateDescriptorPool(device_, &pci, nullptr, &pool_) != VK_SUCCESS) return false;

    // ---- sampler for all combined-image-sampler reads ----
    VkSamplerCreateInfo sci{};
    sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter = VK_FILTER_NEAREST;
    sci.minFilter = VK_FILTER_NEAREST;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    vkCreateSampler(device_, &sci, nullptr, &sampler_);

    auto makeLayout = [&](const std::vector<VkDescriptorSetLayoutBinding>& binds,
                          VkDescriptorSetLayout& out) {
        VkDescriptorSetLayoutCreateInfo li{};
        li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        li.bindingCount = (uint32_t)binds.size();
        li.pBindings = binds.data();
        return vkCreateDescriptorSetLayout(device_, &li, nullptr, &out) == VK_SUCCESS;
    };

    // Reprojection layout: 0..4 CIS, 5..6 storage image.
    std::vector<VkDescriptorSetLayoutBinding> rb(7);
    for (int i = 0; i < 5; i++) { rb[i].binding = i; rb[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; rb[i].descriptorCount = 1; rb[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
    for (int i = 5; i < 7; i++) { rb[i].binding = i; rb[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; rb[i].descriptorCount = 1; rb[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
    if (!makeLayout(rb, reprojSetLayout_)) return false;

    // Denoiser layout: 0..8 CIS, 9..10 storage image.
    std::vector<VkDescriptorSetLayoutBinding> db(11);
    for (int i = 0; i < 9; i++) { db[i].binding = i; db[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; db[i].descriptorCount = 1; db[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
    for (int i = 9; i < 11; i++) { db[i].binding = i; db[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; db[i].descriptorCount = 1; db[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
    if (!makeLayout(db, denoiserSetLayout_)) return false;

    // SR layout: 0..4 CIS, 5..7 storage image.
    std::vector<VkDescriptorSetLayoutBinding> sr(8);
    for (int i = 0; i < 5; i++) { sr[i].binding = i; sr[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; sr[i].descriptorCount = 1; sr[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
    for (int i = 5; i < 8; i++) { sr[i].binding = i; sr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; sr[i].descriptorCount = 1; sr[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
    if (!makeLayout(sr, srSetLayout_)) return false;

    // Tone map layout: 0..1 CIS (colorSR, depthHR).
    std::vector<VkDescriptorSetLayoutBinding> tb(2);
    for (int i = 0; i < 2; i++) { tb[i].binding = i; tb[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; tb[i].descriptorCount = 1; tb[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT; }
    if (!makeLayout(tb, toneMapSetLayout_)) return false;

    // Allocate sets.
    VkDescriptorSetLayout setLays[4] = {reprojSetLayout_, denoiserSetLayout_, srSetLayout_, toneMapSetLayout_};
    VkDescriptorSet sets[4] = {};
    VkDescriptorSetAllocateInfo di{};
    di.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    di.descriptorPool = pool_;
    di.descriptorSetCount = 4;
    di.pSetLayouts = setLays;
    if (vkAllocateDescriptorSets(device_, &di, sets) != VK_SUCCESS) return false;
    reprojSet_ = sets[0]; denoiserSet_ = sets[1]; srSet_ = sets[2]; toneMapSet_ = sets[3];

    auto cis = [&](VkImageView view) {
        VkDescriptorImageInfo ii{};
        ii.sampler = sampler_; ii.imageView = view; ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        return ii;
    };
    auto sto = [&](VkImageView view) {
        VkDescriptorImageInfo ii{};
        ii.imageView = view; ii.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        return ii;
    };

    // ---- Reprojection set ----
    {
        VkDescriptorImageInfo info[7];
        info[0] = cis(colorLDR_.view);
        info[1] = cis(depth_.view);
        info[2] = cis(motion_.view);
        info[3] = cis(colorHistory_.view);
        info[4] = cis(depthHistory_.view);
        info[5] = sto(reprojColor_.view);
        info[6] = sto(reprojConf_.view);
        std::array<VkWriteDescriptorSet, 7> w{};
        for (int i = 0; i < 7; i++) {
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet = reprojSet_; w[i].dstBinding = i; w[i].descriptorCount = 1;
            w[i].descriptorType = (i < 5) ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            w[i].pImageInfo = &info[i];
        }
        vkUpdateDescriptorSets(device_, (uint32_t)w.size(), w.data(), 0, nullptr);
    }

    // ---- Denoiser set ----
    {
        VkDescriptorImageInfo info[11];
        info[0] = cis(colorLDR_.view);   // C_t
        info[1] = cis(depth_.view);      // D_t
        info[2] = cis(normals_.view);    // N_t
        info[3] = cis(motion_.view);     // M_t
        info[4] = cis(noise_.view);      // V_t
        info[5] = cis(reprojColor_.view); // history/reprojected color
        info[6] = cis(depthHistory_.view);
        info[7] = cis(normalHistory_.view);
        info[8] = cis(material_.view);   // material id
        info[9] = sto(denoised_.view);
        info[10] = sto(denoiseConf_.view);
        std::array<VkWriteDescriptorSet, 11> w{};
        for (int i = 0; i < 11; i++) {
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet = denoiserSet_; w[i].dstBinding = i; w[i].descriptorCount = 1;
            w[i].descriptorType = (i < 9) ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            w[i].pImageInfo = &info[i];
        }
        vkUpdateDescriptorSets(device_, (uint32_t)w.size(), w.data(), 0, nullptr);
    }

    // ---- SR set ----
    {
        VkDescriptorImageInfo info[8];
        info[0] = cis(denoised_.view);
        info[1] = cis(depth_.view);
        info[2] = cis(normals_.view);
        info[3] = cis(denoiseConf_.view);
        info[4] = cis(motion_.view);
        info[5] = sto(colorSR_.view);
        info[6] = sto(depthHR_.view);
        info[7] = sto(motionHR_.view);
        std::array<VkWriteDescriptorSet, 8> w{};
        for (int i = 0; i < 8; i++) {
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet = srSet_; w[i].dstBinding = i; w[i].descriptorCount = 1;
            w[i].descriptorType = (i < 5) ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            w[i].pImageInfo = &info[i];
        }
        vkUpdateDescriptorSets(device_, (uint32_t)w.size(), w.data(), 0, nullptr);
    }

    // ---- Tone map set ----
    {
        VkDescriptorImageInfo info[2];
        info[0] = cis(colorSR_.view);
        info[1] = cis(depthHR_.view);
        std::array<VkWriteDescriptorSet, 2> w{};
        for (int i = 0; i < 2; i++) {
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet = toneMapSet_; w[i].dstBinding = i; w[i].descriptorCount = 1;
            w[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[i].pImageInfo = &info[i];
        }
        vkUpdateDescriptorSets(device_, (uint32_t)w.size(), w.data(), 0, nullptr);
    }

    return true;
}

// ---- init ----------------------------------------------------------------

static void transitionLayout(VkCommandBuffer cmd, VkImage image, VkImageAspectFlags aspect,
                             VkImageLayout oldLayout, VkImageLayout newLayout,
                             VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage,
                             VkAccessFlags srcAccess, VkAccessFlags dstAccess) {
    VkImageMemoryBarrier bar{};
    bar.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    bar.oldLayout = oldLayout;
    bar.newLayout = newLayout;
    bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bar.image = image;
    bar.subresourceRange.aspectMask = aspect;
    bar.subresourceRange.levelCount = 1;
    bar.subresourceRange.layerCount = 1;
    bar.srcAccessMask = srcAccess;
    bar.dstAccessMask = dstAccess;
    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &bar);
}

bool DLSS45Recon::init(VkDevice device, VkPhysicalDevice phys,
                       VkRenderPass rendererRenderPass,
                       uint32_t displayW, uint32_t displayH, uint32_t lrScale) {
    device_ = device;
    phys_ = phys;
    scale_ = lrScale;
    hrW_ = displayW; hrH_ = displayH;
    lrW_ = displayW / lrScale; lrH_ = displayH / lrScale;
    compositePass_ = rendererRenderPass;

    // G-buffer images (LR). All but depth are color/sampled + storage targets.
    VkImageUsageFlags colorUsage =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    if (!createImage(colorLDR_, lrW_, lrH_, VK_FORMAT_R16G16B16A16_SFLOAT, colorUsage, VK_IMAGE_ASPECT_COLOR_BIT)) return false;
    if (!createImage(normals_, lrW_, lrH_, VK_FORMAT_R16G16B16A16_SFLOAT, colorUsage, VK_IMAGE_ASPECT_COLOR_BIT)) return false;
    if (!createImage(motion_, lrW_, lrH_, VK_FORMAT_R16G16B16A16_SFLOAT, colorUsage, VK_IMAGE_ASPECT_COLOR_BIT)) return false;
    if (!createImage(material_, lrW_, lrH_, VK_FORMAT_R16G16B16A16_SFLOAT, colorUsage, VK_IMAGE_ASPECT_COLOR_BIT)) return false;
    if (!createImage(noise_, lrW_, lrH_, VK_FORMAT_R16G16B16A16_SFLOAT, colorUsage, VK_IMAGE_ASPECT_COLOR_BIT)) return false;
    if (!createImage(depth_, lrW_, lrH_, VK_FORMAT_D32_SFLOAT,
                     VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                     VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_DEPTH_BIT)) return false;

    // History (LR).
    if (!createImage(colorHistory_, lrW_, lrH_, VK_FORMAT_R16G16B16A16_SFLOAT,
                     VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT)) return false;
    if (!createImage(depthHistory_, lrW_, lrH_, VK_FORMAT_D32_SFLOAT,
                     VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_DEPTH_BIT)) return false;
    if (!createImage(normalHistory_, lrW_, lrH_, VK_FORMAT_R16G16B16A16_SFLOAT,
                     VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT)) return false;

    // Reconstruction (LR) storage targets.
    VkImageUsageFlags storUsage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    if (!createImage(reprojColor_, lrW_, lrH_, VK_FORMAT_R16G16B16A16_SFLOAT, storUsage, VK_IMAGE_ASPECT_COLOR_BIT)) return false;
    if (!createImage(reprojConf_, lrW_, lrH_, VK_FORMAT_R16_SFLOAT, storUsage, VK_IMAGE_ASPECT_COLOR_BIT)) return false;
    if (!createImage(denoised_, lrW_, lrH_, VK_FORMAT_R16G16B16A16_SFLOAT, storUsage, VK_IMAGE_ASPECT_COLOR_BIT)) return false;
    if (!createImage(denoiseConf_, lrW_, lrH_, VK_FORMAT_R16_SFLOAT, storUsage, VK_IMAGE_ASPECT_COLOR_BIT)) return false;

    // HR outputs (storage + sampled).
    if (!createImage(colorSR_, hrW_, hrH_, VK_FORMAT_R16G16B16A16_SFLOAT, storUsage, VK_IMAGE_ASPECT_COLOR_BIT)) return false;
    if (!createImage(depthHR_, hrW_, hrH_, VK_FORMAT_R32_SFLOAT, storUsage, VK_IMAGE_ASPECT_COLOR_BIT)) return false;
    if (!createImage(motionHR_, hrW_, hrH_, VK_FORMAT_R16G16B16A16_SFLOAT, storUsage, VK_IMAGE_ASPECT_COLOR_BIT)) return false;

    if (!createRenderPass()) return false;
    if (!createDescriptors()) return false;

    // Mesh pipeline (needs pool from createDescriptors for its descriptor set).
    if (!createMeshResources()) return false;

    // Compute pipelines.
    if (!reproj_.init(device, reprojSetLayout_)) return false;
    if (!denoiser_.init(device, denoiserSetLayout_)) return false;
    if (!sr_.init(device, srSetLayout_)) return false;
    if (!toneMap_.init(device, toneMapSetLayout_, compositePass_)) return false;

    memset(prevViewProj_, 0, sizeof(prevViewProj_));
    prevViewProj_[0] = prevViewProj_[5] = prevViewProj_[10] = prevViewProj_[15] = 1.0f;
    return true;
}

void DLSS45Recon::initializeHistory(VkCommandBuffer cmd) {
    transitionLayout(cmd, colorHistory_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                     VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                     0, VK_ACCESS_TRANSFER_WRITE_BIT);
    transitionLayout(cmd, depthHistory_.image, VK_IMAGE_ASPECT_DEPTH_BIT,
                     VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                     0, VK_ACCESS_TRANSFER_WRITE_BIT);
    transitionLayout(cmd, normalHistory_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                     VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                     0, VK_ACCESS_TRANSFER_WRITE_BIT);

    VkImageSubresourceRange colorRange{};
    colorRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    colorRange.levelCount = 1;
    colorRange.layerCount = 1;
    const VkClearColorValue clearColor = {{0.0f, 0.0f, 0.0f, 0.0f}};
    vkCmdClearColorImage(cmd, colorHistory_.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         &clearColor, 1, &colorRange);
    const VkClearColorValue clearNormal = {{0.5f, 0.5f, 1.0f, 1.0f}};
    vkCmdClearColorImage(cmd, normalHistory_.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         &clearNormal, 1, &colorRange);

    VkImageSubresourceRange depthRange{};
    depthRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    depthRange.levelCount = 1;
    depthRange.layerCount = 1;
    const VkClearDepthStencilValue clearDepth{1.0f, 0};
    vkCmdClearDepthStencilImage(cmd, depthHistory_.image,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                &clearDepth, 1, &depthRange);

    transitionLayout(cmd, colorHistory_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                     VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    transitionLayout(cmd, depthHistory_.image, VK_IMAGE_ASPECT_DEPTH_BIT,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                     VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    transitionLayout(cmd, normalHistory_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                     VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
}

void DLSS45Recon::updateHistory(VkCommandBuffer cmd) {
    transitionLayout(cmd, colorLDR_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                     VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    transitionLayout(cmd, depth_.image, VK_IMAGE_ASPECT_DEPTH_BIT,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                     VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    transitionLayout(cmd, normals_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                     VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    transitionLayout(cmd, colorHistory_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                     VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    transitionLayout(cmd, depthHistory_.image, VK_IMAGE_ASPECT_DEPTH_BIT,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                     VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    transitionLayout(cmd, normalHistory_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                     VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);

    VkImageCopy colorCopy{};
    colorCopy.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    colorCopy.srcSubresource.layerCount = 1;
    colorCopy.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    colorCopy.dstSubresource.layerCount = 1;
    colorCopy.extent = {lrW_, lrH_, 1};
    vkCmdCopyImage(cmd, colorLDR_.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   colorHistory_.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   1, &colorCopy);

    VkImageCopy depthCopy{};
    depthCopy.srcSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    depthCopy.srcSubresource.layerCount = 1;
    depthCopy.dstSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    depthCopy.dstSubresource.layerCount = 1;
    depthCopy.extent = {lrW_, lrH_, 1};
    vkCmdCopyImage(cmd, depth_.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   depthHistory_.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   1, &depthCopy);
    vkCmdCopyImage(cmd, normals_.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   normalHistory_.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   1, &colorCopy);

    transitionLayout(cmd, colorLDR_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                     VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT);
    transitionLayout(cmd, depth_.image, VK_IMAGE_ASPECT_DEPTH_BIT,
                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                     VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT);
    transitionLayout(cmd, normals_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                     VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT);
    transitionLayout(cmd, colorHistory_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                     VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    transitionLayout(cmd, depthHistory_.image, VK_IMAGE_ASPECT_DEPTH_BIT,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                     VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    transitionLayout(cmd, normalHistory_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                     VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
}

// ---- render ----------------------------------------------------------------

void DLSS45Recon::render(VkCommandBuffer cmd,
                         const GLTFMeshScene& mesh, VkBuffer vertexBuffer, VkBuffer indexBuffer,
                         const float viewMatrix[16], const float projMatrix[16],
                         const float camPos[3],
                         uint32_t swapIndex, VkFramebuffer swapFramebuffer,
                         uint32_t displayW, uint32_t displayH) {
    (void)swapIndex;
    // First frame: transition storage images out of UNDEFINED. History is
    // explicitly cleared below, rejected by the temporal shaders for frame 0,
    // and populated from the current G-buffer at the end of every frame.
    // NOTE: firstFrame_ stays true through the ENTIRE first render so the
    // "back-to-GENERAL" transitions below (which assume prior SHADER_READ_ONLY)
    // do NOT fire on frame 0. It is cleared at the end of render().
    if (firstFrame_) {
        VkPipelineStageFlags all = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        // Storage images -> GENERAL.
        transitionLayout(cmd, reprojColor_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                         VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, all, 0, VK_ACCESS_SHADER_WRITE_BIT);
        transitionLayout(cmd, reprojConf_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                         VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, all, 0, VK_ACCESS_SHADER_WRITE_BIT);
        transitionLayout(cmd, denoised_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                         VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, all, 0, VK_ACCESS_SHADER_WRITE_BIT);
        transitionLayout(cmd, denoiseConf_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                         VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, all, 0, VK_ACCESS_SHADER_WRITE_BIT);
        transitionLayout(cmd, colorSR_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                         VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, all, 0, VK_ACCESS_SHADER_WRITE_BIT);
        transitionLayout(cmd, depthHR_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                         VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, all, 0, VK_ACCESS_SHADER_WRITE_BIT);
        transitionLayout(cmd, motionHR_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                         VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, all, 0, VK_ACCESS_SHADER_WRITE_BIT);
    }
    // Upload camera UBO: viewProj = proj * view; prevViewProj from stored value.
    DLSS45CameraUBO cam{};
    matMul(cam.viewProj, projMatrix, viewMatrix);
    memcpy(cam.prevViewProj, prevViewProj_, sizeof(float) * 16);
    memcpy(cam.camPos, camPos, sizeof(float) * 3);
    cam.camPos[3] = 1.0f;
    cam.resolution[0] = (float)lrW_; cam.resolution[1] = (float)lrH_;
    cam.sunDir[0] = -0.5f; cam.sunDir[1] = -0.8f; cam.sunDir[2] = -0.3f; cam.sunDir[3] = 0.0f;
    void* p;
    vkMapMemory(device_, camUBOMem_, 0, sizeof(cam), 0, &p); memcpy(p, &cam, sizeof(cam)); vkUnmapMemory(device_, camUBOMem_);

    DLSS45SceneUBO sc{};
    sc.skyColor[0] = 0.4f; sc.skyColor[1] = 0.6f; sc.skyColor[2] = 0.9f; sc.skyColor[3] = 1.0f;
    sc.lampColor[0] = 1.0f; sc.lampColor[1] = 0.85f; sc.lampColor[2] = 0.6f; sc.lampColor[3] = 1.0f;
    sc.lampPos[0] = 3.0f; sc.lampPos[1] = 2.0f; sc.lampPos[2] = 1.0f; sc.lampPos[3] = 1.0f;
    vkMapMemory(device_, sceneUBOMem_, 0, sizeof(sc), 0, &p); memcpy(p, &sc, sizeof(sc)); vkUnmapMemory(device_, sceneUBOMem_);

    // ---- 1) G-buffer fill (graphics, LR) ----
    VkClearValue clears[6];
    for (int i = 0; i < 5; i++) { clears[i].color = {{0.0f, 0.0f, 0.0f, 1.0f}}; }
    clears[5].depthStencil = {1.0f, 0};
    VkRenderPassBeginInfo rp{};
    rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rp.renderPass = gbufferPass_;
    rp.framebuffer = gbufferFramebuffer_;
    rp.renderArea.extent = {lrW_, lrH_};
    rp.clearValueCount = 6;
    rp.pClearValues = clears;
    vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport vp{0, 0, (float)lrW_, (float)lrH_, 0, 1};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D scRect{{0, 0}, {lrW_, lrH_}};
    vkCmdSetScissor(cmd, 0, 1, &scRect);

    if (indexBuffer && vertexBuffer && mesh.indexCount()) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, meshLayout_,
                                0, 1, &meshSet_, 0, nullptr);
        VkDeviceSize off = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &vertexBuffer, &off);
        vkCmdBindIndexBuffer(cmd, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, mesh.indexCount(), 1, 0, 0, 0);
    }
    vkCmdEndRenderPass(cmd);

    // Transition G-buffer depth to shader-read for compute sampling.
    transitionLayout(cmd, depth_.image, VK_IMAGE_ASPECT_DEPTH_BIT,
                     VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                     VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                     VK_ACCESS_SHADER_READ_BIT);

    if (firstFrame_) initializeHistory(cmd);

    // ---- 2) Temporal reprojection ----
    // (reads G-buffer + history as SHADER_READ_ONLY; writes reproj as storage GENERAL)
    // Ensure reproj outputs are GENERAL for writing (non-first frames end them SHADER_READ_ONLY).
    if (!firstFrame_) {
        transitionLayout(cmd, reprojColor_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT);
        transitionLayout(cmd, reprojConf_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT);
    }
    reproj_.record(cmd, reprojSet_, lrW_, lrH_, !firstFrame_);

    // Transition reproj storage outputs GENERAL -> SHADER_READ_ONLY for denoiser sampling.
    transitionLayout(cmd, reprojColor_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                     VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                     VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    transitionLayout(cmd, reprojConf_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                     VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                     VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

    // ---- 3) Denoiser ----
    // Ensure denoiser outputs are GENERAL for writing.
    if (!firstFrame_) {
        transitionLayout(cmd, denoised_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT);
        transitionLayout(cmd, denoiseConf_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT);
    }
    denoiser_.record(cmd, denoiserSet_, lrW_, lrH_, !firstFrame_);

    // Transition denoiser storage outputs GENERAL -> SHADER_READ_ONLY for SR sampling.
    transitionLayout(cmd, denoised_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                     VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                     VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    transitionLayout(cmd, denoiseConf_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                     VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                     VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

    // ---- 4) Super resolution ----
    // Ensure SR outputs are GENERAL for writing.
    if (!firstFrame_) {
        transitionLayout(cmd, colorSR_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT);
        transitionLayout(cmd, depthHR_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT);
        // NOTE: motionHR_ is a pure STORAGE output (never sampled), so it stays in
        // GENERAL permanently and must NOT be cycled back from SHADER_READ_ONLY.
    }
    SRPipeline::Params sp{};
    sp.scale = scale_; sp.lrWidth = lrW_; sp.lrHeight = lrH_; sp.edgeTau = 1.0f;
    sr_.record(cmd, srSet_, sp);

    // Transition HR storage outputs GENERAL -> SHADER_READ_ONLY for tone-map sampling.
    transitionLayout(cmd, colorSR_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                     VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                     VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    transitionLayout(cmd, depthHR_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                     VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                     VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

    // ---- 5) Tone-map composite (HR -> swapchain) ----
    toneMap_.record(cmd, toneMapSet_, swapFramebuffer, displayW, displayH);

    // Persist this frame's low-resolution color/depth as the only history
    // consumed by the next frame. This makes temporal state explicit rather
    // than sampling never-written images.
    updateHistory(cmd);

    // Store current viewProj as prev for next frame (motion).
    memcpy(prevViewProj_, cam.viewProj, sizeof(float) * 16);
    firstFrame_ = false;
}

// ---- shutdown ------------------------------------------------------------

void DLSS45Recon::shutdown(VkDevice device) {
    if (!device_) return;
    reproj_.shutdown(device);
    denoiser_.shutdown(device);
    sr_.shutdown(device);
    toneMap_.shutdown(device);

    if (meshPipeline_) vkDestroyPipeline(device, meshPipeline_, nullptr);
    if (meshLayout_) vkDestroyPipelineLayout(device, meshLayout_, nullptr);
    if (gbufferFramebuffer_) vkDestroyFramebuffer(device, gbufferFramebuffer_, nullptr);
    if (gbufferPass_) vkDestroyRenderPass(device, gbufferPass_, nullptr);
    // compositePass_ is the renderer's render pass; do NOT destroy here.
    compositePass_ = VK_NULL_HANDLE;

    for (auto* i : {&colorLDR_, &depth_, &normals_, &motion_, &material_, &noise_,
                    &colorHistory_, &depthHistory_, &normalHistory_,
                    &reprojColor_, &reprojConf_, &denoised_, &denoiseConf_,
                    &colorSR_, &depthHR_, &motionHR_}) {
        if (i->view) vkDestroyImageView(device, i->view, nullptr);
        if (i->image) vkDestroyImage(device, i->image, nullptr);
        if (i->memory) vkFreeMemory(device, i->memory, nullptr);
    }
    if (sampler_) vkDestroySampler(device, sampler_, nullptr);
    if (pool_) vkDestroyDescriptorPool(device, pool_, nullptr);
    if (reprojSetLayout_) vkDestroyDescriptorSetLayout(device, reprojSetLayout_, nullptr);
    if (denoiserSetLayout_) vkDestroyDescriptorSetLayout(device, denoiserSetLayout_, nullptr);
    if (srSetLayout_) vkDestroyDescriptorSetLayout(device, srSetLayout_, nullptr);
    if (toneMapSetLayout_) vkDestroyDescriptorSetLayout(device, toneMapSetLayout_, nullptr);
    if (meshSetLayout_) vkDestroyDescriptorSetLayout(device, meshSetLayout_, nullptr);

    if (camUBO_) vkDestroyBuffer(device, camUBO_, nullptr);
    if (camUBOMem_) vkFreeMemory(device, camUBOMem_, nullptr);
    if (sceneUBO_) vkDestroyBuffer(device, sceneUBO_, nullptr);
    if (sceneUBOMem_) vkFreeMemory(device, sceneUBOMem_, nullptr);

    device_ = VK_NULL_HANDLE;
}
