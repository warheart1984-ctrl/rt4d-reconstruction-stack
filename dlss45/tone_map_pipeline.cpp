#include "tone_map_pipeline.h"

#include <cstdio>
#include <fstream>
#include <vector>

static std::vector<uint32_t> readSpv(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    std::streamsize size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<uint32_t> code((size_t)(size / 4));
    if (!code.empty()) f.read((char*)code.data(), size);
    return code;
}

bool ToneMapPipeline::init(VkDevice device, VkDescriptorSetLayout dsLayout,
                           VkRenderPass renderPass) {
    device_ = device;

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    pcr.offset = 0;
    pcr.size = sizeof(Params);

    VkPipelineLayoutCreateInfo pl{};
    pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl.setLayoutCount = 1;
    pl.pSetLayouts = &dsLayout;
    pl.pushConstantRangeCount = 1;
    pl.pPushConstantRanges = &pcr;
    renderPass_ = renderPass;
    if (vkCreatePipelineLayout(device, &pl, nullptr, &layout_) != VK_SUCCESS) return false;

    auto vcode = readSpv("spirv_rasterize/fullscreen_quad.vert.spv");
    auto fcode = readSpv("spirv_rasterize/tone_map.frag.spv");
    if (vcode.empty() || fcode.empty()) {
        fprintf(stderr, "[RT4D Reconstruction Stack] tonemap: missing SPIR-V\n");
        return false;
    }
    VkShaderModule vmod, fmod;
    VkShaderModuleCreateInfo vsm{};
    vsm.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    vsm.codeSize = vcode.size() * sizeof(uint32_t);
    vsm.pCode = vcode.data();
    vkCreateShaderModule(device, &vsm, nullptr, &vmod);
    VkShaderModuleCreateInfo fsm{};
    fsm.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    fsm.codeSize = fcode.size() * sizeof(uint32_t);
    fsm.pCode = fcode.data();
    vkCreateShaderModule(device, &fsm, nullptr, &fmod);

    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vmod;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fmod;
    stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vp{};
    vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1;
    vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    // Subpass uses a depth attachment -> must provide a (disabled) depth-stencil state.
    VkPipelineDepthStencilStateCreateInfo ds{};
    ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable = VK_FALSE;
    ds.depthWriteEnable = VK_FALSE;
    VkPipelineColorBlendAttachmentState cba{};
    cba.colorWriteMask = 0xf;
    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1;
    cb.pAttachments = &cba;

    VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynCi{};
    dynCi.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynCi.dynamicStateCount = 2;
    dynCi.pDynamicStates = dyn;

    VkGraphicsPipelineCreateInfo gp{};
    gp.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gp.stageCount = 2;
    gp.pStages = stages;
    gp.pVertexInputState = &vi;
    gp.pInputAssemblyState = &ia;
    gp.pViewportState = &vp;
    gp.pRasterizationState = &rs;
    gp.pMultisampleState = &ms;
    gp.pDepthStencilState = &ds;
    gp.pColorBlendState = &cb;
    gp.pDynamicState = &dynCi;
    gp.layout = layout_;
    gp.renderPass = renderPass;
    gp.subpass = 0;

    VkResult r = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gp, nullptr, &pipeline_);
    vkDestroyShaderModule(device, vmod, nullptr);
    vkDestroyShaderModule(device, fmod, nullptr);
    return r == VK_SUCCESS;
}

void ToneMapPipeline::record(VkCommandBuffer cmd, VkDescriptorSet ds, VkFramebuffer fb,
                             uint32_t w, uint32_t h) {
    // A zero-initialized exposure blackens the entire composite even when the
    // reconstruction image is populated. Keep the pass deterministic but use
    // a visible baseline for the current linear-HDR G-buffer contract.
    Params p{};
    p.useACES = 1;
    p.exposure = 1.0f;
    VkRenderPassBeginInfo rp{};
    rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rp.renderPass = renderPass_;
    rp.framebuffer = fb;
    rp.renderArea.extent = {w, h};
    // The renderer's swapchain render pass CLEARs both a color and a depth attachment.
    VkClearValue clears[2]{};
    clears[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    clears[1].depthStencil = {1.0f, 0};
    rp.clearValueCount = 2;
    rp.pClearValues = clears;
    vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(Params), &p);

    VkViewport vp{0, 0, (float)w, (float)h, 0, 1};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc{{0, 0}, {w, h}};
    vkCmdSetScissor(cmd, 0, 1, &sc);

    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRenderPass(cmd);
}

void ToneMapPipeline::shutdown(VkDevice device) {
    if (pipeline_) vkDestroyPipeline(device, pipeline_, nullptr);
    if (layout_) vkDestroyPipelineLayout(device, layout_, nullptr);
    pipeline_ = VK_NULL_HANDLE;
    layout_ = VK_NULL_HANDLE;
}
