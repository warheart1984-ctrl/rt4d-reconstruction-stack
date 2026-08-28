#include "sr_pipeline.h"

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

bool SRPipeline::init(VkDevice device, VkDescriptorSetLayout dsLayout) {
    device_ = device;

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = sizeof(Params);

    VkPipelineLayoutCreateInfo pl{};
    pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl.setLayoutCount = 1;
    pl.pSetLayouts = &dsLayout;
    pl.pushConstantRangeCount = 1;
    pl.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(device, &pl, nullptr, &layout_) != VK_SUCCESS) {
        fprintf(stderr, "[RT4D Reconstruction Stack] sr: pipeline layout failed\n");
        return false;
    }

    auto code = readSpv("spirv_rasterize/sr_classic.comp.spv");
    if (code.empty()) {
        fprintf(stderr, "[RT4D Reconstruction Stack] sr: missing SPIR-V\n");
        return false;
    }

    VkShaderModule mod;
    VkShaderModuleCreateInfo sm{};
    sm.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    sm.codeSize = code.size() * sizeof(uint32_t);
    sm.pCode = code.data();
    if (vkCreateShaderModule(device, &sm, nullptr, &mod) != VK_SUCCESS) return false;

    VkPipelineShaderStageCreateInfo stage{};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = mod;
    stage.pName = "main";

    VkComputePipelineCreateInfo cp{};
    cp.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cp.stage = stage;
    cp.layout = layout_;
    VkResult r = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp, nullptr, &pipeline_);
    vkDestroyShaderModule(device, mod, nullptr);
    return r == VK_SUCCESS;
}

void SRPipeline::record(VkCommandBuffer cmd, VkDescriptorSet ds, const Params& p) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Params), &p);
    uint32_t gx = (p.lrWidth + 7) / 8;
    uint32_t gy = (p.lrHeight + 7) / 8;
    vkCmdDispatch(cmd, gx, gy, 1);
}

void SRPipeline::shutdown(VkDevice device) {
    if (pipeline_) vkDestroyPipeline(device, pipeline_, nullptr);
    if (layout_) vkDestroyPipelineLayout(device, layout_, nullptr);
    pipeline_ = VK_NULL_HANDLE;
    layout_ = VK_NULL_HANDLE;
}
