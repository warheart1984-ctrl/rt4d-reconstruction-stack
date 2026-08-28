#include "ml_weights_loader.h"

#include <cstdio>
#include <cstring>
#include <fstream>

bool MLWeightsLoader::loadFromFile(const std::string& path,
                                   SRNetworkDesc& outDesc,
                                   std::vector<float>& outWeights,
                                   std::vector<float>& outBiases) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        fprintf(stderr, "[RT4D Reconstruction Stack] ml: cannot open %s\n", path.c_str());
        return false;
    }
    std::streamsize sz = f.tellg();
    f.seekg(0, std::ios::beg);

    const size_t descBytes = sizeof(SRNetworkDesc);
    if (sz < (std::streamsize)(descBytes + 8)) {
        fprintf(stderr, "[RT4D Reconstruction Stack] ml: file too small\n");
        return false;
    }
    std::vector<char> buf(sz);
    f.read(buf.data(), sz);

    memcpy(&outDesc, buf.data(), descBytes);
    // Recompute weight count from descriptors to bound read safely.
    size_t totalW = 0;
    auto layerW = [](uint32_t inC, uint32_t outC, uint32_t k) {
        return (size_t)inC * outC * (size_t)k * k;
    };
    totalW += layerW(outDesc.enc1_inCh, outDesc.enc1_outCh, outDesc.enc1_k);
    totalW += layerW(outDesc.enc2_inCh, outDesc.enc2_outCh, outDesc.enc2_k);
    totalW += layerW(outDesc.enc3_inCh, outDesc.enc3_outCh, outDesc.enc3_k);
    totalW += layerW(outDesc.dec_inCh, outDesc.dec_outCh, outDesc.dec_k);
    size_t totalB = outDesc.enc1_outCh + outDesc.enc2_outCh +
                    outDesc.enc3_outCh + outDesc.dec_outCh;

    size_t need = descBytes + totalW * sizeof(float) + totalB * sizeof(float);
    if (buf.size() < need) {
        fprintf(stderr, "[RT4D Reconstruction Stack] ml: truncated file (need %zu, have %zu)\n", need, buf.size());
        return false;
    }

    const float* floats = (const float*)(buf.data() + descBytes);
    outWeights.assign(floats, floats + totalW);
    outBiases.assign(floats + totalW, floats + totalW + totalB);
    fprintf(stderr, "[RT4D Reconstruction Stack] ml: loaded %zu weights, %zu biases, scale=%u\n",
            totalW, totalB, outDesc.scale);
    return true;
}

bool MLWeightsLoader::uploadToGPU(VkDevice device, VkPhysicalDevice phys,
                                  VkDeviceSize weightsBytes, const void* weights,
                                  VkDeviceSize biasBytes, const void* biases,
                                  VkBuffer& weightsBuffer, VkDeviceMemory& weightsMemory,
                                  VkBuffer& biasBuffer, VkDeviceMemory& biasMemory) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(phys, &memProps);

    auto makeBuffer = [&](VkDeviceSize size, const void* data,
                          VkBuffer& buf, VkDeviceMemory& mem) -> bool {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = size;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        if (vkCreateBuffer(device, &bci, nullptr, &buf) != VK_SUCCESS) return false;
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(device, buf, &req);
        uint32_t typeIdx = ~0u;
        for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
            if ((req.memoryTypeBits & (1u << i)) &&
                (memProps.memoryTypes[i].propertyFlags &
                 (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
                 (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
                typeIdx = i; break;
            }
        }
        if (typeIdx == ~0u) return false;
        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = typeIdx;
        if (vkAllocateMemory(device, &ai, nullptr, &mem) != VK_SUCCESS) return false;
        if (vkBindBufferMemory(device, buf, mem, 0) != VK_SUCCESS) return false;
        if (data) {
            void* mapped = nullptr;
            vkMapMemory(device, mem, 0, size, 0, &mapped);
            memcpy(mapped, data, size);
            vkUnmapMemory(device, mem);
        }
        return true;
    };

    if (!makeBuffer(weightsBytes, weights, weightsBuffer, weightsMemory)) return false;
    if (!makeBuffer(biasBytes, biases, biasBuffer, biasMemory)) return false;
    return true;
}
