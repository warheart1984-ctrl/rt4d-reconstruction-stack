#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <vulkan/vulkan.h>

// Mirrors the GLSL NetDesc block in sr_ml_conv.comp.glsl.
struct SRNetworkDesc {
    uint32_t enc1_inCh, enc1_outCh, enc1_k, enc1_stride, enc1_pad, enc1_offW, enc1_offB;
    uint32_t enc2_inCh, enc2_outCh, enc2_k, enc2_stride, enc2_pad, enc2_offW, enc2_offB;
    uint32_t enc3_inCh, enc3_outCh, enc3_k, enc3_stride, enc3_pad, enc3_offW, enc3_offB;
    uint32_t dec_inCh,  dec_outCh,  dec_k, dec_stride,  dec_pad, dec_offW, dec_offB;
    uint32_t scale;
};

class MLWeightsLoader {
public:
    // Parses a flat binary file produced from the ONNX export:
    //   [SRNetworkDesc struct (28 x uint32)] [weights float[]] [biases float[]]
    bool loadFromFile(const std::string& path,
                      SRNetworkDesc& outDesc,
                      std::vector<float>& outWeights,
                      std::vector<float>& outBiases);

    // Uploads weights/biases into storage buffers. Returns handles for bindings 8/9.
    bool uploadToGPU(VkDevice device, VkPhysicalDevice phys,
                     VkDeviceSize weightsBytes, const void* weights,
                     VkDeviceSize biasBytes, const void* biases,
                     VkBuffer& weightsBuffer, VkDeviceMemory& weightsMemory,
                     VkBuffer& biasBuffer, VkDeviceMemory& biasMemory);
};
