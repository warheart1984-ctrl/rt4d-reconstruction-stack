#pragma once

#include <vulkan/vulkan.h>

// RT4D Reconstruction Stack G-buffer contract (Phase 1, classical).
// All low-res (LR) buffers are captured by the base render pass at renderWidth x
// renderHeight; high-res (HR) buffers are produced by the super-resolution pass at
// displayWidth x displayHeight.

struct RT4D_GBufferLR {
    VkImage colorLDR = VK_NULL_HANDLE;       // RGBA16F  raw radiance (pre tone-map)
    VkImageView colorLDRView = VK_NULL_HANDLE;

    VkImage depth = VK_NULL_HANDLE;          // R32F  linear depth from camera
    VkImageView depthView = VK_NULL_HANDLE;

    VkImage normals = VK_NULL_HANDLE;        // RGBA16F  xyz = normal, w = roughness
    VkImageView normalsView = VK_NULL_HANDLE;

    VkImage motionVectors = VK_NULL_HANDLE;  // RG16F  screen-space motion (px/frame)
    VkImageView motionVectorsView = VK_NULL_HANDLE;

    VkImage materialID = VK_NULL_HANDLE;     // R16UI  index into material table
    VkImageView materialIDView = VK_NULL_HANDLE;

    VkImage noiseMeta = VK_NULL_HANDLE;      // R16F  variance / sample-count hint
    VkImageView noiseMetaView = VK_NULL_HANDLE;
};

struct RT4D_HistoryLR {
    VkImage colorHistory = VK_NULL_HANDLE;   // RGBA16F (same format as colorLDR)
    VkImageView colorHistoryView = VK_NULL_HANDLE;

    VkImage depthHistory = VK_NULL_HANDLE;   // R32F (same format as depth)
    VkImageView depthHistoryView = VK_NULL_HANDLE;
};

struct RT4D_ReconstructionLR {
    VkImage reprojectedColor = VK_NULL_HANDLE;          // RGBA16F
    VkImageView reprojectedColorView = VK_NULL_HANDLE;
    VkImage reprojectionConfidence = VK_NULL_HANDLE;    // R16F
    VkImageView reprojectionConfidenceView = VK_NULL_HANDLE;
    VkImage colorDenoised = VK_NULL_HANDLE;             // RGBA16F
    VkImageView colorDenoisedView = VK_NULL_HANDLE;
    VkImage denoiseConfidence = VK_NULL_HANDLE;         // R16F
    VkImageView denoiseConfidenceView = VK_NULL_HANDLE;
};

struct RT4D_UpscaleHR {
    VkImage colorSR = VK_NULL_HANDLE;        // RGBA16F
    VkImageView colorSRView = VK_NULL_HANDLE;
    VkImage depthHR = VK_NULL_HANDLE;        // R32F
    VkImageView depthHRView = VK_NULL_HANDLE;
    VkImage motionVectorsHR = VK_NULL_HANDLE; // RG16F
    VkImageView motionVectorsHRView = VK_NULL_HANDLE;
};
