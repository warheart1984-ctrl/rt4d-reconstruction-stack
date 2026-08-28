#pragma once

// RT4D Reconstruction Stack pass-graph enumeration. Execution order follows
// the internal compatibility enum order.
// the temporal reprojection / denoiser / super-resolution stages are compute
// dispatches between the graphics base (RT4D_GBUFFER_LR) and composite passes.

enum class DLSS45Pass {
    RT4D_GBUFFER_LR,      // base render -> low-res G-buffers
    TEMPORAL_REPROJECTION,// compute
    DENOISER_RR,          // compute (ray reconstruction)
    SUPER_RESOLUTION,     // compute (classic or ML)
    TONE_MAP_COMPOSITE    // graphics -> swapchain
};

enum DLSS45Attachment {
    ATT_COLOR_LDR_LR,
    ATT_DEPTH_LR,
    ATT_NORMALS_LR,
    ATT_MOTION_LR,
    ATT_MATERIAL_LR,
    ATT_COLOR_SR_HR,
    ATT_DEPTH_HR,
    ATT_SWAPCHAIN,
    ATT_COUNT
};
