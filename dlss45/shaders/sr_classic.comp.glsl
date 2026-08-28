#version 450
layout(local_size_x = 8, local_size_y = 8) in;

// Classic edge-aware super-resolution (Phase 1 baseline). Each invocation upsamples
// a 1x1 LR neighborhood into an SR block of size s x s. Inputs: denoised color,
// depth (for edge detection), denoise confidence.

layout(binding = 0) uniform sampler2D uColorDenoised;
layout(binding = 1) uniform sampler2D uDepthLR;
layout(binding = 2) uniform sampler2D uNormalsLR;
layout(binding = 3) uniform sampler2D uDenoiseConfidence;
layout(binding = 4) uniform sampler2D uMotionVectors;

layout(binding = 5, rgba16f) uniform writeonly image2D oColorSR;
layout(binding = 6, r32f)    uniform writeonly image2D oDepthHR;
layout(binding = 7, rg16f)   uniform writeonly image2D oMotionHR;

layout(push_constant) uniform Params {
    uint scale;      // SR scale (2)
    uint lrWidth;    // low-res width
    uint lrHeight;   // low-res height
    float edgeTau;   // edge-sharpening strength
} params;

const float LUM_W = 0.2126;
const float LUM_H = 0.7152;
const float LUM_B = 0.0722;

float luma(vec3 c) { return dot(c, vec3(LUM_W, LUM_H, LUM_B)); }

// Bicubic-ish determinant/tanh contrast sharpening based on local luma variance.
float contrastSig(vec3 around, vec3 center) {
    vec3 dz = around - center;
    return tanh(params.edgeTau * luma(dz));
}

void main() {
    ivec2 pixLR = ivec2(gl_GlobalInvocationID.xy);
    ivec2 lrSize = textureSize(uColorDenoised, 0);
    if (pixLR.x >= lrSize.x || pixLR.y >= lrSize.y) return;
    vec2 uvLR = (vec2(pixLR) + 0.5) / vec2(lrSize);

    vec4 col = texture(uColorDenoised, uvLR);
    vec3 nrm = normalize(texture(uNormalsLR, uvLR).xyz);
    float depth = texture(uDepthLR, uvLR).r;
    vec2 mot = texture(uMotionVectors, uvLR).xy;
    float conf = texture(uDenoiseConfidence, uvLR).r;

    // Simple neighbourhood (for contrast term in edges).
    vec4 colE = textureOffset(uColorDenoised, uvLR, ivec2(1, 0));
    vec4 colW = textureOffset(uColorDenoised, uvLR, ivec2(-1, 0));
    vec4 colS = textureOffset(uColorDenoised, uvLR, ivec2(0, 1));
    vec4 colN = textureOffset(uColorDenoised, uvLR, ivec2(0, -1));

    float cE = contrastSig(colE.rgb, col.rgb);
    float cW = contrastSig(colW.rgb, col.rgb);
    float cS = contrastSig(colS.rgb, col.rgb);
    float cN = contrastSig(colN.rgb, col.rgb);

    // Edge-aware sharpening: preserve edges, smooth interiors (confidence history).
    float sharpen = (cE + cW + cS + cN) * 0.25;
    vec3 sharp = col.rgb + sharpen * conf;

    for (uint oy = 0; oy < params.scale; ++oy) {
        for (uint ox = 0; ox < params.scale; ++ox) {
            ivec2 pixHR = pixLR * int(params.scale) + ivec2(ox, oy);
            imageStore(oColorSR, pixHR, vec4(sharp, 1.0));
            imageStore(oDepthHR, pixHR, vec4(depth));
            imageStore(oMotionHR, pixHR, vec4(mot, 0.0, 0.0));
        }
    }
}
