#version 450
layout(local_size_x = 8, local_size_y = 8) in;

// ML super-resolution (Phase 2 target). Infrastructure-compatible compute skeleton:
// each invocation runs the conv feature extractors over one LR pixel and writes the
// s x s HR patch via pixel shuffle. Weights/biases come from MLWeightsLoader SSBOs.
// The 10 input channels are packed as: [0]=denoised R, [1]=G, [2]=B, [3]=depth,
// [4..6]=normal, [7..8]=motion, [9]=confidence.

layout(binding = 0) uniform sampler2D uColorDenoised;
layout(binding = 1) uniform sampler2D uDepthLR;
layout(binding = 2) uniform sampler2D uNormalsLR;
layout(binding = 3) uniform sampler2D uMotionVectors;
layout(binding = 4) uniform sampler2D uDenoiseConfidence;

layout(binding = 5, rgba16f) uniform writeonly image2D oColorSR;
layout(binding = 6, r32f)    uniform writeonly image2D oDepthHR;
layout(binding = 7, rg16f)   uniform writeonly image2D oMotionHR;

layout(std430, binding = 8) buffer Weights { float w[]; };
layout(std430, binding = 9) buffer Biases  { float b[]; };
layout(std430, binding = 10) readonly buffer NetDesc {
    uint enc1_inCh, enc1_outCh, enc1_k, enc1_stride, enc1_pad, enc1_offW, enc1_offB;
    uint enc2_inCh, enc2_outCh, enc2_k, enc2_stride, enc2_pad, enc2_offW, enc2_offB;
    uint enc3_inCh, enc3_outCh, enc3_k, enc3_stride, enc3_pad, enc3_offW, enc3_offB;
    uint dec_inCh,  dec_outCh,  dec_k, dec_stride,  dec_pad, dec_offW, dec_offB;
    uint scale;
} net;

layout(push_constant) uniform Params {
    uint lrWidth;
    uint lrHeight;
    uint useML;   // 1 = ML path, 0 = fallback
    float _pad0;
} params;

float relu(float x) { return max(x, 0.0); }

// 3x3 conv with ReLU, reading from the 10-channel packed LR inputs via a per-channel
// sampler index. Placeholder index math; real layout resolves from NetDesc offsets.
float conv1(int outCh, ivec2 pix, int inCh) {
    float sum = 0.0;
    for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
            ivec2 p = clamp(pix + ivec2(kx, ky), ivec2(0), ivec2(2047));
            vec2 uv = (vec2(p) + 0.5) / vec2(textureSize(uColorDenoised, 0));
            float v = 0.0;
            if      (inCh == 0) v = texture(uColorDenoised, uv).r;
            else if (inCh == 1) v = texture(uColorDenoised, uv).g;
            else if (inCh == 2) v = texture(uColorDenoised, uv).b;
            else if (inCh == 3) v = texture(uDepthLR, uv).r;
            else if (inCh == 4) v = texture(uNormalsLR, uv).r;
            else if (inCh == 5) v = texture(uNormalsLR, uv).g;
            else if (inCh == 6) v = texture(uNormalsLR, uv).b;
            else if (inCh == 7) v = texture(uMotionVectors, uv).r;
            else if (inCh == 8) v = texture(uMotionVectors, uv).g;
            else                v = texture(uDenoiseConfidence, uv).r;
            float wv = w[(net.enc1_offW / 4) + (((inCh * net.enc1_outCh + outCh) * 9) +
                       ((ky + 1) * 3 + (kx + 1)))];
            sum += v * wv;
        }
    }
    return sum + b[net.enc1_offB / 4 + outCh];
}

void main() {
    ivec2 pixLR = ivec2(gl_GlobalInvocationID.xy);

    // If ML not active, fall through to a basic repeat tile from current color.
    float tr = texture(uColorDenoised,
                       (vec2(pixLR) + 0.5) / vec2(textureSize(uColorDenoised, 0))).r;
    float tg = texture(uColorDenoised,
                       (vec2(pixLR) + 0.5) / vec2(textureSize(uColorDenoised, 0))).g;
    float tb = texture(uColorDenoised,
                       (vec2(pixLR) + 0.5) / vec2(textureSize(uColorDenoised, 0))).b;
    float depth = texture(uDepthLR,
                       (vec2(pixLR) + 0.5) / vec2(textureSize(uDepthLR, 0))).r;
    vec2 mot = texture(uMotionVectors,
                       (vec2(pixLR) + 0.5) / vec2(textureSize(uMotionVectors, 0))).xy;

    // Placeholder ML output (would call conv1 for full 10->32->32->64->3*s^2 graph).
    uint s = max(net.scale, 1u);
    for (uint oy = 0; oy < s; ++oy) {
        for (uint ox = 0; ox < s; ++ox) {
            ivec2 pixHR = pixLR * int(s) + ivec2(ox, oy);
            imageStore(oColorSR, pixHR, vec4(tr, tg, tb, 1.0));
            imageStore(oDepthHR, pixHR, vec4(depth));
            imageStore(oMotionHR, pixHR, vec4(mot, 0.0, 0.0));
        }
    }
}
