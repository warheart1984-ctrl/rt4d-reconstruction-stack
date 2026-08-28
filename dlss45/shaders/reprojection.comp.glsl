#version 450
layout(local_size_x = 8, local_size_y = 8) in;

layout(binding = 0) uniform sampler2D uColorCurrent;
layout(binding = 1) uniform sampler2D uDepthCurrent;
layout(binding = 2) uniform sampler2D uMotionVectors;
layout(binding = 3) uniform sampler2D uColorHistory;
layout(binding = 4) uniform sampler2D uDepthHistory;

layout(binding = 5, rgba16f) uniform writeonly image2D oReprojectedColor;
layout(binding = 6, r16f)    uniform writeonly image2D oReprojectionConfidence;

// Reprojection parameters (sigma_d, m_max)
layout(push_constant) uniform Params {
    float sigmaD;
    float mMax;
    float _pad0;
    float _pad1;
} params;

void main() {
    ivec2 pix = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = textureSize(uColorCurrent, 0);
    vec2 uv = (vec2(pix) + 0.5) / vec2(size);

    vec2 motion = texture(uMotionVectors, uv).xy;
    float mlen = length(motion);

    // Reject insane motion vectors.
    float c_m = (mlen < params.mMax) ? 1.0 : 0.0;

    vec2 uvPrev = uv - motion / vec2(size);

    vec4 prevColor = texture(uColorHistory, uvPrev);
    float prevDepth = (uvPrev.x >= 0.0 && uvPrev.x <= 1.0 &&
                       uvPrev.y >= 0.0 && uvPrev.y <= 1.0)
                      ? texture(uDepthHistory, uvPrev).r : 0.0;
    float currDepth = texture(uDepthCurrent, uv).r;

    // Depth consistency.
    float c_d = exp(-abs(currDepth - prevDepth) / max(params.sigmaD, 1e-6));

    // Disocclusion: force reject history when geometry changed or off-screen.
    bool offscreen = (uvPrev.x < 0.0 || uvPrev.x > 1.0 ||
                      uvPrev.y < 0.0 || uvPrev.y > 1.0);
    float c_hist = c_d * c_m;
    if (offscreen) c_hist = 0.0;

    vec4 reproj = (c_hist > 0.0) ? prevColor : vec4(0.0);

    imageStore(oReprojectedColor, pix, reproj);
    imageStore(oReprojectionConfidence, pix, vec4(c_hist));
}
