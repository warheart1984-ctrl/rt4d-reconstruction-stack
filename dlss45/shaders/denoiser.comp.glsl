#version 450
layout(local_size_x = 8, local_size_y = 8) in;

layout(binding = 0) uniform sampler2D uColorCurrent;   // current radiance C_t
layout(binding = 1) uniform sampler2D uDepthCurrent;   // current depth D_t
layout(binding = 2) uniform sampler2D uNormalsCurrent; // current normals N_t
layout(binding = 3) uniform sampler2D uMotionVectors;  // screen-space motion M_t
layout(binding = 4) uniform sampler2D uNoiseMeta;      // variance V_t
layout(binding = 5) uniform sampler2D uColorHistory;   // reprojected C_{t-1}
layout(binding = 6) uniform sampler2D uDepthHistory;   // reprojected D_{t-1}
layout(binding = 7) uniform sampler2D uNormalsHistory; // reprojected N_{t-1}
layout(binding = 8) uniform sampler2D uMaterialID;     // material id / flags

layout(binding = 9, rgba16f) uniform writeonly image2D oColorDenoised;
layout(binding = 10, r16f)   uniform writeonly image2D oDenoiseConfidence;

// Confidence weights: w_d, w_n, w_m, w_v, sigma_d, sigma_n, m_max, special thresholds
layout(push_constant) uniform Params {
    float wD, wN, wM, wV;
    float sigmaD, mMax, specThreshold, depthGradSlope;
} params;

float depthGradient(vec2 uv, float eps) {
    vec2 e = vec2(eps, 0.0);
    float hy = texture(uDepthCurrent, uv + e).r;
    float hx = texture(uDepthCurrent, uv + vec2(0.0, eps)).r;
    float dx = texture(uDepthCurrent, uv - e).r;
    float dy = texture(uDepthCurrent, uv - vec2(0.0, eps)).r;
    return max(abs(hy - dy), abs(hx - dx));
}

void main() {
    ivec2 pix = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = textureSize(uColorCurrent, 0);
    vec2 uv = (vec2(pix) + 0.5) / vec2(size);

    vec4 cur = texture(uColorCurrent, uv);
    vec4 prev = texture(uColorHistory, uv);
    float Dt = texture(uDepthCurrent, uv).r;
    float Dp = texture(uDepthHistory, uv).r;
    vec4 Nt = texture(uNormalsCurrent, uv);
    vec4 Np = texture(uNormalsHistory, uv);
    vec2 M = texture(uMotionVectors, uv).xy;
    float V = texture(uNoiseMeta, uv).r;
    float mat = texture(uMaterialID, uv).r;

    // Confidence components (heuristic spec).
    float c_d = exp(-abs(Dt - Dp) / max(params.sigmaD, 1e-6));
    vec3 n_a = normalize(Nt.xyz);
    vec3 n_b = normalize(Np.xyz);
    float c_n = max(0.0, dot(n_a, n_b));
    float c_m = (length(M) < params.mMax) ? 1.0 : 0.0;
    float c_v = clamp(1.0 - V, 0.0, 1.0);

    float c_hist = pow(c_d, params.wD) *
                   pow(c_n, params.wN) *
                   pow(c_m, params.wM) *
                   pow(c_v, params.wV);
    c_hist = clamp(c_hist, 0.0, 1.0);

    // Special cases.
    // Disocclusion (off-history): nuke confidence.
    if (c_d < 1e-4 || c_n < 1e-4) c_hist = 0.0;
    // Specular / emissive material (roughness < threshold or flag): bleed clamped.
    if (Nt.w < params.specThreshold) c_hist *= 0.5;
    // High depth gradient (edge / silhouette): warping confidence.
    float grad = depthGradient(uv, 0.001);
    if (grad > params.depthGradSlope) c_hist *= 0.7;

    c_hist = clamp(c_hist, 0.0, 1.0);
    float w_hist = c_hist;
    float w_curr = 1.0 - w_hist;

    vec4 denoised = w_curr * cur + w_hist * prev;

    imageStore(oColorDenoised, pix, denoised);
    imageStore(oDenoiseConfidence, pix, vec4(w_hist));
}
