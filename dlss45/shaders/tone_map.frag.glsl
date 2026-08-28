#version 450
layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 fragColor;

layout(binding = 0) uniform sampler2D uColorSR;
layout(binding = 1) uniform sampler2D uDepthHR;

layout(push_constant) uniform Params {
    uint useACES;   // 1 = ACES-ish tonemap, 0 = Reinhard
    float exposure;
    float _pad0;
    float _pad1;
} params;

vec3 acesFilmic(vec3 x) {
    // ACES fitted curve approximation.
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

vec3 reinhard(vec3 x) { return x / (1.0 + x); }

void main() {
    vec3 hdr = texture(uColorSR, vUV).rgb * params.exposure;
    vec3 mapped = (params.useACES > 0u) ? acesFilmic(hdr) : reinhard(hdr);
    vec3 gamma = pow(max(mapped, vec3(0.0)), vec3(1.0 / 2.2));
    fragColor = vec4(gamma, 1.0);
}
