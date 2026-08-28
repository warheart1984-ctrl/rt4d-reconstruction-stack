#version 450

layout(location = 0) in vec3 vNormal;
layout(location = 1) in vec3 vWorldPos;
layout(location = 2) in vec2 vUV;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 1) uniform HatchUBO {
    vec3 lightPos;
    float _pad0;
    vec3 viewPos;
    float _pad1;
    vec3 baseColor;
    float _pad2;
    vec3 sssColor;
    float hatchProgress;
    mat4 model;
} hatch;

float sssTerm(vec3 N, vec3 L, vec3 V) {
    float nl = max(dot(N, L), 0.0);
    float nv = max(dot(N, V), 0.0);
    return pow(nl * nv, 0.7);
}

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    vec3 N = normalize(vNormal);
    vec3 L = normalize(hatch.lightPos - vWorldPos);
    vec3 V = normalize(hatch.viewPos - vWorldPos);

    float diff = max(dot(N, L), 0.0);
    float sss = sssTerm(N, L, V);

    vec3 base = hatch.baseColor;

    float crackNoise = hash(vUV * 47.0 + hatch.hatchProgress * 10.0);
    float crackMask = step(0.98 - hatch.hatchProgress * 0.5, crackNoise);
    float glowCrack = crackMask * hatch.hatchProgress * 2.0;

    vec3 sssContrib = hatch.sssColor * sss * (1.0 + glowCrack);
    vec3 color = base * diff + sssContrib;

    color += hatch.sssColor * glowCrack * 0.5;

    float fresnel = pow(1.0 - max(dot(N, V), 0.0), 3.0);
    color += hatch.sssColor * fresnel * 0.3 * hatch.hatchProgress;

    outColor = vec4(color, 1.0);
}
