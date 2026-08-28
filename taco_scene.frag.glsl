#version 450

layout(location = 0) in vec3 vNormal;
layout(location = 1) in vec3 vWorldPos;
layout(location = 2) in vec2 vUV;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 1) uniform SceneUBO {
    vec3 sunDir;
    float _pad0;
    vec3 lampPos;
    float _pad1;
    vec3 skyColor;
    float _pad2;
    vec3 lampColor;
    float _pad3;
    mat4 model;
} scene;

float lambert(vec3 n, vec3 l) {
    return max(dot(normalize(n), normalize(l)), 0.0);
}

void main() {
    vec3 N = normalize(vNormal);

    float sun = lambert(N, -scene.sunDir);
    vec3 toLamp = scene.lampPos - vWorldPos;
    float lamp = lambert(N, toLamp);

    vec3 sand = vec3(0.76, 0.70, 0.50);
    vec3 water = vec3(0.2, 0.4, 0.6);
    float blend = smoothstep(-0.5, 0.5, vWorldPos.y);
    vec3 base = mix(water, sand, blend);

    vec3 color = base * (sun * 0.8 + lamp * 1.2) + scene.skyColor * 0.2;

    outColor = vec4(color, 1.0);
}
