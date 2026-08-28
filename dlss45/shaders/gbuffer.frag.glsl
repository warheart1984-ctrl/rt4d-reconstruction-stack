#version 450

layout(location = 0) in vec3 vWorldPos;
layout(location = 1) in vec3 vNormal;
layout(location = 2) in vec2 vUV;
layout(location = 3) in vec2 vMotion;
layout(location = 4) in flat float vMaterialID;

layout(location = 0) out vec4 oColorLDR;
layout(location = 1) out vec4 oNormals;
layout(location = 2) out vec4 oMotion;
layout(location = 3) out vec4 oMaterial;
layout(location = 4) out vec4 oNoiseMeta;

layout(set = 0, binding = 0) uniform Camera {
    mat4 viewProj;
    mat4 prevViewProj;
    vec4 camPos;
    vec4 resolution;
    vec4 sunDir;
} cam;

layout(set = 0, binding = 1) uniform Scene {
    vec4 skyColor;
    vec4 lampColor;
    vec4 lampPos;
} scene;

layout(set = 1, binding = 0) uniform sampler2D uBaseColorTexture;

layout(push_constant) uniform MaterialParams {
    vec4 baseColorFactor;
} material;

void main() {
    vec3 N = normalize(vNormal);
    vec3 L = normalize(-cam.sunDir.xyz);
    vec3 V = normalize(cam.camPos.xyz - vWorldPos);

    float ndl = max(dot(N, L), 0.0);
    vec4 textureColor = texture(uBaseColorTexture, vUV);
    vec3 baseColor = textureColor.rgb * material.baseColorFactor.rgb;
    vec3 col = baseColor * scene.skyColor.xyz * (0.15 + 0.85 * ndl);

    // Simple lamp falloff.
    vec3 toLamp = scene.lampPos.xyz - vWorldPos;
    float dist = length(toLamp);
    vec3 lampDir = toLamp / max(dist, 1e-4);
    float lamp = max(dot(N, lampDir), 0.0) / (1.0 + 0.02 * dist * dist);
    col += baseColor * scene.lampColor.xyz * lamp;

    float roughness = 0.6;
    oColorLDR = vec4(col, 1.0);
    oNormals = vec4(N * 0.5 + 0.5, roughness);
    oMotion = vec4(vMotion, 0.0, 1.0);
    oMaterial = vec4(vMaterialID, 0.0, 0.0, 1.0);
    oNoiseMeta = vec4(0.0, 0.0, 0.0, 1.0);
}
