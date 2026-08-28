#version 450

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;
layout(location = 3) in float inMaterialID;

layout(location = 0) out vec3 vWorldPos;
layout(location = 1) out vec3 vNormal;
layout(location = 2) out vec2 vUV;
layout(location = 3) out flat vec2 vMotion;
layout(location = 4) out flat float vMaterialID;

layout(set = 0, binding = 0) uniform Camera {
    mat4 viewProj;
    mat4 prevViewProj;
    vec4 camPos;      // xyz
    vec4 resolution;  // xy = width,height
    vec4 sunDir;
} cam;

layout(set = 0, binding = 1) uniform Scene {
    vec4 skyColor;
    vec4 lampColor;
    vec4 lampPos;
} scene;

void main() {
    vec4 wp = vec4(inPos, 1.0);
    vWorldPos = inPos;
    vNormal = inNormal;
    vUV = inUV;

    vec4 clipPos = cam.viewProj * wp;
    vec4 prevClip = cam.prevViewProj * wp;

    vec2 ndc = clipPos.xy / clipPos.w;
    vec2 ndcPrev = prevClip.xy / prevClip.w;
    vMotion = (ndc - ndcPrev) * 0.5 * cam.resolution.xy;
    vMaterialID = inMaterialID;

    gl_Position = clipPos;
}
