#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec3 vNormal;
layout(location = 1) out vec3 vWorldPos;
layout(location = 2) out vec2 vUV;

layout(set = 0, binding = 0) uniform CameraUBO {
    mat4 view;
    mat4 proj;
    vec3 camPos;
    vec3 camForward;
    vec3 camRight;
    vec3 camUp;
    vec2 resolution;
    float time;
} cam;

layout(set = 0, binding = 1) uniform BattleUBO {
    vec3 sunDir;
    float _pad0;
    vec3 fogColor;
    float fogDensity;
    mat4 model;
} battle;

void main() {
    vec4 worldPos = battle.model * vec4(inPosition, 1.0);
    gl_Position = cam.proj * cam.view * worldPos;

    vNormal = mat3(battle.model) * inNormal;
    vWorldPos = worldPos.xyz;
    vUV = inUV;
}
