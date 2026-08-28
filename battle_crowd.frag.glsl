#version 450

layout(location = 0) in vec3 vNormal;
layout(location = 1) in vec3 vWorldPos;
layout(location = 2) in vec2 vUV;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 1) uniform BattleUBO {
    vec3 sunDir;
    float _pad0;
    vec3 fogColor;
    float fogDensity;
    mat4 model;
} battle;

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

void main() {
    vec3 N = normalize(vNormal);
    vec3 L = normalize(-battle.sunDir);
    float diff = max(dot(N, L), 0.0);

    vec3 armor = vec3(0.3, 0.25, 0.2);
    vec3 skin = vec3(0.6, 0.4, 0.3);
    float pattern = step(0.5, fract(vUV.x * 4.0 + vUV.y * 4.0));
    vec3 base = mix(armor, skin, pattern);

    float dist = length(vWorldPos - cam.camPos);
    float fog = 1.0 - exp(-battle.fogDensity * dist);

    vec3 color = mix(base * diff, battle.fogColor, fog);

    outColor = vec4(color, 1.0);
}
