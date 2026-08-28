#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in float inMagnitude;
layout(location = 3) in float inConfidence;
layout(location = 4) in vec3 inNormal;
layout(location = 5) in vec3 inViewDir;
layout(location = 6) in vec3 inLightDir;

layout(location = 0) out vec3 vColor;
layout(location = 1) out float vSize;
layout(location = 2) out float vConfidence;
layout(location = 3) out vec3 vNormal;
layout(location = 4) out vec3 vViewDir;
layout(location = 5) out vec3 vLightDir;

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
    gl_Position = cam.proj * cam.view * vec4(inPosition, 1.0);

    vColor = inColor;
    vSize = clamp(inMagnitude * 3.0, 1.0, 12.0);
    vConfidence = inConfidence;
    vNormal = inNormal;
    vViewDir = inViewDir;
    vLightDir = inLightDir;

    gl_PointSize = vSize;
}