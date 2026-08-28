#version 450

layout(location = 0) in vec3 inPositionA;
layout(location = 1) in vec3 inPositionB;
layout(location = 0) out vec3 vEdgeColor;

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
    vec3 pos = (gl_VertexIndex % 2 == 0) ? inPositionA : inPositionB;
    gl_Position = cam.proj * cam.view * vec4(pos, 1.0);

    float t = float(gl_VertexIndex % 2);
    vec3 colA = vec3(0.4, 0.7, 1.0);
    vec3 colB = vec3(1.0, 0.5, 0.3);
    vEdgeColor = mix(colA, colB, t);
}
