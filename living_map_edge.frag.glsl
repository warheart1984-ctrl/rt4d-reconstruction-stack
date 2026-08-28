#version 450

layout(location = 0) in vec3 vEdgeColor;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(vEdgeColor, 0.6);
}
