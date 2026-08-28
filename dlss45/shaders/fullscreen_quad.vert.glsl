#version 450
layout(location = 0) out vec2 vUV;

void main() {
    // Fullscreen triangle (no vertex buffer).
    vec2 pos[3] = vec2[](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    vec2 uv = pos[gl_VertexIndex] * 0.5 + 0.5;
    vUV = uv;
    gl_Position = vec4(pos[gl_VertexIndex], 0.0, 1.0);
}
