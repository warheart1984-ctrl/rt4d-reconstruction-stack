#version 450

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform AtmosphereUBO {
    float time;
    float _pad0;
    float _pad1;
    float _pad2;
    vec3 fogColor;
    float density;
    vec3 windDir;
    float turbulence;
} atm;

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);

    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));

    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float fbm(vec2 p) {
    float v = 0.0;
    float a = 0.5;
    for (int i = 0; i < 4; i++) {
        v += a * noise(p);
        p *= 2.0;
        a *= 0.5;
    }
    return v;
}

void main() {
    float height = vUV.y;

    float fog = smoothstep(0.2, 0.9, height);

    vec2 windUV = vUV * 8.0 + atm.windDir.xz * atm.time * 0.3;
    float smoke = fbm(windUV * atm.turbulence);

    smoke = fog * (0.5 + 0.5 * smoke);

    float flicker = 0.95 + 0.05 * sin(atm.time * 3.7 + vUV.x * 12.0);

    vec3 color = mix(vec3(0.08, 0.08, 0.1), atm.fogColor, smoke * flicker);

    outColor = vec4(color, 1.0);
}
