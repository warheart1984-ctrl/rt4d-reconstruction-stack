#version 450

layout(location = 0) in vec3 vColor;
layout(location = 1) in float vSize;
layout(location = 2) in float vConfidence;
layout(location = 3) in vec3 vNormal;
layout(location = 4) in vec3 vViewDir;
layout(location = 5) in vec3 vLightDir;

layout(location = 0) out vec4 outColor;

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

layout(set = 1, binding = 0) uniform CelParams {
    vec3 baseColor;
    vec3 rimColor;
    float shadowSteps;
    float shadowSmoothness;
    float sssStrength;
    float rimPower;
    float blushStrength;
    vec3 blushColor;
    float specularIntensity;
    float specularSize;
} celParams;

float quantize(float x, float steps, float smoothness) {
    float q = floor(x * steps) / steps;
    return mix(q, x, smoothness);
}

void main() {
    vec2 uv = gl_PointCoord * 2.0 - 1.0;
    float r = length(uv);
    float alpha = smoothstep(1.0, 0.0, r);

    vec3 N = normalize(vNormal);
    vec3 L = normalize(vLightDir);
    vec3 V = normalize(vViewDir);

    float NdotL = max(dot(N, L), 0.0);
    float cel = quantize(NdotL, celParams.shadowSteps, celParams.shadowSmoothness);

    float sss = pow(max(dot(-L, N), 0.0), 1.5) * celParams.sssStrength;

    float rim = pow(1.0 - max(dot(V, N), 0.0), celParams.rimPower);
    vec3 rimLight = celParams.rimColor * rim;

    float blush = celParams.blushStrength * 0.5 * (1.0 - dot(V, N));
    vec3 baseShade = mix(celParams.baseColor, celParams.blushColor, blush) * (cel + sss);
    vec3 finalColor = baseShade + rimLight;

    float glow = mix(0.3, 1.5, vConfidence);
    finalColor *= glow;

    finalColor = clamp(finalColor, 0.0, 1.0);

    outColor = vec4(finalColor, alpha);
}