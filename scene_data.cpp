#include "scene_data.h"
#include <cmath>
#include <cstdlib>
#include <algorithm>

static const float PI = 3.14159265f;

static void lookAtMatrix(float* m, float ex, float ey, float ez,
                          float lx, float ly, float lz,
                          float ux, float uy, float uz) {
    float fx = lx - ex, fy = ly - ey, fz = lz - ez;
    float fl = sqrtf(fx*fx + fy*fy + fz*fz);
    fx /= fl; fy /= fl; fz /= fl;

    // This is the 3D graphics camera used by the GPU raster path.  A prior
    // pseudo-4D basis crossed the view vector with a zero XYZ axis, yielding a
    // zero-length side vector and NaNs.  Keep the RT4D naming at the pass
    // level, but use a stable 3D look-at basis for the actual raster camera.
    float sx = fy * uz - fz * uy;
    float sy = fz * ux - fx * uz;
    float sz = fx * uy - fy * ux;
    float sl = sqrtf(sx*sx + sy*sy + sz*sz);
    sx /= sl; sy /= sl; sz /= sl;
    float ux2 = fy*sz - fz*sy, uy2 = fz*sx - fx*sz, uz2 = fx*sy - fy*sx;

    m[0]=sx;  m[1]=ux2; m[2]=-fx;  m[3]=0;
    m[4]=sy;  m[5]=uy2; m[6]=-fy;  m[7]=0;
    m[8]=sz;  m[9]=uz2; m[10]=-fz; m[11]=0;
    m[12]=-(sx*ex+sy*ey+sz*ez);
    m[13]=-(ux2*ex+uy2*ey+uz2*ez);
    m[14]= fx*ex+fy*ey+fz*ez;
    m[15]=1;
}

static void perspectiveMatrix(float* m, float fovX, float fovY, float aspect, float near, float far) {
    float fX = 1.0f / tanf(fovX * PI / 360.0f);
    float fY = 1.0f / tanf(fovY * PI / 360.0f);
    float rangeInv = 1.0f / (near - far);
    for (int i = 0; i < 16; i++) m[i] = 0;
    m[0] = fX / aspect;
    m[5] = fY;
    m[10] = (near + far) * rangeInv;
    m[11] = -1.0f;
    m[14] = 2.0f * near * far * rangeInv;
}

void LivingMapScene::generateDefault(uint32_t nodeCount) {
    nodes_.resize(nodeCount);
    edges_.clear();

    srand(42);
    for (uint32_t i = 0; i < nodeCount; i++) {
        float angle = (float)i / (float)nodeCount * 2.0f * PI;
        float r = 0.5f + 0.5f * ((float)rand() / RAND_MAX);

        nodes_[i].position[0] = cosf(angle) * r;
        nodes_[i].position[1] = ((float)rand() / RAND_MAX - 0.5f) * 0.4f;
        nodes_[i].position[2] = sinf(angle) * r;

        float hue = (float)i / (float)nodeCount;
        float sat = 0.7f + 0.3f * ((float)rand() / RAND_MAX);
        float val = 0.8f + 0.2f * ((float)rand() / RAND_MAX);
        float c = val * sat;
        float x = c * (1.0f - fabsf(fmodf(hue * 6.0f, 2.0f) - 1.0f));
        float m = val - c;
        if (hue < 1.0f/6.0f)      { nodes_[i].color[0]=c; nodes_[i].color[1]=x; nodes_[i].color[2]=0; }
        else if (hue < 2.0f/6.0f) { nodes_[i].color[0]=x; nodes_[i].color[1]=c; nodes_[i].color[2]=0; }
        else if (hue < 3.0f/6.0f) { nodes_[i].color[0]=0; nodes_[i].color[1]=c; nodes_[i].color[2]=x; }
        else if (hue < 4.0f/6.0f) { nodes_[i].color[0]=0; nodes_[i].color[1]=x; nodes_[i].color[2]=c; }
        else if (hue < 5.0f/6.0f) { nodes_[i].color[0]=x; nodes_[i].color[1]=0; nodes_[i].color[2]=c; }
        else                      { nodes_[i].color[0]=c; nodes_[i].color[1]=0; nodes_[i].color[2]=x; }
        nodes_[i].color[0] += m;
        nodes_[i].color[1] += m;
        nodes_[i].color[2] += m;

        nodes_[i].magnitude = 0.3f + 0.7f * ((float)rand() / RAND_MAX);
        nodes_[i].confidence = 0.2f + 0.8f * ((float)rand() / RAND_MAX);

        // Initializeonormal, view direction, light direction
        // Normal: random on hemisphere facing up
        float nx = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        float ny = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        float nz = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        float nlen = sqrtf(nx*nx + ny*ny + nz*nz);
        nodes_[i].normal[0] = nx / nlen;
        nodes_[i].normal[1] = ny / nlen;
        nodes_[i].normal[2] = nz / nlen;

        // View direction: from camera at (0,1.5,3) looking at origin
        nodes_[i].viewDir[0] = -0.5f;
        nodes_[i].viewDir[1] = 1.0f - 3.0f;
        nodes_[i].viewDir[2] = 3.0f;
        float vlen = sqrtf(nodes_[i].viewDir[0]*nodes_[i].viewDir[0] +
                           nodes_[i].viewDir[1]*nodes_[i].viewDir[1] +
                           nodes_[i].viewDir[2]*nodes_[i].viewDir[2]);
        nodes_[i].viewDir[0] /= vlen;
        nodes_[i].viewDir[1] /= vlen;
        nodes_[i].viewDir[2] /= vlen;

        // Light direction: from above and in front
        nodes_[i].lightDir[0] = 0.5f;
        nodes_[i].lightDir[1] = 0.5f;
        nodes_[i].lightDir[2] = 0.8f;
        float vlen2 = sqrtf(nodes_[i].lightDir[0]*nodes_[i].lightDir[0] +
                            nodes_[i].lightDir[1]*nodes_[i].lightDir[1] +
                            nodes_[i].lightDir[2]*nodes_[i].lightDir[2]);
        nodes_[i].lightDir[0] /= vlen2;
        nodes_[i].lightDir[1] /= vlen2;
        nodes_[i].lightDir[2] /= vlen2;
    }

    for (uint32_t i = 0; i < nodeCount; i++) {
        uint32_t connections = 1 + rand() % 3;
        for (uint32_t c = 0; c < connections; c++) {
            uint32_t j = rand() % nodeCount;
            if (j != i) {
                float dx = nodes_[i].position[0] - nodes_[j].position[0];
                float dy = nodes_[i].position[1] - nodes_[j].position[1];
                float dz = nodes_[i].position[2] - nodes_[j].position[2];
                float dist = sqrtf(dx*dx + dy*dy + dz*dz);
                if (dist < 0.8f) {
                    edges_.push_back({i, j, 1.0f - dist});
                }
            }
        }
    }
}

void LivingMapScene::setCamera(float eyeX, float eyeY, float eyeZ,
                                 float lookX, float lookY, float lookZ,
                                 float fovX, float fovY, float aspect) {
    camPos[0] = eyeX; camPos[1] = eyeY; camPos[2] = eyeZ;
    lookAtMatrix(viewMatrix, eyeX, eyeY, eyeZ, lookX, lookY, lookZ, 0, 1, 0);
    perspectiveMatrix(projMatrix, fovX, fovY, aspect, 0.1f, 100.0f);
}

void LivingMapScene::updateTime(float dt) {
    time += dt;
}

void TacoScene::init() {
    for (int i = 0; i < 16; i++) modelMatrix[i] = (i % 5 == 0) ? 1.0f : 0.0f;
}

void TacoScene::setSunDirection(float x, float y, float z) {
    sunDir[0] = x; sunDir[1] = y; sunDir[2] = z;
}

void TacoScene::setLampPosition(float x, float y, float z) {
    lampPos[0] = x; lampPos[1] = y; lampPos[2] = z;
}

void TacoScene::setSkyColor(float r, float g, float b) {
    skyColor[0] = r; skyColor[1] = g; skyColor[2] = b;
}

void TacoScene::setLampColor(float r, float g, float b) {
    lampColor[0] = r; lampColor[1] = g; lampColor[2] = b;
}

void BattleScene::init() {
    for (int i = 0; i < 16; i++) modelMatrix[i] = (i % 5 == 0) ? 1.0f : 0.0f;
}

void BattleScene::setTime(float t) {
    time = t;
}

void DragonHatchScene::init() {
    for (int i = 0; i < 16; i++) modelMatrix[i] = (i % 5 == 0) ? 1.0f : 0.0f;
}

void DragonHatchScene::setHatchProgress(float t) {
    hatchProgress = std::max(0.0f, std::min(1.0f, t));
}
