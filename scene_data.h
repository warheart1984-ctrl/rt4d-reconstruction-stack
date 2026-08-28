#pragma once

#include <cstdint>
#include <vector>
#include <cmath>

struct NodeData {
    float position[3];
    float color[3];
    float magnitude;
    float confidence;
    float normal[3];
    float viewDir[3];
    float lightDir[3];
};

struct EdgeData {
    uint32_t nodeA;
    uint32_t nodeB;
    float strength;
};

class LivingMapScene {
public:
    void generateDefault(uint32_t nodeCount = 1000);
void setCamera(float eyeX, float eyeY, float eyeZ,
               float lookX, float lookY, float lookZ,
               float fovX, float fovY, float aspect);
    void updateTime(float dt);

    const std::vector<NodeData>& nodes() const { return nodes_; }
    const std::vector<EdgeData>& edges() const { return edges_; }
    uint32_t nodeCount() const { return (uint32_t)nodes_.size(); }
    uint32_t edgeCount() const { return (uint32_t)edges_.size(); }

    float viewMatrix[16] = {};
    float projMatrix[16] = {};
    float camPos[3] = {};
    float time = 0.0f;

private:
    std::vector<NodeData> nodes_;
    std::vector<EdgeData> edges_;
};

class TacoScene {
public:
    void init();
    void setSunDirection(float x, float y, float z);
    void setLampPosition(float x, float y, float z);
    void setSkyColor(float r, float g, float b);
    void setLampColor(float r, float g, float b);

    float sunDir[3] = {-0.5f, -0.8f, -0.3f};
    float lampPos[3] = {3.0f, 2.0f, 1.0f};
    float skyColor[3] = {0.4f, 0.6f, 0.9f};
    float lampColor[3] = {1.0f, 0.85f, 0.6f};
    float modelMatrix[16] = {};
};

class BattleScene {
public:
    void init();
    void setTime(float t);

    float sunDir[3] = {0.3f, -0.7f, 0.5f};
    float fogColor[3] = {0.15f, 0.13f, 0.12f};
    float fogDensity = 0.05f;
    float time = 0.0f;
    float windDir[3] = {1.0f, 0.0f, 0.3f};
    float turbulence = 3.0f;
    float modelMatrix[16] = {};
};

class DragonHatchScene {
public:
    void init();
    void setHatchProgress(float t);

    float lightPos[3] = {2.0f, 3.0f, 1.0f};
    float viewPos[3] = {0.0f, 1.5f, -3.0f};
    float baseColor[3] = {0.15f, 0.1f, 0.08f};
    float sssColor[3] = {0.8f, 0.3f, 0.05f};
    float hatchProgress = 0.0f;
    float modelMatrix[16] = {};
};
