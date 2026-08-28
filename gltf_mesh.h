#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct GLTFMeshVertex {
    float position[3];
    float normal[3];
    float uv[2];
};

class GLTFMeshScene {
public:
    bool load(const char* path);
    void clear();

    const std::vector<GLTFMeshVertex>& vertices() const { return vertices_; }
    const std::vector<uint32_t>& indices() const { return indices_; }
    uint32_t vertexCount() const { return (uint32_t)vertices_.size(); }
    uint32_t indexCount() const { return (uint32_t)indices_.size(); }
    const char* name() const { return name_.c_str(); }

    float modelMatrix[16] = {};
    float boundsMin[3] = {0, 0, 0};
    float boundsMax[3] = {0, 0, 0};
    float center[3] = {0, 0, 0};
    float radius = 1.0f;

private:
    std::string name_;
    std::vector<GLTFMeshVertex> vertices_;
    std::vector<uint32_t> indices_;
};
