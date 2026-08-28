#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct GLTFMeshVertex {
    float position[3];
    float normal[3];
    float uv[2];
    float materialId;
};

struct GLTFPrimitiveRange {
    uint32_t firstIndex = 0;
    uint32_t indexCount = 0;
    uint32_t firstVertex = 0;
    uint32_t vertexCount = 0;
    int32_t materialIndex = -1;
    bool hasSourceUv = false;
};

class GLTFMeshScene {
public:
    bool load(const char* path);
    void clear();

    const std::vector<GLTFMeshVertex>& vertices() const { return vertices_; }
    const std::vector<uint32_t>& indices() const { return indices_; }
    const std::vector<GLTFPrimitiveRange>& primitives() const { return primitives_; }
    uint32_t vertexCount() const { return (uint32_t)vertices_.size(); }
    uint32_t indexCount() const { return (uint32_t)indices_.size(); }
    uint32_t primitiveCount() const { return (uint32_t)primitives_.size(); }
    uint32_t sourceUvPrimitiveCount() const { return sourceUvPrimitiveCount_; }
    uint32_t materialCount() const { return materialCount_; }
    bool usesAnySourceUv() const { return sourceUvPrimitiveCount_ > 0; }
    bool usesOnlySourceUv() const {
        return !primitives_.empty() && sourceUvPrimitiveCount_ == primitives_.size();
    }
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
    std::vector<GLTFPrimitiveRange> primitives_;
    uint32_t sourceUvPrimitiveCount_ = 0;
    uint32_t materialCount_ = 0;
};
