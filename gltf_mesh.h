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

enum class GLTFMissingUvPolicy : uint8_t {
    RejectTexturedPrimitive = 0,
    GeneratePlanarLabeled = 1,
};

enum class GLTFUvOrigin : uint8_t {
    Missing = 0,
    SourceTexcoord0 = 1,
    GeneratedPlanar = 2,
};

struct GLTFLoadOptions {
    GLTFMissingUvPolicy missingUvPolicy =
        GLTFMissingUvPolicy::RejectTexturedPrimitive;
};

struct GLTFTextureData {
    std::string name;
    std::string mimeType;
    std::string sourceUri;
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<uint8_t> rgba;
    bool embedded = false;

    bool decoded() const {
        return width > 0 && height > 0 &&
               rgba.size() == static_cast<size_t>(width) * height * 4;
    }
};

struct GLTFMaterialData {
    std::string name;
    float baseColorFactor[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float metallicFactor = 1.0f;
    float roughnessFactor = 1.0f;
    int32_t baseColorTextureIndex = -1;
    int32_t baseColorTexcoord = 0;
};

struct GLTFPrimitiveRange {
    uint32_t firstIndex = 0;
    uint32_t indexCount = 0;
    uint32_t firstVertex = 0;
    uint32_t vertexCount = 0;
    int32_t materialIndex = -1;
    bool hasSourceUv = false;
    bool hasGeneratedUv = false;
    GLTFUvOrigin uvOrigin = GLTFUvOrigin::Missing;
};

class GLTFMeshScene {
public:
    bool load(const char* path, const GLTFLoadOptions& options = {});
    void clear();

    const std::vector<GLTFMeshVertex>& vertices() const { return vertices_; }
    const std::vector<uint32_t>& indices() const { return indices_; }
    const std::vector<GLTFPrimitiveRange>& primitives() const { return primitives_; }
    const std::vector<GLTFMaterialData>& materials() const { return materials_; }
    const std::vector<GLTFTextureData>& textures() const { return textures_; }
    uint32_t vertexCount() const { return (uint32_t)vertices_.size(); }
    uint32_t indexCount() const { return (uint32_t)indices_.size(); }
    uint32_t primitiveCount() const { return (uint32_t)primitives_.size(); }
    uint32_t sourceUvPrimitiveCount() const { return sourceUvPrimitiveCount_; }
    uint32_t generatedUvPrimitiveCount() const { return generatedUvPrimitiveCount_; }
    uint32_t materialCount() const { return materialCount_; }
    uint32_t decodedTextureCount() const { return decodedTextureCount_; }
    uint32_t sourceTextureMaterialCount() const { return sourceTextureMaterialCount_; }
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
    std::vector<GLTFMaterialData> materials_;
    std::vector<GLTFTextureData> textures_;
    uint32_t sourceUvPrimitiveCount_ = 0;
    uint32_t generatedUvPrimitiveCount_ = 0;
    uint32_t materialCount_ = 0;
    uint32_t decodedTextureCount_ = 0;
    uint32_t sourceTextureMaterialCount_ = 0;
};
