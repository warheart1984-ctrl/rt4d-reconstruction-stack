#include "gltf_mesh.h"

#include <cmath>
#include <cstdio>

namespace {
int fail(const char* message) {
    std::fprintf(stderr, "[gltf-test] FAIL: %s\n", message);
    return 1;
}

int validateMaterialFactors(const GLTFMeshScene& scene) {
    constexpr float expected[6][4] = {
        {0.72f, 0.52f, 0.42f, 1.0f},
        {0.035f, 0.04f, 0.06f, 1.0f},
        {0.28f, 0.30f, 0.34f, 1.0f},
        {0.22f, 0.12f, 0.08f, 1.0f},
        {0.12f, 0.16f, 0.28f, 1.0f},
        {0.08f, 0.25f, 0.30f, 1.0f},
    };
    if (scene.materials().size() != 6)
        return fail("unexpected material factor count");
    for (size_t material = 0; material < scene.materials().size(); ++material)
        for (size_t channel = 0; channel < 4; ++channel)
            if (std::abs(scene.materials()[material].baseColorFactor[channel] -
                         expected[material][channel]) > 1e-6f)
                return fail("base-color factor was not preserved");
    return 0;
}

int validateGeometry(const GLTFMeshScene& scene, bool expectSourceUv,
                     bool expectGeneratedUv) {
    if (scene.vertexCount() != 2210) return fail("unexpected Sentinel vertex count");
    if (scene.indexCount() != 2160) return fail("unexpected Sentinel index count");
    if (scene.primitiveCount() != 5) return fail("unexpected Sentinel primitive count");
    if (scene.sourceUvPrimitiveCount() != (expectSourceUv ? 5u : 0u))
        return fail("unexpected source UV primitive count");
    if (scene.generatedUvPrimitiveCount() != (expectGeneratedUv ? 5u : 0u))
        return fail("unexpected generated UV primitive count");
    if (scene.materialCount() != 6) return fail("unexpected declared material count");

    uint32_t coveredIndices = 0;
    for (const auto& primitive : scene.primitives()) {
        if (primitive.firstIndex != coveredIndices)
            return fail("primitive index ranges are not contiguous");
        if (primitive.firstVertex + primitive.vertexCount > scene.vertexCount())
            return fail("primitive vertex range exceeds the vertex buffer");
        if (primitive.firstIndex + primitive.indexCount > scene.indexCount())
            return fail("primitive index range exceeds the index buffer");
        if (primitive.hasSourceUv != expectSourceUv)
            return fail("primitive source UV label is incorrect");
        if (primitive.hasGeneratedUv != expectGeneratedUv)
            return fail("primitive generated UV label is incorrect");
        const GLTFUvOrigin expectedOrigin = expectSourceUv
            ? GLTFUvOrigin::SourceTexcoord0 : GLTFUvOrigin::GeneratedPlanar;
        if (primitive.uvOrigin != expectedOrigin)
            return fail("primitive UV origin is incorrect");
        if (primitive.materialIndex < 0 ||
            static_cast<uint32_t>(primitive.materialIndex) >= scene.materialCount())
            return fail("primitive material assignment is out of range");

        for (uint32_t i = primitive.firstVertex;
             i < primitive.firstVertex + primitive.vertexCount; ++i) {
            if (scene.vertices()[i].materialId !=
                static_cast<float>(primitive.materialIndex))
                return fail("primitive material assignment was not preserved per vertex");
        }

        for (uint32_t i = primitive.firstIndex;
             i < primitive.firstIndex + primitive.indexCount; ++i) {
            const uint32_t index = scene.indices()[i];
            if (index < primitive.firstVertex ||
                index >= primitive.firstVertex + primitive.vertexCount)
                return fail("primitive-local index was not rebased correctly");
        }
        coveredIndices += primitive.indexCount;
    }
    if (coveredIndices != scene.indexCount())
        return fail("primitive ranges do not cover the index buffer");

    for (const auto& vertex : scene.vertices()) {
        for (float value : vertex.position)
            if (!std::isfinite(value)) return fail("non-finite position");
        for (float value : vertex.normal)
            if (!std::isfinite(value)) return fail("non-finite normal");
        for (float value : vertex.uv)
            if (!std::isfinite(value)) return fail("non-finite UV");
        if (!std::isfinite(vertex.materialId)) return fail("non-finite material ID");
    }
    return 0;
}
}

int main(int argc, char** argv) {
    if (argc != 4)
        return fail("expected base, textured, and missing-UV GLB fixture paths");

    GLTFMeshScene base;
    if (!base.load(argv[1])) return fail("base fixture did not load");
    if (validateGeometry(base, true, false)) return 1;
    if (validateMaterialFactors(base)) return 1;
    if (base.decodedTextureCount() != 0 || base.sourceTextureMaterialCount() != 0)
        return fail("base fixture unexpectedly contains textures");
    if (std::abs(base.materials()[2].metallicFactor - 1.0f) > 1e-6f)
        return fail("base metallic factor was not preserved");

    GLTFMeshScene textured;
    if (!textured.load(argv[2])) return fail("textured fixture did not load");
    if (validateGeometry(textured, true, false)) return 1;
    if (validateMaterialFactors(textured)) return 1;
    if (textured.decodedTextureCount() != 1 ||
        textured.sourceTextureMaterialCount() != 6 ||
        textured.textures().size() != 1 || !textured.textures()[0].decoded() ||
        textured.textures()[0].width != 16 || textured.textures()[0].height != 16 ||
        !textured.textures()[0].embedded)
        return fail("embedded source texture contract failed");

    GLTFMeshScene missingUv;
    if (missingUv.load(argv[3]))
        return fail("default policy accepted a textured primitive without TEXCOORD_0");
    GLTFLoadOptions generatedPolicy{};
    generatedPolicy.missingUvPolicy = GLTFMissingUvPolicy::GeneratePlanarLabeled;
    if (!missingUv.load(argv[3], generatedPolicy))
        return fail("explicit generated-planar policy failed");
    if (validateGeometry(missingUv, false, true)) return 1;
    if (validateMaterialFactors(missingUv)) return 1;

    std::fprintf(stderr,
                 "[gltf-test] PASS: factors, one embedded texture, five source-UV "
                 "ranges, default rejection, and five labeled planar fallbacks\n");
    return 0;
}
