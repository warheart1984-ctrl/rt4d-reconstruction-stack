#include "gltf_mesh.h"

#include <cmath>
#include <cstdio>

namespace {
int fail(const char* message) {
    std::fprintf(stderr, "[gltf-test] FAIL: %s\n", message);
    return 1;
}
}

int main(int argc, char** argv) {
    if (argc != 2) return fail("expected a GLB fixture path");

    GLTFMeshScene scene;
    if (!scene.load(argv[1])) return fail("fixture did not load");
    if (scene.vertexCount() != 2210) return fail("unexpected Sentinel vertex count");
    if (scene.indexCount() != 2160) return fail("unexpected Sentinel index count");
    if (scene.primitiveCount() != 5) return fail("unexpected Sentinel primitive count");
    if (scene.sourceUvPrimitiveCount() != 5)
        return fail("unexpected source UV primitive count");
    if (scene.materialCount() != 6)
        return fail("unexpected declared material count");

    uint32_t coveredIndices = 0;
    for (const auto& primitive : scene.primitives()) {
        if (primitive.firstIndex != coveredIndices)
            return fail("primitive index ranges are not contiguous");
        if (primitive.firstVertex + primitive.vertexCount > scene.vertexCount())
            return fail("primitive vertex range exceeds the vertex buffer");
        if (primitive.firstIndex + primitive.indexCount > scene.indexCount())
            return fail("primitive index range exceeds the index buffer");
        if (!primitive.hasSourceUv)
            return fail("Sentinel primitive lacks source TEXCOORD_0 data");
        if (primitive.materialIndex < 0 ||
            static_cast<uint32_t>(primitive.materialIndex) >= scene.materialCount())
            return fail("primitive material assignment is out of range");

        for (uint32_t i = primitive.firstVertex;
             i < primitive.firstVertex + primitive.vertexCount; ++i) {
            if (scene.vertices()[i].materialId !=
                static_cast<float>(primitive.materialIndex)) {
                return fail("primitive material assignment was not preserved per vertex");
            }
        }

        for (uint32_t i = primitive.firstIndex;
             i < primitive.firstIndex + primitive.indexCount; ++i) {
            const uint32_t index = scene.indices()[i];
            if (index < primitive.firstVertex ||
                index >= primitive.firstVertex + primitive.vertexCount) {
                return fail("primitive-local index was not rebased correctly");
            }
        }
        coveredIndices += primitive.indexCount;
    }
    if (coveredIndices != scene.indexCount())
        return fail("primitive ranges do not cover the index buffer");

    for (const auto& vertex : scene.vertices()) {
        for (float value : vertex.position) if (!std::isfinite(value)) return fail("non-finite position");
        for (float value : vertex.normal) if (!std::isfinite(value)) return fail("non-finite normal");
        for (float value : vertex.uv) if (!std::isfinite(value)) return fail("non-finite UV");
        if (!std::isfinite(vertex.materialId)) return fail("non-finite material ID");
    }

    std::fprintf(stderr,
                 "[gltf-test] PASS: %u vertices, %u indices, %u primitives, "
                 "%u source-UV primitives, %u declared materials\n",
                 scene.vertexCount(), scene.indexCount(), scene.primitiveCount(),
                 scene.sourceUvPrimitiveCount(), scene.materialCount());
    return 0;
}
