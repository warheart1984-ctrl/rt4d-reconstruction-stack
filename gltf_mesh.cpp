#include "gltf_mesh.h"

#define CGLTF_IMPLEMENTATION
#define CGLTF_MALLOC malloc
#define CGLTF_FREE free
#include "cgltf.h"

#include <cmath>
#include <cstdio>
#include <cstring>

void GLTFMeshScene::clear() {
    vertices_.clear();
    indices_.clear();
    primitives_.clear();
    sourceUvPrimitiveCount_ = 0;
    materialCount_ = 0;
    name_.clear();
}

static void computeBounds(GLTFMeshScene& scene) {
    scene.boundsMin[0] = scene.boundsMin[1] = scene.boundsMin[2] = 1e9f;
    scene.boundsMax[0] = scene.boundsMax[1] = scene.boundsMax[2] = -1e9f;
    for (auto& v : scene.vertices()) {
        for (int i = 0; i < 3; i++) {
            if (v.position[i] < scene.boundsMin[i]) scene.boundsMin[i] = v.position[i];
            if (v.position[i] > scene.boundsMax[i]) scene.boundsMax[i] = v.position[i];
        }
    }
    for (int i = 0; i < 3; i++)
        scene.center[i] = 0.5f * (scene.boundsMin[i] + scene.boundsMax[i]);
    float dx = scene.boundsMax[0] - scene.boundsMin[0];
    float dy = scene.boundsMax[1] - scene.boundsMin[1];
    float dz = scene.boundsMax[2] - scene.boundsMin[2];
    scene.radius = 0.5f * std::sqrt(dx * dx + dy * dy + dz * dz);
    if (scene.radius < 1e-6f) scene.radius = 1.0f;
}

bool GLTFMeshScene::load(const char* path) {
    clear();

    cgltf_options opts{};
    cgltf_data* data = nullptr;
    cgltf_result res = cgltf_parse_file(&opts, path, &data);
    if (res != cgltf_result_success) {
        fprintf(stderr, "[GLTF] failed to parse %s (code %d)\n", path, (int)res);
        return false;
    }
    res = cgltf_load_buffers(&opts, data, path);
    if (res != cgltf_result_success) {
        fprintf(stderr, "[GLTF] failed to load buffers %s (code %d)\n", path, (int)res);
        cgltf_free(data);
        return false;
    }

    res = cgltf_validate(data);
    if (res != cgltf_result_success) {
        fprintf(stderr, "[GLTF] validation failed for %s (code %d)\n", path, (int)res);
        cgltf_free(data);
        return false;
    }

    if (data->meshes_count > 0 && data->meshes[0].name) name_ = data->meshes[0].name;
    materialCount_ = static_cast<uint32_t>(data->materials_count);

    for (size_t mi = 0; mi < data->meshes_count; mi++) {
        cgltf_mesh& mesh = data->meshes[mi];
        for (size_t pi = 0; pi < mesh.primitives_count; pi++) {
            cgltf_primitive& prim = mesh.primitives[pi];
            if (prim.type != cgltf_primitive_type_triangles) {
                fprintf(stderr, "[GLTF] skipping non-triangle primitive %zu in mesh %zu\n", pi, mi);
                continue;
            }

            const cgltf_accessor* posAcc = nullptr;
            const cgltf_accessor* normAcc = nullptr;
            const cgltf_accessor* uvAcc = nullptr;
            for (size_t ai = 0; ai < prim.attributes_count; ai++) {
                cgltf_attribute& attr = prim.attributes[ai];
                if (attr.type == cgltf_attribute_type_position) posAcc = attr.data;
                else if (attr.type == cgltf_attribute_type_normal) normAcc = attr.data;
                else if (attr.type == cgltf_attribute_type_texcoord && attr.index == 0) uvAcc = attr.data;
            }
            if (!posAcc) continue;

            size_t vcount = posAcc->count;
            size_t base = vertices_.size();
            vertices_.resize(base + vcount);

            const bool hasNormals = normAcc && normAcc->count == vcount;
            const bool hasSourceUv = uvAcc && uvAcc->count == vcount;
            int32_t materialIndex = -1;
            if (prim.material && data->materials_count > 0) {
                materialIndex = static_cast<int32_t>(prim.material - data->materials);
            }

            for (size_t i = 0; i < vcount; i++) {
                auto& v = vertices_[base + i];
                if (!cgltf_accessor_read_float(posAcc, i, v.position, 3)) {
                    fprintf(stderr, "[GLTF] failed reading position %zu in primitive %zu\n", i, pi);
                    cgltf_free(data);
                    clear();
                    return false;
                }

                if (!hasNormals || !cgltf_accessor_read_float(normAcc, i, v.normal, 3)) {
                    v.normal[0] = 0; v.normal[1] = 1; v.normal[2] = 0;
                }

                if (!hasSourceUv || !cgltf_accessor_read_float(uvAcc, i, v.uv, 2)) {
                    v.uv[0] = 0; v.uv[1] = 0;
                }
                v.materialId = materialIndex >= 0 ? static_cast<float>(materialIndex) : 0.0f;
            }

            const size_t ibase = indices_.size();
            if (prim.indices) {
                const cgltf_accessor* idx = prim.indices;
                const size_t icount = idx->count;
                indices_.resize(ibase + icount);
                for (size_t i = 0; i < icount; i++) {
                    const cgltf_size localIndex = cgltf_accessor_read_index(idx, i);
                    if (localIndex >= vcount) {
                        fprintf(stderr, "[GLTF] primitive %zu index %zu is out of range\n", pi, i);
                        cgltf_free(data);
                        clear();
                        return false;
                    }
                    indices_[ibase + i] = static_cast<uint32_t>(base + localIndex);
                }
            } else {
                indices_.resize(ibase + vcount);
                for (size_t i = 0; i < vcount; ++i)
                    indices_[ibase + i] = static_cast<uint32_t>(base + i);
            }

            GLTFPrimitiveRange range{};
            range.firstIndex = static_cast<uint32_t>(ibase);
            range.indexCount = static_cast<uint32_t>(indices_.size() - ibase);
            range.firstVertex = static_cast<uint32_t>(base);
            range.vertexCount = static_cast<uint32_t>(vcount);
            range.materialIndex = materialIndex;
            range.hasSourceUv = hasSourceUv;
            primitives_.push_back(range);
            if (hasSourceUv) ++sourceUvPrimitiveCount_;
        }
    }

    cgltf_free(data);

    if (vertices_.empty()) {
        fprintf(stderr, "[GLTF] no renderable mesh in %s\n", path);
        return false;
    }

    computeBounds(*this);
    return true;
}
