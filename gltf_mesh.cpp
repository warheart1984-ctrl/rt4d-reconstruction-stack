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

    if (data->meshes_count > 0 && data->meshes[0].name) name_ = data->meshes[0].name;

    for (size_t mi = 0; mi < data->meshes_count; mi++) {
        cgltf_mesh& mesh = data->meshes[mi];
        for (size_t pi = 0; pi < mesh.primitives_count; pi++) {
            cgltf_primitive& prim = mesh.primitives[pi];

            const cgltf_accessor* posAcc = nullptr;
            const cgltf_accessor* normAcc = nullptr;
            const cgltf_accessor* uvAcc = nullptr;
            for (size_t ai = 0; ai < prim.attributes_count; ai++) {
                cgltf_attribute& attr = prim.attributes[ai];
                if (attr.type == cgltf_attribute_type_position) posAcc = attr.data;
                else if (attr.type == cgltf_attribute_type_normal) normAcc = attr.data;
                else if (attr.type == cgltf_attribute_type_texcoord) uvAcc = attr.data;
            }
            if (!posAcc) continue;

            size_t vcount = posAcc->count;
            size_t base = vertices_.size();
            vertices_.resize(base + vcount);

            // cgltf offsets are byte offsets. Keep these as byte pointers until
            // each attribute value is read; adding them to float pointers scaled
            // every non-zero GLB offset by four and produced invalid geometry.
            const char* posData = (const char*)posAcc->buffer_view->buffer->data
                + posAcc->buffer_view->offset + posAcc->offset;
            const char* normData = nullptr;
            const char* uvData = nullptr;
            cgltf_size normStride = 0, posStride = 0, uvStride = 0;
            posStride = posAcc->stride;
            if (normAcc && normAcc->buffer_view) {
                normData = (const char*)normAcc->buffer_view->buffer->data
                    + normAcc->buffer_view->offset + normAcc->offset;
                normStride = normAcc->stride;
            }
            if (uvAcc && uvAcc->buffer_view) {
                uvData = (const char*)uvAcc->buffer_view->buffer->data
                    + uvAcc->buffer_view->offset + uvAcc->offset;
                uvStride = uvAcc->stride;
            }

            for (size_t i = 0; i < vcount; i++) {
                auto& v = vertices_[base + i];
                const char* p = posData + i * posStride;
                v.position[0] = *(const float*)(p);
                v.position[1] = *(const float*)(p + 4);
                v.position[2] = *(const float*)(p + 8);

                if (normData) {
                    const char* n = normData + i * normStride;
                    v.normal[0] = *(const float*)(n);
                    v.normal[1] = *(const float*)(n + 4);
                    v.normal[2] = *(const float*)(n + 8);
                } else {
                    v.normal[0] = 0; v.normal[1] = 1; v.normal[2] = 0;
                }

                if (uvData) {
                    const char* u = uvData + i * uvStride;
                    v.uv[0] = *(const float*)(u);
                    v.uv[1] = *(const float*)(u + 4);
                } else {
                    v.uv[0] = 0; v.uv[1] = 0;
                }
            }

            if (prim.indices) {
                cgltf_accessor* idx = prim.indices;
                size_t icount = idx->count;
                size_t ibase = indices_.size();
                indices_.resize(ibase + icount);
                const char* idata = (const char*)idx->buffer_view->buffer->data
                    + idx->buffer_view->offset + idx->offset;
                cgltf_size istride = idx->stride;
                cgltf_component_type ctype = idx->component_type;
                for (size_t i = 0; i < icount; i++) {
                    const char* p = idata + i * istride;
                    if (ctype == cgltf_component_type_r_16u)
                        indices_[ibase + i] = *(const uint16_t*)p;
                    else if (ctype == cgltf_component_type_r_32u)
                        indices_[ibase + i] = *(const uint32_t*)p;
                    else if (ctype == cgltf_component_type_r_8u)
                        indices_[ibase + i] = *(const uint8_t*)p;
                    else
                        indices_[ibase + i] = *(const uint16_t*)p;
                }
            }
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
