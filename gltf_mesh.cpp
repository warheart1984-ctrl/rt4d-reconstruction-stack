#include "gltf_mesh.h"

#define CGLTF_IMPLEMENTATION
#define CGLTF_MALLOC malloc
#define CGLTF_FREE free
#include "cgltf.h"

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
#endif
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

void GLTFMeshScene::clear() {
    vertices_.clear();
    indices_.clear();
    primitives_.clear();
    materials_.clear();
    textures_.clear();
    sourceUvPrimitiveCount_ = 0;
    generatedUvPrimitiveCount_ = 0;
    materialCount_ = 0;
    decodedTextureCount_ = 0;
    sourceTextureMaterialCount_ = 0;
    name_.clear();
}

static std::string parentDirectory(const char* path) {
    const std::string input(path ? path : "");
    const size_t slash = input.find_last_of("/\\");
    return slash == std::string::npos ? std::string(".") : input.substr(0, slash);
}

static bool decodeTexture(const cgltf_texture& texture, const char* assetPath,
                          GLTFTextureData& output) {
    const cgltf_image* image = texture.image;
    if (!image) return false;

    output.name = texture.name ? texture.name : (image->name ? image->name : "");
    output.mimeType = image->mime_type ? image->mime_type : "";

    int width = 0;
    int height = 0;
    int channels = 0;
    stbi_uc* decoded = nullptr;

    if (image->buffer_view && image->buffer_view->buffer &&
        image->buffer_view->buffer->data) {
        const cgltf_buffer_view* view = image->buffer_view;
        if (view->size > static_cast<cgltf_size>(std::numeric_limits<int>::max()))
            return false;
        const auto* bytes = static_cast<const stbi_uc*>(view->buffer->data) + view->offset;
        decoded = stbi_load_from_memory(bytes, static_cast<int>(view->size),
                                        &width, &height, &channels, 4);
        output.embedded = true;
        output.sourceUri = "embedded-buffer-view";
    } else if (image->uri && std::strncmp(image->uri, "data:", 5) != 0) {
        output.sourceUri = image->uri;
        const std::string fullPath = parentDirectory(assetPath) + "/" + image->uri;
        decoded = stbi_load(fullPath.c_str(), &width, &height, &channels, 4);
    } else {
        fprintf(stderr, "[GLTF] data-URI and extension-only textures are not supported\n");
        return false;
    }

    if (!decoded || width <= 0 || height <= 0) {
        if (decoded) stbi_image_free(decoded);
        return false;
    }

    output.width = static_cast<uint32_t>(width);
    output.height = static_cast<uint32_t>(height);
    output.rgba.assign(decoded, decoded + static_cast<size_t>(width) * height * 4);
    stbi_image_free(decoded);
    return true;
}

static void generatePlanarUv(std::vector<GLTFMeshVertex>& vertices,
                             size_t base, size_t count) {
    std::array<float, 3> minv = {vertices[base].position[0],
                                 vertices[base].position[1],
                                 vertices[base].position[2]};
    std::array<float, 3> maxv = minv;
    for (size_t i = 1; i < count; ++i) {
        for (size_t axis = 0; axis < 3; ++axis) {
            minv[axis] = std::min(minv[axis], vertices[base + i].position[axis]);
            maxv[axis] = std::max(maxv[axis], vertices[base + i].position[axis]);
        }
    }

    std::array<size_t, 3> axes = {0, 1, 2};
    std::sort(axes.begin(), axes.end(), [&](size_t a, size_t b) {
        return (maxv[a] - minv[a]) > (maxv[b] - minv[b]);
    });
    for (size_t i = 0; i < count; ++i) {
        for (size_t uvAxis = 0; uvAxis < 2; ++uvAxis) {
            const size_t axis = axes[uvAxis];
            const float extent = maxv[axis] - minv[axis];
            vertices[base + i].uv[uvAxis] = extent > 1e-8f
                ? (vertices[base + i].position[axis] - minv[axis]) / extent
                : 0.5f;
        }
    }
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

bool GLTFMeshScene::load(const char* path, const GLTFLoadOptions& options) {
    clear();
    if (!path || !path[0]) {
        fprintf(stderr, "[GLTF] asset path is empty\n");
        return false;
    }
    const char* assetPath = path;

    cgltf_options opts{};
    cgltf_data* data = nullptr;
    cgltf_result res = cgltf_parse_file(&opts, assetPath, &data);
    if (res != cgltf_result_success) {
        fprintf(stderr, "[GLTF] failed to parse %s (code %d)\n", assetPath, (int)res);
        return false;
    }
    res = cgltf_load_buffers(&opts, data, assetPath);
    if (res != cgltf_result_success) {
        fprintf(stderr, "[GLTF] failed to load buffers %s (code %d)\n", assetPath, (int)res);
        cgltf_free(data);
        return false;
    }

    res = cgltf_validate(data);
    if (res != cgltf_result_success) {
        fprintf(stderr, "[GLTF] validation failed for %s (code %d)\n", assetPath, (int)res);
        cgltf_free(data);
        return false;
    }

    if (data->meshes_count > 0 && data->meshes[0].name) name_ = data->meshes[0].name;
    materialCount_ = static_cast<uint32_t>(data->materials_count);

    textures_.resize(data->textures_count);
    for (size_t i = 0; i < data->textures_count; ++i) {
        if (decodeTexture(data->textures[i], assetPath, textures_[i])) {
            ++decodedTextureCount_;
        } else {
            fprintf(stderr, "[GLTF] failed to decode base texture %zu in %s\n", i, assetPath);
        }
    }

    materials_.resize(data->materials_count);
    for (size_t i = 0; i < data->materials_count; ++i) {
        const cgltf_material& source = data->materials[i];
        GLTFMaterialData& material = materials_[i];
        material.name = source.name ? source.name : "";
        if (source.has_pbr_metallic_roughness) {
            const auto& pbr = source.pbr_metallic_roughness;
            std::copy(pbr.base_color_factor, pbr.base_color_factor + 4,
                      material.baseColorFactor);
            material.metallicFactor = pbr.metallic_factor;
            material.roughnessFactor = pbr.roughness_factor;
            if (pbr.base_color_texture.texture) {
                const ptrdiff_t textureIndex =
                    pbr.base_color_texture.texture - data->textures;
                if (textureIndex < 0 ||
                    static_cast<size_t>(textureIndex) >= textures_.size() ||
                    !textures_[static_cast<size_t>(textureIndex)].decoded()) {
                    fprintf(stderr, "[GLTF] material %zu references an undecodable base texture\n", i);
                    cgltf_free(data);
                    clear();
                    return false;
                }
                material.baseColorTextureIndex = static_cast<int32_t>(textureIndex);
                material.baseColorTexcoord = pbr.base_color_texture.texcoord;
                ++sourceTextureMaterialCount_;
            }
        }
    }

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
            const bool materialUsesTexture = materialIndex >= 0 &&
                static_cast<size_t>(materialIndex) < materials_.size() &&
                materials_[static_cast<size_t>(materialIndex)].baseColorTextureIndex >= 0;
            if (materialUsesTexture &&
                materials_[static_cast<size_t>(materialIndex)].baseColorTexcoord != 0) {
                fprintf(stderr, "[GLTF] textured primitive %zu requires unsupported TEXCOORD_%d\n",
                        pi, materials_[static_cast<size_t>(materialIndex)].baseColorTexcoord);
                cgltf_free(data);
                clear();
                return false;
            }
            const bool generateUv = materialUsesTexture && !hasSourceUv &&
                options.missingUvPolicy == GLTFMissingUvPolicy::GeneratePlanarLabeled;
            if (materialUsesTexture && !hasSourceUv && !generateUv) {
                fprintf(stderr, "[GLTF] textured primitive %zu has no TEXCOORD_0; "
                                "explicit generated-UV policy required\n", pi);
                cgltf_free(data);
                clear();
                return false;
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
            if (generateUv) generatePlanarUv(vertices_, base, vcount);

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
            range.hasGeneratedUv = generateUv;
            range.uvOrigin = hasSourceUv ? GLTFUvOrigin::SourceTexcoord0
                : (generateUv ? GLTFUvOrigin::GeneratedPlanar : GLTFUvOrigin::Missing);
            primitives_.push_back(range);
            if (hasSourceUv) ++sourceUvPrimitiveCount_;
            if (generateUv) ++generatedUvPrimitiveCount_;
        }
    }

    cgltf_free(data);

    if (vertices_.empty()) {
        fprintf(stderr, "[GLTF] no renderable mesh in %s\n", assetPath);
        return false;
    }

    computeBounds(*this);
    return true;
}
