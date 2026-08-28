# RT4D Reconstruction Stack

[![Build](https://github.com/warheart1984-ctrl/rt4d-reconstruction-stack/actions/workflows/build.yml/badge.svg)](https://github.com/warheart1984-ctrl/rt4d-reconstruction-stack/actions/workflows/build.yml)

A custom Vulkan reconstruction pipeline for native GLB geometry on AMD GPUs.
The current proof runs `armored-sentinel-v1.glb` through:

1. GLB decoding and GPU vertex/index upload
2. Base-color factor and texture ingestion
3. Device-local texture upload through a staging buffer
4. Per-material low-resolution G-buffer draws
5. Temporal reprojection and confidence-weighted denoising
6. Classical super-resolution, tone-map composition, and PNG readback

This project is **not NVIDIA DLSS**. Its current super-resolution path is
classical compute; ML super-resolution and frame generation are not proven.

Project status: proof-stage native GPU stack with verified material and temporal
captures plus portable source-build CI. It is not yet a production renderer or
an artist-approved material pipeline.

See [the capability matrix](docs/CAPABILITIES.md) for the exact boundary
between verified, partial, and unavailable features, and [the roadmap](docs/ROADMAP.md)
for the order in which the stack is intended to grow.

## Verified proof

The verified v0.3 1280x720 material reconstruction capture was produced on an
AMD Radeon RX 480 using RADV. Exact source, binary, fixture, shader, and capture
hashes are in:

- `receipts/rt4d-reconstruction-stack-v0.3-receipt.json`
- `receipts/armored-sentinel-material-reconstruction-v0.3.run.log`
- `receipts/armored-sentinel-material-reconstruction-v0.3.png`

Material status is `provisional_source_materials`: the reconstruction path now
evaluates glTF base-color factors and base-color textures using five primitive
draw ranges. The v0.3 texture is generated diagnostic fixture data embedded in
the GLB; it is source-provided to the runtime but is not artist-authored or
artist-reviewed look development. Metallic/roughness, alpha modes, emissive,
normal maps, and full PBR remain outside this release.

See [the material policy](docs/MATERIAL_POLICY.md) for the source-UV and
generated-planar boundary. The released v0.2 baseline remains available as a
[tagged release](https://github.com/warheart1984-ctrl/rt4d-reconstruction-stack/releases/tag/v0.2).

## Build

Requirements: CMake 3.20+, a C++17 compiler, Vulkan development files, GLFW 3,
`glslc`, and Python 3 when tests are enabled.

```bash
cmake -S . -B build-rt4d-recon
cmake --build build-rt4d-recon -j2
```

## Run the proof

Run from the repository root so shader and fixture paths resolve correctly:

```bash
./build-rt4d-recon/mandala_rasterize recon \
  --asset=fixtures/armored-sentinel-textured-v0.3.glb \
  --frames=3 \
  --capture=receipts/armored-sentinel-material-reconstruction-v0.3.png \
  --debug=gpu-timer
```

`--frames=3` exercises persisted color, depth, and normal history. The first
frame explicitly rejects history; each completed frame supplies history to the
next. Capture frame counts are bounded to 1–600.

For the factor-only comparison using the released Sentinel without an image:

```bash
./build-rt4d-recon/mandala_rasterize recon \
  --asset=armored-sentinel-v1.glb \
  --frames=3 \
  --capture=receipts/armored-sentinel-material-factors-v0.3.png \
  --debug=gpu-timer
```

Textured primitives without `TEXCOORD_0` are rejected by default. The explicit
diagnostic fallback is `--missing-uv=generate-planar-labeled`; generated UVs
are counted and logged separately from source UVs.

The GLB fixtures are intentionally kept in this repository because they are
small and form part of the reproducible proof contract.

The current window is intentionally fixed at 1280x720. Transactional resize
across swapchain, render passes, pipelines, and temporal history is not yet
implemented.

## License

No project-wide license is granted at this stage. Public visibility does not
grant permission to copy, modify, or redistribute the project. Vendored
third-party files retain the license terms and notices contained in those files.
