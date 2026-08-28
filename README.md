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
7. Typed observability exports, motion sequences, and temporal metrics

This project is **not NVIDIA DLSS**. Its current super-resolution path is
classical compute; ML super-resolution and frame generation are not proven.

Project status: proof-stage native GPU stack with verified material, motion,
history-reset, and observability captures plus portable source-build CI. It is
not yet a production renderer or an artist-approved material pipeline.

See [the capability matrix](docs/CAPABILITIES.md) for the exact boundary
between verified, partial, and unavailable features, and [the roadmap](docs/ROADMAP.md)
for the order in which the stack is intended to grow.

## Verified proof

The verified v0.4 proof is a deterministic camera-orbit sequence with an
explicit camera cut. It exports final composites, low-resolution internal
images, per-frame temporal diagnostics, and full-sample plus post-warm-up GPU
timing percentiles on an AMD Radeon RX 480 using RADV. Exact source, binary,
shader, sequence, export, and capture hashes are in:

- `receipts/rt4d-reconstruction-stack-v0.4-receipt.json`
- `receipts/armored-sentinel-motion-v0.4.run.log`
- `receipts/armored-sentinel-motion-v0.4-metrics.json`

Material status is `provisional_source_materials`: the reconstruction path now
evaluates glTF base-color factors and base-color textures using five primitive
draw ranges. The v0.3 texture is generated diagnostic fixture data embedded in
the GLB; it is source-provided to the runtime but is not artist-authored or
artist-reviewed look development. Metallic/roughness, alpha modes, emissive,
normal maps, and full PBR remain outside this release.

See [the observability contract](docs/OBSERVABILITY.md) for visualization and
metric semantics, and [the material policy](docs/MATERIAL_POLICY.md) for the
source-UV and generated-planar boundary. The material classification remains
provisional; v0.4 does not add artist review or full PBR.

The v0.4.1 maintenance proof also exercises every scene selector and hardens
PNG publication. Taco, Battle, and Dragon now produce visible shader-path
diagnostics using the Sentinel mesh as provisional geometry; they are not
authored Taco, battle, or dragon assets. Captures are written to a `.partial`
file, checked for a complete PNG envelope, and only then published. See
`receipts/rt4d-reconstruction-stack-v0.4.1-receipt.json` and
`receipts/six-scene-v0.4.1/`.

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
  --frames=12 \
  --camera-sequence=orbit \
  --camera-cut-frame=6 \
  --sequence-dir=receipts/armored-sentinel-motion-v0.4-sequence \
  --observability-dir=receipts/armored-sentinel-observability-v0.4 \
  --metrics=receipts/armored-sentinel-motion-v0.4-metrics.json \
  --capture=receipts/armored-sentinel-motion-v0.4-final.png
```

The sequence uses frame-indexed camera positions rather than wall-clock time.
Frame zero and the camera-cut frame explicitly reject and clear history. The
observability directory receives cut and final keyframes; the metrics receipt
contains camera samples, motion, confidence, residual, composite-delta, and GPU
timing percentiles. `--metrics` requires `--sequence-dir` so composite deltas
cannot silently become unobserved zeroes. Capture frame counts are bounded to
1–600.

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

The current window is intentionally fixed at 1280x720. Camera cuts use the same
typed reset decision reserved for a future transactional resize, but swapchain,
render-pass, pipeline, export, and history recreation is not yet implemented.

## License

No project-wide license is granted at this stage. Public visibility does not
grant permission to copy, modify, or redistribute the project. Vendored
third-party files retain the license terms and notices contained in those files.
