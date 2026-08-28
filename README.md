# RT4D Reconstruction Stack

[![Build](https://github.com/warheart1984-ctrl/rt4d-reconstruction-stack/actions/workflows/build.yml/badge.svg)](https://github.com/warheart1984-ctrl/rt4d-reconstruction-stack/actions/workflows/build.yml)

A custom Vulkan reconstruction pipeline for native GLB geometry on AMD GPUs.
The current proof runs `armored-sentinel-v1.glb` through:

1. GLB decoding and GPU vertex/index upload
2. Low-resolution G-buffer generation
3. Temporal reprojection
4. Confidence-weighted denoising
5. Classical super-resolution
6. Tone-map composition and PNG readback

This project is **not NVIDIA DLSS**. Its current super-resolution path is
classical compute; ML super-resolution and frame generation are not proven.

Project status: proof-stage native GPU stack with a verified local capture and
portable source-build CI. It is not yet a production renderer or an artist-
approved material pipeline.

See [the capability matrix](docs/CAPABILITIES.md) for the exact boundary
between verified, partial, and unavailable features, and [the roadmap](docs/ROADMAP.md)
for the order in which the stack is intended to grow.

## Verified proof

The verified 1280x720 reconstruction capture was produced on an AMD Radeon RX
480 using RADV. Exact source, binary, asset, shader, and capture hashes are in:

- `receipts/rt4d-reconstruction-stack-v0.2-receipt.json`
- `receipts/armored-sentinel-rt4d-reconstruction-v0.2.run.log`
- `receipts/armored-sentinel-rt4d-reconstruction-v0.2.png`

Material status is `provisional_geometry_based`: the render uses decoded
geometry normals with procedural Lambertian lighting. It is not artist-reviewed
look development. Source-provided `TEXCOORD_0` coordinates and five primitive
material assignments are preserved; the source declares six material slots.
UV authorship has not been independently reviewed, and source textures and
material parameters are not rendered yet.

## Build

Requirements: CMake 3.20+, a C++17 compiler, Vulkan development files, GLFW 3,
and `glslc`.

```bash
cmake -S . -B build-rt4d-recon
cmake --build build-rt4d-recon -j2
```

## Run the proof

Run from the repository root so shader and fixture paths resolve correctly:

```bash
./build-rt4d-recon/mandala_rasterize recon \
  --frames=3 \
  --capture=receipts/armored-sentinel-rt4d-reconstruction-v0.2.png \
  --debug=gpu-timer
```

`--frames=3` exercises persisted color, depth, and normal history. The first
frame explicitly rejects history; each completed frame supplies history to the
next. Capture frame counts are bounded to 1–600.

For the baseline raster path:

```bash
./build-rt4d-recon/mandala_rasterize sentinel \
  --frames=2 \
  --capture=receipts/armored-sentinel-raster-v0.2.png \
  --debug=gpu-timer
```

The GLB fixture is intentionally kept in this repository because it is small
and is part of the reproducible proof contract.

The current window is intentionally fixed at 1280x720. Transactional resize
across swapchain, render passes, pipelines, and temporal history is not yet
implemented.

## License

No project-wide license is granted at this stage. Public visibility does not
grant permission to copy, modify, or redistribute the project. Vendored
third-party files retain the license terms and notices contained in those files.
