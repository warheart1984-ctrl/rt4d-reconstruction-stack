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

## Verified proof

The verified 1280x720 reconstruction capture was produced on an AMD Radeon RX
480 using RADV. Exact source, binary, asset, shader, and capture hashes are in:

- `receipts/rt4d-reconstruction-stack-build-receipt.json`
- `receipts/armored-sentinel-rt4d-reconstruction.run.log`
- `receipts/armored-sentinel-rt4d-reconstruction.png`

Material status is `provisional_geometry_based`: the render uses decoded
geometry normals with procedural Lambertian lighting. It is not artist-reviewed
look development and does not claim authored texture or UV fidelity.

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
  --capture=receipts/armored-sentinel-rt4d-reconstruction.png
```

For the baseline raster path:

```bash
./build-rt4d-recon/mandala_rasterize sentinel \
  --capture=receipts/armored-sentinel-raster.png
```

The GLB fixture is intentionally kept in this repository because it is small
and is part of the reproducible proof contract.

## License

No project-wide license is granted at this stage. Public visibility does not
grant permission to copy, modify, or redistribute the project. Vendored
third-party files retain the license terms and notices contained in those files.
