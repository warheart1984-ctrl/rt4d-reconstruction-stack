---
name: rt4d-reconstruction-stack
description: Build, run, diagnose, and verify the RT4D Reconstruction Stack native Vulkan renderer and its GLB, temporal-history, observability, GPU-timing, PNG-capture, and provenance receipts. Use when a user asks to run RT4D scenes, prove an asset on the GPU, inspect G-buffer or temporal outputs, diagnose a blank capture, or distinguish provisional geometry-based materials from artist-reviewed work.
---

# RT4D Reconstruction Stack

Operate the native Vulkan proof from the repository root. Ground every claim in
the checked-out source, a completed command, and the resulting files.

## Start safely

1. Run `git status --short` and preserve unrelated changes.
2. Confirm the executable target and CLI from `CMakeLists.txt` and
   `./build-rt4d-recon/mandala_rasterize --help`; do not infer flags from older
   notes.
3. Read `docs/CAPABILITIES.md`, `docs/MATERIAL_POLICY.md`, and, for temporal
   work, `docs/OBSERVABILITY.md`.
4. Check `vulkaninfo --summary` when available. Capture mode still creates a
   GLFW Vulkan surface and therefore needs a working display or compositor.
5. Do not install packages, overwrite captures, commit, push, or publish unless
   the user authorizes that action.

## Build and test

Use a source build with tests enabled:

```bash
cmake -S . -B build-rt4d-recon \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON
cmake --build build-rt4d-recon --parallel 2
ctest --test-dir build-rt4d-recon --output-on-failure
```

Requirements are CMake 3.20+, a C++17 compiler, Vulkan development files,
GLFW 3, `glslc`, and Python 3 for tests. ROCm is not required: this is a Vulkan
renderer. The live proof is currently bounded to AMD Radeon RX 480 with RADV;
do not generalize that observation to all AMD GPUs or drivers.

## Use only the supported CLI

The executable is `build-rt4d-recon/mandala_rasterize`. The scene is a
positional argument:

```text
living-map | taco | battle | dragon | sentinel | recon
```

Supported output flags are:

- `--capture=PATH`: write the final 1280x720 PNG.
- `--frames=1..600`: render a bounded number of frames.
- `--sequence-dir=PATH`: write one composite PNG per frame in `recon` mode.
- `--observability-dir=PATH`: export selected G-buffer, motion, confidence,
  reprojection, and next-history diagnostics in `recon` mode.
- `--metrics=PATH`: write temporal metrics and GPU timing percentiles; requires
  `--sequence-dir` and implies GPU timing.
- `--debug=gpu-timer`: print supported GPU timing samples.

Do not claim `--capture-seq`, `--dump-gbuffer`, `--profile`, `--scene=recon`, an
HTTP API, or `/rt4d/*` routes. They are not implemented.

## Run a bounded smoke proof

Choose a new output path, then run:

```bash
./build-rt4d-recon/mandala_rasterize sentinel \
  --frames=1 \
  --debug=gpu-timer \
  --capture=<output-directory>/sentinel.png
python3 .agents/skills/rt4d-reconstruction-stack/scripts/verify_capture.py \
  <output-directory>/sentinel.png
```

Treat success as all of the following: process exit zero, `[capture] OK`, a
complete PNG, a reported adapter, and no Vulkan warning or error. Report a
failure exactly; do not substitute an older capture.

## Run the temporal observability proof

Use `recon` for motion, history, observability, and metrics:

```bash
./build-rt4d-recon/mandala_rasterize recon \
  --asset=fixtures/armored-sentinel-textured-v0.3.glb \
  --frames=12 \
  --camera-sequence=orbit \
  --camera-cut-frame=6 \
  --sequence-dir=<output-directory>/sequence \
  --observability-dir=<output-directory>/observability \
  --metrics=<output-directory>/metrics.json \
  --capture=<output-directory>/final.png
```

Verify the final capture and every sequence PNG with `scripts/verify_capture.py`.
Check that frame zero and the camera-cut frame reset history, and report timing
as a bounded observation rather than a general benchmark.

## Asset and material truth boundary

- Record the asset path, hash, vertex count, index count, primitive count,
  material draw ranges, source-UV count, generated-UV count, and missing-UV
  policy from the actual run.
- Call imported draw ranges, base-color factors, diagnostic texture sampling,
  and generated shading `geometry-based provisional materials`.
- Keep `textureArtistReviewed`, `uvArtistReviewed`,
  `materialAssignmentArtistReviewed`, and `lookDevelopmentArtistReviewed`
  separate and false unless a human artist explicitly reviews them.
- Never relabel `generated_planar` UVs as source-authored UVs.
- Taco, Battle, and Dragon currently use Sentinel geometry to prove their
  shader paths. They are not authored Taco, battle, or dragon assets.
- This is not NVIDIA DLSS. Classical super-resolution is runtime-proven; ML
  super-resolution and frame generation are not.

## Receipt

For a completed run, report:

- source commit or tag and executable SHA-256;
- exact command and exit code;
- GPU adapter, driver, dimensions, frame count, and observed validation count;
- output paths, byte counts, and SHA-256 values;
- history-reset and timing evidence when applicable;
- provisional-material fields separately from artist-review fields;
- unsupported paths and proof limitations.

Read [the AMD Skills and Hyperloom boundary](../../../docs/AMD_SKILLS_AND_HYPERLOOM.md)
before claiming catalog publication or Hyperloom execution.
