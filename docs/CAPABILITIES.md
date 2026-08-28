# Capability matrix

This matrix is the public source of truth for RT4D Reconstruction Stack
capabilities. A compiled shader or declared CLI option is not considered a
verified feature by itself.

| Capability | Status | Evidence and boundary |
|---|---|---|
| Vulkan renderer on AMD RX 480 | Verified | Native 1280x720 captures with zero observed validation messages |
| Multi-primitive GLB ingestion | Verified | Contract test covers 5 primitives, index rebasing, finite attributes, and buffer bounds |
| Source `TEXCOORD_0` ingestion | Verified import | Sentinel reports source UV data on 5/5 primitives; authorship is unreviewed and texture sampling is not connected |
| Material-assignment provenance | Verified import | Five primitive assignments are preserved as per-vertex G-buffer IDs; the source declares six material slots, with no source-material evaluation yet |
| Temporal history | Verified static sequence | Color, depth, and normals persist across a three-frame run; camera-motion/disocclusion quality remains ungraded |
| Temporal reprojection shader | Verified execution | First frame rejects history; later frames consume prior-frame images |
| Confidence denoiser | Verified execution | Compute pass runs; image-quality calibration remains provisional |
| Classical 2x super-resolution | Verified execution | Compute output reaches tone-map and native PNG capture |
| Tone-map composite/readback | Verified | Native swapchain capture produces a non-black RGBA PNG |
| GPU timestamps | Verified basic | Two post-warm-up RT4D samples recorded; not yet a statistically meaningful benchmark |
| Source-build CI | Verified | GitHub Actions configures, compiles shaders/renderer, and runs the GLB contract test |
| Geometry-based materials | Provisional | Procedural lighting from geometry normals; not artist-reviewed |
| Authored textures/PBR materials | Unavailable | Material and texture payloads are not sampled by the current renderer |
| G-buffer/debug visualizers | Unavailable | Unsupported flags are rejected instead of silently accepted |
| Window/swapchain resize | Unavailable | Fixed-size rendering prevents unsafe partial resource recreation |
| ML super-resolution | Declared source only | Shader/weight-loader experiments exist but are not connected or proven |
| Frame generation | Unavailable | No implementation or claim |
| NVIDIA DLSS | Not applicable | This is a custom Vulkan reconstruction stack, not NVIDIA DLSS |

## Evidence policy

- Runtime receipts identify the exact asset, binary, shaders, GPU, frame count,
  output hash, and limitations.
- Cross-driver tests use structural invariants rather than requiring identical
  PNG hashes.
- Generated UVs, fixture art, and geometry-based shading must never be labeled
  authored or artist-approved.
