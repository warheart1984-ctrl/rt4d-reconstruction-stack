# Capability matrix

This matrix is the public source of truth for RT4D Reconstruction Stack
capabilities. A compiled shader or declared CLI option is not considered a
verified feature by itself.

| Capability | Status | Evidence and boundary |
|---|---|---|
| Vulkan renderer on AMD RX 480 | Verified | Native 1280x720 captures with zero observed validation messages |
| Multi-primitive GLB ingestion | Verified | Contract test covers 5 primitives, index rebasing, finite attributes, and buffer bounds |
| Source `TEXCOORD_0` ingestion | Verified import and execution | Sentinel reports source UV data on 5/5 primitives; the diagnostic fixture samples it while recording `usesSourceUv: true`; authorship remains unreviewed |
| Material-assignment provenance | Verified import and execution | Five primitive assignments become five indexed draw ranges; the source declares six material slots |
| Base-color factors | Verified execution | Factors from assigned glTF materials are pushed per draw and multiply the lit base color |
| Embedded base-color image decode | Verified bounded path | PNG through stb_image; the v0.3 proof uses one embedded 16x16 diagnostic PNG |
| External PNG/JPEG base-color image decode | Implemented, not fixture-proven | Relative asset paths are resolved and decoded through stb_image; no v0.3 contract fixture exercises this branch |
| Device-local texture upload | Verified execution | Host-visible staging buffer copies RGBA8 data into sampled `R8G8B8A8_SRGB` device-local images |
| Source base-color texture sampling | Verified diagnostic fixture | One source image is shared by six material descriptors and sampled using source `TEXCOORD_0` |
| Missing textured UV policy | Verified | Default is rejection; opt-in planar generation is labeled and counted separately |
| Temporal history | Verified static sequence | Color, depth, and normals persist across a three-frame run; camera-motion/disocclusion quality remains ungraded |
| Temporal reprojection shader | Verified execution | First frame rejects history; later frames consume prior-frame images |
| Confidence denoiser | Verified execution | Compute pass runs; image-quality calibration remains provisional |
| Classical 2x super-resolution | Verified execution | Compute output reaches tone-map and native PNG capture |
| Tone-map composite/readback | Verified | Native swapchain capture produces a non-black RGBA PNG |
| GPU timestamps | Verified basic | Two post-warm-up RT4D samples recorded; not yet a statistically meaningful benchmark |
| Source-build CI | Verified | GitHub Actions configures, compiles shaders/renderer, and runs the GLB contract test |
| Material look development | Provisional | Base color plus procedural lighting is visible; fixture texture, UVs, and material faces are not artist-reviewed |
| Full glTF PBR | Unavailable | Metallic/roughness, normal, emissive, occlusion, alpha modes, mip chains, and source sampler fidelity are not evaluated |
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
- `source texture` means the runtime decoded it from the selected asset. It does
  not by itself mean an artist authored or approved that image.
