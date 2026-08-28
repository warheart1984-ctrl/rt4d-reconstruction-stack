# RT4D Reconstruction Stack — v1.0

Canonical design document for the RT4D Reconstruction Stack pipeline.
**Hyper-Denoiser (RR) + Sovereign Super-Resolution (SR) + Temporal Frame Weave (FG).**

Phase 1 implements the classical path (temporal reprojection -> denoiser -> edge-aware
super-resolution -> tone-map composite). Phase 2 swaps the SR compute for a small ONNX/ML
model; Phase 3 adds frame generation.

---

## 1. Pass graph

```
[ RT4D base render (LR G-buffer) ]
    |  ColorLDR_LR, Depth_LR, Normals_LR, Motion_LR, MaterialID_LR, NoiseMeta_LR
    v
[ Compute: Temporal Reprojection ]
    |  (reads ColorHistory_LR, DepthHistory_LR)
    |  -> ReprojectedColor_LR, ReprojectionConfidence_LR
    v
[ Compute: Denoiser / Ray Reconstruction ]
    |  -> ColorDenoised_LR, DenoiseConfidence_LR
    v
[ Compute: Super-Resolution LR -> HR ]
    |  (classic now; ML later) -> ColorSR_HR, Depth_HR, Motion_HR
    v
[ Graphics: Tone Map + Composite ]
    v
[ Swapchain Present ]
```

## 2. Buffer contract

### Low-res G-buffers (renderWidth x renderHeight)
| Name              | Format              | Semantics                                  |
|-------------------|---------------------|--------------------------------------------|
| ColorLDR_LR       | RGBA16F             | raw radiance, pre tone-map                 |
| Depth_LR          | R32F                | linear depth from camera                   |
| Normals_LR        | RGBA16F             | xyz = normal, w = roughness                |
| MotionVectors_LR  | RG16F               | screen-space motion (px/frame)             |
| MaterialID_LR     | R16UI               | index into material table                  |
| NoiseMeta_LR      | R16F                | variance / sample-count hint               |

### History
ColorHistory_LR (RGBA16F), DepthHistory_LR (R32F) — same formats as G-buffer.

### Reconstruction
ReprojectedColor_LR (RGBA16F), ReprojectionConfidence_LR (R16F),
ColorDenoised_LR (RGBA16F), DenoiseConfidence_LR (R16F).

### High-res outputs (displayWidth x displayHeight)
ColorSR_HR (RGBA16F), Depth_HR (R32F), MotionVectors_HR (RG16F).

## 3. Confidence heuristics

Per-pixel inputs (LR space): C_t (current), C_{t-1} (history), D_t, D_{t-1}, N_t, N_{t-1},
M_t (motion), V_t (noise), Mat_t (material, optional).

- Depth consistency:  c_d = exp( -|D_t - D_{t-1}| / sigma_d ),  sigma_d in [0.01, 0.05]
- Normal consistency: c_n = max(0, N_t dot N_{t-1})
- Motion validity:    c_m = 1 if ||M_t|| < m_max else 0   (m_max ~ 32 px/frame)
- Noise penalty:      c_v = 1 - V_t  (normalized variance)

Combined:  c_hist = c_d^w_d * c_n^w_n * c_m^w_m * c_v^w_v
Weights:   w_d = 1.0, w_n = 1.0, w_m = 1.0, w_v = 0.5; clamped to [0,1].

Blend:  w_hist = c_hist,  w_curr = 1 - w_hist
Denoised:  C_denoised = w_curr*C_t + w_hist*C_{t-1}
Store DenoiseConfidence_LR = w_hist.

Special cases: disocclusion -> w_hist = 0; specular/emissive material -> w_hist *= 0.5;
high depth gradient (edge) -> w_hist *= 0.7 (avoid haloing).

## 4. ML SR model (Phase 2)

Inputs: ColorDenoised_LR(3), Depth_LR(1), Normals_LR(3), MotionVectors_LR(2),
DenoiseConfidence_LR(1) = 10 channels. Output: s x s HR RGB patch (pixel shuffle).
UNet-lite: Conv(10->32,3x3,ReLU), Conv(32->32), Conv(32->64), Conv(64->3*s^2), PixelShuffle(s).
Loss: L = L1 color + lambda_edge * |grad| L1 + lambda_temp * temporal warp L1.
Export ONNX; deploy via TensorRT or pure Vulkan compute (weights in SSBO).

## 5. Integration order
1. G-buffer contract (this doc) + LR/HR images.
2. Temporal reprojection compute.
3. Denoiser (RR) with confidence heuristics.
4. Classic SR, then swap to ML SR compute.
5. Tone-map composite.
6. Frame generation (Phase 3).
