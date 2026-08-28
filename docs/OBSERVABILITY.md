# Observability and motion contract

v0.4 makes selected internal reconstruction state inspectable without treating
debug visualizations as source textures or quality scores.

## Deterministic motion

`--camera-sequence=orbit` derives every camera position from the integer frame
index and a fixed formula. It does not use wall-clock time. A configured
`--camera-cut-frame=N` starts a second deterministic segment with a fixed
angular offset.

`--sequence-dir=PATH` writes one 1280x720 composite PNG per frame. Screenshot
readback restores the swapchain image to `PRESENT_SRC_KHR`, so later frames do
not rely on an invalid leftover transfer layout.

## History lifecycle

History has an explicit decision for every frame:

- `initial_frame`: clear color/depth/normal history and reject it.
- `none`: consume the prior persisted history.
- `camera_cut`: clear existing history from its shader-read layout and reject it.
- `transactional_resize`: reserved reset reason for the future all-resource
  resize transaction; runtime resize remains unavailable in v0.4.

On an invalid-history frame, previous view-projection is set to the current
view-projection. That prevents an initial frame or camera cut from being
misreported as giant object motion.

## Exported images

`--observability-dir=PATH` exports the camera-cut frame and final frame at
640x360. Each directory includes a machine-readable `manifest.json` and:

- G-buffer color, device depth, encoded normal, and material ID.
- Screen-space motion in pixels.
- Reprojected color and reprojection confidence.
- Persisted color, depth, and normal history for the next frame.

The history filenames end in `-next` because the renderer persists the current
G-buffer at the end of each frame. They are not mislabeled as the history that
was consumed earlier in that same frame.

PNG mappings are visual diagnostics. Motion and confidence are foreground
masked. Motion uses `R=0.5+x/32`, `G=0.5+y/32`, and `B=magnitude/16`. Depth uses
`sqrt(1-deviceDepth)`. The numeric metrics are calculated from float readback
before these mappings.

## Metrics and timing

`--metrics=PATH` requires `--sequence-dir=PATH`, implies GPU timing, and records
every bounded frame:

- Foreground mean, p95, and maximum motion magnitude in pixels.
- Mean reprojection confidence and the ratio at or above 0.5.
- Mean absolute luma residual where reprojection confidence exceeds 0.01.
- Composite-frame mean absolute luma delta.
- Raw GPU frame samples plus interpolated p50, p95, and p99 summaries.
- A second timing summary that explicitly excludes two warm-up frames when the
  run contains more than two samples.

These are observed diagnostics, not perceptual-quality scores or pass/fail
thresholds. Static geometry and a deterministic camera path do not establish
production motion quality, disocclusion quality, or cross-driver image parity.
