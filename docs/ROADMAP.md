# Stability-first roadmap

## v0.2 — Asset and temporal foundation

- Multi-primitive index rebasing and draw-range provenance
- Source-UV and material-assignment provenance, without claiming artist review
- Color/depth/normal temporal history with first-frame invalidation
- Multi-frame capture, GPU timestamps, and strict Vulkan failure propagation
- CPU asset-contract test in CI
- Truthful rejection of unsupported debugging and resize paths

## v0.3 — Material fidelity

- Device-local staging upload for base-color textures
- Per-material draw ranges and base-color factors
- Source-texture fixture with `usesSourceUv` and separate artist-review receipt fields
- Explicit fallback policy for missing UVs; generated planar UVs remain labeled

## v0.4 — Observability and motion

- G-buffer, motion, reprojection-confidence, and history exports
- Deterministic camera-motion capture sequences
- History reset on camera cuts and future transactional resize
- Temporal stability metrics and percentile GPU timing receipts

## v0.5 — Automation and portability

- Offscreen/headless Vulkan capture path
- Optional GPU-runner validation lane
- Broader glTF accessor/material fixtures and fuzzed malformed-input tests
- Release artifacts with source/binary provenance

## Separate experiments

Living Map, WorldDocument, telemetry glyphs, Taco, Battle, Dragon, BHPV, and ML
super-resolution remain separate until each has an isolated contract, licensed
inputs, tests, and a receipt that does not expand the claims of the core stack.
