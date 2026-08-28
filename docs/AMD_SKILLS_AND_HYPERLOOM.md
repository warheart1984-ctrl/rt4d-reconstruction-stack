# AMD Skills and Hyperloom integration boundary

This document separates the installable Agent Skill package from unimplemented
runtime services and from ROCm Hyperloom.

## Current install path

AMD's public catalog uses the standardized Agent Skills layout: a skill folder
containing `SKILL.md`, `skill-card.md`, and `evals/evals.json`. The supported
public installer is the `skills` CLI invoked through `npx`, not an
`amd-skills install` command.

Install this repository's source package directly:

```bash
npx skills add warheart1984-ctrl/rt4d-reconstruction-stack \
  --skill rt4d-reconstruction-stack \
  --agent codex
```

The source skill lives at `.agents/skills/rt4d-reconstruction-stack/`. This is
not an official `amd/skills` catalog listing.

## Why there is no root skill.yaml

No published AMD Skills schema was found for the proposed `skill.yaml` or
`skill.json` fields. In particular, a manifest must not advertise routes or
flags that the program does not implement. The current executable is:

```text
build-rt4d-recon/mandala_rasterize
```

The scene is positional. The implemented output interface is
`--capture`, `--sequence-dir`, `--observability-dir`, `--metrics`, and
`--debug=gpu-timer`. There is no HTTP server, `/rt4d/*` API, `--capture-seq`,
`--dump-gbuffer`, `--profile`, `--scene=recon`, or `rt4d_recon` executable.

Adding an HTTP wrapper could be a future product feature, but describing one in
a manifest would not make it exist or prove its security boundary.

## Official AMD catalog boundary

AMD's current contribution policy accepts federated submissions from AMD-owned
product repositories. This repository is not under an AMD GitHub organization.
It also intentionally grants no project-wide redistribution license, whereas a
catalog contribution needs an explicit license and governance owner.

Therefore the honest current status is:

- Direct installation from this repository: packaged, subject to the
  repository's current license boundary.
- Official AMD catalog installation: unavailable.
- `amd-skills install rt4d-reconstruction-stack`: not a documented command.

If ownership or policy changes, an official submission would still need AMD
eligibility, an explicit license, target-hardware end-to-end evaluation, and a
federation entry maintained through AMD's catalog workflow.

## Hyperloom boundary

ROCm Hyperloom currently optimizes LLM inference workloads and kernels through
its own optimizer, profiling, and session contracts. RT4D is a native Vulkan
graphics renderer, not an LLM serving workload. No validated Hyperloom adapter
for this renderer exists, and the Agent Skill package does not create one.

A future experimental bridge should require all of the following before it is
called an integration:

1. A non-interactive RT4D workload command with a fixed asset, camera sequence,
   frame count, warm-up policy, and output directory.
2. A correctness gate that compares structural invariants, capture completeness,
   history-reset behavior, and validation output before accepting performance.
3. A profiler path that actually understands the Vulkan workload on the target
   AMD driver and GPU; LLM-oriented traces are not interchangeable evidence.
4. A bounded optimization objective using raw samples and percentiles, not a
   single FPS headline.
5. A session receipt containing source commit, binary and shader hashes,
   adapter and driver metadata, candidate changes, outputs, and rollback state.
6. Human review before retaining shader or synchronization changes that alter
   image semantics.

Until those conditions are implemented and run, describe Hyperloom as adjacent
future optimization tooling rather than part of the RT4D runtime.

## Security and evidence notes

- The renderer does not require outbound network access after source and build
  dependencies are present.
- GPU and display-device access cannot be represented truthfully as a generic
  `sandboxed: true` boolean; the host or container must expose a Vulkan-capable
  surface and the relevant device nodes.
- Keep new evidence in a caller-chosen directory and verify every PNG before
  publishing a receipt.
- Preserve source-UV, generated-UV, material-assignment, and artist-review fields
  independently.

## Upstream references

- [AMD Skills repository](https://github.com/amd/skills)
- [AMD Skills requirements](https://github.com/amd/skills/blob/main/docs/skill-requirements.md)
- [AMD Skills contribution policy](https://github.com/amd/skills/blob/main/CONTRIBUTING.md)
- [ROCm Hyperloom](https://github.com/AMD-AGI/Hyperloom)
