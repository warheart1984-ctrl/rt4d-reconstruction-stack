# Material fixtures

`armored-sentinel-textured-v0.3.glb` is generated deterministically from the
released `armored-sentinel-v1.glb`. It embeds a 16x16 diagnostic checker and
connects that texture to each existing base-color material. The checker is
generated test data, not artist-authored or artist-reviewed artwork.
Its fixture metadata records `usesSourceUv: true`; texture, UV, and material
assignment artist-review fields remain separate and false.

`armored-sentinel-textured-missing-uv-v0.3.glb` removes `TEXCOORD_0` from the
same primitives. The normal loader must reject it; tests may load it only with
the explicit `GeneratePlanarLabeled` policy and must record generated UVs as
generated.
Its fixture metadata records `usesSourceUv: false`.

Regenerate both fixtures from the repository root with:

```bash
python3 tools/generate_material_fixtures.py
```
