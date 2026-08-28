# Material and UV policy

This policy keeps runtime provenance separate from creative approval.

## Texture provenance

`source texture` means an image was referenced by the selected glTF asset,
decoded, uploaded through a staging buffer, and sampled by the reconstruction
path. It does not mean the image was made or approved by an artist.

The v0.3 checker is deterministic generated diagnostic data embedded in
`fixtures/armored-sentinel-textured-v0.3.glb`. Its receipt fields are:

- `textureOrigin: generated_diagnostic_checker`
- `usesSourceUv: true`
- `textureArtistReviewed: false`
- `uvArtistReviewed: false`
- `materialAssignmentArtistReviewed: false`

## UV origin

Every primitive has exactly one runtime label:

- `source_texcoord_0`: decoded from the selected asset.
- `generated_planar`: generated from the two largest primitive bounds axes.
- `missing`: no UV data and no texture requires it.

Textured primitives without `TEXCOORD_0` fail by default. Planar generation is
available only through `GeneratePlanarLabeled` or the explicit CLI flag
`--missing-uv=generate-planar-labeled`. It is never renamed to authored UVs.

## Rendering boundary

The v0.3 reconstruction G-buffer evaluates an sRGB base-color texture multiplied
by the glTF base-color factor, then applies the existing procedural lighting.
Per-primitive indexed draw ranges select material descriptors and factors.

This is not full glTF PBR. Metallic/roughness values are preserved by the CPU
loader but not evaluated by the shader. Alpha modes, mip chains, normal maps,
emissive, occlusion, texture transforms, nonzero texture-coordinate sets, data
URIs, KTX/Basis, and source sampler fidelity remain unsupported.
