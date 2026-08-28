#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="$SCRIPT_DIR"
OUTPUT_DIR="$SCRIPT_DIR/spirv_rasterize"

mkdir -p "$OUTPUT_DIR"

if ! command -v glslc &>/dev/null; then
    echo "ERROR: glslc not found"
    exit 1
fi

echo "=== Compiling Rasterization Shaders ==="

compile() {
    local src="$1"
    local stage="$2"
    local name
    name=$(basename "$src" .glsl)
    local out="$OUTPUT_DIR/${name}.spv"
    echo "  $name ($stage) -> $out"
    glslc -fshader-stage="$stage" "$src" -o "$out"
}

for f in "$INPUT_DIR"/*.vert.glsl; do
    [ -f "$f" ] || continue
    compile "$f" vertex
done

for f in "$INPUT_DIR"/*.frag.glsl; do
    [ -f "$f" ] || continue
    compile "$f" fragment
done

for f in "$INPUT_DIR"/*.comp.glsl; do
    [ -f "$f" ] || continue
    compile "$f" compute
done

RECON_DIR="$INPUT_DIR/dlss45/shaders"
if [ -d "$RECON_DIR" ]; then
    echo "=== Compiling RT4D Reconstruction Stack shaders ==="
    for f in "$RECON_DIR"/*.vert.glsl; do
        [ -f "$f" ] || continue
        compile "$f" vertex
    done
    for f in "$RECON_DIR"/*.frag.glsl; do
        [ -f "$f" ] || continue
        compile "$f" fragment
    done
    for f in "$RECON_DIR"/*.comp.glsl; do
        [ -f "$f" ] || continue
        compile "$f" compute
    done
fi

echo ""
echo "=== Done: $(ls "$OUTPUT_DIR"/*.spv 2>/dev/null | wc -l) shaders compiled ==="
ls -la "$OUTPUT_DIR"/*.spv 2>/dev/null
