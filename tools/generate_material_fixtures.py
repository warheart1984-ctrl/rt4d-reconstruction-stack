#!/usr/bin/env python3
"""Generate deterministic v0.3 material fixtures from the released Sentinel.

The embedded checker is diagnostic generated data, not artist-authored art.
"""

from __future__ import annotations

import copy
import argparse
import hashlib
import json
import struct
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "armored-sentinel-v1.glb"
OUT_DIR = ROOT / "fixtures"
SOURCE_SHA256 = "06c76ee61f1582708e50fe208939180ba9b529825052db5afd7f1d2e54e0201e"
JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(
        ">I", zlib.crc32(kind + payload) & 0xFFFFFFFF
    )


def diagnostic_png(width: int = 16, height: int = 16) -> bytes:
    palette = (
        (255, 255, 255, 255),
        (92, 180, 255, 255),
        (255, 190, 88, 255),
        (72, 72, 88, 255),
    )
    rows = bytearray()
    for y in range(height):
        rows.append(0)
        for x in range(width):
            tile = ((x // 4) + (y // 4)) % len(palette)
            rows.extend(palette[tile])
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return signature + png_chunk(b"IHDR", ihdr) + png_chunk(
        b"IDAT", zlib.compress(bytes(rows), 9)
    ) + png_chunk(b"IEND", b"")


def parse_glb(payload: bytes) -> tuple[dict, bytes]:
    magic, version, total = struct.unpack_from("<III", payload, 0)
    if magic != 0x46546C67 or version != 2 or total != len(payload):
        raise ValueError("source is not a valid GLB 2.0 container")
    offset = 12
    chunks: dict[int, bytes] = {}
    while offset < len(payload):
        length, kind = struct.unpack_from("<II", payload, offset)
        offset += 8
        chunks[kind] = payload[offset : offset + length]
        offset += length
    document = json.loads(chunks[JSON_CHUNK].rstrip(b" \x00"))
    return document, chunks[BIN_CHUNK]


def encode_glb(document: dict, binary: bytes) -> bytes:
    json_payload = json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    json_payload += b" " * ((-len(json_payload)) % 4)
    bin_payload = binary + b"\x00" * ((-len(binary)) % 4)
    body = (
        struct.pack("<II", len(json_payload), JSON_CHUNK)
        + json_payload
        + struct.pack("<II", len(bin_payload), BIN_CHUNK)
        + bin_payload
    )
    return struct.pack("<III", 0x46546C67, 2, 12 + len(body)) + body


def textured_document(source_document: dict, source_binary: bytes) -> tuple[dict, bytes]:
    document = copy.deepcopy(source_document)
    texture = diagnostic_png()
    texture_offset = len(source_binary)
    binary = source_binary + texture

    buffer_views = document.setdefault("bufferViews", [])
    image_view = len(buffer_views)
    buffer_views.append(
        {"buffer": 0, "byteOffset": texture_offset, "byteLength": len(texture)}
    )
    document["buffers"][0]["byteLength"] = len(binary)
    document["images"] = [
        {
            "name": "rt4d_generated_diagnostic_checker",
            "bufferView": image_view,
            "mimeType": "image/png",
        }
    ]
    document["samplers"] = [
        {"magFilter": 9729, "minFilter": 9729, "wrapS": 10497, "wrapT": 10497}
    ]
    document["textures"] = [
        {
            "name": "rt4d_generated_diagnostic_checker",
            "sampler": 0,
            "source": 0,
        }
    ]
    for material in document.get("materials", []):
        pbr = material.setdefault("pbrMetallicRoughness", {})
        pbr["baseColorTexture"] = {"index": 0, "texCoord": 0}

    extras = document.setdefault("asset", {}).setdefault("extras", {})
    extras["rt4dMaterialFixture"] = {
        "schema": "rt4d-material-fixture/0.3",
        "sourceAssetSha256": SOURCE_SHA256,
        "textureOrigin": "generated_diagnostic_checker",
        "usesSourceUv": True,
        "textureArtistReviewed": False,
        "uvArtistReviewed": False,
        "materialAssignmentArtistReviewed": False,
        "purpose": "source_texture_upload_and_material_factor_proof",
    }
    return document, binary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true",
        help="verify committed fixtures instead of writing them",
    )
    args = parser.parse_args()
    source_payload = SOURCE.read_bytes()
    actual_hash = hashlib.sha256(source_payload).hexdigest()
    if actual_hash != SOURCE_SHA256:
        raise SystemExit(f"source hash changed: {actual_hash}")

    source_document, source_binary = parse_glb(source_payload)
    textured, binary = textured_document(source_document, source_binary)
    missing_uv = copy.deepcopy(textured)
    for mesh in missing_uv.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            primitive.get("attributes", {}).pop("TEXCOORD_0", None)
    missing_uv["asset"]["extras"]["rt4dMaterialFixture"]["purpose"] = (
        "missing_uv_policy_contract_test"
    )
    missing_uv["asset"]["extras"]["rt4dMaterialFixture"]["usesSourceUv"] = False

    outputs = {
        OUT_DIR / "armored-sentinel-textured-v0.3.glb": encode_glb(textured, binary),
        OUT_DIR / "armored-sentinel-textured-missing-uv-v0.3.glb": encode_glb(
            missing_uv, binary
        ),
    }
    if args.check:
        for path, expected in outputs.items():
            if not path.exists() or path.read_bytes() != expected:
                raise SystemExit(f"fixture is stale: {path.relative_to(ROOT)}")
        return

    OUT_DIR.mkdir(exist_ok=True)
    for path, payload in outputs.items():
        path.write_bytes(payload)


if __name__ == "__main__":
    main()
