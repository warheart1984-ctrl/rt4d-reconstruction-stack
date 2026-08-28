#!/usr/bin/env python3
"""Verify the structural envelope of an RT4D PNG capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_IEND = b"\x00\x00\x00\x00IEND\xaeB\x60\x82"


def inspect_capture(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    if len(data) < 33:
        raise ValueError("capture is too short to contain a complete PNG")
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("capture does not have a PNG signature")
    if data[12:16] != b"IHDR":
        raise ValueError("capture does not begin with an IHDR chunk")
    if not data.endswith(PNG_IEND):
        raise ValueError("capture does not end with a complete IEND chunk")
    width, height = struct.unpack(">II", data[16:24])
    if width == 0 or height == 0:
        raise ValueError("capture reports zero dimensions")
    return {
        "path": str(path),
        "bytes": len(data),
        "width": width,
        "height": height,
        "sha256": hashlib.sha256(data).hexdigest(),
        "pngEnvelopeComplete": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    args = parser.parse_args()
    try:
        result = inspect_capture(args.capture)
    except (OSError, ValueError) as error:
        print(json.dumps({"path": str(args.capture), "error": str(error)}),
              file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
