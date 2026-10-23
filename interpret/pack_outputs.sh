#!/usr/bin/env bash
# Pack lightweight results for download (excludes raw attention .npz and downloaded structures).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${PHYNET_OUT:-$ROOT/interpret/outputs}"
TAR="$ROOT/interpret_outputs_$(date +%Y%m%d_%H%M).tar.gz"
tar -czf "$TAR" -C "$(dirname "$OUT")" --exclude='*.npz' "$(basename "$OUT")"
echo "packed: $TAR ($(du -h "$TAR" | cut -f1))"

