#!/usr/bin/env bash
# Deterministic local/CI entry point for the maintained documentation contract.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

python docs/check_docs.py

echo "OK: documentation conformance passed"
