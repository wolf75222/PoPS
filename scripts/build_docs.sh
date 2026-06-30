#!/usr/bin/env bash
# Transitional documentation check for the docs reset.
#
# The Sphinx and Doxygen sites were removed intentionally. Until the new docs
# are rebuilt, this script is the single local/CI entry point for lightweight
# documentation validation.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

python docs/check_docs.py

echo "OK: documentation reset lint passed"
