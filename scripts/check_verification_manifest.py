#!/usr/bin/env python3
"""Fail closed: validate verification/manifest.toml before any compilation.

The scientific campaign manifest is checked against
``schemas/verification_manifest.v1.json`` (Draft 2020-12). This script does not
compile, bind, run cases, or launch jobs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 test environments
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "verification" / "manifest.toml"
DEFAULT_SCHEMA = ROOT / "schemas" / "verification_manifest.v1.json"


class VerificationManifestError(RuntimeError):
    pass


def _read_text(path: Path, *, kind: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise VerificationManifestError(f"cannot read {kind} {path}: {exc}") from exc


def load_schema(path: Path) -> Draft202012Validator:
    raw = _read_text(path, kind="schema")
    try:
        schema = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise VerificationManifestError(f"invalid JSON in schema {path}: {exc}") from exc
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise VerificationManifestError(f"invalid Draft 2020-12 schema {path}: {exc}") from exc
    return Draft202012Validator(schema)


def load_manifest(path: Path) -> dict:
    raw = _read_text(path, kind="manifest")
    try:
        parsed = tomllib.loads(raw)
    except tomllib.TOMLDecodeError as exc:
        raise VerificationManifestError(f"invalid TOML in {path}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise VerificationManifestError(f"manifest {path} must be a TOML table")
    return parsed


def _format_validation_error(error: ValidationError) -> str:
    location = ".".join(str(part) for part in error.absolute_path)
    if location:
        return f"{location}: {error.message}"
    return error.message


def check_verification_manifest(
    manifest_path: Path = DEFAULT_MANIFEST,
    schema_path: Path = DEFAULT_SCHEMA,
) -> dict:
    validator = load_schema(schema_path)
    instance = load_manifest(manifest_path)
    errors = sorted(validator.iter_errors(instance), key=lambda item: list(item.absolute_path))
    if errors:
        details = "; ".join(_format_validation_error(error) for error in errors)
        raise VerificationManifestError(f"{manifest_path}: {details}")
    return instance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    args = parser.parse_args(argv)
    try:
        check_verification_manifest(args.manifest, args.schema)
    except VerificationManifestError as exc:
        print(f"VERIFICATION-MANIFEST: FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"VERIFICATION-MANIFEST: OK: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
