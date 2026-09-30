"""Receive an externally sealed historical T5 archive without importing PoPS.

The caller supplies the archive seal. This checker never creates owner pins,
changes the frozen scientific oracle, or qualifies another native campaign.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys

ROOT_MANIFEST_SHA = "854974b3447561d22f8c767821cf8ce85eb1c1a8232e2ec3f8f0e56141579a4b"
OWNER_SHA = "7a6d07c39e434b256e08792ff6c89d991c8525068eb0df0c0a37e43c12521bc2"
FROZEN = {"oracle/sol61_integral_feedback_offline_oracle.py": "21041d64af5378a35b5a8717cc0300b6a399d95cc2c9209e66d66e5e2ed54f89",
          "oracle/sol61_integral_feedback_real_countermodels.py": "5a7fd0ddf6b690709d055587f5dda8bde8e20a365976092f8e774eeb5ee35c30",
          "source/test_public_integral_feedback.py": "db1d5153eeb59c8e15e80f22837a9889c30e08f37dc99dab96a1e422fc12962d"}
EXTRAS = set(FROZEN) | {"portable-owner-pins.json", "provenance/root-owner-pins.json",
                        "provenance/root-scientific-reception.json", "provenance/independent-reception.json", "verify_archive.py"}
MAX_BYTES = 64 * 1024 * 1024  # Reception budget; no production file-size limit.


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def strict_json(raw):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def invalid(token):
        raise ValueError("nonfinite JSON token " + token)

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def portable_path(value):
    require(type(value) is str and value and value != "." and ":" not in value and "\\" not in value
            and all(ord(character) >= 32 for character in value), "nonportable archive path")
    path = PurePosixPath(value)
    require(not path.is_absolute() and path.as_posix() == value and all(part not in (".", "..") for part in path.parts),
            "nonportable archive path")
    return value


def regular_tree(root):
    require(root.is_absolute(), "archive root must be absolute")
    require(not root.is_symlink(), "archive root is a symlink")
    require(root.is_dir(), "archive root is not a directory")
    files, directories = set(), set()

    def visit(directory):
        with os.scandir(directory) as entries:
            for entry in entries:
                relative = Path(entry.path).relative_to(root).as_posix()
                mode = entry.stat(follow_symlinks=False).st_mode
                require(not stat.S_ISLNK(mode), "archive symlink entry: " + relative)
                if stat.S_ISDIR(mode):
                    directories.add(relative)
                    visit(Path(entry.path))
                else:
                    require(stat.S_ISREG(mode), "archive nonregular entry: " + relative)
                    files.add(relative)

    visit(root)
    return files, directories


def owner_leaves(value):
    if type(value) is dict:
        if set(value) == {"path", "sha256"}:
            require(type(value["path"]) is str and type(value["sha256"]) is str
                    and re.fullmatch("[0-9a-f]{64}", value["sha256"]), "invalid historical owner pin")
            return [value]
        return [row for item in value.values() for row in owner_leaves(item)]
    if type(value) is list:
        return [row for item in value for row in owner_leaves(item)]
    return []


def receive(root, expected):
    require(type(expected) is str and re.fullmatch("[0-9a-f]{64}", expected), "missing external archive seal")
    root = root.absolute()
    actual_files, actual_directories = regular_tree(root)
    index = root / "manifest.json"
    require(index.is_file() and index.stat().st_size <= MAX_BYTES, "missing/oversized archive manifest")
    raw = index.read_bytes()
    require(digest(raw) == expected, "external archive SHA mismatch")
    data = strict_json(raw)
    require(set(data) == {"schema", "scope", "original_owner_pins_sha256", "native_sha256", "abi_key", "source_commit", "mapping", "files"}
            and data["schema"] == "pops.native-t5-archive@1", "archive contract mismatch")
    require(type(data["files"]) is list and len(data["files"]) == 48, "wrong archive file count")
    rows = {}
    for row in data["files"]:
        require(type(row) is dict and set(row) == {"path", "bytes", "sha256"}, "invalid file inventory row")
        name = portable_path(row["path"])
        require(name not in rows and name != "manifest.json", "duplicate archive inventory path")
        require(type(row["bytes"]) is int and 0 <= row["bytes"] <= MAX_BYTES
                and type(row["sha256"]) is str and re.fullmatch("[0-9a-f]{64}", row["sha256"]), "invalid file pin")
        rows[name] = row
    require(actual_files == set(rows) | {"manifest.json"}, "archive file inventory differs")
    expected_directories = {str(parent) for name in actual_files for parent in PurePosixPath(name).parents if str(parent) != "."}
    require(actual_directories == expected_directories, "archive directory inventory differs")
    require(sum(row["bytes"] for row in rows.values()) <= MAX_BYTES, "archive reception budget exceeded")
    for name, row in rows.items():
        path = root / name
        require(path.stat().st_size == row["bytes"] and digest(path.read_bytes()) == row["sha256"], "archive file pin mismatch: " + name)
    for name, frozen_sha in FROZEN.items():
        require(rows.get(name, {}).get("sha256") == frozen_sha, "frozen historical source/oracle differs: " + name)
    source = root / "provenance/root-owner-pins.json"
    require(data["original_owner_pins_sha256"] == OWNER_SHA and digest(source.read_bytes()) == OWNER_SHA,
            "historical owner seal mismatch")
    original = strict_json(source.read_bytes())
    portable = strict_json((root / "portable-owner-pins.json").read_bytes())
    require(all(data[key] == original[key] == portable[key] for key in ("source_commit", "native_sha256", "abi_key")),
            "archive/original/portable native identity differs")
    leaves = owner_leaves(original)
    source_paths = [row["path"] for row in leaves]
    require(len(leaves) == len(set(source_paths)) == 40 and all(Path(path).is_absolute() for path in source_paths),
            "wrong historical owner file inventory")
    mapping = data["mapping"]
    require(type(mapping) is list and len(mapping) == 40, "wrong archive mapping count")
    translations, targets = {}, set()
    for row in mapping:
        require(type(row) is dict and set(row) == {"source_path", "path", "sha256"}, "invalid mapping row")
        name = portable_path(row["path"])
        require(row["source_path"] not in translations and name not in targets, "duplicate archive mapping source/target")
        require(name in rows and name.startswith("t5-sdk7b/") and Path(row["source_path"]).name == PurePosixPath(name).name
                and rows[name]["sha256"] == row["sha256"], "mapping/file pin mismatch")
        translations[row["source_path"]] = row
        targets.add(name)
    require(set(translations) == set(source_paths) and set(rows) == targets | EXTRAS, "archive mapping/inventory roles differ")

    def translate(value):
        if type(value) is dict:
            if set(value) == {"path", "sha256"}:
                row = translations[value["path"]]
                require(row["sha256"] == value["sha256"], "historical/portable SHA differs")
                return dict(path=row["path"], sha256=value["sha256"])
            return {key: translate(item) for key, item in value.items()}
        if type(value) is list:
            return [translate(item) for item in value]
        return value

    require(translate(original) == portable, "portable pins changed more than the 40 paths")
    spec = importlib.util.spec_from_file_location("frozen_portable_t5_oracle", root / "oracle/sol61_integral_feedback_offline_oracle.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    prior_bytecode = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
        result = module.receive(root / "portable-owner-pins.json")
    finally:
        sys.dont_write_bytecode = prior_bytecode
    historical = strict_json((root / "provenance/root-scientific-reception.json").read_bytes())
    require({key: value for key, value in result.items() if key != "pins_sha256"}
            == {key: value for key, value in historical.items() if key != "pins_sha256"}, "relocation changed scientific reception")
    return dict(status="received", archive_sha256=expected, pinned_files=48, actual_files=len(actual_files),
                total_bytes=sum((root / name).stat().st_size for name in actual_files), symlinks=0,
                relocated_files=40, native_execution=False, scientific_reception=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = receive(args.archive, args.manifest_sha256)
    raw = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(raw)
    print(json.dumps({key: value for key, value in result.items() if key != "scientific_reception"}))


if __name__ == "__main__":
    main()
