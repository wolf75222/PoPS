"""Freeze tracked review inputs from one Git revision; never load a native extension."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

parser = argparse.ArgumentParser()
parser.add_argument("--checkout", type=Path, required=True)
parser.add_argument("--revision", required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
checkout = args.checkout.resolve()
revision = subprocess.check_output(
    ["git", "rev-parse", args.revision + "^{commit}"], cwd=checkout, text=True
).strip()
output = args.output.resolve()
assert not output.exists(), "use a new snapshot directory"
paths = [
    "python",
    "include/pops/runtime/program/amr_program_context.hpp",
    "include/pops/runtime/program/program_cadence_continuation.inc",
    "include/pops/runtime/program/program_context.hpp",
]
archive = subprocess.check_output(["git", "archive", revision, *paths], cwd=checkout)
output.mkdir(parents=True)
with tarfile.open(fileobj=io.BytesIO(archive)) as tree:
    for item in tree:
        if not item.isfile():
            continue
        name = Path(item.name)
        assert not name.is_absolute() and ".." not in name.parts
        assert name.suffix not in {".so", ".dylib"}, "native archive member forbidden"
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(tree.extractfile(item).read())
files = {
    str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(output.rglob("*"))
    if path.is_file()
}
material = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
manifest = {
    "base_commit": revision,
    "source_checkout": str(checkout),
    "files": files,
    "source_tree_sha256": hashlib.sha256(material).hexdigest(),
}
(output / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(
    json.dumps(
        {
            "commit": revision,
            "files": len(files),
            "source_tree_sha256": manifest["source_tree_sha256"],
        }
    )
)
