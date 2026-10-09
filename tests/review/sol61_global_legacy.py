"""Authentic-source separate-process ModuleManifest10 byte/hash comparison."""

import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source / "python"))
    from pops.model import Module, Signature, Rate
    from pops.identity.semantic import semantic_identity_of

    rows = []
    for components in (("density",), ("x", "y"), ("y", "x")):
        module = Module("independent-legacy")
        state = module.state_space("u", components)
        values = module.state_symbols(state)
        module.operator(
            "reaction",
            kind="local_source",
            signature=Signature([state], Rate(state)),
            expr=tuple(-0.31 * value for value in values),
        )
        payload = module.manifest().to_json()
        assert module.manifest().schema_version == 10
        assert "global_quantities" not in module.manifest().to_dict()
        rows.append(
            {
                "components": components,
                "manifest_sha256": hashlib.sha256(payload.encode()).hexdigest(),
                "module_hash": module.module_hash(),
                "semantic_identity": semantic_identity_of(model=module).token,
            }
        )
    print(json.dumps(rows, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
