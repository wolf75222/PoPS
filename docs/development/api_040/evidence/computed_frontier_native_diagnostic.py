#!/usr/bin/env python3
"""Central-only replay of the genuine public fixture with a private cache copy."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--retry", action="store_true")
    args = parser.parse_args()
    if os.environ.get("PYTHONPATH"):
        raise RuntimeError("execute with env -u PYTHONPATH")
    from pops._native_selector import select_native_dimension
    select_native_dimension(2)
    import pops
    import numpy as np
    if not Path(pops.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError("diagnostic requires the genuine installed package under sys.prefix")
    fixture = args.fixture.resolve()
    sys.path.insert(0, str(fixture.parents[4]))
    spec = importlib.util.spec_from_file_location("frozen_computed_frontier_fixture", fixture)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory(prefix="pops-computed-frontier-central-diagnostic-") as temporary:
        cache = Path(temporary) / "cache"
        shutil.copytree(args.cache, cache)
        os.environ["POPS_CACHE_DIR"] = str(cache)
        os.environ["POPS_NATIVE_CACHE_DIR"] = str(cache)
        case, layout, _, _ = module.rotation_case(retry=args.retry)
        artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
        initial = np.zeros((2, module.CELLS, module.CELLS))
        initial[0] = 1.
        subject = artifact.plan.initial_condition_plan.bindings[0].subject
        runtime = pops.bind(artifact, initial_values={subject: initial},
            resources={"execution_context": module.artifact_execution_context(artifact)})
        expected, limit = module.reference(np.array([1., 0.]), .5 if args.retry else 1.)
        receipt = {"package_file": pops.__file__, "sys_prefix": sys.prefix,
                   "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
                   "retry": args.retry, "run_limit": limit.hex(), "numpy_reference": expected.tolist()}
        try:
            pops.run(runtime, t_end=limit, max_steps=1, console=False)
        except RuntimeError as error:
            receipt["error"] = str(error)
        receipt.update(accepted_time=float(runtime.time()).hex(),
                       accepted_macro_step=int(runtime.macro_step()),
                       state=np.asarray(runtime.state_global("rotation")).tolist())
        print(json.dumps(receipt, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
