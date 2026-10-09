"""Run the exact same public Uniform fixture/callsite against a chosen source tree."""

import hashlib
import json
from pathlib import Path
import runpy
import sys

source = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(source / "python"))
import pops  # noqa: E402

fixture = runpy.run_path(str(Path(__file__).with_name("test_sol61_amr_public_original.py")))
case, layout, _, _, _ = fixture["authored"](3, (2, 0, 1), seed=True, uniform=True)
code, resolved = fixture["emit"](case, layout)
print(
    json.dumps(
        {
            "source": str(source),
            "package": pops.__file__,
            "ir": resolved.time._ir_hash(),
            "cpp": hashlib.sha256(code.encode()).hexdigest(),
            "modules": [block.model.module.module_hash() for block in resolved.blocks],
        },
        sort_keys=True,
    )
)
