# Joint reconstruction benchmark v1.1

The first v1 campaign stopped during baseline compilation **before any timing**.
Its preserved receipt is
`outputs/performance-t2-run/00-baseline-compile.stderr.log` in the parent
workspace. The compiled artifact exposes `layout_program_paths` as a `dict`
property; v1 called it as a method in `_dso_sizes`. Both retained installed
SDKs define the same property. No numerical or performance conclusion follows
from the failed receipt.

[`joint_reconstruction_benchmark_v1_1.py`](joint_reconstruction_benchmark_v1_1.py)
corrects that one access and emits schema
`pops.api040.joint-reconstruction-comparison.v1.1`. The [v1 script](joint_reconstruction_benchmark.py),
its [predeclared protocol](joint_reconstruction_benchmark.md), and the failed
receipt remain unchanged. The case, physics, 48×48 grid, five scalar User
components, `FixedDt=1e-4`, 12 steps, equivalence tolerances, warmups,
ABBA order, sample counts, budgets, identity checks, isolated processes,
compiler provenance and output rules are identical to v1. Use the v1 protocol
for interpretation and replace only the script path in its plan and execute
commands. Plan and execute require separate new empty output directories;
the v1 failure directory remains untouched. The v1.1 result carries its own
schema and script SHA-256 so the new run cannot be mistaken for the failed v1.

The regression test
[`test_joint_reconstruction_benchmark_v1_1.py`](../../../tests/python/unit/runtime/test_joint_reconstruction_benchmark_v1_1.py)
checks the real repository artifact property declaration, reproduces the v1
`dict`-not-callable failure with that property shape, confirms that v1.1 hashes
all three DSO classes, and compares the predetermined scenario constants.
Source inspection also confirmed `@property` in both authenticated installed
snapshots `3d06cab` and `cf6dace`. This is a source/API regression check, not
a compile or timing result.

The separately instrumented
[`joint_reconstruction_resource_probe_v1_1.py`](joint_reconstruction_resource_probe_v1_1.py)
imports the v1.1 case after verifying its SHA-256; its v1 predecessor remains
frozen. See [its v1.1 note](joint_reconstruction_resource_probe_v1_1.md).
