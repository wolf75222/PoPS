# Joint reconstruction resource probe v1.1

Use the [original instrumented resource protocol](joint_reconstruction_resource_probe.md)
with [`joint_reconstruction_resource_probe_v1_1.py`](joint_reconstruction_resource_probe_v1_1.py)
in place of the v1 script. This variant pins the SHA-256 of the corrected
[benchmark v1.1](joint_reconstruction_benchmark_v1_1.md) and imports only its
authenticated case and helpers. The fixed snapshots, numerical calculation,
fresh instances, two unprofiled warmups, three profiled runs, native counter
interpretations, 720-second worker budget, identity/ABI checks, and missing-data
rules are unchanged. It emits a separate `resources.v1.1` schema. The v1
resource script and any receipts remain unchanged. No resource measurement has
been run by this patch.
