# Independent actual Serial variants audit

Hooke, GPT-6.1 Sol; offline STD-only, no PoPS/native imports, builds, JIT or environment mutation. Four paths come exclusively from RAW pytest.xml properties, never the `current` symlink. BASE `/Users/romaindespoulain/dev/tmp/pops-api040-native-reception-evidence-20261001/installed-sdkbb416-amr12-variants-serial-dim2`.

Principle → decision: distinguish authentic execution from scientific reception. Result closed rc0, 4 tests, zero failure/error/skip. Identity and all1132 installed source-file inventories before/after are raw-byte identical; result identity/log SHA256 pins match. Every receipt path+SHA256 object was actually hashed (CPP, IR, generated DSO, sidecar, carrier registry, phase states, full checkpoints).

Oracle: arithmetic independently derives scalar Q=T+T² and coupled Q=(a+a²+0.1b²,b+b²+0.2ab) with conserved total, positive branch b stable quadratic root, a bisection90, and z=0.25a+0.5b. This matches independent source oracle06648dd7, rather than recomputing a solver residual. Actual saved NPY active cells at both levels give composite volume1, quantities agreeing with recorded receipts, clocks(.01,1) and(.02,2). Explicit saved values are compared against the arithmetic target at each step.

| width-N | actual file pins | max T oracle error | max Q oracle error | max affine constraint |
|---|---|---|---|---|
| 1-8 | 23 | 7.46069873e-14 | 1.00391917e-13 | 0 |
| 1-16 | 23 | 7.46069873e-14 | 9.65061364e-14 | 0 |
| 2-8 | 25 | 3.76336184e-11 | 5.66261482e-11 | 2.77555756e-17 |
| 2-16 | 25 | 3.76336184e-11 | 5.66290348e-11 | 2.77555756e-17 |

All four cases have exact NPY members accepted↔reloaded and continuous↔replay on levels0/1. Coarse active48/64 at N8 and192/256 at N16; fine active64/256 and256/1024 respectively, so composite volume1. CP accepted/continuous/replay archives are SHA pinned and readable; actual scalar/coupled archives contain71/112 members, checkpoint version12, POPSCAR1 carrier magic, and parseable full JSON manifest. This observer does not perform the full codec geometry/authority/carrier bit validation or strict accepted-contract checker; reader@3 remains the gate.

Recorded scalar native original residual max3.5213272e-13; coupled max9.2072101e-11. Recorded coupled Qbalance max5.6626259e-11 and equation error max3.3066189e-11. Full exact observed figures and receipt hashes are preserved in hooke_serial_variants_archive.actual.json, not inferred from pytest status.

BLOCKER PRESERVED: current reader@3 gate REFUSE on integer controls comparing raw20 with actual typed {'scalar':{'kind':'integer','value':'20'}}. Reader correction must validate the typed contract strictly and preserve older@1/@2 handling. Source-only fake reader fixtures missed this authentic archived shape; do not suppress the refusal, weaken the contract, or infer a science seal from these checks. Root owns revised reader/seals; no native rerun is required for an offline reader correction.

Scope is homogeneous scalar/coupled original FullResidualBasisLU@1, Dim2 Serial, N8/N16; no coarse-fine flux qualification, signed diffusion D coverage, GPU, Dim3 or MPI inference.

Reproduce: `rtk proxy python3 tests/review/hooke_serial_variants_archive.py /Users/romaindespoulain/dev/tmp/pops-api040-native-reception-evidence-20261001/installed-sdkbb416-amr12-variants-serial-dim2`. The script imports only Python stdlib. Run returned0. Full report SHA256 1a89abc79c88b8f36f45f70cb9259aa30af0d40f7ab23d8ae990d33dd4e3e768.
