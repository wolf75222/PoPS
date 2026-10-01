# Independent bb416 / Python relay archive review

Reviewer: Hooke, GPT-6.1 Sol. Offline filesystem/stdlib inspection only; no PoPS import, native execution, build, JIT, environment mutation or production edit. Parent 9ee89b39. Evidence root: `/Users/romaindespoulain/dev/tmp/pops-api040-native-reception-evidence-20261001`.

## Principle → decision → actual evidence

| Principle | Decision | Evidence / oracle | Status |
|---|---|---|---|
| Source, compilation and scientific evidence remain distinct | Both repository build receipts are closed rc0, installed identity rc0 | `native-amr-tag-selection-dim2-01c5b654/build-reception.json` closed 21:52:43 UTC; `native-amr-tag-relay-dim2-fc670393/build-reception.json` closed 22:22:37 UTC on October 1 | Compilation receipts received; no cryptographic C++→DSO graph proof |
| Authenticate actual runtime | Same installed Dim2 MPI DSO and SDK header signature in both builds | DSO SHA256 `488123fe4d7c4ecb3ac604f4eec9e97525a8e745034ea01a69624616d8b09e54`; SDK `bb416020c8971068a97b22b37fa6a2d495b28a8f42c562d9bedceee8a7b6aa5d`; installed native path is pops-api040-ir17 site-packages/pops/_native/dim2/_pops.cpython-312-darwin.so | Receipt path/hash objects and retained wheel native member independently hashed |
| Exact typed Python authority relay | Installed source inventories each contain 1132 entries; differences are precisely `_plans.py` and `_layout_install_projection.py` | Inventory SHA256 before relay `d1d9e07dd99d7f05b59081eb56e2ed6250322fe4a709796c960d6bc00e793c9d`; after relay `bb592911dad61f6bfb15bd0dda602235d8ea2c9dbd57cbd983bab63db3f51b84` | Source delta received; source review in a95b761d is separate |
| Preserve authentic failure | Original installed-sdkbb416 run closed rc1, 1 failure, AttributeError InstallPlan.resolved_tagging | `installed-sdkbb416-amr12-representative-serial-dim2`; identity/source inventory before and after byte-identical | Actual bind failure, preceding relay, preserved |
| Observe actual closed execution | Relay original [1-8] run closed rc0; XML 1 test, zero failure/error/skip | `installed-sdkbb416-relay-amr12-representative-serial-dim2/result.json`, pytest.xml, pytest.log | Actual serial original case passes; no broader claim |
| Keep strict partial coverage guard | Saved accepted coarse active.npy is bool shape(8,8), 48/64 true | ZIP NPY header parsed with ast.literal_eval, exact raw bool bytes counted | Actual N8 partial coverage observed |
| Exact saved-state restart/replay | accepted vs reloaded and continuous vs replay at levels0,1 have identical ZIP member names and every NPY member byte-identical | Files below relay pytest-tmp/test_public_evolved_stage_amr_0/evolved-stage-amr | Independent exact observations; checkpoint manifest reader@3 and acceptance seals remain Root-owned |

Relay run before/after identity and source-files JSON are byte-identical. Identity SHA256 `3e448c93e31dcbad8b6f4b2558f23f7d13205d8669dbe9ab0818cc665209280c`; log SHA256 `dc6545c3a675e5c3c23830ab005ef64ce98224f51d64bef57fb16cb3e2de7392`; both agree with result pins. XML timestamp 2026-10-02T00:23:51.555205+02:00, time76.386s; result wall duration76.9397764171s.

The fixture receipt reports exact payload/manifest equivalence and clocks at macrosteps1 and2. This review independently compares saved level NPY bytes; it does not reconstruct the full strict checkpoint protocol, validate all scientific residuals, qualify MPI2/GPU/Dim3, or promote pytest alone into scientific reception. Build receipts explicitly retain cpp_to_dso_cryptographic_graph_proof=false and native_scientific_acceptance=false.

## Offline commands

`rtk proxy python3` with stdlib pathlib/json/hashlib hashes each receipt path+sha256 object, compares source inventory dictionaries and before/after raw JSON, checks result identity/log pins, and parses XML with xml.etree.ElementTree. `zipfile.ZipFile` compares every member of accepted-level{0,1}.npz against reloaded-level{0,1}.npz and continuous-level{0,1}.npz against replay-level{0,1}.npz. NPY v1 header length uses struct.unpack('<H', raw[8:10]); ast.literal_eval decodes the header, and sum(raw_after_header) counts the active bool values. No installed package is imported.
