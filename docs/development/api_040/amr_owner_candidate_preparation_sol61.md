# Prepare AMR @2 owner inventories, pending ROOT audit

This helper reuses existing `sol61_m18_owner_assemble.py` canonical-root/path/hash helpers and the actual @2 reader admissions. It never imports PoPS, executes a model, modifies native data, signs, emits approval or receives science. It produces a wrapper marked `candidate_pending_ROOT_audit`, scientific_reception=false. The owner_pins_candidate member is a draft inventory only. No current partially failed AMR dataset is promoted.

Supply one spec JSON per actual mode (serial or mpi2), after all four real cells8/16 x widths1/2 cases have complete receipts and clean all-rank raw XML:

```json
{
  "roots": ["/absolute/canonical/immutable/evidence-root", "/absolute/canonical/installed-origin-root"],
  "mode": "serial",
  "ir_version": 16,
  "source_commit": "actual exact 40 hex source revision",
  "native_build_source_commit": "actual exact 40 hex native build revision",
  "abi_key": "actual Native ABI key",
  "native_build_receipt": "/absolute/actual/native-build-receipt.json",
  "identity_before": "/absolute/actual/before-origin-inventory.json",
  "identity_after": "/absolute/actual/after-origin-inventory.json",
  "source_files": {
    "amr": "/absolute/actual/evolved_stage_amr.py",
    "equations": "/absolute/actual/evolved_stage_mms.py",
    "controls": "/absolute/actual/evolved_stage_controls.py",
    "fixture": "/absolute/actual/test_public_evolved_stage_amr.py"
  },
  "junit": [{"rank": 0, "path": "/absolute/actual/result.xml"}],
  "receipts": ["/actual/8-width1/receipt.json", "/actual/8-width2/receipt.json", "/actual/16-width1/receipt.json", "/actual/16-width2/receipt.json"]
}
```

The example contains descriptive placeholders and is intentionally not executable evidence. MPI2 requires raw XML rows for ranks0 and1 with identical batch name inventories. Roots must be canonical, existing and nonoverlapping. Before/after origin inventories are exact dictionaries with native, sdk and package_manifest, each containing actual path+sha256. ROOT extracts those three actual origins from its audited build/run identity evidence (including the current build1132 before/after record); it must retain the original raw build receipt. This utility pins that raw receipt but does not guess its schema or independently reconstruct the build graph. Before/after normalized origin inventories must match exactly and every named origin is rehashed. Retained installed origins must remain available under the declared roots; copy/path translation requires ROOT-reviewed receipt translation and new pins, never silent rewriting.

Run from the chosen source checkout with the existing Python interpreter:

```sh
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops/bin/python tests/review/sol61_amr_owner_candidate.py /absolute/spec.json /absolute/candidate-pending-root-audit.json
```

Output uses exclusive creation and refuses an existing candidate path. The helper verifies actual retained source admissions, IR16/17, actual CPP/IR hash agreement and seven controls, CP12 typed envelope and POPSCAR1 header/shape, receipt/rank identities, carrier registry inventory, raw XML cleanliness/rank properties and closed file-role inventory. Missing CPP, changed identity, missing rank/case, aliases, wrong controls, malformed or failing XML refuse. It does not decide native authenticity merely from a marker: native_evidence=true is part of the candidate assertion ROOT must audit. Source/native revisions and ABI remain explicitly supplied ROOT attestations.

ROOT then audits every candidate, the original build-before/after evidence, actual package/sdk/DSO origins, dataset completeness, raw logs and retained generated sources. ROOT alone may extract the owner_pins_candidate as final owner pins, pin its exact bytes externally and supply the separate immutable @2 ROOT approval. The existing reader @2 runs only after those two external seals are supplied; it independently checks scientific equations, valid/ghost carrier hashes, histories, metadata, diagnostics and lifecycle. This helper never creates either seal or an approval.

SOURCE_ONLY verification: **96 PASS, 0.66s**, including 8 inventory controls plus historical70 and v2 reader18. The inventory positive test uses synthetic bytes/XML with explicit stubs for source/CPP admission to isolate assembly plumbing. It is not a Native dataset, native source admission or scientific receipt; real reader source/CPP protocols remain independently tested. Negative cases cover partial case inventory, changed before/after identity, controls, missing CPP, resealed wrong JUnit rank, file-role alias and failed raw XML. No approval or full receive() is invoked in tests. No ENV mutation, JIT, build or native run occurred.
