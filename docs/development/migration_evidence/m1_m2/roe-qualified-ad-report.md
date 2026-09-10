# Qualified flux Jacobian repair

Production commit: `35bfabc`; strict diagnostic/rollback fixture commit: `781a5cc`.

The Board flux Jacobian differentiated owner-qualified QuantityRef expressions against bare component names, silently producing zero derivatives. It now retains immutable exact conservative coordinates as AD targets. The derivative protocol accepts QuantityRef explicitly and preserves exact owner/component identity; legacy Var differentiation remains supported. Board declaration rollback includes the coordinate tuple; freeze and native formula clone retain it. The common helper repairs both Roe and HLL Jacobians.

Validation used source Python from the isolated followup and unchanged signed candidate5 native/header bytes (Dim2, OpenMP2, MPI1). No native package/source header rebuild or install occurred.

- Source Roe contracts: 8 passed, 1 native case deselected.
- Symbolic/qualified application protocols: 41 passed.
- Full three-file native replay: 11 passed and 2 failed in384.56s, zero skips. Both failures were exact-message mismatches after valid native rejection; no remaining DID NOT RAISE.
- After strict current-message fixture update: both native spectral rejection cases passed in109.04s, zero skips. They assert t=0, macro_step=0, and bit-exact unchanged state.
- Ruff and git diff --check passed; isolated worktree clean.

The current status6 is FiniteVolumeStatus::InvalidWaveSpeed (`include/pops/numerics/spatial/nd/finite_volume.hpp`). The prepared face operator throws before copying candidates to output (`include/pops/numerics/spatial/operators/cartesian_operator.hpp`). Existing no-fallback runtime behavior is preserved.

The whole three-file suite must still be rerun on final candidate6; the historical 11+2 outcome is not relabeled as a full green suite. Evidence: roe-qualified-ad-candidate5.log/.xml/.metadata.json/.results.json, roe-qualified-ad-symbolic.log, roe-qualified-ad-rollback-candidate5.log/.xml/.results.json.
