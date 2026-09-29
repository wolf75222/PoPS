# Independent review of joint benchmark v1.1

Reviewed `PoPS-numerical-bodies` commit `8a34829d07a2f634e5d8f34fee93713309e272ff`
against the preserved v1 runner and companion. This is a source/protocol review,
not a benchmark run.

The exact v1→v1.1 runner diff has only three effects: a descriptive docstring,
the result schema `...comparison.v1.1`, and `_dso_sizes` reading
`artifact.layout_program_paths.values()` rather than calling the dict property.
The fixed case, all acceptance thresholds, 48×48 grid, five-component scalar
User physics, 12-step FixedDt schedule, cold compile, ABBA worker order, sample
counts, timeout budgets, isolation, snapshot/ABI checks, timing sites and
output rules are byte-for-byte unchanged in the runner diff. The old v1 file
and failed receipt are preserved. `_dso_sizes` still includes block, layout
program and top-level DSOs and hashes their actual files.

The companion probe v1.1 changes only its schema, imported runner name and
runner SHA pin. Its embedded SHA
`e59336591c49e5d658cc057bae3ea74b83c506cdec654ff9b929d3bd55319833`
matches the corrected runner bytes. Its case and resource-measurement bodies
are unchanged from the companion v1 diff. The new unit tests reproduce the
old `dict`-not-callable failure, verify the repaired three-DSO inventory and
pin, and check scenario constants; author reports 3/3 source tests, Ruff and
diff-check green. I independently verified the full two-file diff and SHA.

Verdict: v1.1 is a narrow, correctly versioned repair of metadata collection.
It makes no change to the experimental or numerical protocol. No timing,
resource, numerical equivalence or causal performance result follows until
the two authenticated installed snapshots complete a fresh v1.1 campaign in
new output directories.
