# Documentation quality policy

PoPS documents only the public lifecycle and capabilities implemented by the current source tree.
The canonical technical contract is
[`docs/design/SPECIFICATION_TECHNIQUE_FINALE_POPS_ARCHITECTURE.md`](design/SPECIFICATION_TECHNIQUE_FINALE_POPS_ARCHITECTURE.md),
and executable behavior is demonstrated by the scripts under [`examples/final`](../examples/final).
Roadmaps, migration notes, dated validation logs and delivered design drafts are not active user
documentation.

## Active corpus

The maintained root corpus is:

- `README.md`;
- `CONTRIBUTING.md`;
- `SECURITY.md`;
- `CHANGELOG.md`;
- `docs/ARCHITECTURE.md`;
- `docs/ALGORITHMS.md`;
- `docs/CODE_DOCUMENTATION_CONVENTION.md`;
- `docs/CODING_STANDARDS_DECISIONS.md`;
- `docs/DOC_QUALITY.md`;
- `docs/VERSIONING.md`;
- `docs/design/SPECIFICATION_TECHNIQUE_FINALE_POPS_ARCHITECTURE.md`;
- the focused contracts referenced by that specification;
- `docs/docguide/` as the vendored style reference.

`docs/docmap.toml` records ownership, source dependencies and executable checks. A document whose
dependencies changed since its review is reported as stale; release-critical pages must be refreshed
before the release evidence is accepted.

## Rules

1. Document current code and exact capability envelopes, never roadmap intent.
2. Prefer executable examples and generated contracts over copied implementation snippets.
3. Keep one canonical page per topic and link to it instead of cloning contracts.
4. A public name shown in documentation must be importable and lowerable on its advertised route.
5. A native limitation is stated as a capability/refusal rule, not hidden behind generic wording.
6. Every lifecycle example uses `validate -> resolve -> compile -> bind -> run`; internal engines and
   phase records are absent.
7. Presets and manual authoring are documented as equivalent graph builders, never separate runtimes.
8. Update documentation, tests and generated products in the same coherent change.
9. Local links must resolve and generated/reference files must pass their drift checks.
10. Do not use an ignored test or a hand-written success flag as documentation evidence.

Project-specific code documentation rules live in
[`docs/CODE_DOCUMENTATION_CONVENTION.md`](CODE_DOCUMENTATION_CONVENTION.md).

## Checks

Run the deterministic documentation gate with:

```bash
python docs/check_docs.py
```

or through:

```bash
bash scripts/build_docs.sh
```

The check validates the docmap, source/test paths, relative links, punctuation policy and freshness.
The final release gate additionally runs the four installed-package examples, generated-contract
checks and the full conformance suites; `docs/check_docs.py` is intentionally not a substitute for
those executable proofs.
