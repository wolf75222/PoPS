# Documentation quality policy

This repository is in a documentation reset. The former Sphinx site, archived
roadmaps, dated validation figures, and stale design notes were removed. The
next documentation pass must be rebuilt from the current code and from the small
set of retained policy and reference files.

## Retained corpus

The retained documentation is deliberately small:

- `README.md`
- `CONTRIBUTING.md`
- `SECURITY.md`
- `CHANGELOG.md`
- `docs/ARCHITECTURE.md`
- `docs/ALGORITHMS.md`
- `docs/BIBLIOGRAPHY.md`
- `docs/CODE_DOCUMENTATION_CONVENTION.md`
- `docs/CODING_STANDARDS_DECISIONS.md`
- `docs/DOC_QUALITY.md`
- `docs/TRANSLATION_GLOSSARY.md`
- `docs/VERSIONING.md`
- `docs/docguide/`

`docs/docmap.toml`, `docs/check_docs.py`, and `scripts/build_docs.sh` stay as
the minimal guardrail for this transition.

## Rules for the rebuild

1. Document the code that exists, not roadmap intent.
2. Prefer generated API reference and executable examples over copied snippets.
3. Keep one canonical page per topic and link to it instead of duplicating it.
4. Move dated audits, delivered design notes, and run logs out of active docs.
5. Keep user-facing docs in English and avoid non-ASCII punctuation.
6. Update docs with the code change that makes them true.

The vendored Google documentation guide in `docs/docguide/` remains the style
reference. Project-specific code documentation rules live in
`docs/CODE_DOCUMENTATION_CONVENTION.md`.

## Checks

Run the transitional documentation check with:

```bash
python docs/check_docs.py
```

or through the wrapper:

```bash
bash scripts/build_docs.sh
```

When a new Sphinx or Doxygen site is introduced, re-expand this policy, restore
a real docmap, and make the build script own the full site generation again.
