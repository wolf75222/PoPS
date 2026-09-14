# Documentation maintenance

Use implementation behavior as the source of truth. Keep the README focused on installing,
running and navigating PoPS; explain architecture and methods in their canonical guides.
Tutorials remain linear Python scripts with physical authoring before simulation setup.
Design contracts document exact interfaces; dated progress diaries belong in Git history.

The [docmap](docmap.toml) records source dependencies and executable checks for maintained
contract pages. Source changes since a page's review produce freshness diagnostics; they
are review prompts, not proof that prose is wrong or that runtime behavior is qualified.

From the repository root, run:

```bash
bash scripts/build_docs.sh
```

This runs `docs/check_docs.py`: it checks mapped paths and local Markdown/image links across
project guides, tutorials, examples and benchmarks. It does not run simulations or replace
native/release acceptance. Run the affected tutorials and test contracts when their behavior
changes. Keep expected outputs and assumptions in the tutorial's own guide.

Use one explanation per topic, link to actual source and executable examples, and update
paths in tests, CI selectors, manifests and docs when moving files. Keep project prose in
English and follow the code/comment conventions in [CONTRIBUTING.md](../CONTRIBUTING.md).
