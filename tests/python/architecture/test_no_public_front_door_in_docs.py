"""ADC-523: the OLD front doors must not appear in user-facing docs / docstrings.

``pops.compile`` / ``pops.bind`` are the only public compile/bind entry points. A quickstart or a
public docstring must never teach the retired paths -- the low-level ``compile_problem(...)`` driver,
the deleted ``sim.install(...)`` / ``install_program`` methods, a raw ``System.install`` /
``AmrSystem(...)`` construction, or a ``target="system"`` / ``target="amr_system"`` kwarg (the LAYOUT
picks the runtime, never a user string).

Scope: this greps the USER-FACING surface only -- ``README.md`` plus the module docstrings of
``pops/__init__.py`` and ``pops/problem/__init__.py`` (the two docstrings a user reads first).
Internal design /
reference docs (``docs/design/**``, ``docs/ARCHITECTURE.md``, ``docs/ALGORITHMS.md``, the vendored
``docs/docguide/**``) legitimately DESCRIBE the internal mechanism in that vocabulary and are
allowlisted, as is ``CHANGELOG.md`` (history).

The test reads the source tree only; it does not import ``pops`` or ``_pops``.
"""
import ast
import pathlib
import re

import pytest

# tests/python/architecture/<this file> -> repo root is parents[3].
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
POPS = REPO_ROOT / "python" / "pops"

# The retired front doors, as they would appear in a quickstart. Each pattern is anchored so it
# matches a call / attribute use, not an incidental mention of the word.
FORBIDDEN = {
    "compile_problem(": re.compile(r"compile_problem\s*\("),
    "sim.install(": re.compile(r"\.install\s*\("),
    "install_program": re.compile(r"\binstall_program\b"),
    "System.install": re.compile(r"\bSystem\.install\b"),
    "AmrSystem( construction": re.compile(r"\bAmrSystem\s*\("),
    'target="system"': re.compile(r"""target\s*=\s*['"]system['"]"""),
    'target="amr_system"': re.compile(r"""target\s*=\s*['"]amr_system['"]"""),
}


def _hits(text):
    """Return the sorted labels of every forbidden front door found in @p text."""
    return sorted(label for label, pat in FORBIDDEN.items() if pat.search(text))


def test_readme_teaches_only_the_public_front_doors():
    readme = REPO_ROOT / "README.md"
    if not readme.exists():
        pytest.skip("no README.md")
    hits = _hits(readme.read_text())
    assert not hits, (
        "README.md teaches a retired front door %s; use pops.compile(...) / pops.bind(...) and a "
        "layout (Uniform / AMR), never compile_problem / install / target= (ADC-523)." % hits)


@pytest.mark.parametrize("rel", ["__init__.py", "problem/__init__.py"])
def test_public_module_docstring_is_clean(rel):
    path = POPS / rel
    doc = ast.get_docstring(ast.parse(path.read_text(), str(path))) or ""
    hits = _hits(doc)
    assert not hits, (
        "python/pops/%s module docstring teaches a retired front door %s; the public quickstart uses "
        "pops.compile(...) / pops.bind(...), never compile_problem / install / target= (ADC-523)."
        % (rel, hits))
