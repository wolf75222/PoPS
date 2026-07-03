"""ADC-600: architecture gate quarantining the host/prototype hot-path routes.

The prototype/host routes are the JIT ``add_dynamic_block`` (IModel virtual host Rusanov), the AOT
``add_compiled_block`` (host-marshalled production algorithm, no MPI/AMR/GPU) and the
``pops.experimental.PythonFlux`` numpy-host backend.  They may stay for internal tests / prototyping
but must NEVER be taken for the final generic route: the ``pops.compile`` / ``pops.bind`` target
surface must not reference them, a missing production native route must refuse early (no prototype
fallback), and ``pops.experimental`` must not leak onto the ``pops`` root.

These checks are source-only (they do not import ``pops`` / ``_pops``), so they run without a built
native extension.  If a legitimate hit appears in a scanned tree, INVESTIGATE it (the target surface
must not need the host/prototype seams) instead of allowlisting it silently: the failure names the
file and line so the reference can be removed at its source.
"""
import ast
import pathlib
import re


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
POPS = REPO_ROOT / "python" / "pops"

# The TARGET compile/bind surface (Spec 5 sec.11): the Problem authoring type, the pops.compile /
# pops.bind orchestration, and the internal install seam bind calls.  None of these may reference a
# host/prototype route -- their only route is the production native loader.
TARGET_SURFACE = (
    "problem/problem.py",
    "codegen/orchestration.py",
    "runtime/_system_unified_install.py",
)

# The host/prototype route symbols the target surface must never name (AST name / attribute /
# import), plus the experimental (numpy host) package.
FORBIDDEN_NAMES = ("add_dynamic_block", "add_compiled_block")
FORBIDDEN_ATTR_CHAIN = "experimental"  # pops.experimental (host numpy PythonFlux)

# The production-only refusal the compile drivers MUST keep (a silent removal is a fallback hole).
PRODUCTION_ONLY_RE = re.compile(r"require backend='production'")

# The legal route-tier vocabulary (ADC-600): every backend caps row carries one of these.
LEGAL_TIERS = {"production", "prototype", "internal"}


def _read(path):
    return path.read_text(encoding="utf-8")


def _rel(path):
    return path.relative_to(REPO_ROOT).as_posix()


def _forbidden_route_references(path):
    """Line numbers where a host/prototype route is USED as a symbol (name / attribute / import).

    A docstring or comment mentioning a token to EXPLAIN the quarantine is not a usage; the gate is
    that the target surface does not NEED the host/prototype seams.  So the AST is walked for a real
    name / attribute access / import alias of ``add_dynamic_block`` / ``add_compiled_block`` or of the
    ``pops.experimental`` package -- string / comment content is ignored.
    """
    tree = ast.parse(_read(path), filename=str(path))
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            hits.append(node.lineno)
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_NAMES:
            hits.append(node.lineno)
        elif isinstance(node, ast.Attribute) and node.attr == FORBIDDEN_ATTR_CHAIN:
            # pops.experimental (or *.experimental) attribute access.
            hits.append(node.lineno)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            module = getattr(node, "module", None) or ""
            if module == "pops.experimental" or module.endswith(".experimental"):
                hits.append(node.lineno)
            for alias in node.names:
                if alias.name in FORBIDDEN_NAMES or alias.asname in FORBIDDEN_NAMES:
                    hits.append(node.lineno)
                if alias.name == "experimental" and module in ("pops", ""):
                    hits.append(node.lineno)
    return sorted(set(hits))


def test_target_surface_does_not_reference_host_prototype_routes():
    violations = []
    for entry in TARGET_SURFACE:
        path = POPS / entry
        for lineno in _forbidden_route_references(path):
            violations.append("%s:%d" % (_rel(path), lineno))

    assert not violations, (
        "the pops.compile / pops.bind target surface must not reference a host/prototype route "
        "(add_dynamic_block / add_compiled_block / pops.experimental) -- ADC-600. The target route "
        "is the production native loader; a host/prototype route is never a fallback. Investigate "
        "and remove each reference at its source (never allowlist it):\n  " + "\n  ".join(violations))


def test_compile_drivers_keeps_the_production_only_refusal():
    text = _read(POPS / "codegen" / "compile_drivers.py")
    assert PRODUCTION_ONLY_RE.search(text), (
        "compile_drivers.py must keep the production-only refusal string "
        "\"require backend='production'\" (ADC-600): a compiled time program refuses a non-production "
        "backend early, it never falls back to a prototype/host route. A silent removal of this "
        "guard opens a fallback hole -- restore the refusal.")


def test_pops_root_does_not_import_experimental():
    text = _read(POPS / "__init__.py")
    # experimental listed in __all__ (a lazily-reachable submodule name) is fine; an eager IMPORT of
    # the numpy-host package onto the root is what ADC-600 forbids.
    import_lines = [line for line in text.splitlines()
                    if "experimental" in line and "import" in line
                    and not line.lstrip().startswith("#")]
    assert not import_lines, (
        "pops/__init__.py must not import pops.experimental (ADC-600 keeps the numpy-host prototyping "
        "package off the root; PythonFlux is not pops.PythonFlux); found:\n  "
        + "\n  ".join(import_lines))


def test_backend_caps_rows_carry_a_legal_tier():
    """Source-only: every _BACKEND_CAPS row declares a tier in {production, prototype, internal}.

    Parse the assignment in compile_emit.py (no import) and assert each row dict has a ``tier`` key
    whose value is one of the legal tokens, so a report can always name a route's class honestly.
    """
    src = _read(POPS / "codegen" / "compile_emit.py")
    tree = ast.parse(src, filename="compile_emit.py")
    caps_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "_BACKEND_CAPS":
                    caps_node = node.value
    assert isinstance(caps_node, ast.Dict), "could not find the _BACKEND_CAPS dict literal"
    rows = {}
    for key, value in zip(caps_node.keys, caps_node.values, strict=True):
        assert isinstance(key, ast.Constant), "_BACKEND_CAPS keys must be string literals"
        assert isinstance(value, ast.Dict), "_BACKEND_CAPS[%r] must be a dict literal" % key.value
        tier = None
        for k, v in zip(value.keys, value.values, strict=True):
            if isinstance(k, ast.Constant) and k.value == "tier":
                assert isinstance(v, ast.Constant), "tier of %r must be a string literal" % key.value
                tier = v.value
        rows[key.value] = tier
    missing = [b for b, t in rows.items() if t is None]
    illegal = {b: t for b, t in rows.items() if t is not None and t not in LEGAL_TIERS}
    assert not missing, (
        "every _BACKEND_CAPS row must carry a \"tier\" key (ADC-600) so reports name the route class; "
        "missing on: %s" % ", ".join(sorted(missing)))
    assert not illegal, (
        "_BACKEND_CAPS tier values must be in %s (ADC-600); illegal: %s"
        % (sorted(LEGAL_TIERS), illegal))


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
