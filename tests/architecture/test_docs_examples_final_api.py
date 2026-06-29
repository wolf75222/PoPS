"""Guards for active documentation and public examples.

The corrective spec removes the transitional public front doors.  Documentation
and examples are part of the API surface, so they must not advertise old routes
or fallback examples that quietly avoid the compiled path.
"""

import pathlib


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _active_doc_and_example_files():
    roots = [REPO_ROOT / "README.md", REPO_ROOT / "docs", REPO_ROOT / "examples"]
    for root in roots:
        if root.is_file():
            yield root
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix not in {".md", ".rst", ".py"}:
                continue
            if "archive" in path.parts:
                continue
            if path.name in {"SPEC_CORRECTIVE_TASKS.md"}:
                continue
            if path.parent == REPO_ROOT / "docs" and path.suffix != ".md":
                continue
            yield path


def _example_files():
    return sorted((REPO_ROOT / "examples").rglob("*.py"))


def test_active_docs_do_not_advertise_removed_public_front_doors():
    forbidden = (
        "pops.Case",
        "pops.Problem",
        "pops.compile(",
        "pops.bind(",
        "pops.compile,",
        "pops.bind,",
        "pops.Case",
        "pops.Problem",
        "pops.Model",
        "pops.CondensedSchur",
        "pops.FiniteVolume",
        "pops.Spatial",
        "case.block(",
        "add_equation",
        "install_program",
        "CompiledTime",
        "m.compile(",
        ".compile(backend",
        "compile(backend",
        "sim.run(t_end",
        "run(t_end",
        "_get_state",
        "_set_state",
        "_eval_rhs(",
        "get_state(",
        "set_state(",
        "eval_rhs(",
        "set_density(",
        "set_primitive_state(",
        "sim.install(None",
        "pops.Explicit",
        "Explicit.euler",
    )
    offenders = []
    for path in _active_doc_and_example_files():
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                offenders.append("%s contains %s" % (path.relative_to(REPO_ROOT), token))
    assert not offenders, (
        "active docs/examples must use compile_problem -> System/AmrSystem -> install -> "
        "step_cfl only:\n%s" % "\n".join(offenders)
    )


def test_public_examples_do_not_hide_missing_compiled_routes():
    forbidden = (
        "NotImplementedError",
        "compile section is skipped",
        "skip compile",
        "skip compile+install",
        "could not build the .so",
        "except RuntimeError",
        "except (RuntimeError",
        "except NotImplementedError",
    )
    offenders = []
    for path in _example_files():
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                offenders.append("%s contains %s" % (path.relative_to(REPO_ROOT), token))
    assert not offenders, (
        "public examples must fail loudly when the compiled route is missing; no skip/fallback "
        "examples:\n%s" % "\n".join(offenders)
    )


def test_examples_no_skip():
    """TASK-072: exact gate name for public examples that must not hide compile failures."""
    test_public_examples_do_not_hide_missing_compiled_routes()
