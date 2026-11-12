"""The one-shot rejection of N8 lot « reuse across stages » is a test-only C++ patch: no shipped
source names it, and only the native test of the lot applies it."""
from pathlib import Path

import pytest

import pops
import examples.api_v02.models_M01_M14 as early
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from tests.python.support.one_shot_rejection import one_shot_rejection, patch_source
from tests.python.unit.time.test_reuse_across_stages_n8 import _twice_route

ROOT = Path(__file__).resolve().parents[3]
NAMES = ("one_shot_rejection", "one-shot rejection")
MESSAGE = "test-only one-shot rejection"
SHIPPED = ("python", "src", "include", "scripts", "cmake", "examples")
USERS = {
    "tests/python/support/one_shot_rejection.py",
    "tests/python/architecture/test_one_shot_rejection_is_test_only.py",
    "tests/python/integration/runtime/test_reuse_across_stages_n8_native.py",
}


def _texts(*roots):
    for root in roots:
        for path in sorted((ROOT / root).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".hpp", ".cpp", ".h", ".txt", ".cmake",
                                                  ".toml", ".json", ".yml", ".md"}:
                yield path.relative_to(ROOT).as_posix(), path.read_text(errors="replace")


def test_no_shipped_source_names_the_patch():
    assert [path for path, text in _texts(*SHIPPED) if any(name in text for name in NAMES)] == []


def test_only_the_lot_uses_the_patch():
    users = {path for path, text in _texts("tests") if any(name in text for name in NAMES)}
    assert users == USERS


def test_the_patch_refuses_a_source_without_its_single_guard():
    install = "  ctx.install([=](double dt) {\n"
    guard = ('  throw pops::runtime::program::StepAttemptRejected(pops::SolveStatus::'
             'kInvalidEvaluation, "assessment", "Advance assessment \'finite\' rejected the '
             'candidate");')
    body = "if (!(ok)) {\n%s\n}\n" % guard
    patched = patch_source(install + body)
    assert patched.count("std::make_shared<bool>(false)") == 1
    assert patched.index("make_shared<bool>") < patched.index("ctx.install")
    assert patched.split("\n")[patched.split("\n").index(guard) + 2].startswith(
        "if (!*one_shot_rejection)")
    for source in (body, install + install + body, install + body + body, install):
        try:
            patch_source(source)
        except AssertionError:
            continue
        raise AssertionError("the patch accepted %r" % source)


def _cpp(build, monkeypatch, route=None):
    with monkeypatch.context() as patch:
        if route is not None:
            patch.setattr(early, "_traced_recorded_ssprk2", route)
        spec = build()
    resolved = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    return emit_cpp_program(resolved.time, model_graph=ProgramModelGraph.from_resolved_blocks(
        resolved.blocks))


@pytest.mark.parametrize("build, route", [(early.build_m08, None),
                                          (early.build_m10, _twice_route)],
                         ids=["M08", "twice-M10"])
def test_the_emitted_cpp_carries_the_patch_only_when_patched(build, route, monkeypatch):
    source = _cpp(build, monkeypatch, route)
    assert not any(name in source for name in NAMES + (MESSAGE,))
    patched = patch_source(source)
    assert patched.count("std::make_shared<bool>(false)") == 1
    assert patched.count(MESSAGE) == 1


def test_the_patch_needs_a_route_of_its_own(monkeypatch):
    for route in ("N8 reuse across stages twice M10", "one-shot rejection of M08", None):
        with pytest.raises(ValueError, match="does not end with"):
            with one_shot_rejection(monkeypatch, route):
                pass
    with one_shot_rejection(monkeypatch, "N8 reuse across stages M08 one-shot rejection"):
        pass
