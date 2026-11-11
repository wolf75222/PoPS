#!/usr/bin/env python3
"""DM-T1 (spec 0.3 §10, §16.12): emit and compare the generated C++ of the API profiles.

N10 phase 1 (ROM-85). Every translation unit is captured where the real path hands it to the
compiler, with no native module and no compiler:

* the Program unit: ``_compile_resolved_problem(plan)`` runs under ``ir_identity_reuse()`` after
  ``plan.verify()``, as ``pops.compile`` does, and the text is taken at its ``pops_include()``
  call, so the refinement criterion that the driver appends is in it;
* the model units: ``compile_install_models(plan, plan.compile_options)`` runs as
  ``pops.compile`` does, with the real ``compile_model`` and ``compile_native``; only
  ``emit_cpp_native_loader`` is wrapped, to record the text and stop that block before any
  compiler, cache or ``_pops`` call.

Five toolchain facts that need ``_pops`` are fixed, equal at every commit: the native dimension
(the plan's resolved dimension), the C++ standard, the compiler name, the feature key and the ABI
key. A sixth, the Kokkos of a native build, is not selected at all: ``emit`` drops the Kokkos
root overrides (``POPS_KOKKOS_ROOT``, ``Kokkos_ROOT``, ``KOKKOS_ROOT``) from its own environment,
since no ``_pops`` contract can authenticate them and no unit is compiled (CI sets them for its
native tests). Each unit embeds the 64-hex identities of the program, so byte identity also checks
that no identity moved.

    dmt1.py emit OUT.jsonl [--units DIR] [ID ...]     # all 40 profiles when no ID is given
    dmt1.py compare BASE.jsonl HEAD.jsonl [--declared FILE]

``compare`` fails on any difference of a gated profile (the 29 mesh profiles) that ``--declared``
does not list, one profile id per line, ``#`` comments allowed; differences of the 11 array
profiles are reported, not gated (§10, OD18).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The Kokkos root overrides of ``pops.codegen.toolchain._native_kokkos_selection``.
KOKKOS_ROOTS = ("POPS_KOKKOS_ROOT", "Kokkos_ROOT", "KOKKOS_ROOT")
PROFILES = tuple("M%02d" % index for index in range(1, 29)) + tuple(
    "T%02d" % index for index in range(1, 13))
GATED = (
    "M01", "M02", "M03", "M04", "M05", "M06", "M07", "M08", "M10", "M11", "M12", "M13", "M14",
    "M15", "M16", "M17", "M18", "M22", "M24", "M25", "M26", "M27", "T01", "T05", "T07", "T08",
    "T09", "T10", "T12",
)


class _Captured(Exception):
    pass


def _install_capture():
    """Patch the native facts and the two capture points; return the shared state."""
    import pops._native_selector as selector
    import pops.codegen._compile_drivers as drivers
    import pops.codegen._orchestration_compile as orchestration
    import pops.codegen.abi as abi
    import pops.codegen.toolchain as toolchain

    state: dict = {"models": {}}
    selector.select_native_dimension = lambda dimension: state.__setitem__("dimension", dimension)
    toolchain.loader_native_dimension = lambda: state["dimension"]
    abi.loader_native_dimension = lambda: state["dimension"]
    drivers.loader_cxx_std = toolchain.loader_cxx_std = lambda: "c++20"
    drivers._native_kokkos_compiler = toolchain._native_kokkos_compiler = lambda cxx=None: "c++"
    toolchain._native_feature_key = lambda *a, **k: "dmt1-features"
    drivers._abi_key_python = abi._abi_key_python = lambda *a, **k: "dmt1-abi"
    try:
        import pops.runtime._mapped_field_capability as mapped

        mapped.require_mapped_field_native_facts = lambda **k: None
    except ImportError:
        pass

    def capture_include():
        raise _Captured(sys._getframe(1).f_locals["src"])

    drivers.pops_include = capture_include
    real_emit = drivers.emit_cpp_native_loader

    def recording_emit(*args, **kwargs):
        raise _Captured(real_emit(*args, **kwargs))

    drivers.emit_cpp_native_loader = recording_emit
    real_install = orchestration.compile_install_model

    def per_block(name, *args, **kwargs):
        try:
            real_install(name, *args, **kwargs)
            state["models"][name] = None
        except _Captured as captured:
            state["models"][name] = captured.args[0]

    orchestration.compile_install_model = per_block
    return state, drivers, orchestration


def _digest(text: str) -> dict:
    data = text.encode()
    return {"bytes": len(data), "sha": hashlib.sha256(data).hexdigest()}


def _identity_scope(ir_identity_reuse):
    """The scope that ``pops.compile`` opens, at the head and at a base older than #732."""
    import inspect

    if inspect.signature(ir_identity_reuse).parameters:
        return ir_identity_reuse("pops.compile")
    return ir_identity_reuse()


def _emit_one(identifier, state, drivers, orchestration, units):
    import pops
    from examples.api_v02.profile_dispatch import build_profile
    from pops.time._program.serialization import ir_identity_reuse

    spec = build_profile(identifier)
    plan = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    state["dimension"] = plan.resolved_dimension
    with _identity_scope(ir_identity_reuse):
        plan.verify()
        try:
            drivers._compile_resolved_problem(plan)
            raise RuntimeError("the Program driver reached no pops_include() call")
        except _Captured as captured:
            program = captured.args[0]
    state["models"].clear()
    orchestration.compile_install_models(plan, plan.compile_options)
    row = {"id": identifier, "target": plan.target, "program": _digest(program), "models": {}}
    if units is not None:
        (units / ("%s.cpp" % identifier)).write_text(program, encoding="utf-8")
    for block in plan.blocks:
        text = state["models"].get(block.name)
        row["models"][block.name] = None if text is None else _digest(text)
        if units is not None and text is not None:
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in block.name)
            (units / ("%s__%s.cpp" % (identifier, safe))).write_text(text, encoding="utf-8")
    return row


def emit(out: Path, identifiers, units):
    for key in KOKKOS_ROOTS:
        # A root names the Kokkos of a native build; without ``_pops`` nothing authenticates it.
        os.environ.pop(key, None)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "python"))
    state, drivers, orchestration = _install_capture()
    if units is not None:
        units.mkdir(parents=True, exist_ok=True)
    failures = 0
    with out.open("w", encoding="utf-8") as stream:
        for identifier in identifiers:
            start = time.time()
            try:
                row = _emit_one(identifier, state, drivers, orchestration, units)
            except Exception as exc:  # noqa: BLE001 - every profile is reported
                failures += 1
                row = {"id": identifier, "failed": "%s: %s" % (type(exc).__name__, exc),
                       "trace": traceback.format_exc()[-1500:]}
            row["seconds"] = round(time.time() - start, 1)
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            print(identifier, "failed" if "failed" in row else "ok", row["seconds"], flush=True)
    return 1 if failures else 0


def _load(path: Path) -> dict:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["id"]] = row
    return rows


def _units(row: dict) -> dict:
    if "failed" in row:
        return {"failed": row["failed"]}
    units = {"program": row["program"]}
    units.update(("model:%s" % name, value) for name, value in row["models"].items())
    return units


def compare(base: Path, head: Path, declared: Path | None) -> int:
    before, after = _load(base), _load(head)
    allowed = set()
    if declared is not None:
        for line in declared.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                allowed.add(line)
    unknown = sorted(allowed - set(PROFILES))
    if unknown:
        print("declared profiles that do not exist: %s" % unknown)
        return 2
    gated_differences, reported = [], []
    for identifier in PROFILES:
        if identifier not in before or identifier not in after:
            gated_differences.append("%s: missing on one side" % identifier)
            continue
        left, right = _units(before[identifier]), _units(after[identifier])
        if left == right:
            continue
        changed = sorted(key for key in set(left) | set(right) if left.get(key) != right.get(key))
        line = "%s: %s" % (identifier, ", ".join(changed))
        if identifier in GATED and identifier not in allowed:
            gated_differences.append(line)
        else:
            reported.append(line + (" (declared)" if identifier in allowed else " (reported)"))
    for line in reported:
        print("difference", line)
    for line in gated_differences:
        print("UNDECLARED", line)
    print("DM-T1: %d gated profiles, %d undeclared differences, %d reported"
          % (len(GATED), len(gated_differences), len(reported)))
    return 1 if gated_differences else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "DM-T1").splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    emit_parser = sub.add_parser("emit")
    emit_parser.add_argument("out", type=Path)
    emit_parser.add_argument("ids", nargs="*")
    emit_parser.add_argument("--units", type=Path)
    compare_parser = sub.add_parser("compare")
    compare_parser.add_argument("base", type=Path)
    compare_parser.add_argument("head", type=Path)
    compare_parser.add_argument("--declared", type=Path)
    args = parser.parse_args(argv)
    if args.command == "emit":
        identifiers = tuple(args.ids) or PROFILES
        bad = sorted(set(identifiers) - set(PROFILES))
        if bad:
            parser.error("unknown profiles %s" % bad)
        return emit(args.out, identifiers, args.units)
    return compare(args.base, args.head, args.declared)


if __name__ == "__main__":
    sys.exit(main())
