#!/usr/bin/env python3
"""ADC-597: no-fallback compliance matrix for native/generic routes.

This file is intentionally a matrix, not another smoke test.  Each refusal line exercises the
public route that owns the decision, checks the exception type, checks the actionable reason, and
asserts that the route did not merely emit a warning before falling back to an older/simpler path.

The positive lines prove that the supported route next to each refusal still works or is advertised
as available through the structured reports.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import warnings

import pytest
from pops.runtime.system import AmrSystem, System  # ADC-545 advanced runtime seam

pops = pytest.importorskip("pops")

import numpy as np  # noqa: E402

from pops import time as adctime  # noqa: E402
from pops.codegen.loader import CompiledModel  # noqa: E402
from pops.fields import PoissonProblem  # noqa: E402
from pops.fields.bcs import Dirichlet, Periodic  # noqa: E402
from pops.ir.expr import Var  # noqa: E402
from pops.math import laplacian, unknown  # noqa: E402
from pops.mesh import CartesianMesh  # noqa: E402
from pops.mesh.amr import (  # noqa: E402
    AMROutput,
    AllLevels,
    CheckpointPolicy,
    PatchLayout,
    ProperNesting,
    Refine,
    RegridEvery,
    TagUnion,
)
from pops.mesh.layouts import AMR  # noqa: E402
from pops.numerics.reconstruction import WENO5, validate_ghost_depth  # noqa: E402
from pops.numerics.reconstruction.limiters import Minmod  # noqa: E402
from pops.numerics.riemann import HLL, HLLC, Roe, Rusanov  # noqa: E402
from pops.numerics.variables import Primitive  # noqa: E402
from pops.runtime._install_param_routing import route_program_params  # noqa: E402
from pops.solvers.elliptic import FFT, GeometricMG  # noqa: E402


ROOT = Path(__file__).resolve().parents[4]


def _expect_refusal(label, exc_type, fn, needles):
    """Run one no-fallback matrix row.

    A valid refusal must be an exception of the expected type with a route-specific reason.  Python
    warnings are treated as a failure: a warning plus a fallback is precisely what this matrix is
    meant to catch.
    """
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        with pytest.raises(exc_type) as exc:
            fn()
    assert not seen, "%s emitted warning(s) instead of refusing: %r" % (label, seen)
    msg = str(exc.value)
    missing = [needle for needle in needles if needle not in msg]
    assert not missing, "%s: message %r missing %r" % (label, msg, missing)
    return msg


def _scalar_exb_model():
    return pops.Model(
        state=pops.Scalar(),
        transport=pops.ExB(B0=1.0),
        source=pops.NoSource(),
        elliptic=pops.BackgroundDensity(alpha=1.0, n0=0.0),
    )


def _isothermal_model(charge=1.0, cs2=0.5):
    return pops.Model(
        state=pops.FluidState("isothermal", cs2=cs2),
        transport=pops.IsothermalFlux(),
        source=pops.NoSource(),
        elliptic=pops.ChargeDensity(charge=charge),
    )


def _compiled_model(*, target="system", hllc=False, roe=False, wave_speeds=False, prim_names=None):
    return CompiledModel(
        so_path="/no/such/pops-route.so",
        backend="production",
        adder="add_native_block",
        cons_names=["rho", "mx", "my"],
        cons_roles=["density", "momentum_x", "momentum_y"],
        prim_names=list(prim_names or ("rho", "u", "v")),
        n_vars=3,
        gamma=None,
        n_aux=3,
        params={},
        caps={"cpu": True},
        abi_key="SIG|c++|c++23",
        model_hash="modelhash",
        cxx="c++",
        std="c++23",
        hllc=hllc,
        roe=roe,
        wave_speeds=wave_speeds,
        target=target,
    )


def _program_with_context():
    program = adctime.Program("adc597_context")
    dt = program.dt
    state = program.state("gas")
    rhs = program._rhs_legacy(state=state, flux=True, sources=["default"])
    program.commit("gas", program.linear_combine("U1", state + dt * rhs))
    return program


def _compile_bad_program_abi_so():
    cxx = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx:
        pytest.skip("no C++ compiler available for the bad-ABI program route")
    src = (
        'extern "C" const char* pops_program_abi_key() { return "adc597-wrong-abi-key"; }\n'
        'extern "C" const char* pops_program_name() { return "adc597_bad"; }\n'
        'extern "C" const char* pops_program_hash() { return "0"; }\n'
        'extern "C" void pops_install_program(void*) {}\n'
    )
    tmp = tempfile.TemporaryDirectory()
    cpp = Path(tmp.name) / "bad_program_abi.cpp"
    so = Path(tmp.name) / "bad_program_abi.so"
    cpp.write_text(src)
    cmd = [cxx, "-shared", "-fPIC", "-std=c++17", "-O0", str(cpp), "-o", str(so)]
    if sys.platform == "darwin":
        cmd[1:1] = ["-undefined", "dynamic_lookup"]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        tmp.cleanup()
        pytest.skip("could not compile bad-ABI program route: %s" % exc)
    return tmp, str(so)


def _poisson_problem(solver, *bcs):
    phi = unknown("phi")
    rho = Var("rho", "cons")
    return PoissonProblem(unknown=phi, equation=(-laplacian(phi) == rho), bcs=bcs, solver=solver)


def test_refusal_matrix_checks_error_type_reason_and_no_warning():
    rows = [
        (
            "HLL without wave speeds",
            RuntimeError,
            lambda: System(n=8, L=1.0, periodic=True).add_block(
                "s",
                _scalar_exb_model(),
                spatial=pops.FiniteVolume(limiter=Minmod(), riemann=HLL()),
            ),
            ("flux 'hll'", "requires signed wave speeds", "this transport -> 'rusanov'"),
        ),
        (
            "HLLC without contact/star-state capability",
            ValueError,
            lambda: System(n=8, L=1.0, periodic=True)._validate_riemann_capability(
                _compiled_model(hllc=False, prim_names=("rho", "u", "v")),
                pops.FiniteVolume(riemann=HLLC()),
            ),
            ("HLLC", "capability", "hllc_star_state"),
        ),
        (
            "Roe without dissipation capability",
            ValueError,
            lambda: System(n=8, L=1.0, periodic=True)._validate_riemann_capability(
                _compiled_model(roe=False, prim_names=("rho", "u", "v")),
                pops.FiniteVolume(riemann=Roe()),
            ),
            ("Roe", "capability", "roe_dissipation"),
        ),
        (
            "high-order reconstruction with insufficient ghosts",
            ValueError,
            lambda: validate_ghost_depth(WENO5(), available=2, block="plasma"),
            ("WENO5 requires ghost_depth >= 3", "block 'plasma' has ghost_depth=2"),
        ),
        (
            "field solver incompatible with boundary",
            ValueError,
            lambda: _poisson_problem(FFT(), Dirichlet()).validate(),
            ("requires a periodic boundary", "supports_wall_bc is False", "GeometricMG()"),
        ),
        (
            "AMR level envelope exceeded",
            ValueError,
            lambda: AMR(base=CartesianMesh(n=64), max_levels=4).validate(),
            ("max_levels=4", "current native AMR route", "supports max_levels=2"),
        ),
        (
            "AMR refinement ratio unsupported",
            ValueError,
            lambda: AMR(base=CartesianMesh(n=64), ratio=3).validate(),
            ("AMR", "refinement ratio 3", "ratio 2"),
        ),
        (
            "AMR invalid regrid cadence",
            ValueError,
            lambda: RegridEvery(0),
            ("RegridEvery", "steps must be > 0", "FrozenRegrid"),
        ),
        (
            # ADC-542 made dynamic-regrid checkpoints a real feature (format v3), so the old
            # "regrid_every == 0" refusal is gone BY DESIGN; the still-valid refusal on this
            # route is a checkpoint on a block-less engine.
            "AMR checkpoint before any block",
            ValueError,
            lambda: AmrSystem(n=8, L=1.0, periodic=True, regrid_every=3).checkpoint(
                "/tmp/adc597_unwritten_checkpoint"
            ),
            ("AmrSystem.checkpoint", "no blocks installed", "pops.bind"),
        ),
        (
            "program route incompatible with AMR target",
            ValueError,
            lambda: AmrSystem(n=8, L=1.0, periodic=True).add_equation(
                "gas",
                _compiled_model(target="system"),
                spatial=pops.FiniteVolume(limiter=Minmod(), riemann=Rusanov()),
            ),
            ("target='system'", "AmrSystem", "target='amr_system'"),
        ),
    ]
    for label, exc_type, fn, needles in rows:
        _expect_refusal(label, exc_type, fn, needles)


def test_native_abi_header_mismatch_refuses_before_installing_program():
    tmp, so = _compile_bad_program_abi_so()
    try:
        _expect_refusal(
            "native program ABI/header mismatch",
            RuntimeError,
            lambda: System(n=4, L=1.0, periodic=True).install_program(so),
            ("compiled program ABI mismatch", "adc597-wrong-abi-key"),
        )
    finally:
        tmp.cleanup()


def test_descriptor_missing_native_id_refuses_with_loader_reason():
    from pops.numerics.riemann import riemann

    _expect_refusal(
        "descriptor with absent native id",
        LookupError,
        lambda: riemann.User("adc597_missing_riemann"),
        ("adc597_missing_riemann", "not loaded", "load_cpp_library"),
    )


def test_surface_matrix_does_not_use_direct_legacy_bindings():
    """Target surface tests must stay on public adapters, not raw pybind escape hatches."""
    targets = [
        Path(__file__),
        ROOT / "tests/python/unit/physics/test_fv_hll_minmod.py",
        ROOT / "tests/python/unit/runtime/test_spec5_validation_gaps.py",
        ROOT / "tests/python/integration/native_loader/test_spec5_native_capabilities.py",
        ROOT / "tests/python/integration/runtime/test_runtime_inspection_reports.py",
        ROOT / "tests/python/integration/runtime/test_program_runtime_params.py",
    ]
    forbidden = tuple("._s." + name + "(" for name in (
        "add_block", "add_native_block", "add_compiled_block"))
    offenders = []
    for path in targets:
        text = path.read_text()
        for token in forbidden:
            if token in text:
                offenders.append("%s uses %s" % (path.relative_to(ROOT), token))
    assert not offenders, "surface tests must use public routes, not direct legacy bindings: %s" % offenders


def test_fallback_policy_rows_for_refused_routes_are_not_warning_only():
    report = pops.fallback_diagnostics_report()
    rows = {row["key"]: row for row in report["entries"]}
    strict = rows["runtime.limiter_unknown_muscl_ghost"]
    assert strict["policy"] == "refuse_final_route"
    assert "throw" in strict["default_action"]
    for row in rows.values():
        assert row["policy"] != "warning_only", "fallback row %s is warning-only" % row["key"]


def test_positive_matrix_keeps_supported_native_routes_available():
    # Uniform + FV + HLL typed wave speeds.
    n = 16
    sim = System(n=n, L=1.0, periodic=True)
    sim.add_block(
        "ions",
        _isothermal_model(),
        spatial=pops.FiniteVolume(limiter=Minmod(), riemann=HLL(), variables=Primitive()),
        time=pops.Explicit(),
    )
    rho = np.ones((n, n), dtype=np.float64)
    rho[4:8, 4:8] += 0.1
    sim.set_density("ions", rho.ravel())
    sim.step(1e-4)
    assert np.isfinite(np.asarray(sim.density("ions"))).all()

    # Uniform + field problem + native solver.
    assert _poisson_problem(GeometricMG(), Dirichlet()).validate() is True
    assert _poisson_problem(FFT(), Periodic()).validate() is True

    # Program compiled route lowers through the generic ProgramContext.
    emitted = _program_with_context().emit_cpp_program(model=None)
    assert "pops/runtime/program/program_context.hpp" in emitted
    assert "ProgramContext" in emitted
    assert "a later phase" not in emitted

    # AMR + typed tags + declared native field solve support.
    amr_layout = AMR(
        base=CartesianMesh(n=64),
        max_levels=2,
        ratio=2,
        regrid=RegridEvery(4),
        patches=PatchLayout(distribute_coarse=True),
        refine=TagUnion(Refine.on("rho").above(0.05), Refine.on("phi").gradient_above(0.5)),
        nesting=ProperNesting(buffer=1),
        checkpoint=CheckpointPolicy(restartable=True),
        output=AMROutput(fields=["phi"], levels=AllLevels(), include_patch_boxes=True),
    )
    amr_layout.validate()
    amr_report = amr_layout.inspect()
    amr_dict = amr_report["amr_report"] if isinstance(amr_report, dict) else amr_report.to_dict()
    slots = {row["slot"] for row in amr_dict["policies"]}
    assert {"regrid", "refine", "checkpoint", "output"} <= slots
    native_routes = {row.to_dict()["route_id"]: row.to_dict()
                     for row in pops.native_capability_report().routes}
    assert native_routes["layout:AMR"]["status"] == "available"
    assert native_routes["elliptic:geometric_mg"]["status"] == "available"
    assert native_routes["program_context:amr"]["status"] == "available"

    # Runtime params without recompile: pure routing keeps defaults and rejects unknown names.
    per_block, unknown = route_program_params({0: ["k"]}, {"k": 2.0}, {"k": 6.0})
    assert per_block == {0: [6.0]} and unknown == []
    per_block, unknown = route_program_params({0: ["k"]}, {"k": 2.0}, {})
    assert per_block == {0: [2.0]} and unknown == []

    # Structured inspect reports.
    inspected = sim.inspect().to_dict()
    assert inspected["schema_version"] >= 1
    assert "diagnostics" in inspected
    assert "fallbacks" in inspected["diagnostics"]
    assert "solver_events" in inspected["diagnostics"]
