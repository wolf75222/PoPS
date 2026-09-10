"""Public Python Gaussian/mixture qualification through resolved native cell integrals.

Run separately with POPS_NATIVE_DIM=1, 2 and 3. Every run covers uniform N=16/32 and
AMR reprojection N=16->32 and N=32->64, two parameter binds per compiled artifact.
Central profiles also advance through scheduled regrids and checkpoint/restart. Every bind
exports its arrays and independent mathematical oracle, including the nonzero far tails.
"""

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import pops
from pops.analytic import constant, param
from pops.amr import (
    AMRExecution,
    AMRHierarchy,
    AMRRegrid,
    AMRTagging,
    AMRTransfer,
    Buffer,
    ConflictPolicy,
    EqualityPolicy,
    Hysteresis,
    Tag,
)
from pops.codegen import Production
from pops.domain import CartesianDomain
from pops.initial import InitialCondition
from pops.layouts import AMR, Uniform
from pops.lib.amr import StateTransfer
from pops.lib.initial import Analytic, Gaussian
from pops.lib.time import ForwardEuler
from pops.math import ValueExpr, ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.params import RuntimeParam
from pops.projection import ConservativeCellAverage
from pops.time import FixedDt, every
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.integration.runtime.test_dsl_runtime_params import (
    _binary_paths,
    _binary_fingerprints,
)

pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]
POPS_PROCESS_TIMEOUT = 1200
ROOT = Path(__file__).resolve().parents[4]
LEGACY_SOURCE = "bc012531a72ad08dde4ebd1d7548518ad82ae347"


@dataclass(frozen=True, slots=True)
class _LegacyGaussian:
    """Test-only old InitialCondition protocol, exercising the retained native Gaussian route.

    Source: bc012531a72ad08dde4ebd1d7548518ad82ae347,
    python/pops/lib/initial/__init__.py:175-242. This compares two routes in the current
    native library; it does not stand in for executing the complete historical package.
    """

    frame: Any
    center: tuple[tuple[Any, float], ...]
    background: float
    amplitude: float
    inverse_width: float
    native_route = "gaussian_field"
    reprojectable = True
    __pops_ir_immutable__ = True

    def validate_for(self, state):
        declaration = getattr(state, "declaration_ref", None) or state
        components = getattr(declaration, "components", None)
        if components is None:
            components = getattr(getattr(declaration, "space", None), "components", None)
        if not isinstance(components, tuple) or len(components) != 1:
            raise ValueError("legacy Gaussian requires exactly one declared component")
        if getattr(getattr(declaration, "space", None), "frame", None) != self.frame.canonical_id:
            raise ValueError("legacy Gaussian frame differs from the target state frame")
        if tuple(axis for axis, _ in self.center) != self.frame.axes:
            raise ValueError("legacy Gaussian center must cover its exact frame axes")
        if (
            not all(
                math.isfinite(value)
                for value in (
                    self.background,
                    self.amplitude,
                    self.inverse_width,
                    *(value for _, value in self.center),
                )
            )
            or self.inverse_width <= 0
        ):
            raise ValueError("legacy Gaussian requires finite parameters and positive width")
        return True

    def initial_source_options(self):
        return {
            "native_route": self.native_route,
            "frame_id": self.frame.canonical_id,
            "center": {axis.name: value for axis, value in self.center},
            "background": self.background,
            "amplitude": self.amplitude,
            "inverse_width": self.inverse_width,
        }

    def to_data(self):
        data = self.initial_source_options()
        del data["native_route"]
        return {"schema_version": 1, "profile": "gaussian", **data}

    canonical_identity = to_data


def _legacy_profile(frame, dim, weight, tail_sign):
    if tail_sign is not None:
        return _LegacyGaussian(frame, ((frame.axes[0], 0.0),), 0.0, weight, 1.0)
    centers = (0.35,) + (0.55,) * (dim - 1)
    root = math.sqrt(80.0)
    integral = math.prod(
        math.sqrt(math.pi) / (2 * root) * (math.erf(root * (1 - center)) + math.erf(root * center))
        for center in centers
    )
    return _LegacyGaussian(
        frame, tuple(zip(frame.axes, centers, strict=True)), 1 - weight * integral, weight, 80.0
    )


def _normalized_gaussian(frame, centers, width):
    root = math.sqrt(width)
    integral = math.prod(
        math.sqrt(math.pi) / (2 * root) * (math.erf(root * (1 - center)) + math.erf(root * center))
        for center in centers
    )
    return Gaussian(
        frame=frame,
        center=dict(zip(frame.axes, centers, strict=True)),
        background=1 - integral,
        inverse_width=width,
    ).as_analytic()


def _case(target, n, dim, mixture, *, tail_sign=None, legacy_weight=None):
    bounds = ((0.0,) * dim, (1.0,) * dim)
    if tail_sign is not None:
        assert dim == 1 and tail_sign in (-1, 1)
        bounds = ((8.0,), (8.1,)) if tail_sign == 1 else ((-8.1,), (-8.0,))
    frame = CartesianDomain(
        "renamed-mixture-support" if mixture else "neutral-gaussian-support", *bounds
    ).frame()
    model = pops.Model("zero-transport", frame=frame)
    state = model.state("U", components=("rho",))
    (rho,) = state
    flux = model.flux(
        "flux",
        frame=frame,
        state=state,
        components={axis: (0 * rho,) for axis in frame.axes},
        waves={axis: (0 * rho,) for axis in frame.axes},
    )
    rate = model.rate("rate", equation=ddt(state) == -div(flux))
    case = pops.Case("exact-initial-%s" % target)
    block = case.block("renamed-density", model)
    block_state = block[state]
    numerics = DiscretizationPlan()
    numerics.rates.add(
        rate,
        FiniteVolume(
            flux=flux,
            variables=variables.Conservative(state),
            reconstruction=reconstruction.FirstOrder(),
            riemann=riemann.Rusanov(),
        ),
    )
    case.numerics(numerics, block=block)
    program = ForwardEuler(block_state, rate=rate)
    program.step_strategy(FixedDt(0.001))
    case.program(program)
    weight = case.param(RuntimeParam("mixture-weight", default=1.0))
    if tail_sign is not None:
        first = Gaussian(
            frame=frame, center={frame.axes[0]: 0.0}, background=0.0, inverse_width=1.0
        ).as_analytic()
        second = Gaussian(
            frame=frame,
            center={frame.axes[0]: 0.1 * tail_sign},
            background=0.0,
            inverse_width=1.0,
            amplitude=1.0 if mixture else 0.0,
        ).as_analytic()
    else:
        first = _normalized_gaussian(frame, (0.35,) + (0.55,) * (dim - 1), 80.0)
    if tail_sign is None and mixture:
        second = _normalized_gaussian(frame, (0.65,) + (0.45,) * (dim - 1), 120.0)
    elif tail_sign is None:
        from pops.analytic import CellBounds

        second = Analytic(
            frame=frame, components=(constant(1),), cell_integrals=(CellBounds(frame).measure,)
        )
    # Both the physical profile and its explicit integral are composed through public Python.
    profile = Analytic(
        frame=frame,
        components=(
            param(weight) * first.components[0] + (1 - param(weight)) * second.components[0],
        ),
        cell_integrals=(
            param(weight) * first.cell_integrals[0]
            + (1 - param(weight)) * second.cell_integrals[0],
        ),
    )
    if legacy_weight is not None:
        assert not mixture, "a mixture has no single-Gaussian legacy equivalent"
        profile = _legacy_profile(frame, dim, legacy_weight, tail_sign)
    case.initials.add(
        InitialCondition(state=block_state, value=profile, projection=ConservativeCellAverage())
    )
    grid = CartesianGrid(frame=frame, cells=(n,) * dim, periodic=PeriodicAxes(frame.axes))
    if target == "system":
        layout = Uniform(grid)
    else:
        threshold = case.param(RuntimeParam("refine-threshold", default=0.5))
        transfer = AMRTransfer()
        transfer.state(block_state, StateTransfer())
        layout = AMR(
            grid=grid,
            hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
            tagging=AMRTagging(
                rules=(Tag(ValueExpr(block_state) > case.value(threshold)), Buffer(cells=1)),
                hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
                conflict_policy=ConflictPolicy.REFINE_WINS,
            ),
            regrid=AMRRegrid(schedule=every(2, clock=program.clock)),
            transfer=transfer,
            execution=AMRExecution.synchronous(),
        )
    validated = pops.validate(case)
    return pops.resolve(
        validated,
        layout=layout,
        backend=Production(),
        compile_options={"include": str(ROOT / "include")},
    ), validated.resolve(weight)


def _oracle(n, centers, width):
    root = math.sqrt(width)
    factors = []
    total_integral = 1.0
    for center in centers:
        total_integral *= (
            math.sqrt(math.pi)
            / (2 * root)
            * (math.erf(root * (1 - center)) + math.erf(root * center))
        )
        factors.append(
            np.array(
                [
                    math.sqrt(math.pi)
                    * n
                    / (2 * root)
                    * (math.erf(root * ((i + 1) / n - center)) - math.erf(root * (i / n - center)))
                    for i in range(n)
                ]
            )
        )
    result = np.ones((n,) * len(centers))
    for axis, factor in enumerate(factors):
        shape = [1] * len(centers)
        shape[len(centers) - axis - 1] = n
        result *= factor.reshape(shape)
    return 1 - total_integral + result


def _image(simulation, target, n, dim):
    levels = simulation.n_levels()
    states = {}
    for level in range(levels):
        state = (
            simulation.get_state("renamed-density")
            if target == "system"
            else simulation.block_level_state_global("renamed-density", level)
        )
        states[level] = np.array(state, copy=True).reshape((n * 2**level,) * dim)
    clock = {"time": simulation.time(), "macro_step": simulation.macro_step(), "levels": levels}
    if target == "amr_system":
        report = simulation.amr.explain_regrid()
        clock.update(
            regrid_count=report.regrid_count,
            topology_epoch=report.topology_epoch,
            patch_boxes=simulation.patch_boxes(),
        )
    return states, clock


def _assert_neutral_image(states, expected):
    assert states.keys() == expected.keys()
    for level, state in states.items():
        actual = np.asarray(state, dtype=np.float64)
        np.testing.assert_allclose(actual, expected[level], rtol=0, atol=128 * np.finfo(float).eps)
        residual = np.mean(actual.astype(np.longdouble) - 1)
        assert abs(residual) <= 128 * np.finfo(float).eps


def _assert_same_image(actual, expected):
    actual_states, actual_clock = actual
    expected_states, expected_clock = expected
    assert actual_clock == expected_clock
    assert actual_states.keys() == expected_states.keys()
    for level, state in actual_states.items():
        assert state.dtype == expected_states[level].dtype
        np.testing.assert_array_equal(state, expected_states[level])


def _evidence_metadata(
    artifact, simulation, context, before, *, target, n, dim, mixture, weight, tail_sign=None
):
    communicator = context.communicator.handle
    (binding,) = artifact.plan.initial_condition_plan.bindings
    source = binding.source.options.to_data()
    return {
        "schema": "pops.exact-analytic-initial-arrays.v1",
        "target": target,
        "native_dimension": dim,
        "base_cells_per_axis": n,
        "profile": "mixture" if mixture else "gaussian",
        "mixture_weight": weight,
        "tail_sign": tail_sign,
        "dt": 0.001,
        "domain_bounds": ([0.0] * dim, [1.0] * dim)
        if tail_sign is None
        else (([8.0], [8.1]) if tail_sign == 1 else ([-8.1], [-8.0])),
        "native_route": source["native_route"],
        "initial_source": source,
        "legacy_provider_source": None
        if source["native_route"] != "gaussian_field"
        else {
            "commit": LEGACY_SOURCE,
            "path": "python/pops/lib/initial/__init__.py",
            "comparison_scope": "retained old route in the current native library",
        },
        "regrid_every_steps": 2 if target == "amr_system" else None,
        "rank": 0 if communicator is None else int(communicator.rank),
        "ranks": 1 if communicator is None else int(communicator.size),
        "native_storage_precision": artifact.platform_manifest.precision.storage.require(
            "Gaussian qualification storage precision"
        ),
        "platform": artifact.platform_manifest.to_data(),
        "artifact_identity": artifact.artifact_identity.to_data(),
        "bind_identity": simulation.bind_identity.to_data(),
        "test_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "binaries": [
            {"path": str(path), "sha256": digest, "size": size, "mtime_ns": mtime}
            for path, (digest, size, mtime) in before.items()
        ],
    }


def _export_image(tmp_path, record_property, phase, image, expected, metadata):
    states, clock = image
    identifier = (
        "{native_route}-{profile}-{target}-d{native_dimension}-n{base_cells_per_axis}"
    ).format(**metadata)
    if metadata["tail_sign"] is not None:
        identifier += "-tail-%s" % ("positive" if metadata["tail_sign"] == 1 else "negative")
    identifier += "-w%s-r%d" % (str(metadata["mixture_weight"]).replace(".", "p"), metadata["rank"])
    path = tmp_path / (identifier + "-" + phase + ".npz")
    assert not path.exists(), "every qualification image must have its own evidence file"
    arrays = {"actual_level_%d" % level: state for level, state in states.items()}
    arrays.update({"expected_level_%d" % level: state for level, state in expected.items()})
    details = {
        **metadata,
        **clock,
        "phase": phase,
        "array_dtypes": {str(level): str(state.dtype) for level, state in states.items()},
    }
    np.savez(
        path, metadata=np.asarray(json.dumps(details, sort_keys=True, allow_nan=False)), **arrays
    )
    record_property(identifier + "-" + phase, str(path))
    record_property(
        identifier + "-" + phase + "-sha256", hashlib.sha256(path.read_bytes()).hexdigest()
    )


def _legacy_image(target, n, dim, weight, *, tail_sign=None):
    # Old source options contain concrete numbers: compile separately for each weight. The
    # generic route's parameter-rebind/binary-reuse promise is deliberately not assigned here.
    resolved, parameter = _case(target, n, dim, False, tail_sign=tail_sign, legacy_weight=weight)
    (binding,) = resolved.initial_condition_plan.bindings
    assert binding.source.options.to_data()["native_route"] == "gaussian_field"
    artifact = pops.compile(resolved)
    paths = _binary_paths(artifact)
    before = _binary_fingerprints(paths)
    context = artifact_execution_context(artifact)
    simulation = pops.bind(
        artifact, params={parameter: weight}, resources={"execution_context": context}
    )
    image = _image(simulation, target, n, dim)
    metadata = _evidence_metadata(
        artifact,
        simulation,
        context,
        before,
        target=target,
        n=n,
        dim=dim,
        mixture=False,
        weight=weight,
        tail_sign=tail_sign,
    )
    assert _binary_fingerprints(paths) == before
    return image, metadata


@pytest.mark.parametrize("target", ["system", "amr_system"], ids=["uniform", "amr"])
@pytest.mark.parametrize("n", [16, 32])
@pytest.mark.parametrize("mixture", [False, True], ids=["gaussian", "mixture"])
def test_public_exact_initial_neutrality_and_reprojection(
    target, n, mixture, isolated_native_cache, native_cxx, kokkos_root, tmp_path, record_property
):
    del isolated_native_cache, native_cxx, kokkos_root
    configured = os.environ.get("POPS_NATIVE_DIM")
    assert configured in {"1", "2", "3"}, "qualification requires explicit POPS_NATIVE_DIM=1/2/3"
    dim = int(configured)
    resolved, parameter = _case(target, n, dim, mixture)
    (binding,) = resolved.initial_condition_plan.bindings
    assert binding.source.options.to_data()["native_route"] == "analytic_expression"
    artifact = pops.compile(resolved)
    paths = _binary_paths(artifact)
    before = _binary_fingerprints(paths)
    for weight in (0.25, 0.75) if mixture else (1.0, 0.5):
        context = artifact_execution_context(artifact)
        simulation = pops.bind(
            artifact, params={parameter: weight}, resources={"execution_context": context}
        )
        levels = 1 if target == "system" else simulation.n_levels()
        assert levels == (1 if target == "system" else 2)
        expected = {}
        for level in range(levels):
            cells = n * 2**level
            first = _oracle(cells, (0.35,) + (0.55,) * (dim - 1), 80.0)
            second = _oracle(cells, (0.65,) + (0.45,) * (dim - 1), 120.0) if mixture else 1.0
            expected[level] = weight * first + (1 - weight) * second
        metadata = _evidence_metadata(
            artifact,
            simulation,
            context,
            before,
            target=target,
            n=n,
            dim=dim,
            mixture=mixture,
            weight=weight,
        )
        initial = _image(simulation, target, n, dim)
        # The fine-level image records the original analytic reprojection, before any transfer.
        _export_image(tmp_path, record_property, "initial-reprojected", initial, expected, metadata)
        _assert_neutral_image(initial[0], expected)
        assert initial[1]["time"] == 0.0 and initial[1]["macro_step"] == 0
        assert _binary_fingerprints(paths) == before
        if not mixture:
            legacy, legacy_metadata = _legacy_image(target, n, dim, weight)
            legacy_metadata["paired_generic_artifact_identity"] = metadata["artifact_identity"]
            _export_image(
                tmp_path, record_property, "initial-reprojected", legacy, expected, legacy_metadata
            )
            _assert_neutral_image(legacy[0], expected)
            assert legacy[1]["levels"] == initial[1]["levels"]
            assert legacy[1]["time"] == 0.0 and legacy[1]["macro_step"] == 0
            for level, actual in initial[0].items():
                np.testing.assert_allclose(
                    actual, legacy[0][level], rtol=0, atol=128 * np.finfo(float).eps
                )

        # Cross the scheduled regrid at step 2, then checkpoint off the regrid boundary.
        pops.run(simulation, t_end=0.003, max_steps=3)
        saved = _image(simulation, target, n, dim)
        _export_image(tmp_path, record_property, "checkpoint", saved, expected, metadata)
        _assert_neutral_image(saved[0], expected)
        assert saved[1]["time"] == 0.003 and saved[1]["macro_step"] == 3
        assert saved[1]["levels"] == levels
        if target == "amr_system":
            assert saved[1]["regrid_count"] == initial[1]["regrid_count"] + 1
            assert saved[1]["topology_epoch"] == initial[1]["topology_epoch"] + 1
        checkpoint = simulation.checkpoint(tmp_path / ("checkpoint-w%s" % weight))
        record_property("checkpoint-w%s" % weight, str(checkpoint))
        assert _binary_fingerprints(paths) == before

        # The uninterrupted and restarted paths both execute step 4's scheduled regrid.
        pops.run(simulation, t_end=0.005, max_steps=2)
        uninterrupted = _image(simulation, target, n, dim)
        _export_image(tmp_path, record_property, "uninterrupted", uninterrupted, expected, metadata)
        _assert_neutral_image(uninterrupted[0], expected)
        assert uninterrupted[1]["time"] == 0.005 and uninterrupted[1]["macro_step"] == 5
        assert uninterrupted[1]["levels"] == levels
        if target == "amr_system":
            assert uninterrupted[1]["regrid_count"] == saved[1]["regrid_count"] + 1
            assert uninterrupted[1]["topology_epoch"] == saved[1]["topology_epoch"] + 1
        assert _binary_fingerprints(paths) == before

        simulation.restart(checkpoint)
        restored = _image(simulation, target, n, dim)
        _export_image(tmp_path, record_property, "restored", restored, expected, metadata)
        _assert_same_image(restored, saved)
        assert _binary_fingerprints(paths) == before
        pops.run(simulation, t_end=0.005, max_steps=2)
        replayed = _image(simulation, target, n, dim)
        _export_image(tmp_path, record_property, "replayed", replayed, expected, metadata)
        _assert_neutral_image(replayed[0], expected)
        _assert_same_image(replayed, uninterrupted)
        assert _binary_fingerprints(paths) == before


def _tail_oracle(n, sign, center):
    """Independent CPython erfc bound difference, with no additive background."""
    lower, upper = (8.0, 8.1) if sign == 1 else (-8.1, -8.0)
    dx = (upper - lower) / n
    values = []
    for cell in range(n):
        lo, hi = lower + cell * dx - center, lower + (cell + 1) * dx - center
        # Reflect negative intervals before evaluating positive erfc arguments.
        if hi < 0:
            lo, hi = -hi, -lo
        values.append(math.sqrt(math.pi) / (2 * dx) * (math.erfc(lo) - math.erfc(hi)))
    return np.asarray(values)


@pytest.mark.parametrize("sign", [-1, 1], ids=["negative-tail", "positive-tail"])
@pytest.mark.parametrize("mixture", [False, True], ids=["gaussian", "mixture"])
def test_public_exact_initial_preserves_nonzero_far_tails(
    sign, mixture, isolated_native_cache, native_cxx, kokkos_root, tmp_path, record_property
):
    del isolated_native_cache, native_cxx, kokkos_root
    configured = os.environ.get("POPS_NATIVE_DIM")
    assert configured in {"1", "2", "3"}, "qualification requires explicit POPS_NATIVE_DIM"
    if configured != "1":
        pytest.skip("far-tail qualification is declared for native Dim=1")
    n = 8
    resolved, parameter = _case("system", n, 1, mixture, tail_sign=sign)
    artifact = pops.compile(resolved)
    paths = _binary_paths(artifact)
    before = _binary_fingerprints(paths)
    first, second = _tail_oracle(n, sign, 0.0), _tail_oracle(n, sign, 0.1 * sign)
    for weight in (0.25, 0.75) if mixture else (1.0, 0.5):
        context = artifact_execution_context(artifact)
        simulation = pops.bind(
            artifact, params={parameter: weight}, resources={"execution_context": context}
        )
        actual = np.asarray(simulation.get_state("renamed-density"), dtype=np.float64).reshape(n)
        expected = weight * first + ((1 - weight) * second if mixture else 0.0)
        metadata = _evidence_metadata(
            artifact,
            simulation,
            context,
            before,
            target="system",
            n=n,
            dim=1,
            mixture=mixture,
            weight=weight,
            tail_sign=sign,
        )
        _export_image(
            tmp_path,
            record_property,
            "initial",
            _image(simulation, "system", n, 1),
            {0: expected},
            metadata,
        )
        assert np.all(expected > 0) and np.all(actual > 0)
        # An erf(hi)-erf(lo) implementation returns zero on every one of these cells.
        # There is no absolute tolerance that could hide cancellation of this tiny signal.
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=0)
        if not mixture:
            legacy, legacy_metadata = _legacy_image("system", n, 1, weight, tail_sign=sign)
            legacy_metadata.update(
                paired_generic_artifact_identity=metadata["artifact_identity"],
                classification="invalid-old-nonzero-cell-integral",
                invariant="a positive Gaussian has a strictly positive integral on every cell",
            )
            _export_image(
                tmp_path,
                record_property,
                "invalid-zero-tail",
                legacy,
                {0: expected},
                legacy_metadata,
            )
            # Classify the actually executed old failure, never equality to an invalid baseline.
            assert np.all(legacy[0][0] == 0)
        assert _binary_fingerprints(paths) == before
