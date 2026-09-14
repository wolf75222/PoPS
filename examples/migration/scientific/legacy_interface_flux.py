"""Compatibility witness for the pre-migration external NumericalFlux protocol.

Retained until the replacement completes native qualification; not a new physics example.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pops
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import (
    DiscretizationPlan,
    FiniteVolume,
    reconstruction,
    riemann,
    variables,
)
from pops.time import (
    FixedDt,
)


HERE = Path(__file__).resolve().parents[1]
SOURCE_ROOT = HERE.parents[1]
DEFAULT_CELLS = 16


from .runtime import _run_case, _state_summary


def _frame(name):
    return Rectangle(name, lower=(0, 0), upper=(1, 1)).frame(Cartesian2D())


def _layout(frame, cells, *, periodic=True):
    return Uniform(
        CartesianGrid(
            frame=frame,
            cells=(cells, cells),
            periodic=PeriodicAxes(frame.axes) if periodic else None,
        )
    )


def _component_source(manifest: Any) -> bytes:
    template = (HERE / "native_scalar_flux.cpp.in").read_text(encoding="ascii")
    replacements = {
        "@COMPONENT_ID@": json.dumps(manifest.component_id),
        "@SEMANTIC_DIGEST@": json.dumps(manifest.semantic_digest.token),
        "@MANIFEST_DIGEST@": json.dumps(manifest.manifest_digest.token),
    }
    for marker, value in replacements.items():
        template = template.replace(marker, value)
    if any(marker in template for marker in replacements):
        raise RuntimeError("native component template contains an unresolved marker")
    return template.encode("ascii")


def _native_flux_source_component(work_dir: Path) -> Any:
    from pops import interfaces
    from pops.external import build_source_package_manifest, load
    from pops.model import ComponentManifest

    interface = interfaces.NumericalFlux
    manifest = ComponentManifest(
        uri="pops://examples.migration/scalar-upwind",
        component_type="numerical_flux",
        version="1.0.0",
        facets=interface.facets,
        signature={
            "generic": True,
            "state_components": 1,
            "native_interface": interface.signature_declaration(),
        },
        interfaces=interface.manifest_declarations(),
        target={
            "variants": [
                {
                    "dimension": 2,
                    "scalar": "float64",
                    "device": "cpu",
                    "features": [],
                }
            ]
        },
        entry_points={"interface_table": "pops_component_interface_v1"},
    )
    source = _component_source(manifest)
    work_dir.mkdir(parents=True, exist_ok=True)
    source_name = "native_scalar_flux.cpp"
    (work_dir / source_name).write_bytes(source)
    package_data = build_source_package_manifest(
        components={"scalar_upwind": manifest},
        payloads={source_name: ("source", source)},
    )
    package_path = work_dir / "native_scalar_flux.pops.json"
    package_path.write_text(json.dumps(package_data, indent=2), encoding="utf-8")
    return load(package_path).require("scalar_upwind", interface=interfaces.NumericalFlux)()


def _native_flux_component(work_dir: Path) -> Any:
    from pops.external import compile_component

    component = _native_flux_source_component(work_dir)
    return compile_component(
        component,
        include=os.environ.get("POPS_INCLUDE"),
        native_dimension=2,
    )


def _imported_primitive_case(
    cells: int, component: Any
) -> tuple[Any, Any, dict[str, np.ndarray], float]:
    from pops.boundary import TransportBoundarySet
    from pops.boundary.transport import Outflow
    from pops.mesh.boundaries import (
        BlockInterfaceSide,
        ConservativeInterface,
    )

    frame = _frame("imported_primitive_square")
    model = pops.Model("imported_primitive_advection", frame=frame)
    state = model.state("U", components=("u",))
    (u,) = state
    velocity = model.vector("velocity", frame=frame, components={axis: 1.0 for axis in frame.axes})
    flux = model.flux(
        "advection",
        frame=frame,
        state=state,
        components={axis: (u,) for axis in frame.axes},
        waves={axis: (1.0,) for axis in frame.axes},
    )
    rate = model.rate("advection_rate", equation=ddt(state) == -div(flux))
    case = pops.Case("imported_native_primitive_case")
    left_block = case.block("left", model)
    right_block = case.block("right", model)
    left_state = left_block[state]
    right_state = right_block[state]
    finite_volume = FiniteVolume(
        flux=flux,
        variables=variables.Conservative(state),
        reconstruction=reconstruction.FirstOrder(),
        riemann=riemann.ScalarUpwind(velocity=velocity),
    )
    boundaries = frame.boundaries

    def numerics(subject: Any) -> DiscretizationPlan:
        plan = DiscretizationPlan()
        plan.rates.add(rate, finite_volume)
        plan.boundaries.add(
            TransportBoundarySet({boundary: Outflow(state=subject) for boundary in boundaries.all})
        )
        return plan

    left_plan = numerics(left_state)
    right_plan = numerics(right_state)
    ConservativeInterface(
        "native_scalar_interface",
        left=BlockInterfaceSide(left_state, boundaries.x_max),
        right=BlockInterfaceSide(right_state, boundaries.x_min),
        numerical_flux=component,
        permutation=(0,),
        right_normal_translation=1.0,
    ).attach(left_plan, right_plan)
    case.numerics(left_plan, block=left_block)
    case.numerics(right_plan, block=right_block)

    program = pops.Program("imported_primitive_forward_euler")
    left_time = program.state(left_state)
    right_time = program.state(right_state)
    stage = program.stage("evaluate", c=0)
    left_rhs = program.value("left_rhs", rate(left_time.n), at=stage)
    right_rhs = program.value("right_rhs", rate(right_time.n), at=stage)
    dt = 1.0e-3
    program.commit(
        left_time.next,
        program.value("left_next", left_time.n + dt * left_rhs, at=left_time.next.point),
    )
    program.commit(
        right_time.next,
        program.value("right_next", right_time.n + dt * right_rhs, at=right_time.next.point),
    )
    program.step_strategy(FixedDt(dt))
    case.program(program)
    initial = {
        "left": np.ones((1, cells, cells), dtype=np.float64),
        "right": np.full((1, cells, cells), 3.0, dtype=np.float64),
    }
    return case, _layout(frame, cells, periodic=False), initial, dt


def _imported_native_primitive(args: argparse.Namespace) -> dict[str, Any]:
    component = _native_flux_component(args.work_dir / "native-component")
    case, layout, initial, dt = _imported_primitive_case(args.cells, component)
    initial_total = sum(float(np.mean(values)) for values in initial.values())
    runtime, report, artifact = _run_case(
        case,
        layout,
        initial,
        dt=dt,
        steps=args.steps,
        components=(component,),
    )
    final_total = sum(
        float(np.mean(np.asarray(runtime.state_global(block)))) for block in ("left", "right")
    )
    return {
        "workflow": "imported-native-primitive",
        "accepted_steps": report.accepted_steps,
        "final_time": float(runtime.time()),
        "artifact": artifact.artifact_identity.token,
        "component_id": component.component_id,
        "component_binary": component.binary_identity.token,
        "source_package": component.source_package.token,
        "left": _state_summary(runtime, "left"),
        "right": _state_summary(runtime, "right"),
        "paired_mean_change": final_total - initial_total,
    }
