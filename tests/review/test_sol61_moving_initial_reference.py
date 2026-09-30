"""Independent real Case/resolve and initial storage preflight; no native/JIT."""

from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pops
import pytest
from pops.codegen._plans import _canonicalize_initial_value_mapping
from pops.domain import CartesianDomain
from pops.frames import Cartesian1D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.analytic import coordinate, sin, time
from pops.math import ddt, div
from pops.mesh import (
    CartesianGrid,
    GeometryEvolution,
    LayoutPlanBuilder,
    MovingControlVolumes,
    PeriodicAxes,
)
from pops.mesh._layout_plan_contracts import NormalizedGeometry
from pops.numerics import DiscretizationPlan, FiniteVolume, reconstruction, riemann, variables
from pops.projection import ConservativeCellAverage
from pops.runtime._bind_validation import _layout_mesh, validate_bound_initial_values
from pops.time import FixedDt, MovingFieldProjection


def public_case(*, multiple=False):
    frame = CartesianDomain("independent-reference", (0.0,), (2.0,)).frame(Cartesian1D())
    model = pops.Model("independent-moving-transport", frame=frame)
    state = model.state("particles", components=("last", "first", "middle"))
    flux = model.flux(
        "transport",
        frame=frame,
        state=state,
        components={axis: tuple(0.31 * q for q in state) for axis in frame.axes},
        waves={axis: (0.31,) * 3 for axis in frame.axes},
    )
    equation = model.rate("balance", equation=ddt(state) == -div(flux))
    case = pops.Case("independent-moving-storage")
    program = pops.Program("explicit-reference-step")
    x = coordinate(frame, frame.axes[0])
    motion = GeometryEvolution((x + 0.021 * sin(3.141592653589793 * x) * time(program.clock),))
    grids = {}
    handles = {}
    for name, n in (("slow", 13), ("fast", 7)) if multiple else (("slow", 13),):
        block = case.block(name, model)
        spatial = DiscretizationPlan()
        spatial.rates.add(
            equation,
            FiniteVolume(
                flux=flux,
                variables=variables.Conservative(state),
                reconstruction=reconstruction.FirstOrder(),
                riemann=riemann.Rusanov(),
            ),
        )
        case.numerics(spatial, block=block)
        case.initials.add(
            InitialCondition(
                state=block[state], value=BindArray(), projection=ConservativeCellAverage()
            )
        )
        temporal = program.state(block[state])
        geometry = program.geometry_state(temporal.n, evolution=motion)
        endpoint = program.reynolds_update(
            geometry,
            physical_rate=equation(temporal.n),
            projection=MovingFieldProjection((Fraction(1, 2), Fraction(1, 2)), (1, 0)),
            geometry_tolerance=1e-13,
            at=temporal.next.point,
        )
        program.commit(temporal.next, endpoint)
        grids[name] = MovingControlVolumes(
            Uniform(CartesianGrid(frame=frame, cells=(n,), periodic=PeriodicAxes(frame.axes))),
            evolution=motion,
        )
        handles[name] = block[state]
    program.step_strategy(FixedDt(0.003))
    case.program(program)
    if not multiple:
        plan = pops.resolve(pops.validate(case), layout=grids["slow"])
    else:
        subjects = case.layout_subjects()
        builder = LayoutPlanBuilder(case.owner_path.canonical())
        providers = {}
        for name, descriptor in grids.items():
            layout = builder.layout(name + "-grid", descriptor)
            providers[layout] = descriptor
            builder.assign_block(next(b for b in subjects.blocks if b.local_id == name), layout)
            builder.assign_state(
                next(s for s in subjects.states if s.block_ref.local_id == name), layout
            )
        declared = builder.resolve(**subjects.to_dict())
        plan = pops.resolve(pops.validate(case), layout=declared, layout_providers=providers)
    return plan, grids, handles


def check_public(plan, supplied):
    values = _canonicalize_initial_value_mapping(plan.initial_condition_plan, supplied)
    instances = {row.name: {"components": 3} for row in plan.blocks}
    return validate_bound_initial_values(
        SimpleNamespace(precision="double"),
        SimpleNamespace(instances=instances),
        plan.layout,
        values,
    )


@pytest.mark.parametrize("multiple", (False, True))
def test_real_moving_case_uses_each_exact_block_reference(multiple):
    plan, grids, handles = public_case(multiple=multiple)
    assert all(not hasattr(layout, "mesh") for layout in grids.values())
    arrays = {
        handles[name]: np.zeros((3, 13 if name == "slow" else 7), dtype=np.float64)
        for name in handles
    }
    assert check_public(plan, arrays) == []
    wrong = {**arrays, handles["slow"]: np.zeros((3, 12))}
    errors = check_public(plan, wrong)
    assert len(errors) == 1 and "slow" in errors[0] and "complete state" in errors[0]
    if multiple:
        swapped = {
            handles["slow"]: arrays[handles["fast"]],
            handles["fast"]: arrays[handles["slow"]],
        }
        assert len(check_public(plan, swapped)) == 2
    with pytest.raises((KeyError, ValueError), match="initial|subject|registered|author|binding"):
        foreign_plan, _, foreign_handles = public_case(multiple=multiple)
        assert foreign_plan is not plan
        check_public(plan, {foreign_handles["slow"]: np.zeros((3, 13))})


@pytest.mark.parametrize(
    "shape,dtype",
    (((13, 3), np.float64), ((13,), np.float64), ((3, 15), np.float64), ((3, 13), np.float32)),
)
def test_typed_reference_refuses_transpose_bare_halo_and_precision(shape, dtype):
    plan, _, handles = public_case()
    errors = check_public(plan, {handles["slow"]: np.zeros(shape, dtype=dtype)})
    assert len(errors) == 1
    assert "complete state" in errors[0] or "declared precision" in errors[0]


class Projected:
    def __init__(self, geometry):
        self.geometry = geometry

    def normalized_geometry(self):
        return self.geometry


def geometry(cells):
    rank = len(cells)
    return NormalizedGeometry(
        "pops://coordinates/independent-reference@1",
        "pops://measure/independent-reference@1",
        tuple("xyz"[:rank]),
        (0.0,) * rank,
        (1.0,) * rank,
        cells,
    )


@pytest.mark.parametrize("cells", ((11,), (11, 7), (11, 7, 5)))
def test_ranked_reference_contract_is_open_exact_and_native_axis_order(cells):
    layout = Projected(geometry(cells))
    assert _layout_mesh(layout) == cells
    subject = SimpleNamespace

    class Key:
        block_ref = subject(local_id="material")

    args = subject(instances={"material": {"components": 2}})
    manifest = subject(precision="double")
    assert (
        validate_bound_initial_values(
            manifest, args, layout, {Key(): np.zeros((2, *reversed(cells)))}
        )
        == []
    )
    if len(cells) > 1:
        assert (
            "complete state"
            in validate_bound_initial_values(
                manifest, args, layout, {Key(): np.zeros((2, *cells))}
            )[0]
        )


@pytest.mark.parametrize("invalid", (None, SimpleNamespace(cells=(13,)), (13,)))
def test_invalid_projection_cannot_fall_back_to_correct_mesh(invalid):
    layout = Projected(invalid)
    layout.mesh = SimpleNamespace(cells=(13,))
    assert _layout_mesh(layout) is None


def test_bad_ranked_geometry_refuses_at_its_authority_constructor():
    for cells, axes, lower, upper in (
        ((13, 7), ("x",), (0.0,), (1.0,)),
        ((0,), ("x",), (0.0,), (1.0,)),
        ((13,), ("x",), (1.0,), (0.0,)),
    ):
        with pytest.raises(
            (ValueError, TypeError), match="rank|positive|axis|above|dimension|one|length"
        ):
            NormalizedGeometry(
                "pops://coordinates/test@1", "pops://measure/test@1", axes, lower, upper, cells
            )


def test_existing_amr_runtime_projection_precedes_normalized_projection():
    class Adaptive(Projected):
        def capabilities(self):
            return {"layout": "amr"}

        def runtime_layout_data(self):
            return {"grid": {"cells": [9, 5]}}

        def normalized_geometry(self):
            raise AssertionError("AMR must keep its established runtime route")

    assert _layout_mesh(Adaptive(None)) == (9, 5)


def test_exact_contract_rejects_subclass_projection():
    class GeometrySubclass(NormalizedGeometry):
        pass

    honest = geometry((13,))
    subclass = GeometrySubclass(
        honest.coordinate_system,
        honest.cell_measure,
        honest.axis_names,
        honest.lower,
        honest.upper,
        honest.cells,
    )
    assert _layout_mesh(Projected(subclass)) is None


def test_reference_rank_does_not_relax_native_spatial_layout_dimensions():
    from pops.mesh._layout_plan_contracts import NativeSpatialLayout

    reference = NormalizedGeometry(
        "pops://coordinates/higher-reference@1",
        "pops://measure/higher-reference@1",
        ("x", "y", "z", "w"),
        (0.0,) * 4,
        (1.0,) * 4,
        (3, 4, 5, 6),
    )
    assert reference.dimension == 4  # Deliberately rank-generic reference contract.
    with pytest.raises(ValueError, match="only dimensions 1, 2, and 3"):
        NativeSpatialLayout(
            "independent",
            reference.coordinate_system,
            reference.cell_measure,
            reference.axis_names,
            reference.cells,
            reference.lower,
            reference.upper,
            (True,) * 4,
            "cell",
            {"kind": "replicated"},
        )


@pytest.mark.parametrize("multiple", (False, True))
def test_real_parent_method_refuses_same_honest_moving_arrays(monkeypatch, multiple):
    import ast
    from pathlib import Path
    import subprocess
    from pops.runtime import _bind_validation as validation

    root = Path(__file__).resolve().parents[2]
    source = subprocess.run(
        ["rtk", "git", "show", "470dfb08^:python/pops/runtime/_bind_validation.py"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    tree = ast.parse(source)
    definition = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_layout_mesh"
    )
    module = ast.Module(body=[definition], type_ignores=[])
    namespace = dict(vars(validation))
    exec(compile(module, "parent470dfb08:_layout_mesh", "exec"), namespace)
    plan, _, handles = public_case(multiple=multiple)
    arrays = {
        handle: np.zeros((3, 13 if name == "slow" else 7)) for name, handle in handles.items()
    }
    assert check_public(plan, arrays) == []
    monkeypatch.setattr(validation, "_layout_mesh", namespace["_layout_mesh"])
    errors = check_public(plan, arrays)
    assert len(errors) == len(handles)
    assert all("reference-grid cells" in error for error in errors)
