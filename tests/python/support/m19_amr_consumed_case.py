"""Non-kinetic declared Helmholtz output consumed through two physical maps."""

from fractions import Fraction
import pops
from pops.model import (
    Module,
    Rate,
    FieldSpace,
    Handle,
    OwnerPath,
    PhysicalSupport,
    PhysicalDimension,
)
from pops.fields import FieldProblem, FieldBoundary, FieldDiscretization, bcs
from pops.fields.methods import CellCenteredSecondOrder
from pops.math import Reaction
from pops._ir.elliptic import DivCoeffGrad
from pops.solvers import CompositeFieldGMRES
from pops.time import FailRun, FixedDt
from pops.numerics.terms import SourceTerm
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.mesh import (
    CartesianGrid,
    PeriodicAxes,
    LayoutPlanBuilder,
    LayoutRepresentation,
    LayoutMappingOperation,
    LayoutSynchronization,
    PhysicalSupportMap,
    AxisQuadrature,
    native_physical_mapping,
)

WEIGHTS = (
    (Fraction(1, 8), Fraction(1, 8), Fraction(1, 4), Fraction(1, 2)),
    (Fraction(-1, 4), Fraction(0), Fraction(1, 4), Fraction(0)),
)
NAMES = ("reservoir_z", "zeta_laminated_material", "reservoir_a")


def build(directory, *, reverse=False, provider_factory=None):
    if provider_factory is None:
        provider_factory = fault_provider
    from pops.params import RuntimeParam

    unit = PhysicalDimension()
    slab = PhysicalSupport((("eta", "normalized thickness"), ("x", "periodic rod")))
    rod = PhysicalSupport((("x", "periodic rod"),))
    frame = Rectangle("storage x then eta", (0, 0), (1, 1)).frame(Cartesian2D())
    case = pops.Case("AMR equation-owned scalar and changing hierarchy")
    threshold = case.param(RuntimeParam("source refinement threshold", default=2.2))
    source = Module("nonkinetic temperature load", frame=frame)
    space = source.state_space(
        "thermal coefficients",
        ("unused", "heat_load"),
        support=slab,
        units=(unit, unit),
        sampling="cell_average",
    )

    @source.operator("change_load", signature=space >> Rate(space), kind="local_source")
    def change_load(state):
        return (0 * state[0], 32 + 0 * state[1])

    destinations = []
    for i, name in enumerate((NAMES[0], NAMES[2])):
        model = Module("thermal reservoir " + str(i), frame=frame)
        evolved = model.state_space(
            "accumulation",
            ("spectator", "enthalpy", "marker"),
            support=rod,
            units=(unit, unit, unit),
            sampling="cell_average",
        )
        incoming = model.field_space(
            "applied temperature", ("delivered",), support=rod, units=(unit,), sampling="cell"
        )
        factor = i + 1

        @model.operator(
            "consume_mapped_temperature",
            signature=(evolved, incoming) >> Rate(evolved),
            kind="local_source",
        )
        def rhs(state, field, factor=factor):
            return (0 * state[0], factor * field[0], 0 * state[2])

        block = case.block(name, model)
        destinations.append(
            (block, block[model.state_handle(evolved)], block[model.field_handle(incoming)], model)
        )
        if i == 0:
            src = case.block(NAMES[1], source)
    source_state = src[source.state_handle(space)]
    phi = Handle("screened_temperature", kind="field", owner=OwnerPath.model("thermal potential"))
    physical = FieldProblem(
        "screened thermal response",
        unknowns=(phi,),
        equations=(Reaction(phi, 1) - DivCoeffGrad(phi, 0.125) == source.state_symbols(space)[1],),
        boundaries=(
            FieldBoundary(phi, bcs.BoundaryCondition(bcs.AllPhysicalBoundaries(), bcs.Periodic())),
        ),
        unknown_spaces={
            phi: FieldSpace(
                "temperature", ("temperature",), support=slab, units=(unit,), sampling="cell"
            )
        },
        coordinate_units=(unit, unit),
    )
    field = case.field(
        physical,
        FieldDiscretization(
            method=CellCenteredSecondOrder(),
            boundaries=(),
            observation_axes=(1, 0),
            solver=CompositeFieldGMRES(max_iter=100, restart=25, rel_tol=1e-12, abs_tol=1e-12),
        ),
    )
    program = pops.Program("solve once and consume two mapped outputs")
    current = program.state(source_state)
    solution = field.observe(
        program.solve(field, values={source_state: current.n}, at=current.n.point).consume(
            action=FailRun()
        )
    )
    port = solution.mapping_port(field[phi], factor=1)
    maps = []
    order = (
        tuple(reversed(tuple(enumerate(destinations))))
        if reverse
        else tuple(enumerate(destinations))
    )
    for i, (block, state, target, model) in order:
        stage = program.state(state)
        mapping = PhysicalSupportMap(
            slab,
            rod,
            source_axes=(1, 0),
            target_axes=(1,),
            native_dimension=2,
            reductions=(AxisQuadrature(1, 0, 1, 4, unit, weights=WEIGHTS[i]),),
        )
        context = solution.publish_mapped(
            mapping, {(target, "delivered"): port}, states={state: stage.n}
        )
        rate = program.rhs(
            state=stage.n,
            fields=context,
            terms=[SourceTerm(block[model.operator_handle("consume_mapped_temperature")])],
        )
        program.commit(
            stage.next,
            program.value(
                "integrated " + block.local_id, stage.n + program.dt * rate, at=stage.next.point
            ),
        )
        maps.append((mapping, target, i))
    source_rate = program.rhs(
        state=current.n, terms=[SourceTerm(src[source.operator_handle("change_load")])]
    )
    program.commit(
        current.next,
        program.value("changed load", current.n + program.dt * source_rate, at=current.next.point),
    )
    program.step_strategy(FixedDt(1 / 64))
    case.program(program)
    from pops.initial import InitialCondition
    from pops.lib.initial import Constant
    from pops.projection import ConservativeCellAverage
    from pops.lib.initial import Analytic
    from pops.analytic import x, y, cos
    import math

    case.initials.add(
        InitialCondition(
            state=source_state,
            value=Analytic(
                frame=frame,
                components=(
                    -19 + 0 * x(frame),
                    2 + 0.1 * cos(2 * math.pi * x(frame)) + 0.07 * cos(2 * math.pi * y(frame)),
                ),
            ),
            projection=ConservativeCellAverage(),
        )
    )
    for i, item in enumerate(destinations):
        case.initials.add(
            InitialCondition(
                state=item[1],
                value=Constant((11.0 + i, -101.0 - i, 37.0 + i)),
                projection=ConservativeCellAverage(),
            )
        )
    validated = pops.validate(case)
    subjects = validated.layout_subjects()
    builder = LayoutPlanBuilder(validated.owner_path.canonical())
    grids = (
        Uniform(CartesianGrid(frame=frame, cells=(3, 4), periodic=PeriodicAxes(frame.axes))),
        Uniform(CartesianGrid(frame=frame, cells=(1, 3), periodic=PeriodicAxes(frame.axes))),
    )
    from pops.amr import (
        AMRHierarchy,
        AMRTagging,
        Tag,
        Buffer,
        Hysteresis,
        EqualityPolicy,
        ConflictPolicy,
        AMRRegrid,
        AMRTransfer,
        AMRExecution,
    )
    from pops.time import every
    from pops.layouts import AMR
    from pops.lib.amr import StateTransfer, ConservativeInjection, CoarseFineInjection
    from pops.math import ValueExpr

    adapted = []
    for index, grid in enumerate(grids):
        transfer = AMRTransfer()
        handles = (source_state,) if index == 0 else tuple(item[1] for item in destinations)
        for handle in handles:
            transfer.state(
                handle,
                StateTransfer(
                    prolongation=ConservativeInjection(), coarse_fine=CoarseFineInjection()
                ),
            )
        condition = (
            ValueExpr(source_state)["heat_load"] > validated.value(threshold)
            if index == 0
            else ValueExpr(destinations[0][1])["marker"] > validated.value(threshold)
        )
        adapted.append(
            AMR(
                grid=grid.mesh,
                hierarchy=AMRHierarchy(max_levels=2, ratios=((2, 2) if index == 0 else (1, 2),)),
                tagging=AMRTagging(
                    rules=(Tag(condition), Buffer(cells=0)),
                    hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
                    conflict_policy=ConflictPolicy.REFINE_WINS,
                ),
                regrid=AMRRegrid(schedule=every(1, clock=program.clock)),
                transfer=transfer,
                execution=AMRExecution.synchronous(),
            ).resolve_for_case(validated.resolve)
        )
    grids = tuple(adapted)
    layouts = (
        builder.layout("laminate storage", grids[0]),
        builder.layout("reservoir storage", grids[1]),
    )
    for block in subjects.blocks:
        builder.assign_block(block, layouts[block.local_id != NAMES[1]])
    for row in subjects.states:
        builder.assign_state(row, layouts[row.block_ref.local_id != NAMES[1]])
    for row in subjects.fields:
        builder.assign_field(row, layouts[1] if row.block_ref else layouts[0])
    requirements = []
    for mapping, target, _i in maps:
        (requirement,) = builder.require_mapping(
            *layouts,
            source=field,
            target=target,
            operation=LayoutMappingOperation(mapping.operation_abi),
            synchronization=LayoutSynchronization.PROGRAM_POINT_V1,
            source_representation=LayoutRepresentation.CELL_FIELD_V1,
            target_representation=LayoutRepresentation.CELL_FIELD_V1,
            source_observation=port,
            target_component="delivered",
            physical_map=mapping,
        )
        requirements.append(requirement)
    providers = tuple(provider_factory(r, directory) for r in requirements)
    layout = builder.resolve(**subjects.to_dict(), providers=providers)
    return pops.resolve(
        validated,
        layout=layout,
        layout_providers=dict(zip(layouts, grids, strict=True)),
        components=tuple(p.component for p in providers),
        compile_options={"model_source_policy": "require"},
    )


DT = 1 / 64


def initial():
    import numpy as np

    x = (np.arange(3) + 0.5) / 3
    eta = (np.arange(4) + 0.5) / 4
    source = np.stack(
        (
            np.full((4, 3), -19.0),
            2 + 0.1 * np.cos(2 * np.pi * x)[None, :] + 0.07 * np.cos(2 * np.pi * eta)[:, None],
        )
    )
    result = {NAMES[1]: source}
    for i, name in enumerate((NAMES[0], NAMES[2])):
        result[name] = np.stack(
            (np.full((3, 1), 11.0 + i), np.full((3, 1), -101.0 - i), np.full((3, 1), 37.0 + i))
        )
    return result


def fault_provider(requirement, directory):
    """Genuine external Transfer V2 test DSO; the core and physical integral are unchanged."""
    import json
    from pathlib import Path
    from pops import interfaces
    from pops.external import load, build_source_package_manifest
    from pops.model import ComponentManifest
    from pops.mesh.layout_mapping import NativeLayoutMapping

    root = Path(directory) / requirement.qualified_id.rsplit("::", 1)[-1]
    path = root / "amr-fault.pops.json"
    if not path.exists():
        regular = native_physical_mapping(requirement, directory)
        data = regular.component.component_manifest.to_data()
        kwargs = {key: value for key, value in data.items() if key != "digests"}
        kwargs["signature"] = {
            **kwargs["signature"],
            "test_fault": {
                "contract": "amr-mapped-finite-rejection@1",
                "target": "highest observed integral rank",
            },
        }
        manifest = ComponentManifest(**kwargs)
        cpp = (root / "physical_map.cpp").read_text()
        needle = "  return pops::component::apply_physical_support_integral(operation, request->source, request->destination, status, &request->execution);"
        if cpp.count(needle) != 1:
            raise ValueError("exact generated integral callback not found")
        body = r"""  const int code = pops::component::apply_physical_support_integral(operation, request->source, request->destination, status, &request->execution);
  if (code != 0) return code;
  int rank=0;
#ifdef POPS_HAS_MPI
  if (std::strcmp(request->execution.communicator_identity, "serial") != 0 &&
      std::strcmp(request->execution.communicator_identity, POPS_EXECUTION_NONCOLLECTIVE_IDENTITY_V1) != 0) {
    MPI_Comm_rank(MPI_Comm_f2c(static_cast<MPI_Fint>(request->execution.communicator_f_handle)), &rank);
  }
#endif
  if (const char* trace=std::getenv("POPS_AMR_FIELD_MAP_TRACE")) { if (auto* file=std::fopen(trace,"a")) { std::fprintf(file,"%d\n",rank);std::fclose(file); } }
  if (const char* target=std::getenv("POPS_AMR_FIELD_MAP_FAULT")) {
    char* end=nullptr;errno=0;long parsed=std::strtol(target,&end,10);
    if (errno || !end || end==target || *end || parsed<0 || parsed>INT_MAX) return 93;
    if (parsed==rank && request->destination.data) {
      auto* destination=static_cast<double*>(request->destination.data);
      const double invalid=std::numeric_limits<double>::quiet_NaN();
      if (request->destination.memory_space==POPS_MEMORY_SPACE_HOST_V1) {
        destination[0]=invalid;
      } else {
        if (!pops::component::physical_transfer_detail::supports_device_context(
                request->execution, request->destination.memory_space)) return 94;
        using ExecutionSpace=Kokkos::DefaultExecutionSpace;
        const auto execution=pops::component::physical_transfer_detail::execution_instance<ExecutionSpace>(request->execution);
        Kokkos::parallel_for("m19_fault_exact_residence", Kokkos::RangePolicy<ExecutionSpace>(execution,0,1),
                            AmrTestNonfiniteDestination{destination,invalid});
        execution.fence();
      }
      if (const char* trace=std::getenv("POPS_AMR_FIELD_MAP_TRACE")) { if(auto* file=std::fopen(trace,"a")){std::fprintf(file,"injected:%d\n",rank);std::fclose(file);} }
    }
  }
  return 0;"""
        cpp = cpp.replace(needle, body).replace(
            "#include <cstring>",
            "#include <cstring>\n#ifdef POPS_HAS_MPI\n#include <mpi.h>\n#endif\n#include <cstdlib>\n#include <cstdio>\n#include <cerrno>\n#include <climits>\n#include <limits>\n"
            "struct AmrTestNonfiniteDestination {\n"
            "  double* destination;\n"
            "  double invalid;\n"
            "  KOKKOS_FUNCTION void operator()(const int) const { destination[0]=invalid; }\n"
            "};",
        )
        # Catalog semantic identity changes with the declared numerical test extension.
        cpp = cpp.replace(
            regular.component.component_manifest.semantic_digest.token,
            manifest.semantic_digest.token,
        ).replace(
            regular.component.component_manifest.manifest_digest.token,
            manifest.manifest_digest.token,
        )
        raw = cpp.encode()
        (root / "amr_fault.cpp").write_bytes(raw)
        path.write_text(
            json.dumps(
                build_source_package_manifest(
                    components={"map": manifest}, payloads={"amr_fault.cpp": ("source", raw)}
                )
            )
        )
    component = load(path).require("map", interface=interfaces.Transfer)()
    if (
        component.component_manifest.to_data()["signature"]["physical_map"]
        != requirement.physical_map.to_data()
    ):
        raise ValueError("foreign physical map")
    return NativeLayoutMapping(component, (requirement,))
