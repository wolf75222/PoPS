"""Public Source realization of an equation-owned scalar mapped to another layout."""
from fractions import Fraction
from pathlib import Path
import pops
import pytest
from pops.model import Module, Rate, FieldSpace, Handle, OwnerPath, PhysicalSupport, PhysicalDimension
from pops.fields import FieldProblem, FieldBoundary, FieldDiscretization, bcs
from pops.fields.methods import CellCenteredSecondOrder
from pops.math import laplacian, Reaction
from pops.solvers import CG
from pops.time import FailRun, FixedDt
from pops.numerics.terms import SourceTerm
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.mesh import CartesianGrid, PeriodicAxes, LayoutPlanBuilder, LayoutRepresentation, LayoutMappingOperation, LayoutSynchronization, PhysicalSupportMap, native_physical_mapping


def route(tmp_path, *, adaptive=False):
    unit = PhysicalDimension()
    x = PhysicalSupport((("x", "shared interval"),))
    xy = PhysicalSupport((("x", "shared interval"), ("eta", "laminate")))
    frame = Rectangle("native", (0,0), (1,1)).frame(Cartesian2D())
    case = pops.Case("equation-owned mapped outputs")
    if adaptive:
        from pops.params import RuntimeParam
        threshold = case.param(RuntimeParam("refinement threshold", default=-1.))
    src = Module("equation input", **({"frame": frame} if adaptive else {}))
    su = src.state_space("load", ("a", "b"), support=x, units=(unit,unit), sampling="cell_average")
    sb = case.block("material", src)
    dst = Module("independent destination", **({"frame": frame} if adaptive else {}))
    du = dst.state_space("evolved", ("s", "t", "u"), support=xy, units=(unit,unit,unit), sampling="cell_average")
    df = dst.field_space("temperature input", ("mapped",), support=xy, units=(unit,), sampling="cell")
    @dst.operator("consume", signature=(du,df) >> Rate(du), kind="local_source")
    def consume(state, field):
        return (field[0], 2*field[0], 0*state[2])
    db = case.block("destination", dst)
    state = sb[src.state_handle(su)]
    out = db[dst.state_handle(du)]
    phi = Handle("solution", kind="field", owner=OwnerPath.model("physical unknown"))
    load = src.state_symbols(su)[0]
    physical = FieldProblem("screened", unknowns=(phi,), equations=(Reaction(phi,1)-laplacian(phi)==load,),
        boundaries=(FieldBoundary(phi,bcs.BoundaryCondition(bcs.AllPhysicalBoundaries(),bcs.Periodic())),),
        unknown_spaces={phi:FieldSpace("physical temperature", ("temperature",),support=x,units=(unit,),sampling="cell")},
        coordinate_units=(unit,))
    if adaptive:
        from pops.solvers import CompositeFieldGMRES
        solver = CompositeFieldGMRES(max_iter=100, restart=25)
    else:
        solver = CG(max_iter=100,rel_tol=1e-10,abs_tol=1e-12)
    field = case.field(physical, FieldDiscretization(method=CellCenteredSecondOrder(), boundaries=(),
        observation_axes=(0,),solver=solver))
    p = pops.Program("solve map publish")
    a,b = p.state(state),p.state(out)
    solved = field.observe(p.solve(field,values={state:a.n},at=a.n.point).consume(action=FailRun()))
    port = solved.mapping_port(field[phi])
    target = db[dst.field_handle(df)]
    mapping = PhysicalSupportMap(x,xy)
    context = solved.publish_mapped(mapping,{(target,"mapped"):port},states={out:b.n})
    rhs=p.rhs(state=b.n,fields=context,terms=[SourceTerm(db[dst.operator_handle("consume")])])
    p.commit(b.next,p.value("evolve",b.n+p.dt*rhs,at=b.next.point))
    p.commit(a.next,p.value("retain",1*a.n,at=a.next.point))
    p.step_strategy(FixedDt(.01));case.program(p)
    if adaptive:
        from pops.initial import InitialCondition
        from pops.lib.initial import Constant
        from pops.projection import ConservativeCellAverage
        for handle, values in ((state, (2., 3.)), (out, (5., 6., 7.))):
            case.initials.add(InitialCondition(state=handle, value=Constant(values),
                                              projection=ConservativeCellAverage()))
    validated=pops.validate(case);subjects=validated.layout_subjects()
    builder=LayoutPlanBuilder(validated.owner_path.canonical())
    descriptors=(Uniform(CartesianGrid(frame=frame,cells=(8,1),periodic=PeriodicAxes(frame.axes))),Uniform(CartesianGrid(frame=frame,cells=(8,4),periodic=PeriodicAxes(frame.axes))))
    if adaptive:
        from pops.layouts import AMR
        from pops.amr import AMRHierarchy, AMRRegrid, AMRTagging, AMRTransfer, AMRExecution, Tag, Buffer, Hysteresis, EqualityPolicy, ConflictPolicy
        from pops.lib.amr import StateTransfer
        transfers = []
        for handle in (state, out):
            transfer = AMRTransfer(); transfer.state(handle, StateTransfer()); transfers.append(transfer)
        descriptors = tuple(AMR(grid=descriptor.mesh, hierarchy=AMRHierarchy(max_levels=2,
            ratios=((2, 1) if index == 0 else (2, 2),)), tagging=AMRTagging(rules=(Tag(__import__("pops.math", fromlist=["ValueExpr"]).ValueExpr((state,out)[index])[("a","s")[index]] > validated.value(threshold)), Buffer(cells=0)),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD), conflict_policy=ConflictPolicy.REFINE_WINS),
            regrid=AMRRegrid.frozen(), transfer=transfers[index], execution=AMRExecution.synchronous())
            for index, descriptor in enumerate(descriptors))
        descriptors = tuple(descriptor.resolve_for_case(validated.resolve) for descriptor in descriptors)
    layouts=(builder.layout("source storage",descriptors[0]),builder.layout("destination storage",descriptors[1]))
    for block in subjects.blocks:builder.assign_block(block,layouts[block.local_id=="destination"])
    for row in subjects.states:builder.assign_state(row,layouts[row.block_ref.local_id=="destination"])
    for row in subjects.fields:builder.assign_field(row,layouts[1] if row.block_ref else layouts[0])
    requirement,=builder.require_mapping(*layouts,source=field,target=target,
        operation=LayoutMappingOperation.PHYSICAL_PULLBACK_V1,synchronization=LayoutSynchronization.PROGRAM_POINT_V1,
        source_representation=LayoutRepresentation.CELL_FIELD_V1,target_representation=LayoutRepresentation.CELL_FIELD_V1,
        source_observation=port,target_component="mapped",physical_map=mapping)
    provider=native_physical_mapping(requirement,tmp_path)
    layout=builder.resolve(**subjects.to_dict(),providers=(provider,))
    return pops.resolve(validated,layout=layout,layout_providers=dict(zip(layouts,descriptors)),components=(provider.component,),compile_options={"include":str(Path(__file__).resolve().parents[2]/"include")})


def test_public_route_resolves_and_emits_private_scalar_candidate(tmp_path):
    resolved=route(tmp_path)
    from pops.codegen.program_slicing import slice_program
    from pops.time._program.detach import detach_compiled_program
    from pops.codegen.program_models import ProgramModelGraph
    from pops.codegen.program_graph_lowering import emit_program_graph
    assignments={r.subject.local_id:r.layout for r in resolved.layout_plan.assignments if r.subject_kind=="block"}
    sources=[]
    for layout in resolved.layout_plan.layouts:
        blocks=tuple(b for b in resolved.blocks if assignments[b.name]==layout.handle)
        p=detach_compiled_program(slice_program(resolved.time,tuple(b.name for b in blocks)))
        sources.append(emit_program_graph(p.to_graph(),lowering_program=p,model_graph=ProgramModelGraph.from_resolved_blocks(blocks),target="system",field_plans={}))
    assert any("mapped_field_" in source for source in sources)
    assert sum("publish_field_components" in source for source in sources)==1


@pytest.mark.parametrize("adaptive", (False, True))
def test_consumed_field_mapping_scratch_retains_solve_or_state_authority(tmp_path, adaptive):
    from pops.codegen.program_slicing import slice_program
    from pops.time._program.detach import detach_compiled_program
    from pops.codegen.program_models import ProgramModelGraph
    from pops.codegen.program_graph_lowering import emit_program_graph
    from pops.fields._mapped_publication import validate_pack

    resolved = route(tmp_path, adaptive=adaptive)
    assignments = {row.subject.local_id: row.layout for row in resolved.layout_plan.assignments
                   if row.subject_kind == "block"}
    packs = imports = 0
    for layout in resolved.layout_plan.layouts:
        blocks = tuple(block for block in resolved.blocks if assignments[block.name] == layout.handle)
        program = detach_compiled_program(slice_program(resolved.time, tuple(block.name for block in blocks)))
        source = emit_program_graph(
            program.to_graph(), lowering_program=program,
            model_graph=ProgramModelGraph.from_resolved_blocks(blocks),
            target="amr_system" if adaptive else "system", field_plans={})
        for node in program._values:
            if node.op == "field_map_pack":
                packs += 1
                _, _, solve = validate_pack(node)
                for subslot, suffix in ((0, ""), (1, "_status")):
                    token = f"mapped_field_{node.id}{suffix}"
                    if adaptive:
                        assert (f"{token}_pointer = &ctx.hierarchy_field_scratch("
                                f"{solve.id}, {node.id}, {subslot}, 1, 0)") in source
                        assert f"{token}_pointer = &ctx.scalar_scratch(" not in source
                    else:
                        assert f"{token}_pointer = &ctx.scalar_scratch(" in source
            elif node.op == "layout_map_import":
                imports += 1
                # Destination candidates still derive from their authenticated
                # State input. They must not be rerouted to the source solve.
                assert f"u{node.id}_pointer = &ctx.scalar_scratch(" in source
                assert f"u{node.id}_pointer = &ctx.hierarchy_field_scratch(" not in source
        if adaptive and any(node.op == "field_publication" for node in program._values):
            # This destination has no local solve: its mapped field still needs
            # a complete hierarchy publication before the provider-backed RHS.
            assert not any(node.op == "solve_linear" for node in program._values)
            assert "HierarchyBarrierKind::field_publication" in source
            assert source.index("ctx.publish_staged_field_components();") < source.index(
                "ctx.prepare_provider_values(", source.index("ctx.publish_staged_field_components();"))
    assert packs == imports == 1
