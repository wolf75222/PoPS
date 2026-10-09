"""SOURCE_ONLY actual public FieldProblem/Program witnesses, no native calls or JIT."""
from pathlib import Path
import sys
import unittest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

import pops
from pops.domain import CartesianDomain
from pops.fields import (FieldBoundary, FieldDiscretization, FieldProblem, bcs,
    CellCenteredNonlinearCoupled, SpatialInteractionKernel, CellVolumeMeasure,
    CellMidpoint, DirectSpatialInteraction, FieldInteractionQuadrature)
from pops.frames import Cartesian2D
from pops.math import SpatialInteraction, Reaction, ValueExpr, DivCoeffGrad
from pops.model import Handle, OwnerPath
from pops.solvers import Newton
from pops.time import FailRun, FixedDt


def build(order=(0, 1), *, interaction=True, candidate=False, budget=32*1024*1024):
    frame = CartesianDomain("physical-box", lower=(.3, -.4), upper=(2.3, 2.6)).frame(Cartesian2D())
    model = pops.Model("loads", frame=frame)
    load = model.state("forcing", components=("f", "g"))
    case = pops.Case("original nonlocal finite witness")
    block = case.block("load", model, states=(load,))
    unknowns = tuple(Handle(name, kind="field", owner=OwnerPath.model("mixed unknowns")) for name in ("rho", "I"))
    rho, integral = unknowns
    kernel = SpatialInteractionKernel(2, lambda x, y: 1 + .2*x[0] - .3*y[1] + .05*x[1]*y[0])
    equations = (Reaction(rho, 1+ValueExpr(rho)**2)-DivCoeffGrad(integral,
        .01*(1+ValueExpr(rho)**2) if candidate else .01) == load[0],
        Reaction(integral, 1+.1*ValueExpr(integral)**2)-SpatialInteraction(rho, kernel) == load[1])
    problem = FieldProblem("mixed original", unknowns=tuple(unknowns[i] for i in order),
        equations=tuple(equations[i] for i in order), boundaries=tuple(FieldBoundary(unknowns[i],
        bcs.BoundaryCondition(bcs.AllPhysicalBoundaries(), bcs.Periodic())) for i in order))
    quadrature = FieldInteractionQuadrature(CellVolumeMeasure(), CellMidpoint(), DirectSpatialInteraction(budget)) if interaction else None
    method = CellCenteredNonlinearCoupled(finite_difference_step=1e-7,
        face_policy="Arithmetic@1" if candidate else None,
        coefficient_evaluation="PerCandidate@1" if candidate else None, interaction=quadrature)
    field = case.field(problem, FieldDiscretization(method=method, boundaries=(), solver=Newton(tolerance=1e-10)))
    program = pops.Program("full F with original interaction")
    current = program.state(block[load])
    request = field.bind_program_inputs(program=program, values={block[load]: current.n},
        at=current.next.point, solver=field.default_program_solver())
    return case, field, program, current, request, block, load, frame


def emit(args, *, amr=False):
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph
    from pops.initial import InitialCondition
    from pops.lib.initial import BindArray
    from pops.layouts import Uniform, AMR
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.projection import ConservativeCellAverage

    case, field, program, current, request, block, load, frame = args
    solved = program.solve(request, solver=field.default_program_solver()).consume(action=FailRun())
    observed = field.observe(solved)
    for unknown in field._field_registry.resolved_registration(field).operator.unknowns:
        program.record_scalar(unknown.local_id, program.sum(observed[unknown]))
    program.commit(current.next, program.value("fixed loads", 1*current.n, at=current.next.point))
    program.step_strategy(FixedDt(.01))
    case.program(program)
    case.initials.add(InitialCondition(state=block[load], value=BindArray(), projection=ConservativeCellAverage()))
    grid = CartesianGrid(frame=frame, cells=(8, 6), periodic=PeriodicAxes(frame.axes))
    layout = Uniform(grid)
    if amr:
        from pops.amr import AMRExecution, AMRHierarchy, AMRRegrid, AMRTagging, AMRTransfer, Buffer, Tag, Hysteresis, EqualityPolicy, ConflictPolicy
        from pops.lib.amr import StateTransfer
        from pops.params import RuntimeParam
        from pops.time import every
        transfer = AMRTransfer()
        transfer.state(block[load], StateTransfer())
        threshold = case.param(RuntimeParam("refinement threshold", default=.5))
        layout = AMR(grid=grid, hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
            tagging=AMRTagging(rules=(Tag(ValueExpr(block[load])["f"] > case.value(threshold)), Buffer(cells=1)),
                hysteresis=Hysteresis(0, EqualityPolicy.HOLD), conflict_policy=ConflictPolicy.REFINE_WINS),
            regrid=AMRRegrid(schedule=every(1000, clock=program.clock)), transfer=transfer,
            execution=AMRExecution.synchronous())
    resolved = pops.resolve(pops.validate(case), layout=layout)
    return emit_cpp_program(resolved.time, model=ProgramModelGraph.from_resolved_blocks(resolved.blocks), target="amr_system" if amr else "system")


class OriginalInteractionSource(unittest.TestCase):
    def test_original_sign_order_and_ir_version(self):
        for order in ((0, 1), (1, 0)):
            for candidate in (False, True):
                args = build(order, candidate=candidate)
                program, request, field = args[2], args[4], args[1]
                token = program.solve(request, solver=field.default_program_solver())._token
                self.assertEqual(program._serialize()["version"], 20)
                term, = token.attrs["source_contract"]["interactions"]["terms"]
                self.assertEqual((term["row"], term["column"]), (order.index(1), order.index(0)))
                self.assertEqual(dict(term["scale"]), {"kind": "integer", "value": "-1"})
                self.assertEqual(token.attrs["contract"], "pops.spatial-field-residual@4")
                self.assertFalse(any(v.op == "spatial_interaction" for v in program._values))
                self.assertEqual(token.attrs["solve_request"]["derivative"]["scheme"], "central_full_residual")

    def test_physics_without_realization_refuses(self):
        with self.assertRaisesRegex(ValueError, "explicit FieldInteractionQuadrature"):
            build(interaction=False)

    def test_resealed_term_mutations_refused(self):
        from pops.fields._original_field_interaction import validate_interactions
        for key, value in (("row", True), ("column", 1), ("scale", {"kind":"integer", "value":"1"}),
                           ("kernel", {"contract":"pops.spatial-interaction-kernel@1", "dimension":2, "tree":["constant", "0x1.0000000000000p+0"], "units":None})):
            args = build(); source = args[4].problem.source_contract
            from pops.time._program.serialization import _json_ready
            data = _json_ready(source["interactions"])
            data["terms"][0][key] = value
            with self.assertRaises((ValueError, TypeError)):
                validate_interactions(data, source["field_problem"], source["unknown_components"])

    def test_budget_and_unknown_axes_refuse(self):
        for budget in (True, 0, -1, 2**64):
            with self.assertRaises(ValueError):
                DirectSpatialInteraction(budget)
        with self.assertRaises(ValueError):
            SpatialInteractionKernel(2, lambda x, y: pops.math.Var("z", "foreign"))

    def test_exact_uint64_budget_through_public_request_and_emission(self):
        from pops.fields._original_field_interaction import interaction_identity_data, interaction_budget
        from pops.identity import canonical_bytes
        for budget in (2**63-1, 2**63, 2**64-1):
            args = build(budget=budget)
            data = args[4].problem.source_contract["interactions"]
            projected = interaction_identity_data(data)
            canonical_bytes(projected)
            self.assertEqual(interaction_budget(data["realization"]), budget)
            code = emit(args)
            self.assertIn(str(budget)+"ULL", code)

    def test_actual_request_revalidates_realization(self):
        # Actual request revalidation, with its registered physical FieldProblem.
        from pops.fields._program_nonlinear_problem import validate_nonlinear_field_request
        args = build()
        request = args[4]
        token = args[2].solve(request, solver=args[1].default_program_solver())._token
        validate_nonlinear_field_request(args[2], token)
        realization = request.problem.source_contract["interactions"]["realization"]
        self.assertEqual(realization["contract"], "pops.original-field-interaction-realization@1")
        self.assertEqual(realization["quadrature"], "pops.cell-midpoint@1")

    def test_actual_public_uniform_and_amr_emission(self):
        out = REPO / "outputs" / "original-field-nonlocal"
        out.mkdir(parents=True, exist_ok=True)
        for amr in (False, True):
            for candidate in (False, True):
                code = emit(build((1, 0), candidate=candidate), amr=amr)
                self.assertNotIn("seal_original_field_source(", code)
                self.assertNotIn("prepare_closed_original_interaction(", code)
                if amr:
                    self.assertIn("original_candidate_interaction(", code)
                    self.assertIn("solve_candidate_interaction(" if candidate else "solve_interaction(", code)
                else:
                    self.assertIn("ctx.spatial_interaction(", code)
                self.assertIn("interaction_0(index, 0)", code)
                (out / (("amr" if amr else "uniform") + ("-candidate" if candidate else "-frozen") + ".cpp")).write_text(code)


if __name__ == "__main__":
    print("SOURCE_ONLY", pops.__file__, "native module loaded:", "_pops" in sys.modules, flush=True)
    unittest.main()
