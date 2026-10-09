"""Shared-callsite legacy payload exporter. Runs unchanged on parent and candidate."""
import hashlib
import json
from pathlib import Path
import sys

# argv supplies a source checkout and output; importing any installed PoPS is forbidden.
source, output = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
sys.path.insert(0, str(source / "python"))
import pops
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.domain import CartesianDomain
from pops.fields import CellCenteredNonlinearCoupled, FieldBoundary, FieldDiscretization, FieldProblem, bcs
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.lib.initial import BindArray
from pops.layouts import Uniform
from pops.math import Reaction, DivCoeffGrad, ValueExpr
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.model import Handle, OwnerPath
from pops.projection import ConservativeCellAverage
from pops.solvers import Newton
from pops.time import FailRun, FixedDt

results = {}
for width in (2, 3):
    for mode in ("literal", "captured", "candidate"):
        frame = CartesianDomain("legacy-box", lower=(0.,0.), upper=(1.,1.)).frame(Cartesian2D())
        model = pops.Model("load", frame=frame)
        load = model.state("forcing", components=tuple("f%d"%i for i in range(width)))
        case = pops.Case("legacy-original-field")
        block = case.block("forcing", model, states=(load,))
        unknowns = tuple(Handle("q%d"%i, kind="field", owner=OwnerPath.model("field-product")) for i in range(width))
        equations = []
        for i, q in enumerate(unknowns):
            coefficient = .01 if mode=="literal" else .01*(1+ValueExpr(load)[0]**2) if mode=="captured" else .01*(1+ValueExpr(q)**2)
            equations.append(Reaction(q, 1+ValueExpr(q)**2)-DivCoeffGrad(q,coefficient)==load[i])
        problem = FieldProblem("legacy-product", unknowns=unknowns, equations=tuple(equations),
            boundaries=tuple(FieldBoundary(q, bcs.BoundaryCondition(bcs.AllPhysicalBoundaries(),bcs.Periodic())) for q in unknowns))
        method = CellCenteredNonlinearCoupled(finite_difference_step=1e-7,
            face_policy="Arithmetic@1" if mode!="literal" else None,
            coefficient_evaluation="PerCandidate@1" if mode=="candidate" else None)
        field = case.field(problem, FieldDiscretization(method=method,boundaries=(),solver=Newton(tolerance=1e-10)))
        program = pops.Program("legacy-program")
        current=program.state(block[load])
        request=field.bind_program_inputs(program=program,values={block[load]:current.n},at=current.next.point,solver=field.default_program_solver())
        solved=program.solve(request,solver=field.default_program_solver()).consume(action=FailRun())
        observed=field.observe(solved)
        for q in field._field_registry.resolved_registration(field).operator.unknowns: program.record_scalar(q.local_id,program.sum(observed[q]))
        program.commit(current.next,program.value("fixed forcing",1*current.n,at=current.next.point))
        program.step_strategy(FixedDt(.01));case.program(program)
        case.initials.add(InitialCondition(state=block[load],value=BindArray(),projection=ConservativeCellAverage()))
        resolved=pops.resolve(pops.validate(case),layout=Uniform(CartesianGrid(frame=frame,cells=(8,6),periodic=PeriodicAxes(frame.axes))))
        graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks)
        cpp=emit_cpp_program(resolved.time,model=graph)
        results['%d-%s'%(width,mode)]={"IR":resolved.time._semantic_serialize(),
            "CPP":cpp,"discretization":method.to_data(),"Modules":{b.name:b.model.module.manifest().to_dict() for b in resolved.blocks}}
output.write_text(json.dumps(results,sort_keys=True,separators=(',',':'))+'\n')
print('SOURCE_ONLY',pops.__file__,'profiles',len(results),'sha256',hashlib.sha256(output.read_bytes()).hexdigest())
