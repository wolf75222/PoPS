"""Public dt-bound emission experiment; nonzero exit explicitly records a gap.

Use the resolved Case/ModelGraph route, without forging a private SSA state node.
It must emit both the isolated bound and step body to count as received.
"""
import json
from pathlib import Path

import pops
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.time import FixedDt, Program

ROOT = Path(__file__).resolve().parents[2]
out = ROOT / "outputs/sol61-dot-all-a77-independent"
out.mkdir(parents=True, exist_ok=True)
results = []


def install_bound(program, state):
    program.set_dt_bound(lambda P, cfl: cfl / (1 + P.dot_all(state.n, state.n)))


for kind in ("existing_state", "readonly_block"):
    frame = Rectangle("dt_frame", (0., 0.), (1., 1.)).frame(Cartesian2D())
    m = pops.Model("dt_model", frame=frame)
    q = m.state("U", components=("a", "b"))
    case = pops.Case("dt_case")
    block = case.block("fluid", m)
    p = Program("dt_program")
    u = p.state(block[q])
    old = u.n
    p.commit(u.next, p.value("identity", (old[0], old[1]), at=u.next.point))
    p.step_strategy(FixedDt(.1))
    bound = u if kind == "existing_state" else p.state(case.block("bound_data", m)[q])
    install_bound(p, bound)
    assert p._serialize()["version"] == 7
    case.program(p)
    layout = Uniform(CartesianGrid(frame=frame, cells=(4, 4), periodic=PeriodicAxes(frame.axes)))
    row = {"case": kind, "version": 7, "python_package": str(Path(pops.__file__).resolve())}
    try:
        resolved = pops.resolve(pops.validate(case), layout=layout)
        graph = ProgramModelGraph.from_resolved_blocks(resolved.blocks)
        cpp = emit_cpp_program(resolved.time, model_graph=graph)
        assert "ctx.dot_all(" in cpp
        row["emission"] = "received"
    except (KeyError, ValueError) as error:
        row.update(emission="FAILED", error_type=type(error).__name__, error=str(error))
    results.append(row)
(out / "dt-bound-receipt.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
if any(row["emission"] != "received" for row in results):
    raise SystemExit(1)
