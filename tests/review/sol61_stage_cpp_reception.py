"""Opt-in actual generated C++ syntax gate; no native loader, link or execution."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def as_amr(case, program, uniform):
    from pops.amr import AMRExecution, AMRHierarchy, AMRRegrid, AMRTagging, AMRTransfer
    from pops.amr import Tag, Buffer, Hysteresis, EqualityPolicy, ConflictPolicy
    from pops.layouts import AMR
    from pops.lib.amr import StateTransfer
    from pops.params import RuntimeParam
    from pops.time import every
    from pops._ir.handle_expr import ValueExpr
    threshold = case.param(RuntimeParam("review refinement threshold", default=.5))
    transfer = AMRTransfer()
    states = tuple(row.state for row in program._time_states.values())
    references = states  # Program State handles are already block-qualified.
    for state in references:
        transfer.state(state, StateTransfer())
    return AMR(grid=uniform.mesh, hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(rules=(Tag(ValueExpr(references[0])[0] > case.value(threshold)),Buffer(cells=1)),
                          hysteresis=Hysteresis(0, EqualityPolicy.HOLD),conflict_policy=ConflictPolicy.REFINE_WINS),
        regrid=AMRRegrid(schedule=every(1000,clock=program.clock)),transfer=transfer,
        execution=AMRExecution.synchronous())


def emitted_cases(root):
    sys.path.insert(0,str(root/"python"))
    import pops
    assert Path(pops.__file__).resolve() == root/"python/pops/__init__.py"
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph
    witnesses = runpy.run_path(str(root/"tests/review/test_sol61_stage_additive_independent.py"))
    for amr in (False,True):
        for partition in (False,True):
            if partition:
                case,program,layout,_ = witnesses["partitioned_witness"](captured_Q=True)
            else:
                case,program,layout,*_ = witnesses["witness"]()
            if amr:
                layout = as_amr(case,program,layout)
            resolved = pops.resolve(pops.validate(case),layout=layout)
            cpp = emit_cpp_program(resolved.time,model=ProgramModelGraph.from_resolved_blocks(resolved.blocks),
                target="amr_system" if amr else "system",field_plans=resolved.field_plans)
            yield ("amr" if amr else "uniform")+("-partition" if partition else "-one-carrier"),cpp


def legacy_records(root):
    sys.path.insert(0,str(root/"python"))
    sys.path.insert(0,str(root/"examples/migration/scientific"))
    import pops
    assert Path(pops.__file__).resolve() == root/"python/pops/__init__.py"
    from api040_m27_mixed_linear import build_case
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph
    implicit = runpy.run_path(str(root/"tests/python/unit/time/test_implicit_stage_request.py"))
    cases = [("mixed-linear",build_case(16)),("mixed-linear-permuted",build_case(16,permuted=True)),
             ("implicit-stage",implicit["make_source_stage"]()),
             ("implicit-stage-nonlinear-map",implicit["make_source_stage"](nonlinear=True))]
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian2D
    from pops.mesh import CartesianGrid,PeriodicAxes
    from pops.layouts import Uniform
    from pops.time import FixedDt
    from pops.initial import InitialCondition
    from pops.lib.initial import BindArray
    from pops.projection import ConservativeCellAverage
    for pairing in (False,True):
        frame = CartesianDomain("legacy-domain",lower=(0,0),upper=(1,1)).frame(Cartesian2D())
        model = pops.Model("legacy-model",frame=frame)
        state = model.state("density",components=("a","b","c"))
        case = pops.Case("legacy-dot" if pairing else "legacy-copy")
        block = case.block("material",model,states=(state,))
        program = pops.Program("legacy-program")
        u = program.state(block[state])
        if pairing:
            program.record_scalar("all-components",program.dot_all(u.n,u.n))
        program.commit(u.next,program.value("keep",1*u.n,at=u.next.point))
        program.step_strategy(FixedDt(.01))
        case.program(program)
        case.initials.add(InitialCondition(state=block[state],value=BindArray(),projection=ConservativeCellAverage()))
        layout = Uniform(CartesianGrid(frame=frame,cells=(5,4),periodic=PeriodicAxes(frame.axes)))
        cases.append(("vector-pairing" if pairing else "state-copy",(case,layout)))
    result = {}
    for name,(case,layout,*_) in cases:
        resolved = pops.resolve(pops.validate(case),layout=layout)
        cpp = emit_cpp_program(resolved.time,model=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
        result[name] = {"ir_sha256":resolved.time._ir_hash(),"cpp_sha256":hashlib.sha256(cpp.encode()).hexdigest()}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--compiler",type=Path,default=Path("/usr/bin/clang++"))
    parser.add_argument("--dependency-prefix",type=Path,required=True)
    parser.add_argument("--openmp-include",type=Path)
    parser.add_argument("--emit-only",action="store_true")
    parser.add_argument("--legacy-only",action="store_true")
    args = parser.parse_args(argv)
    root,out = args.source_root.resolve(),args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    if args.legacy_only:
        records = legacy_records(root)
        (out/"legacy-six.json").write_text(json.dumps(records,indent=2,sort_keys=True)+"\n")
        print(json.dumps({"legacy_cases":len(records),"path":str(out/"legacy-six.json")}))
        return 0
    flags = [str(args.compiler),"-std=c++20","-fsyntax-only","-fno-fast-math","-DPOPS_NATIVE_DIM=2",
             "-DPOPS_RUNTIME_SHARED_EXCEPTION_ABI","-DPOPS_HAS_KOKKOS","-DKOKKOS_DEPENDENCE",
             "-DPOPS_HAS_MPI","-DPOPS_HAS_PARALLEL_HDF5","-I"+str(root/"include"),
             "-I"+str(args.dependency_prefix.resolve()/"include")]
    if args.openmp_include:
        flags += ["-Xpreprocessor","-fopenmp","-I"+str(args.openmp_include.resolve())]
    record = {"schema":"sol61.stage-cpp-source-reception@1","scope":"source_and_header_compatibility_only",
              "source_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip(),
              "native_execution":False,"mode":"emit-only" if args.emit_only else "fsyntax-only","cases":{}}
    failures = 0
    for name,cpp in emitted_cases(root):
        source = out/(name+".cpp")
        source.write_text(cpp)
        command = flags+[str(source)]
        if args.emit_only:
            result = None
        else:
            result = subprocess.run(command,capture_output=True,text=True,timeout=120,check=False)
            (out/(name+".stdout")).write_text(result.stdout)
            (out/(name+".stderr")).write_text(result.stderr)
        record["cases"][name] = {"cpp_sha256":sha(source),"command":command,
            "returncode":None if result is None else result.returncode}
        failures += int(result is not None and result.returncode != 0)
    (out/"receipt.json").write_text(json.dumps(record,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"cases":len(record["cases"]),"failed_syntax":failures,"receipt":str(out/"receipt.json")}))
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
