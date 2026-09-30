"""Independent exact-parent/candidate symbolic contract and DAG reception.

Source-only subprocesses; no native selector, package install or environment edit.
The receipt explicitly separates compatibility checks from a confirmed regression.
"""
import ast
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile


ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = "908508a7ce8ccc05c6241f6239818380fd0760e3"
PARENT = "bc37b0af4012a10fa3e9bd7ecaccd1f7befc897b"
OUT = ROOT / "outputs/sol61-layering-908508a"
OUT.mkdir(parents=True, exist_ok=True)


def snapshot(revision):
    archive = subprocess.check_output(["git", "archive", revision, "python"], cwd=ROOT)
    directory = OUT / revision[:8]
    directory.mkdir(exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        bundle.extractall(directory, filter="data")
    return directory / "python", sha256(archive).hexdigest()


FIXTURE = r'''
import json, pathlib, pops
from fractions import Fraction
from decimal import Decimal
from pops.linalg import FiniteSupport, FiniteLinearMap
from pops._ir import finite_linear
from pops._ir.expr import Const, Expr, _wrap
from pops.model.hash_data import canonical_hash_data
from pops.time.expressions import encode_expressions
from pops.numerics import reconstruction, riemann
from pops.frames import Cartesian2D
from pops import Model, Case, Program
from pops.time import FixedDt
from pops.model.handles import Handle
from collections.abc import Mapping

def data(value):
    if isinstance(value,Handle):return value.inspect()
    if isinstance(value,Mapping):return {k:data(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [data(v) for v in value]
    return canonical_hash_data(value)

def lower(values):
    hook=getattr(finite_linear,"lower_finite_scalars",None)
    return hook(values) if hook else tuple(_wrap(x) for x in values)

def encode(values):
    roots=lower(values)
    graph=encode_expressions(roots,None)
    return {"canonical":canonical_hash_data(roots),
            "encoded":canonical_hash_data(graph[:2]),
            "joint_count":sum(row[0]=="finite_linear_v1" for row in graph[1])}

s=FiniteSupport("ordered",("a","b"))
mapping=FiniteLinearMap(s,s,((1.,2.),(-3.,4.)))
v=mapping.apply(s.bind((Fraction(1,7),Decimal("0.125"))))
cases={
 "plain_joint":v.components,
 "finite_first_arithmetic":(v[0]+2*v[1],v[1]-1),
 "scalar_first_mixed":(Const(1)+v[0],Const(2)+v[1]),
 "finite_vector_arithmetic":(v+v*Fraction(2,3)).components,
 "unary":(-v[0],abs(v[1])),
 "comparison":(v[0]>0,v[1]<=3),
 "solve":mapping.solve(s.bind((2,3))).components,
}
receipt={"source":str(pathlib.Path(pops.__file__).resolve()),
         "cases":{name:encode(values) for name,values in cases.items()}}

model=Model("review_policy",frame=Cartesian2D())
state=model.species("U",state=("a","b"))
other=model.species("V",state=("c",))
scalar=reconstruction.User(lambda p:p(0)+Fraction(1,4)*(p(1)-p(-1)),formal_order=2)
joint=reconstruction.User(lambda p:(p(0)[0]+p(1,other)[0],p(0)[1]-p(-1)[0]),
                          state=state,sampling=(other,),formal_order=1)
face=riemann.User(body=lambda l,r,fl,fr,sp:(fl[0]+r[1],fr[1]-l[0]),state=state,
                  stability=lambda l,r,fl,fr,sp:sp)
receipt["policies"]={name:{"options":data(value.options),
    "roots":canonical_hash_data((value.expression,) if isinstance(value.expression,Expr)
                                 else tuple(value.expression))}
    for name,value in (("scalar",scalar),("joint",joint),("face",face))}
receipt["handles"]=data((state,other))

def program_image(mixed):
    m=Model("review_program_model",frame=Cartesian2D())
    q=m.state("U",components=("a","b"))
    case=Case("review_program_case");block=case.block("fluid",m,states=(q,))
    p=Program("review_program");u=p.state(block[q]);vec=mapping.apply(s.bind(u.n))
    if mixed:
        value=p.value("mapped",(u.n[0]+vec[0],u.n[1]+vec[1]),at=u.next.point)
    else:
        value=vec.materialize(p,"mapped",template=u.n,at=u.next.point)
    p.commit(u.next,value);p.step_strategy(FixedDt(.1));assert p.validate()
    return {"hash":p._ir_hash(),"attrs":canonical_hash_data(value.attrs),
            "joint_count":sum(row[0]=="finite_linear_v1" for row in value.attrs["expression_nodes"])}
receipt["programs"]={"plain":program_image(False),"mixed":program_image(True)}

failures={}
def refusal(name,fn):
    try:fn()
    except (TypeError,ValueError,IndexError,NotImplementedError) as e:
        failures[name]={"type":type(e).__name__,"message":str(e)}
    else:raise AssertionError("missing refusal: "+name)
refusal("wrong_support",lambda:mapping.apply(FiniteSupport("other",s.dofs).bind((1,2))))
refusal("permuted_support",lambda:mapping.apply(FiniteSupport(s.name,tuple(reversed(s.dofs))).bind((1,2))))
refusal("nan_coefficient",lambda:FiniteLinearMap(s,s,((float("nan"),0),(0,1))))
refusal("inf_input",lambda:s.bind((float("inf"),0)))
refusal("wrong_width",lambda:s.bind((1,)))
refusal("vector_scalar_reconstruction",lambda:reconstruction.User(lambda p:(p(0),p(1)),formal_order=1))
refusal("bad_offset",lambda:reconstruction.User(lambda p:p(True),formal_order=1))
refusal("face_missing_stability",lambda:riemann.User(body=lambda l,r,fl,fr,sp:(fl[0],fr[1]),state=state))
refusal("face_wrong_width",lambda:riemann.User(body=lambda l,r,fl,fr,sp:(fl[0],),state=state,
                                               stability=lambda l,r,fl,fr,sp:sp))
refusal("truth_value",lambda:bool(v[0]>0))
receipt["refusals"]=failures
print(json.dumps(receipt,sort_keys=True))
'''

receipts, archives = {}, {}
for revision in (PARENT, CANDIDATE):
    source_path, archives[revision] = snapshot(revision)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("POPS_NATIVE_DIM", None)
    environment["PYTHONPATH"] = str(source_path)
    run = subprocess.run([sys.executable, "-c", FIXTURE], cwd=OUT, env=environment,
                         capture_output=True, text=True)
    if run.returncode:
        raise RuntimeError(revision + " fixture failed:\n" + run.stderr[-3000:])
    receipts[revision] = json.loads(run.stdout)
    assert Path(receipts[revision]["source"]).is_relative_to(source_path)

before, after = receipts[PARENT], receipts[CANDIDATE]
checks = 0
encoded_differences = []
for name in before["cases"]:
    assert before["cases"][name]["canonical"] == after["cases"][name]["canonical"], name
    checks += 1
    if before["cases"][name]["encoded"] == after["cases"][name]["encoded"]:
        checks += 1
    else:
        encoded_differences.append(name)
assert before["policies"] == after["policies"]
assert before["handles"] == after["handles"]
assert before["programs"]["plain"] == after["programs"]["plain"]
assert before["refusals"].keys() == after["refusals"].keys()
refusal_message_changes = []
refusal_type_changes = []
for name in before["refusals"]:
    if before["refusals"][name]["type"] != after["refusals"][name]["type"]:
        refusal_type_changes.append(name)
    if before["refusals"][name]["message"] != after["refusals"][name]["message"]:
        refusal_message_changes.append(name)
    checks += 1
checks += 3

# Test every AST scope and resolve relative imports; do not relax the gate.
candidate_python = OUT / CANDIDATE[:8] / "python"
for relative, forbidden in (
    ("linalg/finite.py", ("pops.",)),
    ("numerics/reconstruction/user.py", ("pops._ir", "pops.physics")),
    ("numerics/reconstruction/joint.py", ("pops._ir", "pops.physics")),
    ("numerics/riemann/user.py", ("pops._ir", "pops.physics")),
    ("_ir/finite_linear.py", ("pops.linalg", "pops.model", "pops.time", "pops.physics")),
):
    tree = ast.parse((candidate_python / "pops" / relative).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets = [item.name for item in node.names]
        elif isinstance(node, ast.ImportFrom):
            package = ("pops." + relative[:-3].replace("/", ".")).split(".")[:-1]
            prefix = package[:len(package) - node.level + 1] if node.level else []
            target = ".".join(prefix + ([node.module] if node.module else []))
            targets = [target]
        else:
            continue
        assert not any(target.startswith(forbidden) for target in targets), (relative, targets)
    checks += 1
gate = "tests/python/architecture/test_import_graph.py"
assert subprocess.check_output(["git", "show", PARENT + ":" + gate], cwd=ROOT) == \
       subprocess.check_output(["git", "show", CANDIDATE + ":" + gate], cwd=ROOT)
checks += 1

# Preserve the actual failure as a named receipt; this is not a passing parity check.
assert before["cases"]["scalar_first_mixed"]["joint_count"] == 1
assert after["cases"]["scalar_first_mixed"]["joint_count"] == 2
assert before["programs"]["mixed"]["joint_count"] == 1
assert after["programs"]["mixed"]["joint_count"] == 2
assert before["programs"]["mixed"]["hash"] != after["programs"]["mixed"]["hash"]
result = {"parent": PARENT, "candidate": CANDIDATE, "archive_sha256": archives,
          "compatible_checks": checks, "receipts": receipts,
          "encoded_differences": encoded_differences,
          "refusal_message_changes": refusal_message_changes,
          "refusal_type_changes": refusal_type_changes,
          "confirmed_regression": "mixed Expr/finite projections duplicate application and change Program hash"}
(OUT / "receipt.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(checks, "checks passed (including 10 refused conditions; exception differences recorded)")
print("CONFIRMED REGRESSION: mixed Program application count 1 -> 2; Program hash differs")
print("Changed finite encodings:", encoded_differences)
print("Changed refusal wording:", refusal_message_changes)
print("Changed refusal types:", refusal_type_changes)
print("Receipt:", OUT / "receipt.json")
