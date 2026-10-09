"""Independent URI, lazy IR and emitter refusal reception on exact 21a56b9."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
URI = "pops.program.dot-all@1"


def fixture():
    from pops import Case, Model
    from pops.frames import Cartesian2D
    from pops.time import FixedDt, Program
    m = Model("independent_pairing", frame=Cartesian2D())
    q = m.state("U", components=("a", "b", "c", "d", "e"))
    case = Case("independent_pairing")
    b = case.block("fluid", m, states=(q,))
    p = Program("independent_pairing")
    p.step_strategy(FixedDt(.1))
    return m, q, case, p, p.state(b[q])


def emit(m, p):
    from pops.codegen.module_lowering import lower_and_validate
    from pops.codegen.program_codegen import emit_cpp_program
    model, _ = lower_and_validate(m, facade=m)
    return emit_cpp_program(p, model=model)


def commit_identity(p, u):
    p.commit(u.next, p.value("identity", tuple(u.n[i] for i in range(5)), at=u.next.point))


@pytest.mark.parametrize("region", ("flat", "branch", "nested_branch", "while", "dt_bound"))
def test_real_version_selection_is_conditional_and_recursive(region):
    m, q, case, p, u = fixture()
    p.dot(u.n, u.n)
    p.norm2(u.n)
    p.norm_inf(u.n)
    assert p._serialize()["version"] == 5
    if region == "flat":
        node = p.dot_all(u.n, u.n)
        assert dict(node.attrs) == {"kind": "dot_all", "component_contract": URI}
    elif region == "branch":
        p.branch(p.norm2(u.n) > 0, lambda P: P.dot_all(u.n, u.n), lambda P: P.dot(u.n, u.n))
    elif region == "nested_branch":
        condition = p.norm2(u.n) > 0
        p.branch(condition,
                 lambda P: P.branch(condition, lambda PP: PP.dot_all(u.n, u.n), lambda PP: PP.dot(u.n, u.n)),
                 lambda P: P.dot(u.n, u.n))
    elif region == "while":
        p.while_(u.n, lambda P, x: P.dot_all(x, x) > 0, lambda P, x: x)
    else:
        bound = p.state(case.block("bound_data", m, states=(q,))[q])
        p.set_dt_bound(lambda P, cfl: P.dot_all(bound.n, bound.n) * cfl)
    assert p._serialize()["version"] == 7
    commit_identity(p, u)
    if region != "dt_bound":
        source = emit(m, p)
        assert "ctx.dot_all(" in source and "ctx.dot(" in source
    else:
        # Exact author parent lacks the separate central readonly-input map fix.
        # This branch receives serialization only, without claiming dt-bound emission.
        nodes = p._serialize()["dt_bound"]["nodes"]
        assert any(node["op"] == "reduce" and node["attrs"].get("kind") == "dot_all" for node in nodes)


@pytest.mark.parametrize("contract", (None, "pops.program.dot-all@99", "dot_all", 1))
def test_forged_pairing_contract_is_refused_by_real_emitter(contract):
    m, _, _, p, u = fixture()
    node = p.dot_all(u.n, u.n)
    attrs = {"kind": "dot_all"}
    if contract is not None:
        attrs["component_contract"] = contract
    object.__setattr__(node, "attrs", attrs)
    commit_identity(p, u)
    with pytest.raises(ValueError, match="exact vector pairing contract"):
        emit(m, p)


def test_unknown_reduction_is_refused_instead_of_dot_fallback():
    m, _, _, p, u = fixture()
    node = p.dot_all(u.n, u.n)
    object.__setattr__(node, "attrs", {"kind": "future_pairing"})
    commit_identity(p, u)
    with pytest.raises(ValueError, match="unsupported Program reduction"):
        emit(m, p)


def test_invalid_public_operands_do_not_publish_values():
    m, q, case, p, u = fixture()
    other = p.state(case.block("other", m, states=(q,))[q])
    for a, b, text in ((u.n, 1., "State/RHS"), (u.n, other.n, "same block")):
        before = len(p._values)
        with pytest.raises(ValueError, match=text):
            p.dot_all(a, b)
        assert len(p._values) == before
    left = p.scalar_field("left", ncomp=2)
    right = p.scalar_field("right", ncomp=3)
    before = len(p._values)
    with pytest.raises(ValueError, match="component count"):
        p.dot_all(left, right)
    assert len(p._values) == before
    before = len(p._values)
    with pytest.raises(NotImplementedError, match="explicit Program block owner"):
        p.dot_all(left, left)
    assert len(p._values) == before


def test_foreign_program_and_forged_reordered_space_refuse_before_publication():
    from pops import Model
    from pops.frames import Cartesian2D
    from pops.time import Program
    m, q, case, p, _ = fixture()
    block = case.block("joint", m, states=(q,))
    reordered = Model("reordered", frame=Cartesian2D())
    r = reordered.state("V", components=("e", "d", "c", "b", "a"))
    other = case.block("reordered", reordered, states=(r,))
    u, v = p.state(block[q]), p.state(other[r])
    foreign = Program("foreign").state(block[q])
    left, right, external = u.n, v.n, foreign.n
    before = len(p._values)
    with pytest.raises(ValueError):
        p.dot_all(left, external)
    assert len(p._values) == before
    # Suppress the earlier block mismatch deliberately to reach the exact Space check.
    object.__setattr__(right, "block", left.block)
    with pytest.raises(ValueError):
        p.dot_all(left, right)
    assert len(p._values) == before


def test_exact_legacy_program_hash_and_attrs_against_parent(tmp_path):
    parent = subprocess.check_output(("git", "archive", "8bf5ae0", "python"), cwd=ROOT)
    import io
    import tarfile
    with tarfile.open(fileobj=io.BytesIO(parent)) as bundle:
        bundle.extractall(tmp_path, filter="data")
    body = r'''
import json
from pops import Case, Model
from pops.frames import Cartesian2D
from pops.time import Program, FixedDt
m=Model("legacy_pairing",frame=Cartesian2D());q=m.state("U",components=("a","b","c"))
b=Case("legacy_pairing").block("fluid",m,states=(q,));p=Program("legacy_pairing");u=p.state(b[q])
p.step_strategy(FixedDt(.1))
for name in ("dot","norm2","norm_inf"):
 n=getattr(p,name)(u.n,u.n) if name=="dot" else getattr(p,name)(u.n)
 p.record_scalar(name,n)
p.commit(u.next,p.value("identity",tuple(u.n[i] for i in range(3)),at=u.next.point))
assert p.validate()
print(json.dumps({"hash":p._ir_hash(),"ir":p._serialize()} ,sort_keys=True))
'''
    def run(path):
        env = dict(os.environ, PYTHONPATH=str(path))
        env.pop("POPS_NATIVE_DIM", None)
        return json.loads(subprocess.check_output((sys.executable, "-c", body), cwd=tmp_path, env=env, text=True))
    old, new = run(tmp_path / "python"), run(ROOT / "python")
    assert old == new and old["ir"]["version"] == 5
