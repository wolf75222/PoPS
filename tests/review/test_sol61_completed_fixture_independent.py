"""SOURCE_ONLY public admission and synthetic independent Fraction contraction witnesses."""
from fractions import Fraction
from pathlib import Path
import ast

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]


def contraction(rows, geometries, *, source_name="T1", transposed=False, include_covered=False):
    # No PoPS action, accumulation, check_saved or author reference provides expected values.
    contributors=[]
    for row, geo in zip(rows,geometries,strict=True):
        mask=row["active"] if not include_covered else np.ones_like(row["active"])
        for index in zip(*np.where(mask),strict=True):
            contributors.append(tuple(Fraction(float(geo["centers"][axis][index])) for axis in (0,1))+
                                (Fraction(float(geo["cell_volumes"][index])),Fraction(float(row[source_name][index]))))
    results=[]
    for row,geo in zip(rows,geometries,strict=True):
        computed=np.empty(row["active"].shape)
        for index in np.ndindex(computed.shape):
            x0,x1=(Fraction(float(geo["centers"][axis][index])) for axis in (0,1))
            total=Fraction()
            for y0,y1,volume,value in contributors:
                kernel=Fraction(1,4)+x0*y1-2*y0+Fraction(1,2)*x1
                if transposed:
                    kernel=Fraction(1,4)+y0*x1-2*x0+Fraction(1,2)*y1
                total+=kernel*volume*value
            computed[index]=float(total)
        results.append(computed)
    flat=np.concatenate([result[row["active"]] for result,row in zip(results,rows,strict=True)])
    return results,{"sum":float(sum(Fraction(float(v)) for v in flat)),
                    "abs_sum":float(sum(abs(Fraction(float(v))) for v in flat)),
                    "min":float(min(flat)),"max":float(max(flat))}


def synthetic_tables():
    rows,geometries=[],[]
    for n in (4,8):
        y,x=np.meshgrid((np.arange(n)+.5)/n,(np.arange(n)+.5)/n,indexing="ij")
        active=(x>=.5) if n==4 else (x<.5)
        t0=.15+.02*x
        t1=.25+.01*y+.03*x
        rows.append({"active":active,"T0":t0,"T1":t1,"Q0":t0+t0*t0+.1*t1*t1})
        geometries.append({"centers":np.stack((x,y)),"cell_volumes":np.full((n,n),1/n**2)})
    return rows,geometries


def test_fixture_contraction_matches_independent_exact_fraction_and_discriminates_countermodels():
    from tests.python.support.completed_original_interaction import reference
    rows,geometries=synthetic_tables()
    expected,reduced=contraction(rows,geometries)
    actual,actual_reduced=reference({"rows":rows,"geometry":geometries},"T1")
    for left,right in zip(actual,expected,strict=True):
        np.testing.assert_allclose(left,right,atol=2e-15,rtol=0)
    for kind,value in reduced.items():
        assert abs(actual_reduced[kind]-value)<2e-13
    assert reduced["min"]<0<reduced["max"]
    for mutation in ({"transposed":True},{"source_name":"Q0"},{"source_name":"T0"},{"include_covered":True}):
        wrong,wrong_reduced=contraction(rows,geometries,**mutation)
        assert any(np.max(np.abs(a-b))>.001 for a,b in zip(wrong,expected,strict=True))
        assert any(abs(wrong_reduced[kind]-reduced[kind])>.001 for kind in reduced)


@pytest.mark.parametrize("width",(1,2))
def test_real_public_fixture_keeps_original_request_and_global_source_identity(width):
    import pops
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph
    from tests.python.support.completed_original_interaction import build
    from tests.python.support.evolved_stage_amr import build as physical_build

    old_case,_=physical_build(8,width)
    case,layout,profile=build(width)
    old=old_case._time_registry.program
    program=case._time_registry.program
    old_solve=next(v for v in old._values if v.op=="solve_spatial_field")
    solve=next(v for v in program._values if v.op=="solve_spatial_field")
    assert old_solve.attrs==solve.attrs
    assert tuple(v.id for v in old_solve.inputs)==tuple(v.id for v in solve.inputs)
    assert [v.op for v in program._values[:len(old._values)]]==[v.op for v in old._values]
    source=next(v for v in program._values if v.id==profile["source_value_id"])
    interaction=next(v for v in program._values if v.id==profile["interaction_value_id"])
    assert source.block is source.state_ref is source.space is None
    assert source.attrs["component"]==width-1
    assert interaction.block is interaction.state_ref is None
    assert profile["storage"]!=profile["source_ssa_name"]
    assert {v.attrs["diagnostic"] for v in program._values if v.op=="record_scalar"}=={
        "ir19."+("T0" if width==1 else "T1")+"."+kind for kind in ("sum","abs_sum","min","max")}
    resolved=pops.resolve(pops.validate(case),layout=layout)
    assert resolved.time._serialize()["version"]==19
    ic=resolved.initial_condition_plan
    assert len(ic.bindings)==width+1
    assert all(ic.canonical_subject(row.subject) is row.subject for row in ic.bindings)
    cpp=emit_cpp_program(resolved.time,model=ProgramModelGraph.from_resolved_blocks(resolved.blocks),target="amr_system")
    assert cpp.count("ctx.reduce_closed_original_interaction(")==8
    # Restrict to each actually emitted composite definition: Accept precedes sealing.
    for region in cpp.split("ctx.seal_original_field_source(")[1:]:
        assert region.index("ctx.prepare_closed_original_interaction(")<region.index("ctx.reduce_closed_original_interaction(")
    assert "kFullResidualBasisLU" in cpp


def test_saved_fixture_disclaims_cellwise_native_I_and_keeps_separate_checkpoint_paths():
    test=(ROOT/"tests/python/integration/runtime/test_public_completed_original_interaction.py").read_text()
    helper=(ROOT/"tests/python/support/completed_original_interaction.py").read_text()
    tree=ast.parse(test)
    phases=next(node for node in ast.walk(tree) if isinstance(node,ast.For) and isinstance(node.iter,ast.Tuple)
                and len(node.iter.elts)==3 and isinstance(node.iter.elts[0],ast.Tuple)
                and isinstance(node.iter.elts[0].elts[0],ast.Constant) and node.iter.elts[0].elts[0].value=="accepted")
    assert [row.elts[0].value for row in phases.iter.elts]==["accepted","continuous","replay"]
    assert 'checkpoints["reloaded"] = checkpoint(world,owner,directory,"reloaded")' in test
    assert 'phase+"-checkpoint"' in test
    assert 'assert digest(row["path"]) == row["sha256"]' in test
    assert '"native_I_array_available":False' in test and '"entire_I_per_cell_qualified":False' in test
    assert '"full_nonlocal_newton_qualified":False' in test
    assert '"I_array_role":"COMPUTED_REFERENCE_ONLY"' in helper
    assert 'native.output_state_local_pieces(name, level)' in helper
    assert '"ir.json",component.dump_ir' in helper and '"cpp",component.dump_cpp' in helper
    assert '"initial_condition_plan_identity":artifact.plan.initial_condition_plan.identity.token' in test
