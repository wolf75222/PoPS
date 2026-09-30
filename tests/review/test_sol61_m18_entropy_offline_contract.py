"""Pure source/protocol/math checks; no invented positive native saved states."""
import importlib.util
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
SCRIPT=ROOT/"tests/review/sol61_m18_entropy_offline_oracle.py"
spec=importlib.util.spec_from_file_location("independent_m18_protocol_test",SCRIPT)
oracle=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=oracle
spec.loader.exec_module(oracle)
EXAMPLE=ROOT/"examples/migration/scientific/api040_m18_entropy.py"


def test_pending_contract_forbids_pops_import_and_fabricates_no_states(tmp_path):
    code='''import importlib.abc,runpy,sys
class NoPoPS(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path=None,target=None):
        if name=="pops" or name.startswith("pops."): raise AssertionError("PoPS import forbidden")
sys.meta_path.insert(0,NoPoPS())
sys.argv=[sys.argv[1]]
runpy.run_path(sys.argv[0],run_name="__main__")
'''
    result=subprocess.run([sys.executable,"-I","-c",code,str(SCRIPT)],cwd=tmp_path,
                          capture_output=True,text=True,check=True)
    contract=json.loads(result.stdout)
    assert contract["status"]=="pending_external_native_receipts"
    assert contract["external_owner_sha256_required"] is True
    assert len(contract["phases"])==10 and contract["constants"]["unknowns"]==3
    assert list(tmp_path.iterdir())==[]


@pytest.mark.parametrize("seal",[None,"","f"*63,"g"*64])
def test_missing_owner_seal_refused_before_reading_data(tmp_path,seal):
    with pytest.raises(ValueError,match="external owner seal"):
        oracle.receive(tmp_path/"not-an-owner-file.json",seal)


def test_current_authored_three_unknown_readonly_original_residual_contract():
    oracle.authored_contract(EXAMPLE.read_text())


@pytest.mark.parametrize("before,after,message",[
    ('{"dual": seed}','{"dual": seed, "target": data_state.n}',"three dual unknowns"),
    ('captures={"target": data_state.n}','captures={"target": data_state.next}',"readonly target"),
    ('tolerance=2.e-11','tolerance=2.e-8',"original tolerance"),
    ('max_iterations=12','max_iterations=6',"Newton budget"),
    ('FixedDt(.01)','FixedDt(.02)',"duration authority"),
    ('QUADRATURE.residual(unknowns["dual"], target)','QUADRATURE.residual(unknowns["dual"], 2*target)',"original three-moment"),
    ('program.commit(dual_state.next, solved[dual_block])',
     'program.commit(dual_state.next, solved[dual_block])\n    program.commit(data_state.next, data_state.n)',"only the dual"),
])
def test_authored_shortcuts_and_relaxations_are_refused(before,after,message):
    source=EXAMPLE.read_text()
    altered=source.replace(before,after)
    assert altered!=source
    with pytest.raises(ValueError,match=message):
        oracle.authored_contract(altered)


@pytest.mark.parametrize("target,label",[
    ((1.,0.,.5),"interior"),((1.,0.,1.1),"outside"),((1.,1.,1.),"boundary"),
    ((1.,0.,0.),"boundary"),((0.,0.,0.),"boundary"),((0.,1.,0.),"outside"),
    ((1.,2.,1.),"outside"),((1.,.25,.1),"outside"),
])
def test_polygon_cone_is_distinct_from_numerical_outcome(target,label):
    assert oracle.cone(target)==label


def test_entropy_integer_nullspace_certificate_is_independent_of_newton():
    # A five-node mathematical vector, not a synthetic native campaign/state.
    probability=np.array([.1,.2,.4,.2,.1])
    direction=np.array([-1.,4.,-6.,4.,-1.])/1024
    np.testing.assert_array_equal(oracle.BASIS@direction,np.zeros(3))
    assert oracle.entropy(probability+direction)>oracle.entropy(probability)


@pytest.mark.parametrize("values",[(0.,.2,.4,.2,.1),(np.inf,.2,.4,.2,.1),(np.nan,.2,.4,.2,.1)])
def test_entropy_refuses_boundary_nonfinite_and_overflow_populations(values):
    with pytest.raises(ValueError,match="strictly positive"):
        oracle.entropy(np.asarray(values))


def test_finite_populations_do_not_mask_entropy_overflow():
    with pytest.raises(ValueError,match="entropy overflow"):
        oracle.entropy(np.full(5,1.e308))


def test_partition_empty_peer_and_replication_have_explicit_support():
    full=[[[0,0],[5,4]]]
    empty=[]
    assert oracle.ownership([dict(dual=full,target=full),dict(dual=empty,target=empty)],2)=="distributed"
    assert oracle.ownership([dict(dual=full,target=full)]*2,2)=="replicated"
    left=[[[0,0],[2,4]]]
    right=[[[2,0],[5,4]]]
    assert oracle.ownership([dict(dual=left,target=left),dict(dual=right,target=right)],2)=="distributed"


@pytest.mark.parametrize("rows",[
    [dict(dual=[[[0,0],[4,4]]],target=[[[0,0],[4,4]]])],
    [dict(dual=[[[0,0],[5,4]]],target=[])],
    [dict(dual=[[[0,0],[6,4]]],target=[[[0,0],[6,4]]])],
    [dict(dual=[[[0,0],[5,4]],[[0,0],[5,4]]],target=[[[0,0],[5,4]]])],
])
def test_missing_foreign_or_overlapping_owners_are_refused(rows):
    with pytest.raises(ValueError):
        oracle.ownership(rows,1)


def test_signed_zero_and_bool_clock_are_distinct_in_canonical_checkpoint_protocol():
    assert oracle.cbor(dict(time=0..hex(),macro_step=1))!=oracle.cbor(dict(time=(-0.).hex(),macro_step=1))
    assert oracle.cbor(dict(time=.01.hex(),macro_step=1))!=oracle.cbor(dict(time=.01.hex(),macro_step=True))


def test_no_junit_witness_cannot_qualify_a_native_campaign():
    with pytest.raises(ValueError,match="JUnit missing"):
        oracle.check_junit(b"<testsuite/>",{},0)


def test_real_public_source_ir_and_narrow_countermodels_preserve_three_unknowns():
    # Public source resolution/emission contract only; no compile(), bind() or run().
    import pops
    from examples.migration.scientific.api040_m18_entropy import make_case
    case,layout,_=make_case()
    plan=pops.resolve(pops.validate(case),layout=layout)
    ir=plan.time._serialize(include_provenance=False)
    oracle.compiled_contract(ir,plan.time._ir_hash())
    for key,value in (("output_count",2),("max_iter",6),("capture_names",[]),("product_widths",[6,3])):
        changed=copy.deepcopy(ir)
        attrs=next(node["attrs"] for node in changed["nodes"] if node["op"]=="solve_coupled_implicit")
        attrs[key]=value
        # This is an explicit negative protocol hash, not native/owner authority.
        counter_hash=oracle.digest(json.dumps(changed,sort_keys=True,separators=(",",":" )).encode())
        with pytest.raises(ValueError,match="three-unknown"):
            oracle.compiled_contract(changed,counter_hash)


def test_current_three_distinct_version_contracts_are_not_legacy_package_abi_six():
    assert "inline constexpr int kAbiVersion = 5;" in (ROOT/"include/pops/runtime/module_capabilities.hpp").read_text()
    assert "inline constexpr int kNativeSystemPackageAbiVersion = 7;" in (
        ROOT/"include/pops/runtime/system/native_package_capability.hpp").read_text()
    assert "NATIVE_SYSTEM_PACKAGE_ABI_VERSION = 7" in (ROOT/"python/pops/codegen/_compile_emit.py").read_text()
