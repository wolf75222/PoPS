"""Source/independent-oracle checks only; native coverage has a separate receiver."""
import numpy as np
import pops
import pytest
import re
from examples.api_v02.method_spatial_bdf2 import build_spatial_bdf2
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from tests.python.integration.runtime.test_api_v02_spatial_bdf2_native import fourier_reference


def test_spatial_bdf2_emits_a_real_residual_and_consumes_history():
    spec=build_spatial_bdf2()
    resolved=pops.resolve(pops.validate(spec.case),layout=spec.layout)
    source=emit_cpp_program(resolved.time,target="system",
        model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    assert "spatial_implicit" in source
    assert "HistoryInterval requires an exact retained publication" in source
    assert "BDF2_previous_member" in source
    assert "consistent_theta_startup" in source
    assert spec.case._time._ir_hash() != build_spatial_bdf2(history_control=True).case._time._ir_hash()


def test_fourier_oracle_satisfies_the_full_discrete_bdf2_relation():
    ny,nx=10,12
    rng=np.random.default_rng(42)
    initial=rng.normal(size=(ny,nx))
    h=.005
    current,previous,_=fourier_reference(initial,h,7)
    _,older,_=fourier_reference(initial,h,6)
    laplacian=sum(n*n*(np.roll(current,1,axis)+np.roll(current,-1,axis)-2*current)
                  for n,axis in ((ny,0),(nx,1)))
    np.testing.assert_allclose(3*current-4*previous+older,.2*h*laplacian,
                               rtol=0,atol=2.e-15)


def _amr_source(spec):
    resolved=pops.resolve(pops.validate(spec.case),layout=spec.layout)
    return emit_cpp_program(resolved.time,target="amr_system",
        model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks))


def test_amr_interval_and_prebarrier_control_copy_into_owned_spatial_workspace():
    source=_amr_source(build_spatial_bdf2(hierarchy="partial"))
    assert "interval_owner_" in source
    assert "stage_spatial_hierarchy_previous(" in source
    assert "HistoryInterval has no matching live clock/attempt" in source
    assert "same_attempt(ctx.resource_attempt())" in source
    # The actual continuation stages an owned previous value, suspends once,
    # solves the whole hierarchy, then publishes each level in the resume arm.
    staged = source.index("ctx.stage_spatial_hierarchy_previous(")
    suspended = source.index("ctx.suspend_hierarchy_barrier(")
    solved = source.index("ctx.solve_spatial_hierarchy(")
    published = source.index("_workspace->candidate(ctx.level())")
    assert staged < suspended < solved < published
    assert source.count("ctx.suspend_hierarchy_barrier(") == 1
    assert "HierarchyBarrierKind::nonlinear_solve" in source
    assert "ctx.reconcile_spatial_hierarchy_previous(" in source
    assert re.search(r"stage_spatial_hierarchy_previous\(\d+,0,\(\*branch_\d+_s0_owner\)", source)


@pytest.mark.parametrize("hierarchy", ["uniform", "partial"])
def test_startup_diffusion_budget_uses_its_consuming_fraction(hierarchy):
    spec=build_spatial_bdf2(hierarchy=hierarchy)
    resolved=pops.resolve(pops.validate(spec.case),layout=spec.layout)
    source=emit_cpp_program(resolved.time,
        target="system" if hierarchy=="uniform" else "amr_system",
        model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    # The lazy startup arm consumes h/3 times the observed rate. Its constituent
    # evaluation must not reject h*frequency before that real consumer runs.
    weighted = re.findall(r"partition_frequency_\d+_0 = .*1.*3.*diffusion_frequency_\d+",source)
    # Both targets emit one executable startup evaluation. The AMR path uses
    # the continuation rather than duplicating an unexecuted flat fallback.
    assert len(weighted) == 1
    assert not re.search(r"dt\s*\*\s*diffusion_frequency_\d+\s*<=",source)
    assert re.search(r"dt \* partition_frequency_\d+_0 <=",source)
    assert "combined_transport_diffusion_stability" in source


def test_amr_prebarrier_scalar_is_captured_by_value_in_the_resume_arm():
    spec=build_spatial_bdf2(hierarchy="partial")
    program=spec.case._time
    begin=next(value for value in program._values if value.op=="interval_begin")
    program.record_scalar("retained_post_barrier_capture",begin)
    source = _amr_source(spec)
    variable = f"interval_begin_{begin.id}"
    declaration = source.index(f"const pops::Real {variable} =")
    suspended = source.index("ctx.suspend_hierarchy_barrier(")
    publication = source.index("_workspace->candidate(ctx.level())")
    record = source.index(f'ctx.record_scalar("retained_post_barrier_capture", {variable});')
    assert declaration < suspended < publication < record
    # Both deferred callbacks own scalar captures. Only stable native fields
    # are explicitly borrowed; the local interval scalar must never be.
    captures = re.findall(r"(?:^\s*|}, )\[([^]\n]*)\]\(\) \{\n    auto& ctx = \*ctx_owner;", source[suspended:record], re.MULTILINE)
    assert len(captures) == 2
    assert all(capture.startswith("=") for capture in captures)
    assert all(f"&{variable}" not in capture for capture in captures)
    assert "same_attempt(ctx.resource_attempt())" in source


def test_amr_control_after_spatial_barrier_is_lazy_collective_and_owned():
    spec=build_spatial_bdf2(hierarchy="partial")
    program=spec.case._time
    begin=next(value for value in program._values if value.op=="interval_begin")
    state=next(value for value in program._values if value.op=="state")
    program.branch(begin>0,lambda builder:builder.value("left",1*state),
        lambda builder:builder.value("right",1*state),name="post_barrier_control")
    source = _amr_source(spec)
    start = source.index('// ProfileScope "node:post_barrier_control"')
    end = source.index('ctx.profile_record("node:post_barrier_control"', start)
    branch = source[start:end]
    assert source.index("_workspace->candidate(ctx.level())") < start
    predicate = f"if ((interval_begin_{begin.id} > pops::Real(0)))"
    assert predicate in branch
    assert "branch choice differs across execution ranks" in branch
    assert "all_reduce_min" in branch and "all_reduce_max" in branch
    true_arm, false_arm = branch.split("} else {", 1)
    assert 'ProfileScope "node:left"' in true_arm
    assert 'ProfileScope "node:right"' not in true_arm
    assert 'ProfileScope "node:right"' in false_arm
    assert 'ProfileScope "node:left"' not in false_arm
    assert "std::shared_ptr<pops::MultiFab<pops::kNativeDimension>> branch_" in branch
    for arm in (true_arm, false_arm):
        prepared = arm.index("ctx.own_structural_field(")
        voted = arm.index("pops::collectively_rethrow_exception(")
        installed = arm.index("_owner = std::move(structured_pending_0)")
        assert prepared < voted < installed
        assert "ctx.require_program_field_access(" in arm
