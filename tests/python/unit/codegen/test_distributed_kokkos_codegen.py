"""Distributed device source proofs; native tests authenticate actual dispatch separately."""
import json
import pytest
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_source_map import source_map
from pops.time._program.detach import detach_compiled_program
from tests.python.integration.runtime.test_api_v02_distributed_kokkos_native import _build


@pytest.mark.parametrize('kind',('arithmetic','import','empty','nested','overflow','bounds','nonfinite'))
@pytest.mark.parametrize('ranks',(1,3))
def test_distributed_native_receivers_have_closed_device_maps(kind,ranks):
    p=_build(kind,ranks).case._time
    assert p.validate()
    cpp=emit_cpp_program(p)
    assert cpp==emit_cpp_program(detach_compiled_program(p))
    rows=[json.loads(line.split('@1 ',1)[1]) for line in cpp.splitlines()
          if '// pops-distributed-map-placement@1 ' in line]
    assert len(rows)==2 and all(row['backend']=='Kokkos::DefaultExecutionSpace' for row in rows)
    assert 'Kokkos::parallel_for("pops_distributed_map_' in cpp
    assert 'distributed::map<pops::kNativeDimension>' in cpp
    assert 'pops_finite_kokkos_launch_count' in cpp
    assert '::host_mirror_type' in cpp and '::HostMirror' not in cpp
    assert 'finite_structural::ScalarProductPack<pops::kNativeDimension>' in cpp
    assert '_packer->pack(' in cpp
    # Every accelerated region uses its authenticated SSA-local execution lane,
    # including allocation consensus, host staging and the device rank capture.
    assert 'distributed_lane' not in cpp
    for row in rows:
        lane = f"distributed_{row['ssa']}_lane"
        assert f"const auto& {lane} = distributed_{row['ssa']}_registry.lane();" in cpp
        assert f"distributed::phase({lane}," in cpp
        assert f"= {lane}.rank();" in cpp
    assert source_map(cpp)['inventories'][0]['program_hash']==p._ir_hash()
    regions = cpp.split('KOKKOS_FUNCTION void operator()(')[1:]
    assert regions
    for region in regions:
        kernel=region.split('\n};',1)[0]
        assert 'MPI_' not in kernel and 'ctx.' not in kernel and 'throw ' not in kernel
    assert 'std::int64_t(9007199254740995)' in cpp


def test_replicated_container_checks_references_without_replicating_shards():
    cpp=emit_cpp_program(_build('nested',3).case._time)
    assert 'reference_registry.common(reference->domain)' in cpp
    assert 'resolve_array<pops::kNativeDimension>(reference_registry,reference->version' in cpp
    assert 'agree_i64(std::int64_t(reference->version.ordinal)' in cpp
    assert 'reference_registry.common(reference)' in cpp
    assert 'reference->domain->capacity()==9223372036854775807' in cpp
    assert 'reference->values' not in cpp
    assert 'reference=pops::runtime::program::distributed::requests_array<' in cpp


def test_direct_branch_exit_capture_is_shared_by_distributed_lowering():
    from pops.time import Program
    from pops.algebra import Distributed
    from pops.codegen.program_emit_distributed_kokkos import admission
    from typed_program_support import typed_state
    p = Program("distributed_captured_exit")
    q = typed_state(p, "material", state_name="U")
    _ = q.n
    d = Distributed(p)
    domain = d.domain((0, 3), capacity=3)
    external = p.product(x=p.float64(7), flag=p.boolean(True))
    result = d.local_map(domain, lambda b, i: b.capture(lambda c:
        c.branch(Distributed(c).gid(i) == 0, lambda e: external,
                 lambda e: e.product(x=e.float64(1)/e.float64(Distributed(e).gid(i)),
                                     flag=e.boolean(False)))), byte_capacity=4096)
    reason, plan = admission(result)
    assert reason is None and external.id in {v.id for v in plan[-1]}
    p.require_success(result)
    p.commit(q.next, p.value("accepted", q.n, at=q.next.point))
    cpp = emit_cpp_program(p)
    assert 'Kokkos::parallel_for("pops_distributed_map_' in cpp
    assert cpp == emit_cpp_program(detach_compiled_program(p))


@pytest.mark.parametrize("exit_kind", ("direct", "branch", "fold"))
def test_host_request_closure_captures_only_used_projection_leaves(exit_kind):
    from pops.time import Program
    from pops.algebra import Distributed
    from typed_program_support import typed_state

    p = Program("distributed_projected_exit_" + exit_kind)
    q = typed_state(p, "material", state_name="U")
    _ = q.n
    d = Distributed(p)
    domain = d.domain((0, 3), capacity=3)
    external = p.product(nested=p.product(domain=domain, offset=p.int64(0)), unused=p.float64(7))

    def body(b, index):
        if exit_kind == "direct":
            offset = external["nested"]["offset"]
        elif exit_kind == "branch":
            offset = b.branch(Distributed(b).gid(index) == 0,
                              lambda c: external["nested"]["offset"],
                              lambda c: Distributed(c).gid(index))
        else:
            offset = b.fold(Distributed(b).gid(index), 1,
                            lambda c, carry, k: external["nested"]["offset"])
        return Distributed(b).index(external["nested"]["domain"], offset)

    requests = d.requests(domain, domain, body)
    p.record_scalar("count", p.float64(d.size(d.domain_of(requests))))
    p.commit(q.next, p.value("accepted", q.n, at=q.next.point))
    cpp = emit_cpp_program(p)
    assert cpp == emit_cpp_program(detach_compiled_program(p))
    assert "::requests(" in cpp
    # The outer Product is not captured wholesale. Its unrelated sibling does
    # not become a transport obligation for this local request closure.
    assert f"capture_{requests.id}_{external.id}" not in cpp


@pytest.mark.parametrize("auxiliary", (False, True))
def test_complete_amr_regrid_projection_closures_emit_after_resolve(auxiliary):
    import pops
    from pops.codegen._orchestration_compile import build_program_model_graph
    from examples.api_v02.continuation_regrid_amr import build_continuation_regrid_amr
    from examples.api_v02.continuation_regrid_auxiliary import build_continuation_regrid_auxiliary

    if auxiliary:
        spec, providers = build_continuation_regrid_auxiliary()
        plan = pops.resolve(pops.validate(spec.case), layout=spec.layout,
                            layout_providers=providers)
    else:
        spec = build_continuation_regrid_amr()
        plan = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    cpp = emit_cpp_program(plan.time, model_graph=build_program_model_graph(plan),
                           target="amr_system", field_plans=plan.field_plans)
    assert "::requests(" in cpp
    assert "level_continuation_" in cpp
