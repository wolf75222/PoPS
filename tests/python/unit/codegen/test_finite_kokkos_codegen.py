"""Source/typing tests only; device execution has a separate native receiver."""
import json
import re

import pytest

from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_emit_finite_kokkos import admission
from pops.codegen.program_source_map import source_map
from pops.time import Program
from pops.time._program.detach import detach_compiled_program
from typed_program_support import typed_state


def _case(body):
    p = Program("portable_finite_map")
    q = typed_state(p, "material", state_name="U"); q.n
    domain = p.range_set(7, capacity=2**63-1, ownership="replicated")
    output = p.map(domain, lambda b,i:body(b,i,domain))
    p.commit(q.next,p.value("accepted",q.n,at=q.next.point))
    assert p.validate()
    cpp = emit_cpp_program(p)
    assert cpp == emit_cpp_program(detach_compiled_program(p))
    return output, cpp


def test_product_arithmetic_is_a_real_kokkos_kernel_with_exact_source_map():
    output, cpp = _case(lambda b,i,d:b.product(
        index=b.index_value(i)+2**53+3, value=b.float64(b.index_value(i))*2,
        nested=b.product(even=b.index_value(i)%2==0)))
    assert admission(output)[0] is None
    assert 'Kokkos::parallel_for("pops_finite_map_' in cpp
    assert 'KOKKOS_FUNCTION void operator()(const std::int64_t index) const' in cpp
    assert 'Kokkos::DefaultExecutionSpace' in cpp
    assert '::host_mirror_type' in cpp and '::HostMirror' not in cpp
    assert 'pops_finite_kernel_launches.fetch_add' in cpp
    data = source_map(cpp)
    assert {'finite_map','int64_op','product','scalar_op'} <= {x['op'] for x in data['operations']}


def test_lazy_branch_keeps_inactive_division_inside_the_device_branch():
    _, cpp = _case(lambda b,i,d:b.branch(b.index_value(i)==0,
        lambda c:c.float64(7), lambda c:c.float64(1)/c.float64(c.index_value(i))))
    kernel = cpp.split('KOKKOS_FUNCTION void operator()(',1)[1].split('\n};',1)[0]
    assert '} else {' in kernel
    assert kernel.index('} else {') < kernel.index(' / ')
    assert 'MPI_' not in kernel and 'ctx.' not in kernel and 'throw ' not in kernel


def test_captured_gather_stages_values_and_retains_instance_comparisons():
    def body(b,i,d):
        return b.index_value(b.index(d,6-b.index_value(i)))
    _, cpp = _case(body)
    assert 'identity' in cpp and 'index validation' in cpp
    assert 'finite_kokkos::allocate<' in cpp


@pytest.mark.parametrize("product", (False, True))
def test_direct_captured_branch_exit_is_a_device_capture(product):
    p = Program("captured_branch_exit")
    q = typed_state(p, "material", state_name="U"); q.n
    domain = p.range_set(3, capacity=3, ownership="replicated")
    external = p.product(x=p.float64(7), flag=p.boolean(True)) if product else p.float64(7)
    output = p.map(domain, lambda b, i: b.branch(b.index_value(i) == 0,
        lambda c: external,
        lambda c: c.product(x=c.float64(1)/c.float64(c.index_value(i)), flag=c.boolean(False))
                  if product else c.float64(1)/c.float64(c.index_value(i))))
    p.commit(q.next, p.value("accepted", q.n, at=q.next.point))
    reason, captures = admission(output)
    assert reason is None and external.id in {v.id for v in captures}
    cpp = emit_cpp_program(p)
    assert 'KOKKOS_FUNCTION void operator()(' in cpp
    assert cpp == emit_cpp_program(detach_compiled_program(p))


def test_float64_construction_on_amr_is_scalar_algebra_without_array_storage():
    p = Program("amr_scalar_construction")
    q = typed_state(p, "material", state_name="U"); q.n
    factor = p.float64(1.)
    p.commit(q.next, p.value("accepted", factor*q.n, at=q.next.point))
    cpp = emit_cpp_program(p, target="amr_system")
    assert "nonfinite Float64 construction" in cpp
    assert "ctx.attach_affine_flux_expression" in cpp
    assert "finite_kokkos::allocate" not in cpp


@pytest.mark.parametrize('unsupported',('nested_array','domain_creation','ordered_scan'))
def test_unsupported_effect_or_layout_has_explicit_host_placement(unsupported):
    def body(b,i,d):
        if unsupported=='nested_array': return b.map(d,lambda c,j:c.index_value(j))
        if unsupported=='domain_creation': return b.size(b.range_set(2,capacity=2,ownership='replicated'))
        return b.scan(d,b.int64(0),lambda c,s,j:c.product(state=s+1,value=c.index_value(j)),order='ascending')['state']
    output, cpp = _case(body)
    reason, _ = admission(output)
    assert reason is not None
    rows=[json.loads(line.split('@1 ',1)[1]) for line in cpp.splitlines()
          if '// pops-finite-map-placement@1 ' in line]
    assert any(row['ssa']==output.id and row['backend']=='host' and row['reason']==reason for row in rows)


def test_host_literal_fault_injection_stays_outside_kernel_captures():
    _, cpp = _case(lambda b,i,d:b.float64(970113.125))
    literal='pops::Real('+float(970113.125).hex()+')'
    assert cpp.count(literal)==1
    assert cpp.index(literal)<cpp.index('Kokkos::parallel_for("pops_finite_map_')


def test_empty_domain_keeps_kernel_launch_guard_and_no_partial_publication():
    _, cpp = _case(lambda b,i,d:b.float64(b.index_value(i)))
    launch=cpp.index('Kokkos::parallel_for("pops_finite_map_')
    check=cpp.index('finite Kokkos map failed checked arithmetic')
    publication=cpp.index('_buffer = pops_algebra_finite::make_values',launch)
    assert cpp.rfind(' > 0)',0,launch)>0
    assert launch<check<publication


@pytest.mark.parametrize('kind',('arithmetic','gather','empty','overflow','bounds','nonfinite'))
def test_native_receivers_build_closed_device_maps_before_compilation(kind):
    from tests.python.integration.runtime.test_api_v02_finite_kokkos_native import _build
    spec=_build(kind)
    program=spec.case._time
    assert program.validate()
    cpp=emit_cpp_program(program)
    # Output host mirrors are refreshed after dispatch. Reusing a destination
    # must not redeclare allocation failure auxiliaries in one C++ scope.
    helper_declarations=re.findall(r'bool (finite_\w+_exhausted) = false;',cpp)
    scoped_declarations=re.findall(r'\{\s+bool (finite_\w+_exhausted) = false;',cpp)
    assert helper_declarations == scoped_declarations
    assert len(helper_declarations)>len(set(helper_declarations))
    rows=[json.loads(line.split('@1 ',1)[1]) for line in cpp.splitlines()
          if '// pops-finite-map-placement@1 ' in line]
    assert rows and all(row['backend']=='Kokkos::DefaultExecutionSpace' for row in rows)
    assert source_map(cpp)['inventories'][0]['program_hash']==program._ir_hash()
