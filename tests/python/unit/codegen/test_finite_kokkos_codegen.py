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
    q = typed_state(p, "material", state_name="U")
    _ = q.n
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
    # The output columns stay in the target memory space: no host mirror (DM7, DM8).
    assert 'finite_collective::publish(' in cpp
    assert 'host_mirror_type' not in cpp and '::HostMirror' not in cpp
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
    assert 'finite_collective::allocate_view(' in cpp
    assert 'finite_kokkos::View<' in cpp


@pytest.mark.parametrize("product", (False, True))
def test_direct_captured_branch_exit_is_a_device_capture(product):
    p = Program("captured_branch_exit")
    q = typed_state(p, "material", state_name="U")
    _ = q.n
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
    q = typed_state(p, "material", state_name="U")
    _ = q.n
    factor = p.float64(1.)
    p.commit(q.next, p.value("accepted", factor*q.n, at=q.next.point))
    cpp = emit_cpp_program(p, target="amr_system")
    assert "nonfinite Float64 construction" in cpp
    assert "ctx.attach_affine_flux_expression" in cpp
    assert "finite_kokkos::allocate" not in cpp


@pytest.mark.parametrize('unsupported',('nested_array','domain_creation','ordered_scan'))
def test_unsupported_effect_or_layout_has_explicit_host_placement(unsupported):
    def body(b,i,d):
        if unsupported=='nested_array':
            return b.map(d,lambda c,j:c.index_value(j))
        if unsupported=='domain_creation':
            return b.size(b.range_set(2,capacity=2,ownership='replicated'))
        return b.scan(d,b.int64(0),lambda c,s,j:c.product(state=s+1,value=c.index_value(j)),order='ascending')['state']
    output, cpp = _case(body)
    reason, _ = admission(output)
    assert reason is not None
    rows=[json.loads(line.split('@1 ',1)[1]) for line in cpp.splitlines()
          if '// pops-finite-map-placement@1 ' in line]
    assert any(row['ssa']==output.id and row['backend']=='host' and row['reason']==reason for row in rows)


def test_host_literal_fault_injection_stays_outside_kernel_captures():
    _, cpp = _case(lambda b,i,d:b.float64(970113.125))
    literal='pops::Real('+(970113.125).hex()+')'
    assert cpp.count(literal)==1
    assert cpp.index(literal)<cpp.index('Kokkos::parallel_for("pops_finite_map_')


def test_empty_domain_keeps_kernel_launch_guard_and_no_partial_publication():
    _, cpp = _case(lambda b,i,d:b.float64(b.index_value(i)))
    launch=cpp.index('Kokkos::parallel_for("pops_finite_map_')
    check=cpp.index('finite Kokkos map failed checked arithmetic')
    publication=cpp.index('finite_collective::publish(',launch)
    assert cpp.rfind(' > 0)',0,launch)>0
    assert launch<check<publication


@pytest.mark.parametrize('kind',('arithmetic','gather','empty','overflow','bounds','nonfinite'))
def test_native_receivers_build_closed_device_maps_before_compilation(kind):
    from tests.python.integration.runtime.test_api_v02_finite_kokkos_native import _build
    spec=_build(kind)
    program=spec.case._time
    assert program.validate()
    cpp=emit_cpp_program(program)
    # Reusing a destination must not redeclare allocation failure auxiliaries
    # in one C++ scope.
    helper_declarations=re.findall(r'bool (finite_\w+_exhausted) = false;',cpp)
    scoped_declarations=re.findall(r'\{\s+bool (finite_\w+_exhausted) = false;',cpp)
    assert helper_declarations == scoped_declarations
    # Reusable host helpers own their exception/vote temporaries. Every output
    # column is allocated in the target memory space before dispatch and
    # published as the Array after the failure check of the SAME map; no
    # output has a host mirror (DM7, DM8).
    assert not re.findall(r'finite_collective::mirror(?:_copy)?\(finite_\d+_output_', cpp)
    columns = re.findall(r'finite_collective::allocate_view\((finite_\d+_output_\d+), ', cpp)
    assert columns
    for column in columns:
        allocation = cpp.index('finite_collective::allocate_view(%s, ' % column)
        launch = cpp.index('Kokkos::parallel_for("pops_finite_map_', allocation)
        check = cpp.index('finite_collective::first_failure(', launch)
        publication = cpp.index('finite_collective::publish(', check)
        assert allocation < launch < check < publication
        assert column in cpp[publication:cpp.index(';', publication)]
    rows=[json.loads(line.split('@1 ',1)[1]) for line in cpp.splitlines()
          if '// pops-finite-map-placement@1 ' in line]
    assert rows and all(row['backend']=='Kokkos::DefaultExecutionSpace' for row in rows)
    assert source_map(cpp)['inventories'][0]['program_hash']==program._ir_hash()


def _rows(cpp, tag):
    return [json.loads(line.split(tag + ' ', 1)[1]) for line in cpp.splitlines()
            if '// ' + tag + ' ' in line]


def _program(name, build):
    p = Program(name)
    q = typed_state(p, "material", state_name="U")
    _ = q.n
    selected = build(p, q)
    p.commit(q.next, p.value("accepted", selected, at=q.next.point))
    assert p.validate()
    cpp = emit_cpp_program(p)
    assert cpp == emit_cpp_program(detach_compiled_program(p))
    return cpp


def test_device_map_publishes_labelled_columns_without_boxing_or_host_staging():
    output, cpp = _case(lambda b,i,d:b.product(
        k=b.index_value(i), x=b.float64(b.index_value(i)), flag=b.index_value(i)%2==0))
    assert admission(output)[0] is None
    prefix = 'finite_%d' % output.id
    start = cpp.index('const auto %s_array = [&]() POPS_FINITE_HOST_NOINLINE -> '
                      'pops_algebra_finite::DeviceArray {' % prefix)
    region = cpp[start:cpp.index('}();', start)]
    # Kokkos Tools tells array data from O(1) control by these labels (DM8).
    for column in range(3):
        assert '"pops_finite_array:%s_output_%d"' % (prefix, column) in region
    assert '"pops_finite_control:%s_failure"' % prefix in region
    assert 'finite_collective::first_failure(%s_failure, ' % prefix in region
    assert 'finite_collective::publish(%s_array, ' % prefix in region
    for host_form in ('std::any', 'mirror', 'host_mirror_type', 'for (std::int64_t'):
        assert host_form not in region
    # Replication is checked once on the published columns by a device fingerprint.
    assert 'finite_collective::uniform_device(%s_array, ' % prefix in region
    assert _rows(cpp, 'pops-finite-map-placement@1') == [{
        'backend': 'Kokkos::DefaultExecutionSpace', 'reason': None, 'ssa': output.id,
        'storage': 'device columns in the default memory space; no host staging'}]
    for getter in ('pops_finite_device_to_host_bytes', 'pops_finite_host_to_device_bytes',
                   'pops_finite_host_transfer_count'):
        assert 'extern "C" std::uint64_t %s() {' % getter in cpp


def test_host_placed_map_writes_host_columns_and_publishes_them_once():
    output, cpp = _case(lambda b,i,d:b.size(b.range_set(2,capacity=2,ownership='replicated'))
                        + b.index_value(i))
    reason, _ = admission(output)
    assert reason == 'missing device lowering: finite_range_set'
    assert _rows(cpp, 'pops-finite-map-placement@1') == [{
        'backend': 'host', 'reason': reason, 'ssa': output.id,
        'storage': 'host evaluation; device columns published once (counted host phase)'}]
    assert 'pops_finite_host_map_invocations.fetch_add(1, std::memory_order_relaxed);' in cpp
    prefix = 'finite_%d' % output.id
    assert cpp.count('finite_collective::allocate_host(%s_host_0, ' % prefix) == 1
    publication = 'finite_collective::publish_host(%s_array, ' % prefix
    assert cpp.count(publication) == 1 and '&pops_finite_residency' in cpp.split(publication)[1][:200]
    assert 'finite_collective::allocate_values(' not in cpp
    assert 'finite_collective::assign_any(' not in cpp


def test_reduce_reading_a_device_array_is_one_serial_device_region():
    def build(p, q):
        domain = p.range_set(5, capacity=8, ownership="replicated")
        values = p.map(domain, lambda b,i:b.float64(b.index_value(i)))
        total = p.reduce(values, p.product(s=p.float64(0), n=p.int64(0)),
                         lambda b,s,x:b.product(s=s['s']+x, n=s['n']+1), order='ascending')
        return total['s']*q.n
    cpp = _program("ordered_device_fold", build)
    rows = _rows(cpp, 'pops-finite-ordered-placement@1')
    assert [(row['backend'], row['schedule'], row['reason']) for row in rows] == [
        ('Kokkos::DefaultExecutionSpace', 'serial region (ordered, IR4)', None)]
    prefix = 'finite_%d' % rows[0]['ssa']
    launch = cpp.index('Kokkos::parallel_for("pops_finite_ordered_%d", ' % rows[0]['ssa'])
    # One work item runs the iterations in order (IR4); the host reads only the
    # control scalars back: the failure, its iteration and the final carry.
    assert 'Kokkos::IndexType<std::int64_t>>(0,1), ' in cpp[launch:cpp.index('\n', launch)]
    assert 'for (std::int64_t step = 0; step < ' in cpp
    assert '"pops_finite_control:%s_control"' % prefix in cpp
    copy = cpp.index('finite_collective::mirror_copy(%s_control_host, %s_control, '
                     % (prefix, prefix))
    assert launch < copy
    assert 'finite_collective::read<' not in cpp


def test_scan_publishes_device_columns_from_its_serial_region():
    def build(p, q):
        domain = p.range_set(4, capacity=4, ownership="replicated")
        scanned = p.scan(domain, p.int64(0), lambda b,s,i:b.product(
            state=s+b.index_value(i), value=b.float64(s)), order="ascending")
        return p.float64(scanned["state"])*q.n
    cpp = _program("ordered_device_scan", build)
    rows = _rows(cpp, 'pops-finite-ordered-placement@1')
    assert [(row['backend'], row['schedule'], row['storage']) for row in rows] == [
        ('Kokkos::DefaultExecutionSpace', 'serial region (ordered, IR4)',
         'device columns in the default memory space; no host staging')]
    prefix = 'finite_%d' % rows[0]['ssa']
    launch = cpp.index('Kokkos::parallel_for("pops_finite_ordered_%d", ' % rows[0]['ssa'])
    assert cpp.index('finite_collective::publish(%s_array, ' % prefix) > launch
    assert re.search(r'"pops_finite_array:%s_output_\d+"' % prefix, cpp)


def test_fold_carrying_an_array_is_sequenced_on_the_host_around_device_maps():
    def build(p, q):
        domain = p.range_set(3, capacity=3, ownership="replicated")
        start = p.map(domain, lambda b,i:b.float64(b.index_value(i)))
        p.fold(start, 2, lambda b,old,k:b.map(b.domain(old), lambda c,i:old[i]+1.))
        return q.n
    cpp = _program("array_carry_fold", build)
    rows = _rows(cpp, 'pops-finite-ordered-placement@1')
    assert [(row['backend'], row['schedule'], row['reason']) for row in rows] == [
        ('host', 'host loop', 'ordered carry holds an Array: the host sequences the iterations')]
    maps = _rows(cpp, 'pops-finite-map-placement@1')
    assert len(maps) == 2 and all(row['backend'] == 'Kokkos::DefaultExecutionSpace' for row in maps)


def test_host_array_access_reads_each_leaf_once_and_counts_it():
    def build(p, q):
        domain = p.range_set(4, capacity=4, ownership="replicated")
        values = p.map(domain, lambda b,i:b.product(k=b.index_value(i), x=b.float64(b.index_value(i))))
        return values[p.index(domain, 2)]['x']*q.n
    cpp = _program("device_array_access", build)
    reads = [line for line in cpp.splitlines() if 'finite_collective::read<' in line]
    assert len(reads) == 2
    assert all('&pops_finite_residency, ctx.prepared_execution_lane())' in line for line in reads)
    assert 'Array access requires the same live domain instance and an in-range Index' in cpp
