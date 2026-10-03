"""Generic strict frozen-data boundary, and the genuine public component counterexample."""
from dataclasses import dataclass
from types import MappingProxyType
import pytest
from pops.problem._snapshot import AuthoringSnapshot


@dataclass(frozen=True)
class Receipt:
    __pops_snapshot_immutable_data__=1
    payload: object
    def snapshot_data(self): return {'payload':'declared receipt'}


def test_stored_leaves_cannot_be_hidden_by_projection():
    left=AuthoringSnapshot({'record':Receipt(('A',MappingProxyType({'pin':b'abc'})))})
    right=AuthoringSnapshot({'record':Receipt(('A',MappingProxyType({'pin':b'abd'})))})
    assert left.hash != right.hash
    assert '$immutable_data' in left.to_dict()['record']


@pytest.mark.parametrize('value', ([1], {'hidden':2}, lambda:1, object()))
def test_mutable_opaque_and_executable_leaves_refused(value):
    with pytest.raises(TypeError,match='immutable-data'):
        AuthoringSnapshot({'record':Receipt(value)})


def test_forged_marker_does_not_make_mutable_object_immutable():
    class Mutable:
        __pops_snapshot_immutable_data__=1
        def snapshot_data(self): return {}
    with pytest.raises(TypeError,match='frozen dataclass'):
        AuthoringSnapshot({'record':Mutable()})


def test_projection_cannot_mutate_frozen_fields_during_authentication():
    @dataclass(frozen=True)
    class Mutating:
        __pops_snapshot_immutable_data__=1
        number:int=1
        def snapshot_data(self):
            object.__setattr__(self,'number',2)
            return {}
    with pytest.raises(ValueError,match='mutated stored authority'):
        AuthoringSnapshot({'record':Mutating()})


def test_frozen_and_ordinary_cycles_stay_refused():
    cyclic=Receipt(None)
    object.__setattr__(cyclic,'payload',cyclic)
    with pytest.raises(ValueError,match='reference cycle'):
        AuthoringSnapshot({'record':cyclic})
    ordinary=[];ordinary.append(ordinary)
    with pytest.raises(ValueError,match='reference cycle'):
        AuthoringSnapshot({'ordinary':ordinary})


def test_method_code_identity_is_retained_even_with_same_projected_data():
    before=AuthoringSnapshot({'record':Receipt(1)})
    original=Receipt.snapshot_data
    try:
        Receipt.snapshot_data=lambda self: {'payload':'declared receipt'}
        after=AuthoringSnapshot({'record':Receipt(1)})
        assert before.hash != after.hash
    finally: Receipt.snapshot_data=original


def test_genuine_component_validate_resolve_no_executable_registry_cycle(tmp_path):
    from pops import interfaces
    from pops.fields import ExternalFieldSolver
    from tests.python.integration.native_loader.test_external_field_solver_runtime import _component,_topology_source,_solver_source,_program
    from tests.python.integration._final_field_program import passive_field_model,resolve_periodic_field_program
    topology=_component(tmp_path,name='topology',interface=interfaces.FieldTopology,source_factory=_topology_source)
    solver=_component(tmp_path,name='solver',interface=interfaces.FieldSolver,source_factory=_solver_source,
        manifest_parameters=({'name':'answer','kind':'runtime'},),instance_parameters={'answer':7})
    provider=ExternalFieldSolver(topology=topology,solver=solver)
    plan=resolve_periodic_field_program(passive_field_model('projection counterexample',coefficient=0.),_program,
        name='projection counterexample',block_name='material',target='system',n=8,field_solver=provider,components=(topology,solver))
    plan.verify()
    assert len(plan.component_inputs)==2


@pytest.mark.parametrize('marker',(True,2,'1'))
def test_marker_types_and_versions_are_exact(marker):
    @dataclass(frozen=True)
    class Foreign:
        __pops_snapshot_immutable_data__=marker
        value:int=1
        def snapshot_data(self): return {}
    with pytest.raises(TypeError,match='contract'):
        AuthoringSnapshot({'record':Foreign()})


def test_callable_record_is_not_implicitly_a_data_value():
    @dataclass(frozen=True)
    class Callback:
        coefficient:int=2
        def __call__(self): return self.coefficient
    with pytest.raises(TypeError,match='callable record'):
        AuthoringSnapshot({'record':Receipt(Callback())})


def test_live_backing_mapping_change_is_visible_in_stored_authority():
    raw={'pin':b'one'}
    record=Receipt(MappingProxyType(raw))
    before=AuthoringSnapshot({'record':record})
    raw['pin']=b'two'
    assert before.hash != AuthoringSnapshot({'record':record}).hash


_method_global_pin = {'abi': 4}

def test_data_record_method_globals_remain_authenticated():
    @dataclass(frozen=True)
    class WithGlobal:
        __pops_snapshot_immutable_data__ = 1
        value: int = 1
        def snapshot_data(self):
            return {'abi': _method_global_pin['abi']}
    before = AuthoringSnapshot({'record': WithGlobal()})
    try:
        _method_global_pin['abi'] = 5
        assert before.hash != AuthoringSnapshot({'record': WithGlobal()}).hash
    finally:
        _method_global_pin['abi'] = 4


def test_extra_forced_stored_authority_is_not_ignored():
    record = Receipt(1)
    object.__setattr__(record, 'undeclared_native_authority', {'pin': b'changed'})
    with pytest.raises(TypeError, match='undeclared stored fields'):
        AuthoringSnapshot({'record': record})
