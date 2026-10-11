"""Prepared diagonal-MG authority; native execution is qualified separately."""
from dataclasses import replace
from types import SimpleNamespace

import pytest


def _provider_and_node(**options):
    from pops.solvers import preconditioners
    from pops.solvers.preconditioners import _program_preconditioner_provider
    descriptor=preconditioners.TensorDiagonalMG(**options)
    provider=_program_preconditioner_provider(descriptor)
    return provider, SimpleNamespace(attrs={
        'preconditioner_provider':provider.authority(),
        'preconditioner_options':provider.prepare_options(descriptor.options,where='test')})


def _context(**changes):
    from pops.solvers._prepared_preconditioner_registry import PreparedPreconditionerNativeContext
    values=dict(operator_contract={'schema_version':1,'coefficient_shape':[2,2],
        'frozen_capture_authority':'unit-test-operator-authority'},
        coefficient_field_expr='frozen_owned_tensor',coefficient_dimension=2,
        geometry_expr='exact_geometry',boundary_expr='exact_field_boundary')
    values.update(changes)
    return PreparedPreconditionerNativeContext(**values)


def test_native_emission_uses_resolved_owned_coefficient_and_canonical_controls():
    provider,node=_provider_and_node()
    expression=provider.emit(node,[],'prototype','distribution',native_context=_context())
    assert 'make_tensor_diagonal_mg_sessions<pops::kNativeDimension>(' in expression
    assert 'prototype, frozen_owned_tensor, exact_geometry, exact_field_boundary, distribution, 2, 2, 2, 50)' in expression
    assert provider.authority()['native_context_schema']=='pops.prepared-preconditioner.native-context@1'
    assert provider.native_component.entry_headers==('pops/numerics/elliptic/mg/tensor_diagonal_preconditioner.hpp',)


def test_tensor_provider_refuses_missing_or_untyped_operator_context():
    provider,node=_provider_and_node()
    for context in (None,{},SimpleNamespace(**_context().contract_data())):
        with pytest.raises(ValueError,match='typed native operator context'):
            provider.emit(node,[],'prototype','distribution',native_context=context)


@pytest.mark.parametrize('changes',(
    {'operator_contract':{}},{'coefficient_dimension':True},{'coefficient_dimension':4},
    {'coefficient_field_expr':'pointer; unchecked()'},{'geometry_expr':'geometry\nstatement'},
))
def test_native_context_refuses_incomplete_or_non_expression_binding(changes):
    with pytest.raises(ValueError):
        _context(**changes)


def test_operator_contract_changes_collective_provider_identity_even_with_same_generated_names():
    provider,node=_provider_and_node()
    first=provider.emit(node,[],'prototype','distribution',native_context=_context())
    second=provider.emit(node,[],'prototype','distribution',native_context=_context(
        operator_contract={'schema_version':1,'coefficient_shape':[2,2],
                          'frozen_capture_authority':'another-qualified-operator'}))
    assert first!=second
    renamed=replace(_context(),coefficient_field_expr='renamed_owned_tensor')
    assert renamed.contract_data()==_context().contract_data()


@pytest.mark.parametrize('options',(
    {'minimum_coarse_extent':1},{'pre_sweeps':-1},{'post_sweeps':True},
    {'bottom_sweeps':0},{'unknown_option':2},
))
def test_exact_options_reject_out_of_domain_or_unrecognized_values(options):
    with pytest.raises((TypeError,ValueError)):
        _provider_and_node(**options)


def test_nonsingular_scalar_bicgstab_use_admitted_and_cg_not_certified():
    from pops.solvers.krylov._prepared_method_registry import prepared_krylov_method_provider_by_id
    provider,_node=_provider_and_node()
    nullspace={'schema_version':1,'contract':{'declaration':'nonsingular'}}
    for method in ('pops.krylov.bicgstab','pops.krylov.gmres'):
        provider.validate_use(method_provider=prepared_krylov_method_provider_by_id(method).authority(),
            components=1,nullspace_contract=nullspace,where='test')
    with pytest.raises(ValueError,match='symmetric CG'):
        provider.validate_use(method_provider=prepared_krylov_method_provider_by_id('pops.krylov.cg').authority(),
            components=1,nullspace_contract=nullspace,where='test')
