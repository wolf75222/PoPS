"""Independent real-source reference/geometry contract audit; no Native."""
from pathlib import Path


def test_static_forcing_reference_uses_zero_operator_scratch_not_warm_phi():
    root=Path(__file__).resolve().parents[2]
    source=(root/'include/pops/numerics/elliptic/amr/composite_fac_poisson.hpp').read_text()
    start=source.index('  Real composite_forcing_norm_()')
    body=source[start:source.index('  static Real relative_residual_',start)]
    assert 'level->residual_operator_view.set_val(Real(0))' in body
    assert body.index('set_val(Real(0))')<body.index('compute_level_residual_')<body.index('apply_flux_mismatch')<body.index('return composite_residual_norm_()')
    assert 'levels_[level]->phi' not in body
    reference=source.index('const Real reference = composite_forcing_norm_();')
    assert reference<source.index('const Real initial = composite_residual_norm_();',reference)


def test_fixture_builtin_Field_layout_inherits_hierarchy_not_arbitrary_provider():
    root=Path(__file__).resolve().parents[2]
    native=(root/'src/runtime/amr/amr_system.cpp').read_text()
    start=native.index('request.hierarchy.levels.push_back(EllipticBuildRequest<Dim>{')
    request=native[start:native.index('});',start)]
    assert 'layout.patches()' in request and 'layout.distribution()' in request and 'state.local_rank()' in request
    fac=(root/'include/pops/numerics/elliptic/amr/composite_fac_poisson.hpp').read_text()
    assert 'phi(request.boxes, request.distribution, request.local_rank, 1,' in fac
