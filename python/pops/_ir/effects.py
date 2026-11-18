"""The effect table of the Program IR (spec IR5; G02 lot L, design note §5.1).

One row per Program operation, under the op name the authoring layer emits. Region safety, device
placement and the two effect words of a GraphFunction snapshot are read from this table. The private
lists that held them are gone: ``_PURE``, ``_COLLECTIVE``, ``_RENDEZVOUS``, ``_FINITE_CHECKED``,
``_PARTIAL`` and ``_NATIVE_FIELD_ALGEBRA`` of ``graph_functions``; ``_REGION_SAFE_OPS`` of the
structured-control emitter; ``_OPS``, ``_ORDERED_OPS`` and ``_HOST_REGIONS`` of the finite Kokkos
emitter. A new operation gets its row here, or every reader refuses it.

A row is a set of words:

``graph``
    the operation has a GraphFunction effect contract: a GraphFunction may contain it. This is a
    language-effect profile, not a list of physical models: creating state or history resources,
    publishing and reading an ambient timestep are not GraphFunction operations.
``fails``
    it may raise a recoverable failure; a GraphFunction that contains it outside a ``capture`` has
    the effect word ``failure``.
``algorithm`` or ``rendezvous``
    it enters a collective. ``algorithm``: the collective belongs to the operation itself (a
    reduction, an exchange, a native solve). ``rendezvous``: the lowering adds an agreement on a
    failure or a decision (row E2 of the G02 design note; lot F2 moves it to the failure key).
    Either gives a GraphFunction the effect word ``collective``.
``region``
    the structured-control emitter has checked its lane-uniform failure contract: it may appear in a
    structured region. The operation has no recoverable failure or a lane-uniform native status;
    the word is added only after checking the emitter's failure rendezvous, never because an
    operation is numerically useful.
``device`` or ``ordered``
    it has a device lowering inside the body of a finite map and of an ordered serial region
    (``device``), or inside an ordered region only (``ordered``).
``host-region``
    a control operation whose nested blocks run where it runs; on the host route of a device map,
    the host executes them.

A row without words records an emitted operation for which lot L grants none of the checked
capabilities. It makes no claim about the operation's runtime purity or placement.

The fields that the note adds later come with their readers: the variants a site may raise (lot F1),
the domain row of a partial operation (lot P), suspensions (lot F2) and the resource an allocation
draws from (lot K1). This module imports nothing from the package, so every reader may import it.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

_WORDS = frozenset(("graph", "fails", "algorithm", "rendezvous", "region", "device", "ordered",
                    "host-region"))


@dataclass(frozen=True)
class Effect:
    """The effect profile of one Program operation."""
    graph: bool
    fails: bool
    collective: str | None
    region: bool
    device: str | None
    host_region: bool


def _row(words: str) -> Effect:
    tokens = words.split()
    if len(set(tokens)) != len(tokens) or not set(tokens) <= _WORDS:
        raise ValueError("malformed effect row: %r" % words)
    if {"algorithm", "rendezvous"} <= set(tokens) or {"device", "ordered"} <= set(tokens):
        raise ValueError("conflicting effect words: %r" % words)
    if not {"fails", "algorithm", "rendezvous"}.isdisjoint(tokens) and "graph" not in tokens:
        raise ValueError("an effect word without a GraphFunction contract: %r" % words)
    collective = next((word for word in ("algorithm", "rendezvous") if word in tokens), None)
    device = next((word for word in ("device", "ordered") if word in tokens), None)
    return Effect(graph="graph" in tokens, fails="fails" in tokens, collective=collective,
                  region="region" in tokens, device=device, host_region="host-region" in tokens)


# The spatial operations already refused after synchronization. Their operation names and effect
# rows have one spelling here while G10 owns these transitional operators.
_POST_SYNC_SPATIAL_ROWS = {
    "apply_laplacian_coeff":                '',
    "divergence":                           '',
    "gradient":                            '',
    "laplacian":                           '',
}
POST_SYNC_SPATIAL_OPS = frozenset(_POST_SYNC_SPATIAL_ROWS)

_ROWS = {
    **_POST_SYNC_SPATIAL_ROWS,
    "accept_increment":                     '',
    # acceptance_guard: see stencil_flux_divergence.
    "acceptance_guard":                    'graph fails algorithm region',
    "affine_moment_update":                 '',
    "apply":                                '',
    "apply_in":                            'graph fails algorithm region',
    "apply_out":                           'graph fails algorithm region',
    "attempt_start":                        '',
    "bool_constant":                       'graph region device',
    "bool_op":                             'graph region device',
    "branch":                              'graph fails rendezvous region device host-region',
    "bundle_remap_validate":               'graph fails rendezvous region',
    "bundle_validate":                     'graph fails rendezvous region',
    "capture_result":                      'graph region host-region',
    "cell_compare":                        'graph region',
    "cfl":                                 'region',
    "compare":                             'graph region device',
    "composite_integral":                   '',
    "composite_pair":                       '',
    "condensed_coeffs":                     '',
    "condensed_energy":                     '',
    "condensed_reconstruct":                '',
    "condensed_rhs":                        '',
    "continuation_accept":                 'graph fails rendezvous region',
    "continuation_accepted_candidate":     'graph fails rendezvous region',
    "continuation_assessment_checks":      'graph fails rendezvous region',
    "continuation_assessment_ok":          'graph fails rendezvous region',
    "continuation_attest":                 'graph fails rendezvous region',
    "continuation_candidate_base":         'graph fails rendezvous region',
    "continuation_candidate_value":        'graph fails rendezvous region',
    "continuation_child_base":             'graph fails rendezvous region',
    "continuation_child_reports":          'graph fails rendezvous region',
    "continuation_consume_increment":      'graph fails rendezvous region',
    "continuation_forest_child_phase":     'graph fails rendezvous region',
    "continuation_forest_interval":        'graph fails rendezvous region',
    "continuation_forest_level":           'graph fails rendezvous region',
    "continuation_forest_level_count":     'graph fails rendezvous region',
    "continuation_freeze":                 'graph fails rendezvous region',
    "continuation_open":                    '',
    "continuation_parent_open":            'graph fails rendezvous region',
    "continuation_parent_publish":         'graph fails rendezvous region',
    "continuation_proposal":               'graph fails rendezvous region',
    "continuation_publish":                 '',
    "continuation_read":                   'graph fails rendezvous region',
    "continuation_regrid_base":            'graph fails rendezvous region',
    "continuation_regrid_child_publish":   'graph fails rendezvous region',
    "continuation_regrid_integral":        'graph fails rendezvous region',
    "continuation_regrid_parent_open":     'graph fails rendezvous region',
    "continuation_regrid_parent_publish":  'graph fails rendezvous region',
    "continuation_regrid_read":            'graph fails rendezvous region',
    "continuation_regrid_source":          'graph fails rendezvous region',
    "continuation_regrid_source_count":    'graph fails rendezvous region',
    "continuation_regrid_templates":       'graph fails rendezvous region',
    "continuation_regular_phase":          'graph fails rendezvous region',
    "continuation_sync_open":              'graph fails rendezvous region',
    "continuation_sync_publish":           'graph fails rendezvous region',
    "continuation_synchronized_fields":    'graph fails rendezvous region',
    "continuation_transform":              'graph fails rendezvous region',
    "continuation_working_value":          'graph fails rendezvous region',
    "coupled_rate":                         '',
    "coupled_rate_out":                     '',
    "diffusive_rhs":                        '',
    "distributed_balanced_domain":         'graph fails algorithm',
    "distributed_csr":                     'graph fails algorithm',
    "distributed_domain":                  'graph fails algorithm',
    "distributed_gid":                     'graph device',
    "distributed_import":                  'graph fails algorithm',
    "distributed_import_csr":              'graph fails algorithm',
    "distributed_index":                   'graph device',
    "distributed_index_argument":          'graph device',
    "distributed_local_at":                'graph device',
    "distributed_map":                     'graph fails algorithm',
    "distributed_offset_argument":         'graph',
    "distributed_rank":                    'graph device',
    "distributed_reduce":                  'graph fails algorithm',
    "distributed_requests":                'graph fails algorithm',
    "distributed_scan":                    'graph fails algorithm',
    "distributed_scatter_reduce":          'graph fails algorithm',
    "distributed_size":                    'graph fails algorithm device',
    "distributed_value_domain":            'graph fails algorithm device',
    "field_component":                     'graph region',
    "field_evolved_state":                  '',
    "field_gradient":                      'graph fails algorithm region',
    "field_map_pack":                       '',
    "field_problem_apply":                 'graph fails algorithm region',
    "field_problem_coefficients":          'graph fails algorithm region',
    "field_problem_load":                  'graph fails algorithm region',
    "field_publication":                    '',
    "field_sample":                         '',
    "field_state_cell_mean":                '',
    "field_warm_start":                     '',
    "fill_boundary":                        '',
    "finite_at":                           'graph fails rendezvous region device',
    "finite_domain":                       'graph region device',
    "finite_index":                        'graph fails rendezvous region device',
    "finite_index_argument":               'graph region device',
    "finite_index_value":                  'graph region device',
    "finite_map":                          'graph fails rendezvous region',
    "finite_range_set":                    'graph fails rendezvous region',
    "finite_real":                         'graph fails rendezvous region device',
    "finite_same_domain":                  'graph fails rendezvous region device',
    "finite_scan":                         'graph fails rendezvous region',
    "finite_size":                         'graph region device',
    "geometry_state":                       '',
    "graph_argument":                       '',
    "history":                              '',
    "history_interval_begin":               '',
    "history_interval_check":               '',
    "history_interval_duration":            '',
    "hmin":                                'region',
    "implicit_source":                      '',
    "input_fields":                         '',
    "int64_cast":                          'graph fails rendezvous region device',
    "int64_compare":                       'graph region device',
    "int64_constant":                      'graph region device',
    "int64_op":                            'graph fails rendezvous region device',
    "integral_candidate":                   '',
    "interval_begin":                       '',
    # interval_check and interval_end: root-issued sidecars stay lexical captures; every delivery
    # rechecks the exact context, point and live attempt, and converges a failure before leaving.
    "interval_check":                      'region',
    "interval_end":                        'region',
    "invocation_level":                    'graph region',
    "iteration_is_ok":                     'graph region',
    "iteration_payload":                   'graph region',
    "iteration_success":                   'graph fails rendezvous region',
    "layout_map_export":                    '',
    "layout_map_import":                    '',
    "linear_combine":                      'graph region',
    "linear_source":                        '',
    "local_transform":                      '',
    "logical_and":                          '',
    "matrix_free_operator":                'graph fails algorithm region',
    "max_wave_speed":                       '',
    "pointwise_expression":                'graph fails region',
    "post_synchronization":                 '',
    "principal_rate":                       '',
    "product":                             'graph region device',
    "product_get":                         'graph region device',
    "project":                              '',
    "provenance_bind":                     'graph region device',
    "range":                                '',
    "reached_duration":                     '',
    "record_balance_term":                  '',
    "record_scalar":                        '',
    "reduce":                              'graph algorithm region',
    "region_argument":                     'graph region ordered',
    "region_index":                        'graph region ordered',
    "requested_dt":                        'region',
    "result_error":                        'graph region',
    "result_is_ok":                        'graph region',
    "result_match":                        'graph fails rendezvous region host-region',
    "result_ok":                           'graph region',
    "result_payload":                      'graph region',
    "result_success":                      'graph fails rendezvous region',
    "reynolds_update":                      '',
    "rhs":                                  '',
    "rhs_jacvec":                           '',
    "scalar_expression":                   'graph fails rendezvous region device',
    "scalar_field":                        'region',
    "scalar_isfinite":                     'graph region device',
    "scalar_op":                           'graph fails rendezvous region device',
    "solve_coupled_implicit":               '',
    "solve_evidence":                       '',
    "solve_fields":                         '',
    "solve_fields_from_blocks":             '',
    "solve_implicit_source":                '',
    "solve_linear":                        'graph fails algorithm region',
    "solve_local_linear":                   '',
    "solve_local_nonlinear":                '',
    "solve_outcome":                       'graph fails algorithm region',
    "solve_outcome_component":             'graph fails algorithm region',
    "solve_spatial_field":                  '',
    "solve_spatial_nonlinear":              '',
    "source":                               '',
    "spatial_interaction":                  '',
    "state":                               'region',
    # PreparedStencilFlux (and acceptance_guard) converges allocation, halo, kernel and validity
    # failures; a guarded capture keeps its exchanges across the region's lexical lifetime.
    "stencil_flux_divergence":             'graph fails algorithm region',
    "store_history":                        '',
    "structured_fold":                     'graph fails rendezvous region',
    "structured_iterate":                  'graph fails rendezvous region host-region',
    "subcycle":                             '',
    "synchronize":                          '',
    "variant_construct":                   'graph region',
    "variant_is":                          'graph fails rendezvous region',
    "variant_payload":                     'graph region',
    "variant_require_success":             'graph fails rendezvous region',
    "variant_unreachable":                 'graph fails region',
    "vector_field":                        'graph region',
    "where":                               'graph region',
    "while":                                '',
}

EFFECTS = MappingProxyType({op: _row(words) for op, words in _ROWS.items()})
del _ROWS, _POST_SYNC_SPATIAL_ROWS


def effect(op: str) -> Effect:
    """The row of @p op; an operation with no row has no checked effect contract."""
    try:
        return EFFECTS[op]
    except KeyError:
        raise ValueError("Program operation has no row in the effect table: %s" % op) from None


def has_graph_contract(op: str) -> bool:
    row = EFFECTS.get(op)
    return row is not None and row.graph


def graph_effect_words(op: str) -> frozenset[str]:
    """The GraphFunction effect words of @p op alone, without its nested blocks."""
    row = EFFECTS.get(op)
    if row is None or not row.graph:
        raise ValueError("GraphFunction operation has no declared effect contract: %s" % op)
    words = set()
    if row.fails:
        words.add("failure")
    if row.collective is not None:
        words.add("collective")
    return frozenset(words)


def region_safe(op: str) -> bool:
    row = EFFECTS.get(op)
    return row is not None and row.region


def device_lowered(op: str, *, ordered: bool = False) -> bool:
    """Whether @p op has a device lowering in a map body, or in an ordered body when @p ordered."""
    row = EFFECTS.get(op)
    if row is None or row.device is None:
        return False
    return row.device == "device" or ordered


def host_region(op: str) -> bool:
    row = EFFECTS.get(op)
    return row is not None and row.host_region
