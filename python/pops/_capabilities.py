"""pops._capabilities -- structured native and descriptor capability reports (facade).

The private descriptor-catalog report walks the inert catalogs (Riemann / reconstruction
/ limiter / projection bricks, the mesh layouts, the solver / field catalogs) and reports, per
entry, its name / category / native id / availability / requirements. It is PURE: it imports
only the pure-stdlib authoring packages, never ``_pops``, and runs nothing -- it instantiates
each catalogued descriptor and reads its declared metadata.

This is the introspectable counterpart of the hand-written ``pops.capabilities()`` (the runtime
doctor's dispatch table): that one mirrors what the compiled runtime can dispatch, this one is
sourced straight from the typed descriptors, so the two cannot silently disagree about which
bricks exist.

The implementation was split across three sibling modules for the 500-line cap (ADC-619):
``_capabilities_common`` (the inert value objects + shared route helpers),
``_capabilities_report`` (the native ``capability_report`` value object + route-row builders),
and ``_capabilities_inspect`` (the descriptor-catalog walk, the C++ cross-check, and the AMR
report). Public inspection is exclusively ``pops.inspect(object)``; this module only shares
implementation records between the doctor, validation and inspection layers.
"""
from __future__ import annotations

from pops._capabilities_common import (  # noqa: F401  (re-exported at the historical path)
    CapabilityEntry,
    CapabilityMatrix,
    CapabilityRouteMatrix,
    CapabilityRouteRow,
    _availability_status,
    _axis_for_route,
    _flag_value,
    _route_status_from_availability,
    _status_from_flag,
    _unsupported_error,
)
from pops._capabilities_report import (  # noqa: F401  (re-exported at the historical path)
    NativeCapabilityReport,
    _feature_backend,
    _feature_layout,
    _feature_platform,
    _flag_error_message,
    _inventory_rows,
    _module_capabilities,
    _native_capability_report_from_extension,
    _route_from_native_dict,
    _row,
    _support_rows,
    native_capability_matrix,
    native_capability_report,
)
from pops._capabilities_inspect import (  # noqa: F401  (re-exported at the historical path)
    AmrReport,
    CapabilityMismatchError,
    _LAYOUT_NATIVE_FLAG,
    _amr_policy_rows,
    _cross_check,
    _entry_from_brick,
    _layout_amr_report,
    _native_amr_context,
    _native_amr_envelope,
    _native_rows,
    _walk_brick_catalog,
    _walk_class_catalog,
    _descriptor_catalog_report,
)

__all__ = ["CapabilityMatrix", "CapabilityEntry",
           "CapabilityMismatchError", "AmrReport",
           "CapabilityRouteRow", "CapabilityRouteMatrix", "NativeCapabilityReport",
           "native_capability_report", "native_capability_matrix"]
