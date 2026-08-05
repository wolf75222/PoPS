# Declarative manifest of the AMR per-route block-build SEAM translation units (ADC-593).
#
# WHY THIS FILE EXISTS
#   The _pops extension used to carry ~20 hand-written .cpp files, one per
#   (transport, flux) numeric combination under amr/block/**. Each was a 10-29 line function that
#   instantiates ONE leaf of
#   the template product in its own translation unit -- a deliberate BUILD-MEMORY mitigation
#   (ADC-335 / ADC-342 / ADC-359): the full product (~1700 leaves) in one TU exceeds 7 GB at -O3
#   under Kokkos, so per-flux TUs parallelize and cap peak memory. That mitigation is correct and
#   stays. What was wrong was the GROWTH STRATEGY: a new Riemann or reconstruction meant a new
#   hand-written runtime file.
#
#   This manifest is the SINGLE declarative list those TUs are now generated from.
#   src/CMakeLists.txt configures one template per row into ${build}/src/generated_seams/; Python
#   and native tests consume the central AMR object target instead of generating a second product.
#   The generated .cpp is byte-equivalent in symbols and semantics to
#   the deleted hand-written file; only a "generated" header comment is added.
#
# THE GROWTH RULE (acceptance criterion of ADC-593)
#   Adding a Riemann or reconstruction = ONE ROW here + the make_block_<flux> / dispatch_amr_*
#   template in the headers (that is NUMERICS, not binding glue). NO new hand-written runtime leaf.
#   This manifest is NOT the descriptor registry: the declarative registry is brick_catalog.hpp (its
#   Python mirror brick_catalog.py) for transports and routes.py _REGISTRY["riemann"] for fluxes;
#   tests/python/architecture/test_runtime_builder_manifest.py asserts every row's (transport, flux) is legal
#   there, so the manifest cannot invent a route.
#
# ROW FORMAT (fields separated by "|", one row per string in the list):
#   template | side | transport | flux | symbol | out_subdir | out_name
#     template   template stem under src/runtime/builders/templates/<template>.cpp.in
#     side       amr_block (validated manifest category)
#     transport  exb | isothermal | compressible (must be a brick_catalog transport id)
#     flux       -                       for a transport-only seam (whole make_block dispatcher)
#                rusanov|hll|hllc|roe     for a flux-subdivided seam (must be a routes.py riemann id)
#     symbol     the pops::detail:: seam function the header declares and the facade calls
#     out_subdir sub-path under the generated dir (mirrors the old hand-written location)
#     out_name   generated file basename (mirrors the old hand-written basename)
#
# Templates (one per distinct FILE SHAPE, read from the deleted originals):
#   amr_block_transport_seam    build_amr_block_for(<ctor>, ...)                         [iso, exb]
#   amr_block_flux_seam         build_amr_block_for_flux(<ctor>, ..., dispatch_amr_block_<flux>)   [comp x flux]
# NOT generated (kept hand-written -- unique shape, classified in docs/design/pybind-binding-audit.md):
#   amr/block/compressible/amr_block_compressible.cpp        thin riemann DISPATCHER (one per transport)
#
# Uniform System no longer participates in this legacy template product. Its compiled package
# materializes one `PreparedSystemBlock<Dim>` and publishes it through `install_prepared_block`.

set(POPS_SEAM_COMBINATIONS
    # --- AMR multi-block side ----------------------------------------------------------------------
    "amr_block_transport_seam|amr_block|exb|-|build_amr_block_exb|amr/block/base|amr_block_exb.cpp"
    "amr_block_transport_seam|amr_block|isothermal|-|build_amr_block_isothermal|amr/block/base|amr_block_isothermal.cpp"
    "amr_block_flux_seam|amr_block|compressible|rusanov|build_amr_block_compressible_rusanov|amr/block/compressible|amr_block_compressible_rusanov.cpp"
    "amr_block_flux_seam|amr_block|compressible|hll|build_amr_block_compressible_hll|amr/block/compressible|amr_block_compressible_hll.cpp"
    "amr_block_flux_seam|amr_block|compressible|hllc|build_amr_block_compressible_hllc|amr/block/compressible|amr_block_compressible_hllc.cpp"
    "amr_block_flux_seam|amr_block|compressible|roe|build_amr_block_compressible_roe|amr/block/compressible|amr_block_compressible_roe.cpp"
    "amr_block_flux_seam|amr_block|compressible|roe_hll_rusanov_recovery|build_amr_block_compressible_roe_hll_rusanov_recovery|amr/block/compressible|amr_block_compressible_roe_hll_rusanov_recovery.cpp"
)

# Expand one manifest row into a generated seam .cpp under @p out_root, appending the generated path to
# the list variable named by @p out_var (in the caller's scope). The central runtime object target is
# the single generator and consumer authority. The template chooses the constructor / flux tokens from
# the row via configure_file @VAR@s.
function(pops_generate_seam row out_root out_var)
  string(REPLACE "|" ";" _cols "${row}")
  list(GET _cols 0 _tmpl)
  list(GET _cols 1 SEAM_SIDE)
  list(GET _cols 2 SEAM_TRANSPORT)
  list(GET _cols 3 SEAM_FLUX)
  list(GET _cols 4 SEAM_SYMBOL)
  list(GET _cols 5 _subdir)
  list(GET _cols 6 _name)

  if(NOT SEAM_SIDE STREQUAL "amr_block")
    message(FATAL_ERROR "pops_generate_seam: retired non-AMR side '${SEAM_SIDE}' in row: ${row}")
  endif()

  # The AMR transport constructor reads its exact route fields from the prepared build arguments.
  if(_tmpl STREQUAL "amr_block_transport_seam" OR _tmpl STREQUAL "amr_block_flux_seam")
    set(_spec "a.spec")
  else()
    message(FATAL_ERROR "pops_generate_seam: unknown template '${_tmpl}' in row: ${row}")
  endif()

  # Transport ctor expression resolved against the exact AMR model specification.
  if(SEAM_TRANSPORT STREQUAL "exb")
    set(SEAM_TR_CTOR "ExBVelocity{Real(${_spec}.B0)}")
  elseif(SEAM_TRANSPORT STREQUAL "isothermal")
    set(SEAM_TR_CTOR "IsothermalFlux{Real(${_spec}.cs2), Real(${_spec}.vacuum_floor)}")
  elseif(SEAM_TRANSPORT STREQUAL "compressible")
    set(SEAM_TR_CTOR "CompressibleFlux{Real(${_spec}.gamma)}")
  else()
    message(FATAL_ERROR "pops_generate_seam: unknown transport '${SEAM_TRANSPORT}' in row: ${row}")
  endif()

  set(_out "${out_root}/${_subdir}/${_name}")
  # @ONLY: substitute ONLY the @SEAM_*@ tokens (never a bare ${...}), so the generated C++ keeps its own
  # ${...}-free body untouched. All row-varying pieces are resolved above into SEAM_* variables.
  configure_file("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/templates/${_tmpl}.cpp.in" "${_out}" @ONLY)

  # Append to the caller's list without a leading empty element (list(APPEND) on an empty var yields a
  # clean single-element list, so no stray "" source that CMake would try to resolve to a file).
  set(_acc "${${out_var}}")
  list(APPEND _acc "${_out}")
  set(${out_var} "${_acc}" PARENT_SCOPE)
endfunction()
