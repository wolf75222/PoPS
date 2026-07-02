# Declarative manifest of the per-route block-build SEAM translation units (ADC-593).
#
# WHY THIS FILE EXISTS
#   The _pops extension used to carry ~20 hand-written .cpp files, one per
#   (side, transport, flux) numeric combination (system/isothermal/*, system/compressible/*,
#   amr/block/**, amr/compiled/**). Each was a 10-29 line function that instantiates ONE leaf of
#   the template product in its own translation unit -- a deliberate BUILD-MEMORY mitigation
#   (ADC-335 / ADC-342 / ADC-359): the full product (~1700 leaves) in one TU exceeds 7 GB at -O3
#   under Kokkos, so per-flux TUs parallelize and cap peak memory. That mitigation is correct and
#   stays. What was wrong was the GROWTH STRATEGY: a new Riemann or reconstruction meant a new
#   hand-written pybind file.
#
#   This manifest is the SINGLE declarative list those TUs are now generated from
#   (python/CMakeLists.txt and tests/CMakeLists.txt configure_file a template per row into
#   ${build}/generated_seams/). The generated .cpp is byte-equivalent in symbols and semantics to
#   the deleted hand-written file; only a "generated" header comment is added.
#
# THE GROWTH RULE (acceptance criterion of ADC-593)
#   Adding a Riemann or reconstruction = ONE ROW here + the make_block_<flux> / dispatch_amr_*
#   template in the headers (that is NUMERICS, not bindings). NO new hand-written pybind file. This
#   manifest is NOT the descriptor registry: the declarative registry is brick_catalog.hpp (its
#   Python mirror brick_catalog.py) for transports and routes.py _REGISTRY["riemann"] for fluxes;
#   tests/python/architecture/test_pybind_seam_manifest.py asserts every row's (transport, flux) is legal
#   there, so the manifest cannot invent a route.
#
# ROW FORMAT (fields separated by "|", one row per string in the list):
#   template | side | transport | flux | symbol | out_subdir | out_name
#     template   template stem under python/bindings/templates/<template>.cpp.in
#     side       system | amr_block | amr_compiled (audit category; documentation only)
#     transport  exb | isothermal | compressible (must be a brick_catalog transport id)
#     flux       -                       for a transport-only seam (whole make_block dispatcher)
#                rusanov|hll|hllc|roe     for a flux-subdivided seam (must be a routes.py riemann id)
#     symbol     the pops::detail:: seam function the header declares and the facade calls
#     out_subdir sub-path under the generated dir (mirrors the old hand-written location)
#     out_name   generated file basename (mirrors the old hand-written basename)
#
# Templates (one per distinct FILE SHAPE, read from the deleted originals):
#   system_transport_seam   build_block_for(<transport ctor>, ...)                       [exb]
#   system_flux_seam        build_block_for_make(<ctor>, ..., make_block_<flux>(...))    [iso/comp x flux]
#   amr_block_transport_seam    build_amr_block_for(<ctor>, ...)                         [iso, exb]
#   amr_block_flux_seam         build_amr_block_for_flux(<ctor>, ..., dispatch_amr_block_<flux>)   [comp x flux]
#   amr_compiled_transport_seam build_amr_compiled_for(<ctor>, ...)                      [iso, exb]
#   amr_compiled_flux_seam      build_amr_compiled_for_flux(<ctor>, ..., dispatch_amr_compiled_<flux>) [comp x flux]
#
# NOT generated (kept hand-written -- unique shapes, classified in docs/design/pybind-binding-audit.md):
#   system/base/system_polar.cpp            verbatim polar visitor body (not a template leaf)
#   amr/block/compressible/amr_block_compressible.cpp        thin riemann DISPATCHER (one per transport)
#   amr/compiled/compressible/amr_compiled_compressible.cpp  thin riemann DISPATCHER (one per transport)

set(POPS_SEAM_COMBINATIONS
    # --- System side -------------------------------------------------------------------------------
    "system_transport_seam|system|exb|-|build_block_exb|system/base|system_exb.cpp"
    "system_flux_seam|system|isothermal|rusanov|build_block_isothermal_rusanov|system/isothermal|system_isothermal_rusanov.cpp"
    "system_flux_seam|system|isothermal|hll|build_block_isothermal_hll|system/isothermal|system_isothermal_hll.cpp"
    "system_flux_seam|system|compressible|rusanov|build_block_compressible_rusanov|system/compressible|system_compressible_rusanov.cpp"
    "system_flux_seam|system|compressible|hll|build_block_compressible_hll|system/compressible|system_compressible_hll.cpp"
    "system_flux_seam|system|compressible|hllc|build_block_compressible_hllc|system/compressible|system_compressible_hllc.cpp"
    "system_flux_seam|system|compressible|roe|build_block_compressible_roe|system/compressible|system_compressible_roe.cpp"
    # --- AMR multi-block side ----------------------------------------------------------------------
    "amr_block_transport_seam|amr_block|exb|-|build_amr_block_exb|amr/block/base|amr_block_exb.cpp"
    "amr_block_transport_seam|amr_block|isothermal|-|build_amr_block_isothermal|amr/block/base|amr_block_isothermal.cpp"
    "amr_block_flux_seam|amr_block|compressible|rusanov|build_amr_block_compressible_rusanov|amr/block/compressible|amr_block_compressible_rusanov.cpp"
    "amr_block_flux_seam|amr_block|compressible|hll|build_amr_block_compressible_hll|amr/block/compressible|amr_block_compressible_hll.cpp"
    "amr_block_flux_seam|amr_block|compressible|hllc|build_amr_block_compressible_hllc|amr/block/compressible|amr_block_compressible_hllc.cpp"
    "amr_block_flux_seam|amr_block|compressible|roe|build_amr_block_compressible_roe|amr/block/compressible|amr_block_compressible_roe.cpp"
    # --- AMR single-block compiled side ------------------------------------------------------------
    "amr_compiled_transport_seam|amr_compiled|exb|-|build_amr_compiled_exb|amr/compiled/base|amr_compiled_exb.cpp"
    "amr_compiled_transport_seam|amr_compiled|isothermal|-|build_amr_compiled_isothermal|amr/compiled/base|amr_compiled_isothermal.cpp"
    "amr_compiled_flux_seam|amr_compiled|compressible|rusanov|build_amr_compiled_compressible_rusanov|amr/compiled/compressible|amr_compiled_compressible_rusanov.cpp"
    "amr_compiled_flux_seam|amr_compiled|compressible|hll|build_amr_compiled_compressible_hll|amr/compiled/compressible|amr_compiled_compressible_hll.cpp"
    "amr_compiled_flux_seam|amr_compiled|compressible|hllc|build_amr_compiled_compressible_hllc|amr/compiled/compressible|amr_compiled_compressible_hllc.cpp"
    "amr_compiled_flux_seam|amr_compiled|compressible|roe|build_amr_compiled_compressible_roe|amr/compiled/compressible|amr_compiled_compressible_roe.cpp"
)

# Expand one manifest row into a generated seam .cpp under @p out_root, appending the generated path to
# the list variable named by @p out_var (in the caller's scope). Both python/CMakeLists.txt (the _pops
# module) and tests/CMakeLists.txt (the pops_runtime_{system,amr} OBJECT libs) call this so the generation
# is defined ONCE. The template chooses the ctor / flux tokens from the row via configure_file @VAR@s.
function(pops_generate_seam row out_root out_var)
  string(REPLACE "|" ";" _cols "${row}")
  list(GET _cols 0 _tmpl)
  list(GET _cols 1 SEAM_SIDE)
  list(GET _cols 2 SEAM_TRANSPORT)
  list(GET _cols 3 SEAM_FLUX)
  list(GET _cols 4 SEAM_SYMBOL)
  list(GET _cols 5 _subdir)
  list(GET _cols 6 _name)

  # The object the transport ctor reads its ModelSpec fields off, VERBATIM from the deleted originals:
  # the System-side seams read `model`, the AMR multi-block seams read `a.spec`, the AMR compiled seams
  # read `spec`. Keyed by the file shape (template stem), never guessed.
  if(_tmpl MATCHES "^system_")
    set(_spec "model")
  elseif(_tmpl STREQUAL "amr_block_transport_seam" OR _tmpl STREQUAL "amr_block_flux_seam")
    set(_spec "a.spec")
  elseif(_tmpl STREQUAL "amr_compiled_transport_seam" OR _tmpl STREQUAL "amr_compiled_flux_seam")
    set(_spec "spec")
  else()
    message(FATAL_ERROR "pops_generate_seam: unknown template '${_tmpl}' in row: ${row}")
  endif()

  # Transport ctor expression, VERBATIM from dispatch_transport (block_seam.hpp), resolved against the
  # spec object of this file shape.
  if(SEAM_TRANSPORT STREQUAL "exb")
    set(SEAM_TR_CTOR "ExBVelocity{Real(${_spec}.B0)}")
  elseif(SEAM_TRANSPORT STREQUAL "isothermal")
    set(SEAM_TR_CTOR "IsothermalFlux{Real(${_spec}.cs2), Real(${_spec}.vacuum_floor)}")
  elseif(SEAM_TRANSPORT STREQUAL "compressible")
    set(SEAM_TR_CTOR "CompressibleFlux{Real(${_spec}.gamma)}")
  else()
    message(FATAL_ERROR "pops_generate_seam: unknown transport '${SEAM_TRANSPORT}' in row: ${row}")
  endif()

  # make_block_hll is the only System flux that forwards wave_speed_cache (it is the only flux that
  # engages it); the other fluxes end at positivity_floor. This is the sole per-flux body difference on
  # the System side, so it is a template @VAR@ rather than a separate template.
  if(SEAM_FLUX STREQUAL "hll")
    set(SEAM_MAKE_EXTRA_ARGS ", aa.wave_speed_cache")
  else()
    set(SEAM_MAKE_EXTRA_ARGS "")
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
