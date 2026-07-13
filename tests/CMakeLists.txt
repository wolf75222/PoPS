# Test architecture: GoogleTest for C++, pytest for Python, CTest for orchestration.
# All C++ tests live under tests/cpp/** and are registered through wrappers below.

option(POPS_TESTS_FAST_O0 "Compile test executables at -O0 for faster CI correctness builds" OFF)
if(POPS_TESTS_FAST_O0)
  add_compile_options(
    $<$<AND:$<CONFIG:Release,RelWithDebInfo,MinSizeRel>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-O0>)
endif()

include(FetchContent)
find_package(GTest 1.17 CONFIG QUIET)
if(NOT GTest_FOUND)
  message(STATUS "GoogleTest not found in the environment; fetching v1.17.0")
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG 52eb8108c5bdec04579160ae17225d66034bd723) # v1.17.0
  FetchContent_MakeAvailable(googletest)
  set(_pops_gtest_include_dir "${googletest_SOURCE_DIR}/googletest/include")
endif()
include(GoogleTest)

if(_pops_gtest_include_dir)
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/gtest")
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/gtest/gtest.h"
       "#pragma once\n#include \"${_pops_gtest_include_dir}/gtest/gtest.h\"\n")
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/cpp/test_sources.cmake)

add_library(pops_gtest_main STATIC cpp/support/gtest_main.cpp)
target_include_directories(pops_gtest_main PRIVATE
  "$<TARGET_PROPERTY:GTest::gtest,INTERFACE_INCLUDE_DIRECTORIES>")
target_compile_features(pops_gtest_main PUBLIC cxx_std_${POPS_CXX_STD})
set_target_properties(pops_gtest_main PROPERTIES NO_SYSTEM_FROM_IMPORTED ON)
if(_pops_gtest_include_dir)
  target_include_directories(pops_gtest_main BEFORE PUBLIC "${CMAKE_CURRENT_BINARY_DIR}")
  target_include_directories(pops_gtest_main BEFORE PUBLIC "${_pops_gtest_include_dir}")
  target_compile_options(pops_gtest_main BEFORE PRIVATE "-I${_pops_gtest_include_dir}")
  target_compile_options(pops_gtest_main BEFORE PRIVATE "-iquote${_pops_gtest_include_dir}")
endif()

set(_pops_gtest_xml_dir "${CMAKE_BINARY_DIR}/test-results/gtest")
file(MAKE_DIRECTORY "${_pops_gtest_xml_dir}")

function(pops_test_source out name)
  if(NOT DEFINED POPS_CPP_TEST_SOURCE_${name})
    message(FATAL_ERROR "No C++ test source registered for ${name}")
  endif()
  set(${out} "${CMAKE_SOURCE_DIR}/${POPS_CPP_TEST_SOURCE_${name}}" PARENT_SCOPE)
endfunction()

function(pops_infer_test_labels out source)
  set(_tier unit)
  set(_domain runtime)
  set(_speed fast)
  if("${source}" MATCHES "/integration/([^/]+)/")
    set(_tier integration)
    set(_domain "${CMAKE_MATCH_1}")
    set(_speed medium)
  elseif("${source}" MATCHES "/unit/([^/]+)/")
    set(_tier unit)
    set(_domain "${CMAKE_MATCH_1}")
    set(_speed fast)
  elseif("${source}" MATCHES "/regression/([^/]+)/")
    set(_tier regression)
    set(_domain "${CMAKE_MATCH_1}")
    set(_speed medium)
  endif()

  set(_labels ${_tier} ${_domain} ${_speed})
  if(_domain STREQUAL "mpi")
    set(_labels backend mpi medium)
  elseif(_domain STREQUAL "native_loader")
    set(_labels integration codegen native_loader compiler medium)
  endif()
  set(${out} ${_labels} PARENT_SCOPE)
endfunction()

function(pops_add_gtest_suite)
  set(options ENABLE_EXPORTS NO_DISCOVER)
  set(oneValueArgs NAME TIMEOUT RESOURCE_LOCK)
  set(multiValueArgs SOURCES EXTRA_LIBS LABELS COMPILE_DEFINITIONS RESOURCE_GROUPS)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  if(NOT ARG_NAME)
    message(FATAL_ERROR "pops_add_gtest_suite requires NAME")
  endif()
  if(NOT ARG_SOURCES)
    pops_test_source(_src ${ARG_NAME})
    set(ARG_SOURCES "${_src}")
  endif()
  if(NOT ARG_LABELS)
    list(GET ARG_SOURCES 0 _first_source)
    pops_infer_test_labels(ARG_LABELS "${_first_source}")
  endif()
  if(NOT ARG_TIMEOUT)
    set(ARG_TIMEOUT 600)
  endif()

  add_executable(${ARG_NAME} ${ARG_SOURCES})
  set_target_properties(${ARG_NAME} PROPERTIES NO_SYSTEM_FROM_IMPORTED ON)
  target_link_libraries(${ARG_NAME} PRIVATE pops::pops pops_dev_options GTest::gtest pops_gtest_main ${ARG_EXTRA_LIBS})
  target_include_directories(${ARG_NAME} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp/support"
    "${CMAKE_CURRENT_BINARY_DIR}")
  if(_pops_gtest_include_dir)
    target_include_directories(${ARG_NAME} BEFORE PRIVATE "${_pops_gtest_include_dir}")
    target_compile_options(${ARG_NAME} BEFORE PRIVATE "-I${_pops_gtest_include_dir}")
    target_compile_options(${ARG_NAME} BEFORE PRIVATE "-iquote${_pops_gtest_include_dir}")
  endif()
  target_compile_features(${ARG_NAME} PRIVATE cxx_std_${POPS_CXX_STD})
  if(ARG_ENABLE_EXPORTS)
    set_target_properties(${ARG_NAME} PROPERTIES ENABLE_EXPORTS ON)
  endif()
  if(ARG_COMPILE_DEFINITIONS)
    target_compile_definitions(${ARG_NAME} PRIVATE ${ARG_COMPILE_DEFINITIONS})
  endif()

  set(_pops_test_properties
    LABELS "${ARG_LABELS}"
    TIMEOUT ${ARG_TIMEOUT})
  if(ARG_RESOURCE_LOCK)
    list(APPEND _pops_test_properties RESOURCE_LOCK "${ARG_RESOURCE_LOCK}")
  endif()
  if(ARG_RESOURCE_GROUPS)
    list(APPEND _pops_test_properties RESOURCE_GROUPS "${ARG_RESOURCE_GROUPS}")
  endif()

  if(NOT ARG_NO_DISCOVER)
    gtest_discover_tests(${ARG_NAME}
      DISCOVERY_MODE POST_BUILD
      DISCOVERY_TIMEOUT 60
      XML_OUTPUT_DIR "${_pops_gtest_xml_dir}/${ARG_NAME}"
      PROPERTIES ${_pops_test_properties})
  endif()
endfunction()

function(pops_add_test name)
  pops_add_gtest_suite(NAME ${name})
endfunction()

function(pops_add_mpi_rank_test name nproc)
  if(NOT TARGET ${name})
    message(FATAL_ERROR "MPI CTest entry ${name}_np${nproc} needs target ${name}")
  endif()
  add_test(NAME ${name}_np${nproc}
           COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${nproc}
                   ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${name}>
                   --gtest_output=xml:${_pops_gtest_xml_dir}/${name}_np${nproc}.xml
                   ${MPIEXEC_POSTFLAGS})
  set_tests_properties(${name}_np${nproc} PROPERTIES
    LABELS "backend;mpi;medium"
    PROCESSORS ${nproc}
    TIMEOUT 600)
endfunction()

function(pops_add_mpi_gtest_suite name)
  set(options)
  set(oneValueArgs)
  set(multiValueArgs EXTRA_LIBS RANKS COMPILE_DEFINITIONS)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  if(NOT ARG_RANKS)
    set(ARG_RANKS 1 2 4)
  endif()
  pops_test_source(_src ${name})
  pops_add_gtest_suite(NAME ${name}
    SOURCES "${_src}"
    EXTRA_LIBS ${ARG_EXTRA_LIBS}
    COMPILE_DEFINITIONS ${ARG_COMPILE_DEFINITIONS}
    LABELS backend mpi medium
    NO_DISCOVER)
  foreach(_np IN LISTS ARG_RANKS)
    pops_add_mpi_rank_test(${name} ${_np})
  endforeach()
endfunction()

function(pops_native_loader_definitions out)
  set(${out}
    POPS_TEST_CXX="${CMAKE_CXX_COMPILER}"
    POPS_TEST_INCLUDE="${CMAKE_SOURCE_DIR}/include"
    POPS_TEST_CXX_STD="c++${CMAKE_CXX_STANDARD}"
    POPS_TEST_TMPDIR="${CMAKE_CURRENT_BINARY_DIR}"
    PARENT_SCOPE)
endfunction()

set_property(GLOBAL APPEND PROPERTY JOB_POOLS pops_heavy_tu=${POPS_HEAVY_TU_POOL})

function(pops_cap_opt_for_kokkos_ram target)
  target_compile_options(${target} PRIVATE
    $<$<AND:$<CONFIG:Release,RelWithDebInfo,MinSizeRel>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-O0>)
  set_target_properties(${target} PROPERTIES JOB_POOL_COMPILE pops_heavy_tu)
endfunction()

# --- ADC-336 (P0-A) : les deux TU lourdes du runtime compilees UNE SEULE FOIS par config -----------
# Avant, ~23 executables de test re-listaient python/bindings/system/base/system.cpp et/ou python/bindings/amr/amr_system.cpp en source
# (chacune une compilation cc1plus de plusieurs Go / centaines de secondes -- la meme TU recompilee
# jusqu'a 20x dans un build de tests). On les compile UNE fois dans deux bibliotheques OBJECT, splicees
# dans chaque executable consommateur EXACTEMENT comme si la .cpp y etait encore listee : memes objets,
# donc tous les symboles (y compris POPS_EXPORT) presents, et -rdynamic les exporte toujours.
# Parite bit-a-bit : la cible de reference est ce que ces tests compilaient deja -- pops::pops seul,
# AUCUN pops_dev_options, AUCUN define POPS_* (les POPS_TEST_* des loaders natifs sont PRIVATE a leur .cpp
# et inutilises par la TU lourde) -- donc on n'ajoute rien.
# -O0 PUBLIC : cape la TU lourde ET se propage a la .cpp de chaque test consommateur, pour que le chemin
# add_compiled_model (instancie dans la .cpp du test) reste au MEME -O que le chemin add_block (dans cet
# objet partage). L'ancien cap par cible couvrait les deux a -O0 ; le preserver garde l'ensemble
# (TU, flags) identique a aujourd'hui et evite toute derive de bits par contraction FMA a -O3 (GCC).
# Le pool ninja pops_heavy_tu (taille 1, declare ci-dessus) reste PRIVATE : seule la TU lourde se
# serialise (anti-OOM), pas les petites .cpp de test.
# _pops (python/CMakeLists.txt) n'est PAS touche : il compile deja chaque TU une fois avec ses propres
# flags de production (POPS_HEADER_SIG/visibilite/...), et AUCUN preset ne co-construit tests+python
# (POPS_BUILD_TESTS et POPS_BUILD_PYTHON sont mutuellement exclusifs dans CMakePresets.json) -- cette
# bibliotheque est donc strictement tests-only et ne peut pas etre partagee avec _pops.
# ADC-593: the per-route block-build seam TUs (system/{isothermal,compressible}, amr/block, amr/compiled)
# are GENERATED from the same declarative manifest the _pops module uses (python/bindings/
# seam_combinations.cmake), into ${CMAKE_CURRENT_BINARY_DIR}/generated_seams here. The C++ test targets
# splice the SAME generated leaves the module compiles, so a manifest row is the ONE place a (transport,
# flux) leaf is registered for both the module and the tests. amr_system.cpp / system.cpp CALL the seams,
# so the generated leaves must be compiled/linked wherever those heavy TUs are.
include(${CMAKE_CURRENT_SOURCE_DIR}/../python/bindings/seam_combinations.cmake)
set(POPS_GENERATED_SEAMS_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated_seams)
set(POPS_SYSTEM_GENERATED_SEAMS "")
set(POPS_AMR_GENERATED_SEAMS "")
foreach(_seam_row ${POPS_SEAM_COMBINATIONS})
  string(REPLACE "|" ";" _cols "${_seam_row}")
  list(GET _cols 1 _side)
  if(_side STREQUAL "system")
    pops_generate_seam("${_seam_row}" "${POPS_GENERATED_SEAMS_DIR}" POPS_SYSTEM_GENERATED_SEAMS)
  else()
    pops_generate_seam("${_seam_row}" "${POPS_GENERATED_SEAMS_DIR}" POPS_AMR_GENERATED_SEAMS)
  endif()
endforeach()

# ADC-632: system.cpp is split by responsibility into sibling TUs sharing the binding-private
# system_impl.hpp (install / fields / io / profiling / program); the OBJECT lib compiles the same
# six TUs the _pops module does, so every consumer keeps the full System facade symbol set.
add_library(pops_runtime_system OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/../python/bindings/system/base/system.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../python/bindings/system/base/system_install.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../python/bindings/system/base/system_fields.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../python/bindings/system/base/system_io.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../python/bindings/system/base/system_profiling.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../python/bindings/system/base/system_program.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../python/bindings/system/base/system_polar.cpp
            ${POPS_SYSTEM_GENERATED_SEAMS})
# ADC-335: the AMR transport-seam TUs (multi-block build_amr_block_<t> + single-block
# build_amr_compiled_<t>). amr_system.cpp CALLS them, so they must be compiled/linked wherever
# amr_system.cpp is -- the pops_runtime_amr object lib AND the ENABLE_EXPORTS native-loader tests that
# re-list amr_system.cpp directly (below). One list, reused, so none is forgotten. The two thin riemann
# DISPATCHERS (amr_block_compressible.cpp / amr_compiled_compressible.cpp) stay hand-written; the per-flux
# leaves come from POPS_AMR_GENERATED_SEAMS.
set(POPS_AMR_SEAM
    ${CMAKE_CURRENT_SOURCE_DIR}/../python/bindings/amr/block/compressible/amr_block_compressible.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../python/bindings/amr/compiled/compressible/amr_compiled_compressible.cpp
    ${POPS_AMR_GENERATED_SEAMS})
add_library(pops_runtime_amr    OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/../python/bindings/amr/amr_system.cpp ${POPS_AMR_SEAM})
foreach(_rt pops_runtime_system pops_runtime_amr)
  target_link_libraries(${_rt} PUBLIC pops::pops ${CMAKE_DL_LIBS})
  target_compile_options(${_rt} PUBLIC
    $<$<AND:$<CONFIG:Release,RelWithDebInfo,MinSizeRel>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-O0>)
  set_target_properties(${_rt} PROPERTIES JOB_POOL_COMPILE pops_heavy_tu)
endforeach()

# Generated C++ solver kernel used by test_solver_codegen_generated.
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  add_test(
    NAME test_component_catalog_generated
    COMMAND ${CMAKE_COMMAND} -E env "PYTHONDONTWRITEBYTECODE=1"
            "${Python3_EXECUTABLE}"
            "${CMAKE_SOURCE_DIR}/scripts/generate_component_catalog.py" --check)
  set_tests_properties(test_component_catalog_generated PROPERTIES LABELS "architecture;generated")

  set(_pops_gen_solver_hpp "${CMAKE_CURRENT_BINARY_DIR}/pops_generated_solver.hpp")
  file(GLOB_RECURSE _pops_gen_solver_pysrc CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/python/pops/*.py")
  add_custom_command(
    OUTPUT "${_pops_gen_solver_hpp}"
    COMMAND ${CMAKE_COMMAND} -E env "PYTHONDONTWRITEBYTECODE=1"
            "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/scripts/gen_solver_kernel.py"
            "${_pops_gen_solver_hpp}"
    DEPENDS "${CMAKE_SOURCE_DIR}/scripts/gen_solver_kernel.py" ${_pops_gen_solver_pysrc}
    COMMENT "Generating custom-solver C++ kernel"
    VERBATIM)
  add_custom_target(pops_gen_solver_kernel DEPENDS "${_pops_gen_solver_hpp}")
endif()

set(POPS_CPP_STANDARD_TESTS
  test_profiler
  test_cache_manager
  test_checkpoint_cache
  test_external_brick
  test_external_brick_json
  test_runtime_environment
  test_residual_operator
  test_capability_report
  test_canonical_identity
  test_box2d
  test_fab2d
  test_box_array
  test_multifab
  test_sync_residence
  test_reduce
  test_reduce_abs_sum
  test_dense_eig
  test_route_ids
  test_brick_catalog
  test_component_interfaces
  test_fill_boundary
  test_fill_boundary_cache
  test_copy_schedule_cache
  test_physical_bc
  test_geometry
  test_refinement
  test_ref_ratio
  test_amr_hierarchy
  test_cluster
  test_regrid
  test_flux_register
  test_coverage_mask
  test_patch_range
  test_cf_interface
  test_program_reflux_ledger
  test_amr_diagnostics
  test_load_balance
  test_poisson_smoother
  test_poisson_smoother_cache
  test_geometric_mg
  test_poisson_convergence
  test_poisson_fft
  test_elliptic_operator
  test_elliptic_problem
  test_elliptic_composite_rhs
  test_composite_fac_adjacent
  test_composite_fac_nlevel
  test_composite_fac_poisson
  test_composite_fac_variable_eps
  test_composite_fac_tensor
  test_amr_composite_poisson
  test_poisson_disc
  test_solve_robust
  test_cut_cell
  test_cut_fraction
  test_cut_fraction_mg_identity
  test_cut_cell_anisotropic
  test_variable_epsilon
  test_screened_poisson
  test_anisotropic_epsilon
  test_cut_cell_anisotropic_multibox
  test_full_tensor_operator
  test_polar_poisson_mms
  test_polar_tensor_elliptic_mms
  test_polar_schur_multibox
  test_krylov_solver
  test_numerical_defaults
  test_fallback_diagnostics
  test_structured_solver_diagnostics
  test_public_validation_errors
  test_program_context_schur_free
  test_platform_manifest
  test_generic_krylov
  test_solver_codegen_generated
  test_dispatch_tags
  test_scheme_dispatch
  test_model_registry
  test_spatial_discretisation
  test_primitive_recon
  test_system_abstraction
  test_system_coupler
  test_two_species_minimal
  test_coupled_source
  test_coupling_operator_contract
  test_system_two_explicit
  test_assembler_driver
  test_amr_system_coupler
  test_amr_source_covered_cells
  test_amr_composite_source_conservation
  test_amr_layout_guard
  test_amr_stride_cadence
  test_system_hardening
  test_imex_partial
  test_imex_transport
  test_multirate_stride
  test_cfl_dt
  test_adaptive_multirate
  test_user_time_integrator
  test_diffusion
  test_amr_diffusion
  test_amr_spatial_parity
  test_amr_positivity_floor
  test_disc_domain_mask
  test_eb_transport
  test_embedded_boundary_generic
  test_aux_extra
  test_aux_names
  test_aux_halo
  test_aux_coupler_bz
  test_aux_system_bz
  test_aux_composite
  test_amr_aux_bz
  test_amr_system_bz_pop
  test_amr_system_bz_multibox
  test_aux_single_source
  test_weno_convergence
  test_splitting
  test_imex_ap
  test_ap_limit
  test_polar_transport_mms
  test_polar_system_step
  test_polar_mms_vr
  test_polar_fluid_transport
  test_polar_lorentz_source
  test_dynamic_model
  test_block_builder
  test_block_inverse
  test_weno5_ssprk3
  test_positivity_floor
  test_wave_speed_cache_engagement
  test_variable_role
  test_variable_user_role
  test_isothermal_vacuum_floor
  test_roe_flux
  test_riemann_capabilities
  test_newton_robustness
  test_elliptic_interface
  test_field_context
  test_system_registries
)

foreach(_test IN LISTS POPS_CPP_STANDARD_TESTS)
  pops_add_test(${_test})
endforeach()

# The Cartesian-vs-polar radial-diffusion comparison runs two full advections and sits close to the
# 600 s default cap on slow CI runners (timed out on identical binaries across consecutive master
# runs, ADC-624); labels stay inferred (unit physics fast).
pops_add_gtest_suite(NAME test_polar_ring_advection TIMEOUT 1800)

if(TARGET test_solver_codegen_generated AND TARGET pops_gen_solver_kernel)
  target_include_directories(test_solver_codegen_generated PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
  add_dependencies(test_solver_codegen_generated pops_gen_solver_kernel)
endif()

# Tests that need runtime object libraries, dlopen symbols, or injected native-loader compiler data.
pops_test_source(_src test_facade_routing)
pops_add_gtest_suite(NAME test_facade_routing SOURCES "${_src}" EXTRA_LIBS ${CMAKE_DL_LIBS} pops_runtime_system)

pops_test_source(_src test_exb_seam)
pops_add_gtest_suite(NAME test_exb_seam SOURCES "${_src}" EXTRA_LIBS ${CMAKE_DL_LIBS} pops_runtime_system)

pops_test_source(_src test_coupled_fieldsolve)
pops_add_gtest_suite(NAME test_coupled_fieldsolve SOURCES "${_src}" EXTRA_LIBS ${CMAKE_DL_LIBS} pops_runtime_system)

pops_test_source(_src test_config_model_validation)
pops_add_gtest_suite(NAME test_config_model_validation SOURCES "${_src}" EXTRA_LIBS ${CMAKE_DL_LIBS} pops_runtime_system pops_runtime_amr)

pops_test_source(_src test_native_aux_named)
pops_add_gtest_suite(NAME test_native_aux_named SOURCES "${_src}" EXTRA_LIBS ${CMAKE_DL_LIBS} pops_runtime_system)

pops_test_source(_src test_compiled_model_parity)
pops_add_gtest_suite(NAME test_compiled_model_parity SOURCES "${_src}" EXTRA_LIBS ${CMAKE_DL_LIBS} pops_runtime_system)

pops_test_source(_src test_weno5_compiled_model)
pops_add_gtest_suite(NAME test_weno5_compiled_model SOURCES "${_src}" EXTRA_LIBS ${CMAKE_DL_LIBS} pops_runtime_system)

pops_test_source(_src test_amr_compiled_model)
pops_add_gtest_suite(NAME test_amr_compiled_model SOURCES "${_src}" EXTRA_LIBS ${CMAKE_DL_LIBS} pops_runtime_amr)

pops_test_source(_src test_amr_seed_no_refine)
pops_add_gtest_suite(NAME test_amr_seed_no_refine SOURCES "${_src}" EXTRA_LIBS ${CMAKE_DL_LIBS} pops_runtime_amr)

pops_test_source(_src test_amr_regrid_variable)
pops_add_gtest_suite(NAME test_amr_regrid_variable SOURCES "${_src}" EXTRA_LIBS ${CMAKE_DL_LIBS} pops_runtime_amr)

pops_test_source(_src test_amr_potential)
pops_add_gtest_suite(NAME test_amr_potential SOURCES "${_src}" EXTRA_LIBS ${CMAKE_DL_LIBS} pops_runtime_system pops_runtime_amr)

pops_test_source(_src test_amr_distribute_coarse_serial)
pops_add_gtest_suite(NAME test_amr_distribute_coarse_serial SOURCES "${_src}" EXTRA_LIBS ${CMAKE_DL_LIBS} pops_runtime_amr)

pops_test_source(_src test_amr_native_loader)
pops_native_loader_definitions(_native_defs)
pops_add_gtest_suite(NAME test_amr_native_loader SOURCES "${_src}" EXTRA_LIBS ${CMAKE_DL_LIBS} pops_runtime_amr ENABLE_EXPORTS RESOURCE_LOCK native_loader COMPILE_DEFINITIONS ${_native_defs})

pops_test_source(_src test_external_riemann_dispatch)
pops_native_loader_definitions(_native_defs)
set(_external_riemann_defs ${_native_defs})
if(POPS_HAS_KOKKOS)
  list(APPEND _external_riemann_defs
    POPS_TEST_KOKKOS_INC="$<JOIN:$<TARGET_PROPERTY:Kokkos::kokkos,INTERFACE_INCLUDE_DIRECTORIES>, >"
    POPS_TEST_KOKKOS_DEFS="$<JOIN:$<TARGET_PROPERTY:Kokkos::kokkos,INTERFACE_COMPILE_DEFINITIONS>, >"
    POPS_TEST_KOKKOS_OPTS="$<JOIN:$<TARGET_PROPERTY:Kokkos::kokkos,INTERFACE_COMPILE_OPTIONS>, >")
endif()
pops_add_gtest_suite(NAME test_external_riemann_dispatch SOURCES "${_src}" EXTRA_LIBS ${CMAKE_DL_LIBS} ENABLE_EXPORTS RESOURCE_LOCK native_loader COMPILE_DEFINITIONS ${_external_riemann_defs})

pops_test_source(_src test_amr_weno5_native)
pops_native_loader_definitions(_native_defs)
pops_add_gtest_suite(NAME test_amr_weno5_native SOURCES "${_src}" EXTRA_LIBS ${CMAKE_DL_LIBS} pops_runtime_amr ENABLE_EXPORTS RESOURCE_LOCK native_loader COMPILE_DEFINITIONS ${_native_defs})

pops_test_source(_src test_amr_riemann_native)
pops_native_loader_definitions(_native_defs)
pops_add_gtest_suite(NAME test_amr_riemann_native SOURCES "${_src}" EXTRA_LIBS ${CMAKE_DL_LIBS} pops_runtime_amr ENABLE_EXPORTS RESOURCE_LOCK native_loader COMPILE_DEFINITIONS ${_native_defs})

pops_test_source(_src test_amr_imex_native)
pops_native_loader_definitions(_native_defs)
pops_add_gtest_suite(NAME test_amr_imex_native SOURCES "${_src}" EXTRA_LIBS ${CMAKE_DL_LIBS} pops_runtime_amr ENABLE_EXPORTS RESOURCE_LOCK native_loader COMPILE_DEFINITIONS ${_native_defs})

pops_test_source(_src test_amr_system_contract)
pops_add_gtest_suite(NAME test_amr_system_contract SOURCES "${_src}" EXTRA_LIBS ${CMAKE_DL_LIBS} pops_runtime_amr)

pops_test_source(_src test_amr_system_twoblock)
pops_add_gtest_suite(NAME test_amr_system_twoblock SOURCES "${_src}" EXTRA_LIBS ${CMAKE_DL_LIBS} pops_runtime_amr)

pops_test_source(_src test_amr_named_field)
pops_add_gtest_suite(NAME test_amr_named_field SOURCES "${_src}" EXTRA_LIBS ${CMAKE_DL_LIBS} pops_runtime_amr)

pops_test_source(_src test_amr_multiblock_compiled)
pops_add_gtest_suite(NAME test_amr_multiblock_compiled SOURCES "${_src}" EXTRA_LIBS ${CMAKE_DL_LIBS} pops_runtime_amr)

pops_test_source(_src test_amr_multiblock_substeps)
pops_add_gtest_suite(NAME test_amr_multiblock_substeps SOURCES "${_src}" EXTRA_LIBS ${CMAKE_DL_LIBS} pops_runtime_amr)

pops_test_source(_src test_amr_multiblock_coupled_source)
pops_add_gtest_suite(NAME test_amr_multiblock_coupled_source SOURCES "${_src}" EXTRA_LIBS ${CMAKE_DL_LIBS} pops_runtime_amr)

pops_test_source(_src test_amr_coupled_source_role_strict)
pops_add_gtest_suite(NAME test_amr_coupled_source_role_strict SOURCES "${_src}" EXTRA_LIBS ${CMAKE_DL_LIBS} pops_runtime_amr)

pops_test_source(_src test_amr_multiblock_imex)
pops_add_gtest_suite(NAME test_amr_multiblock_imex SOURCES "${_src}" EXTRA_LIBS ${CMAKE_DL_LIBS} pops_runtime_amr)

pops_test_source(_src test_amr_multiblock_regrid_union)
pops_add_gtest_suite(NAME test_amr_multiblock_regrid_union SOURCES "${_src}" EXTRA_LIBS ${CMAKE_DL_LIBS} pops_runtime_amr)

pops_test_source(_src test_amr_history_ring)
pops_add_gtest_suite(NAME test_amr_history_ring SOURCES "${_src}" EXTRA_LIBS ${CMAKE_DL_LIBS} pops_runtime_amr)

pops_test_source(_src test_amr_transfer_properties)
pops_add_gtest_suite(NAME test_amr_transfer_properties SOURCES "${_src}" EXTRA_LIBS ${CMAKE_DL_LIBS} pops_runtime_amr)

pops_test_source(_src test_aux_runtime_bz)
pops_add_gtest_suite(NAME test_aux_runtime_bz SOURCES "${_src}" EXTRA_LIBS ${CMAKE_DL_LIBS} pops_runtime_system)

pops_test_source(_src test_coupled_source_stack_limit)
pops_add_gtest_suite(NAME test_coupled_source_stack_limit SOURCES "${_src}" EXTRA_LIBS ${CMAKE_DL_LIBS} pops_runtime_system)

pops_test_source(_src test_aux_te)
pops_add_gtest_suite(NAME test_aux_te SOURCES "${_src}" EXTRA_LIBS ${CMAKE_DL_LIBS} pops_runtime_system)

pops_test_source(_src test_program_runtime)
pops_add_gtest_suite(NAME test_program_runtime SOURCES "${_src}" EXTRA_LIBS ${CMAKE_DL_LIBS} pops_runtime_system)

# ADC-538: the native ProgramContext contract + the history-ring checkpoint round-trip both drive a
# real System (the ring memory is co-distributed with block 0), so they link the heavy runtime object
# library like test_program_runtime -- NOT the standalone-core POPS_CPP_STANDARD_TESTS list.
pops_test_source(_src test_program_context_contract)
pops_add_gtest_suite(NAME test_program_context_contract SOURCES "${_src}" EXTRA_LIBS ${CMAKE_DL_LIBS} pops_runtime_system)

pops_test_source(_src test_checkpoint_history)
pops_add_gtest_suite(NAME test_checkpoint_history SOURCES "${_src}" EXTRA_LIBS ${CMAKE_DL_LIBS} pops_runtime_system)

# ADC-626: selective history persistence + deterministic ring replay. Drives a real System (installs a
# native ramp step closure and calls rebuild_history_slots), so it links the heavy runtime object
# library like test_checkpoint_history above.
pops_test_source(_src test_checkpoint_history_policy)
pops_add_gtest_suite(NAME test_checkpoint_history_policy SOURCES "${_src}" EXTRA_LIBS ${CMAKE_DL_LIBS} pops_runtime_system)

pops_test_source(_src test_program_loader)
pops_native_loader_definitions(_native_defs)
pops_add_gtest_suite(NAME test_program_loader SOURCES "${_src}" EXTRA_LIBS ${CMAKE_DL_LIBS} pops_runtime_system ENABLE_EXPORTS RESOURCE_LOCK native_loader COMPILE_DEFINITIONS ${_native_defs})

# ADC-538: the ABI symbol-presence fence compiles a stub problem.so at test time and dlsyms the Program
# + module-metadata symbol contract -- gated exactly like test_program_loader (ENABLE_EXPORTS so the
# System seam symbols export; native_loader lock so the runtime compiles are serialized).
pops_test_source(_src test_program_abi_symbols)
pops_native_loader_definitions(_native_defs)
pops_add_gtest_suite(NAME test_program_abi_symbols SOURCES "${_src}" EXTRA_LIBS ${CMAKE_DL_LIBS} pops_runtime_system ENABLE_EXPORTS RESOURCE_LOCK native_loader COMPILE_DEFINITIONS ${_native_defs})

pops_test_source(_src test_native_loader_param_overflow)
pops_native_loader_definitions(_native_defs)
pops_add_gtest_suite(NAME test_native_loader_param_overflow SOURCES "${_src}" EXTRA_LIBS ${CMAKE_DL_LIBS} pops_runtime_system ENABLE_EXPORTS RESOURCE_LOCK native_loader COMPILE_DEFINITIONS ${_native_defs})

pops_test_source(_src test_module_metadata)
pops_add_gtest_suite(NAME test_module_metadata SOURCES "${_src}" EXTRA_LIBS ${CMAKE_DL_LIBS})

# ADC-622: external-brick registry PER-.so ISOLATION. TWO fixture .so, each registering DISJOINT brick
# ids from external_brick.hpp (header-light, NO Kokkos / _pops), are dlopen'd in one process by
# test_external_brick_isolation, which asserts each .so's manifest lists only ITS OWN bricks. The fix
# under test is the hidden-visibility BrickRegistry (POPS_BRICK_LOCAL): without it, on Linux/GCC the
# header-only registry singleton is STB_GNU_UNIQUE and the loader unifies it across the two .so.
# -fno-gnu-unique on the fixture .so is the belt-and-suspenders backstop (GCC only; guarded).
function(pops_add_brick_fixture_so name source)
  add_library(${name} SHARED "${CMAKE_CURRENT_SOURCE_DIR}/${source}")
  target_link_libraries(${name} PRIVATE pops::pops)
  target_compile_features(${name} PRIVATE cxx_std_${POPS_CXX_STD})
  set_target_properties(${name} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  # Reproduce a REAL brick .so build: -fno-gnu-unique so GCC never emits the registry singleton as
  # STB_GNU_UNIQUE (defense in depth behind the hidden-visibility fix). GCC only -- Clang / MSVC reject
  # the flag and never emit GNU_UNIQUE anyway.
  target_compile_options(${name} PRIVATE $<$<CXX_COMPILER_ID:GNU>:-fno-gnu-unique>)
endfunction()
pops_add_brick_fixture_so(pops_iso_fixture_a cpp/integration/runtime/external_brick_fixture_a.cpp)
pops_add_brick_fixture_so(pops_iso_fixture_b cpp/integration/runtime/external_brick_fixture_b.cpp)

pops_test_source(_src test_external_brick_isolation)
pops_add_gtest_suite(NAME test_external_brick_isolation SOURCES "${_src}" EXTRA_LIBS ${CMAKE_DL_LIBS}
  COMPILE_DEFINITIONS
    POPS_ISO_FIXTURE_A_SO="$<TARGET_FILE:pops_iso_fixture_a>"
    POPS_ISO_FIXTURE_B_SO="$<TARGET_FILE:pops_iso_fixture_b>")
add_dependencies(test_external_brick_isolation pops_iso_fixture_a pops_iso_fixture_b)


if(POPS_HAS_MPI)
  set(POPS_MPI_RANKS_test_mpi_polar_schur 1 2 4)
  set(POPS_MPI_RANKS_test_mpi_mbox_parity 1 2 4)
  set(POPS_MPI_RANKS_test_mpi_hybrid_mbox_parity 1 2 4)
  set(POPS_MPI_RANKS_test_mpi_cutcell_multibox 1 2 4)
  set(POPS_MPI_RANKS_test_mpi_amr_compiled_parity 1 2 4)
  set(POPS_MPI_RANKS_test_mpi_amr_twoblock_parity 1 2 4)
  set(POPS_MPI_RANKS_test_mpi_amr_distributed_coarse 1 2 4)
  set(POPS_MPI_RANKS_test_mpi_composite_fac 1 2 4)
  set(POPS_MPI_RANKS_test_amr_regrid_mpi_parity 1 2 4)
  set(POPS_MPI_RANKS_test_mpi_system_solve_fields 1 2 4)
  set(POPS_MPI_RANKS_test_mpi_system_fft 1 2 4)
  set(POPS_MPI_RANKS_test_mpi_coupled_source 1 2 4)
  set(POPS_MPI_RANKS_test_mpi_system_gather_scatter 1 2 4)
  set(POPS_MPI_RANKS_test_mpi_system_io_gather 1 2 4)
  set(POPS_MPI_RANKS_test_mpi_array_reduce 4)
  set(POPS_MPI_RANKS_test_mpi_coupler_inject 4)
  set(POPS_MPI_RANKS_test_mpi_fft_distributed 4)
  set(POPS_MPI_RANKS_test_mpi_fillboundary 4)
  set(POPS_MPI_RANKS_test_mpi_poisson 4)
  set(POPS_MPI_RANKS_test_mpi_redistribute 4)
  set(POPS_MPI_RANKS_test_mpi_smoke 4)

  set(POPS_CPP_MPI_ONLY_TESTS
    test_mpi_polar_schur
    test_mpi_mbox_parity
    test_mpi_hybrid_mbox_parity
    test_mpi_cutcell_multibox
    test_mpi_amr_compiled_parity
    test_mpi_amr_twoblock_parity
    test_mpi_amr_distributed_coarse
    test_mpi_composite_fac
    test_amr_regrid_mpi_parity
    test_mpi_system_solve_fields
    test_mpi_system_fft
    test_mpi_coupled_source
    test_mpi_system_gather_scatter
    test_mpi_system_io_gather
    test_mpi_array_reduce
    test_mpi_coupler_inject
    test_mpi_fft_distributed
    test_mpi_fillboundary
    test_mpi_poisson
    test_mpi_redistribute
    test_mpi_smoke
  )

  foreach(_test IN LISTS POPS_CPP_MPI_ONLY_TESTS)
    set(_extra_libs)
    # test_mpi_system_* + test_mpi_coupled_source need the System runtime; test_mpi_amr_* +
    # test_amr_regrid_mpi_parity need the AMR runtime (AmrSystem). These are PREFIX matches --
    # a `$`-anchored pattern would only match the bare prefix and leave e.g.
    # test_mpi_amr_twoblock_parity / test_mpi_system_solve_fields without their runtime object
    # lib, so the link fails with undefined AmrSystem/System symbols.
    if(_test MATCHES "^test_mpi_system_" OR _test STREQUAL "test_mpi_coupled_source")
      list(APPEND _extra_libs ${CMAKE_DL_LIBS} pops_runtime_system)
    elseif(_test MATCHES "^test_mpi_amr_" OR _test STREQUAL "test_amr_regrid_mpi_parity")
      list(APPEND _extra_libs ${CMAKE_DL_LIBS} pops_runtime_amr)
    endif()
    pops_add_mpi_gtest_suite(${_test} EXTRA_LIBS ${_extra_libs} RANKS ${POPS_MPI_RANKS_${_test}})
  endforeach()

  foreach(_np 1 2 4)
    pops_add_mpi_rank_test(test_krylov_solver ${_np})
  endforeach()
  foreach(_np 1 2 4)
    pops_add_mpi_rank_test(test_fill_boundary_cache ${_np})
  endforeach()
  foreach(_np 1 2 4)
    pops_add_mpi_rank_test(test_copy_schedule_cache ${_np})
  endforeach()
  foreach(_np 2 4)
    pops_add_mpi_rank_test(test_amr_system_bz_multibox ${_np})
  endforeach()
endif()

get_property(_pops_dir_targets DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY BUILDSYSTEM_TARGETS)
set(_pops_capped 0)
foreach(_t ${_pops_dir_targets})
  get_target_property(_t_sources ${_t} SOURCES)
  get_target_property(_t_libs ${_t} LINK_LIBRARIES)
  if(_t_sources MATCHES "python/(amr_)?system\.cpp" OR _t_libs MATCHES "pops_runtime_(system|amr)")
    pops_cap_opt_for_kokkos_ram(${_t})
    math(EXPR _pops_capped "${_pops_capped}+1")
  endif()
endforeach()
message(STATUS "pops tests: ${_pops_capped} heavy runtime targets capped (-O0 + pops_heavy_tu pool)")
