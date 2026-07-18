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
  # GoogleTest is a build-only test dependency.  Its upstream install rules must never leak into
  # `cmake --install` for the PoPS SDK (and must not require unbuilt gmock archives when a package
  # consumer smoke installs the header-only core).
  set(INSTALL_GTEST OFF CACHE BOOL "Do not install PoPS' private GoogleTest dependency" FORCE)
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
  set(options ENABLE_EXPORTS NO_DISCOVER RUNTIME_DISCOVERY)
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

  set(_pops_test_labels "${ARG_LABELS};cpp-target:${ARG_NAME}")

  if(NOT ARG_NO_DISCOVER)
    if(ARG_RUNTIME_DISCOVERY)
      # Runtime discovery is an explicit escape hatch for a future suite whose
      # parameterized or generated names cannot be recovered from its sources.
      # Keep it POST_BUILD: PRE_TEST would enumerate every executable on each
      # CTest inventory and again on execution, which is especially costly for
      # CI's label-authenticated shards.
      set(_pops_test_list "${ARG_NAME}_TESTS")
      set(_pops_test_properties TIMEOUT ${ARG_TIMEOUT})
      if(ARG_RESOURCE_LOCK)
        list(APPEND _pops_test_properties RESOURCE_LOCK "${ARG_RESOURCE_LOCK}")
      endif()
      gtest_discover_tests(${ARG_NAME}
        DISCOVERY_MODE POST_BUILD
        DISCOVERY_TIMEOUT 60
        XML_OUTPUT_DIR "${_pops_gtest_xml_dir}/${ARG_NAME}"
        TEST_LIST "${_pops_test_list}"
        PROPERTIES ${_pops_test_properties})

      # ``gtest_discover_tests(PROPERTIES ...)`` transports an alternating
      # property/value list through a generated CMake script.  Assign list-valued
      # LABELS/RESOURCE_GROUPS after discovery so CTest receives atomic labels.
      set(_pops_list_properties_file
        "${CMAKE_CURRENT_BINARY_DIR}/${ARG_NAME}_list_properties.cmake")
      set(_pops_resource_groups_assignment "")
      if(ARG_RESOURCE_GROUPS)
        set(_pops_resource_groups_assignment
          "    RESOURCE_GROUPS [==[${ARG_RESOURCE_GROUPS}]==]\n")
      endif()
      file(GENERATE OUTPUT "${_pops_list_properties_file}" CONTENT
"foreach(_pops_discovered_test IN LISTS ${_pops_test_list})
  set_tests_properties(\"\${_pops_discovered_test}\" PROPERTIES
    LABELS [==[${_pops_test_labels}]==]
${_pops_resource_groups_assignment}  )
endforeach()
")
      set_property(DIRECTORY APPEND PROPERTY TEST_INCLUDE_FILES
        "${_pops_list_properties_file}")
    else()
      # All current PoPS suites use ordinary TEST/TEST_F declarations.  Register
      # them at configure time so CTest reads one CTestTestfile.cmake instead of
      # two generated include scripts per executable on every inventory/run.
      # Omitting SKIP_DEPENDENCY intentionally makes CMake reconfigure whenever
      # a test declaration changes.
      gtest_add_tests(
        TARGET ${ARG_NAME}
        SOURCES ${ARG_SOURCES}
        TEST_LIST _pops_discovered_tests)
      if(NOT _pops_discovered_tests)
        message(FATAL_ERROR
          "No GoogleTest cases found in ${ARG_NAME}; use NO_DISCOVER for a manually registered suite")
      endif()

      set_tests_properties(${_pops_discovered_tests} PROPERTIES
        LABELS "${_pops_test_labels}"
        TIMEOUT "${ARG_TIMEOUT}")
      if(ARG_RESOURCE_LOCK)
        set_tests_properties(${_pops_discovered_tests} PROPERTIES
          RESOURCE_LOCK "${ARG_RESOURCE_LOCK}")
      endif()
      if(ARG_RESOURCE_GROUPS)
        set_tests_properties(${_pops_discovered_tests} PROPERTIES
          RESOURCE_GROUPS "${ARG_RESOURCE_GROUPS}")
      endif()

      # Preserve the per-case GoogleTest XML artifacts previously provided by
      # gtest_discover_tests(XML_OUTPUT_DIR) without adding CTest include files.
      file(MAKE_DIRECTORY "${_pops_gtest_xml_dir}/${ARG_NAME}")
      foreach(_pops_discovered_test IN LISTS _pops_discovered_tests)
        set_property(TEST "${_pops_discovered_test}" APPEND PROPERTY ENVIRONMENT
          "GTEST_OUTPUT=xml:${_pops_gtest_xml_dir}/${ARG_NAME}/${_pops_discovered_test}.xml")
      endforeach()
    endif()
  endif()
endfunction()

function(pops_add_test name)
  pops_add_gtest_suite(NAME ${name})
endfunction()

function(pops_add_mpi_rank_test name nproc)
  set(options)
  set(oneValueArgs GTEST_FILTER)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "" ${ARGN})
  if(NOT TARGET ${name})
    message(FATAL_ERROR "MPI CTest entry ${name}_np${nproc} needs target ${name}")
  endif()
  set(_gtest_filter_arg)
  if(ARG_GTEST_FILTER)
    list(APPEND _gtest_filter_arg "--gtest_filter=${ARG_GTEST_FILTER}")
  endif()
  add_test(NAME ${name}_np${nproc}
           COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${nproc}
                   ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${name}>
                   --gtest_output=xml:${_pops_gtest_xml_dir}/${name}_np${nproc}.xml
                   ${_gtest_filter_arg}
                   ${MPIEXEC_POSTFLAGS})
  set_tests_properties(${name}_np${nproc} PROPERTIES
    LABELS "backend;mpi;medium"
    ENVIRONMENT "POPS_TEST_EXPECT_RANKS=${nproc}"
    PROCESSORS ${nproc}
    TIMEOUT 600)
endfunction()

function(pops_add_mpi_rank_parity_test name)
  set(options)
  set(oneValueArgs SIGNATURE_PREFIX)
  set(multiValueArgs RANKS)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  if(NOT TARGET ${name})
    message(FATAL_ERROR "MPI rank-parity CTest entry needs target ${name}")
  endif()
  if(NOT ARG_SIGNATURE_PREFIX MATCHES "^[A-Za-z0-9_.:-]+$")
    message(FATAL_ERROR
      "MPI rank-parity suite ${name} needs an alphanumeric SIGNATURE_PREFIX")
  endif()
  list(LENGTH ARG_RANKS _rank_count)
  if(_rank_count LESS 2)
    message(FATAL_ERROR "MPI rank-parity suite ${name} needs at least two RANKS")
  endif()

  set(_seen_ranks)
  set(_max_rank 0)
  set(_previous_rank 0)
  foreach(_rank IN LISTS ARG_RANKS)
    if(NOT _rank MATCHES "^[1-9][0-9]*$")
      message(FATAL_ERROR "MPI rank-parity suite ${name} has invalid rank ${_rank}")
    endif()
    if(_rank IN_LIST _seen_ranks)
      message(FATAL_ERROR "MPI rank-parity suite ${name} repeats rank ${_rank}")
    endif()
    if(_rank LESS_EQUAL _previous_rank)
      message(FATAL_ERROR "MPI rank-parity suite ${name} RANKS must be sorted")
    endif()
    list(APPEND _seen_ranks ${_rank})
    set(_previous_rank ${_rank})
    set(_max_rank ${_rank})
  endforeach()

  set(_test_name ${name}_rank_parity)
  add_test(NAME ${_test_name}
    COMMAND ${CMAKE_COMMAND}
      "-DPOPS_MPIEXEC_EXECUTABLE=${MPIEXEC_EXECUTABLE}"
      "-DPOPS_MPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
      "-DPOPS_MPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}"
      "-DPOPS_MPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
      "-DPOPS_MPI_TEST_EXECUTABLE=$<TARGET_FILE:${name}>"
      "-DPOPS_MPI_RANKS=${ARG_RANKS}"
      "-DPOPS_MPI_SIGNATURE_PREFIX=${ARG_SIGNATURE_PREFIX}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/run_mpi_rank_parity.cmake")
  set_tests_properties(${_test_name} PROPERTIES
    LABELS "backend;mpi;medium"
    PROCESSORS ${_max_rank}
    TIMEOUT 600)
endfunction()

function(pops_add_mpi_gtest_suite name)
  set(options)
  set(oneValueArgs)
  set(multiValueArgs EXTRA_LIBS RANKS RANK_PARITY COMPILE_DEFINITIONS)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  if(ARG_RANKS AND ARG_RANK_PARITY)
    message(FATAL_ERROR "MPI suite ${name} cannot mix RANKS and RANK_PARITY")
  endif()
  if(NOT ARG_RANKS)
    if(NOT ARG_RANK_PARITY)
      set(ARG_RANKS 1 2 4)
    endif()
  endif()
  pops_test_source(_src ${name})
  pops_add_gtest_suite(NAME ${name}
    SOURCES "${_src}"
    EXTRA_LIBS ${ARG_EXTRA_LIBS}
    COMPILE_DEFINITIONS ${ARG_COMPILE_DEFINITIONS}
    LABELS backend mpi medium
    NO_DISCOVER)
  if(ARG_RANK_PARITY)
    pops_add_mpi_rank_parity_test(${name}
      RANKS ${ARG_RANK_PARITY}
      SIGNATURE_PREFIX "POPS_MPI_PARITY_SIGNATURE_${name}")
  else()
    foreach(_np IN LISTS ARG_RANKS)
      pops_add_mpi_rank_test(${name} ${_np})
    endforeach()
  endif()
endfunction()

# A lifecycle test needs complete control over main(): its first executable statement must be the
# embedding application's MPI_Init_thread, before PoPS or a test framework can touch MPI.  Keep this
# registration separate from the GoogleTest wrapper so that ownership and finalization are proved by
# a real process lifecycle rather than by a fixture or mocked state.
function(pops_add_mpi_standalone_suite name)
  set(options)
  set(oneValueArgs)
  set(multiValueArgs RANKS)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  if(NOT ARG_RANKS)
    set(ARG_RANKS 1 2)
  endif()

  pops_test_source(_src ${name})
  add_executable(${name} "${_src}")
  set_target_properties(${name} PROPERTIES NO_SYSTEM_FROM_IMPORTED ON)
  target_link_libraries(${name} PRIVATE pops::pops pops_dev_options)
  target_compile_features(${name} PRIVATE cxx_std_${POPS_CXX_STD})

  foreach(_np IN LISTS ARG_RANKS)
    add_test(NAME ${name}_np${_np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${_np}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${name}> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(${name}_np${_np} PROPERTIES
      LABELS "backend;mpi;medium"
      PROCESSORS ${_np}
      TIMEOUT 60)
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

function(pops_cap_opt_for_kokkos_ram target)
  target_compile_options(${target} PRIVATE
    $<$<AND:$<CONFIG:Release,RelWithDebInfo,MinSizeRel>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-O0>)
  set_target_properties(${target} PROPERTIES JOB_POOL_COMPILE pops_heavy_test_tu)
endfunction()

# The central object targets are defined, configured and populated exclusively by src/CMakeLists.txt.
# Test executables below consume them through EXTRA_LIBS; this file never repeats runtime sources.

# Generated C++ solver kernel used by test_solver_codegen_generated.
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  add_test(
    NAME test_component_catalog_generated
    COMMAND ${CMAKE_COMMAND} -E env
            "PYTHONDONTWRITEBYTECODE=1"
            "PYTHONPATH=${CMAKE_SOURCE_DIR}/python"
            "${Python3_EXECUTABLE}"
            "${CMAKE_SOURCE_DIR}/scripts/generate_component_catalog.py" --check)
  set_tests_properties(test_component_catalog_generated PROPERTIES
    LABELS "python;architecture;generated")

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
  test_step_attempt_rejected_header_only
  test_world_communicator
  test_residual_operator
  test_capability_report
  test_canonical_identity
  test_box2d
  test_fab2d
  test_box_array
  test_multifab
  test_multiblock_interface_scheduler
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
  test_prepared_boundary_plan
  test_geometry
  test_refinement
  test_ref_ratio
  test_amr_hierarchy
  test_amr_tensor_fac_provider
  test_cluster
  test_regrid
  test_flux_register
  test_coverage_mask
  test_patch_range
  test_cf_interface
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
  test_numerical_defaults
  test_pure_field_algebra_extreme_dot
  test_scaled_scalar
  test_flux_interfaces
  test_fallback_diagnostics
  test_structured_solver_diagnostics
  test_public_validation_errors
  test_program_context_schur_free
  test_platform_manifest
  test_generic_krylov
  test_krylov_workspace_reentrancy
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
  test_field_nullspace
  test_field_context
  test_system_registries
)

foreach(_test IN LISTS POPS_CPP_STANDARD_TESTS)
  pops_add_test(${_test})
endforeach()
# The generic hierarchy tensor-provider contract is owned by AmrSystem at runtime.  Keep the test in
# the standard inventory, but link the AMR facade implementation that supplies its provider registry.
target_link_libraries(test_amr_tensor_fac_provider PRIVATE pops_runtime_amr)
target_link_libraries(test_multiblock_interface_scheduler PRIVATE pops_runtime_amr)
pops_cap_opt_for_kokkos_ram(test_multiblock_interface_scheduler)

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
pops_native_loader_definitions(_native_defs)
pops_add_gtest_suite(NAME test_native_aux_named SOURCES "${_src}" EXTRA_LIBS ${CMAKE_DL_LIBS} pops_runtime_system ENABLE_EXPORTS RESOURCE_LOCK native_loader COMPILE_DEFINITIONS ${_native_defs})

pops_test_source(_src test_compiled_model_parity)
pops_add_gtest_suite(NAME test_compiled_model_parity SOURCES "${_src}")

pops_test_source(_src test_weno5_compiled_model)
pops_add_gtest_suite(NAME test_weno5_compiled_model SOURCES "${_src}" EXTRA_LIBS ${CMAKE_DL_LIBS} pops_runtime_system)

pops_test_source(_src test_amr_compiled_model)
pops_add_gtest_suite(NAME test_amr_compiled_model SOURCES "${_src}")

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
pops_add_gtest_suite(NAME test_amr_native_loader SOURCES "${_src}" EXTRA_LIBS ${CMAKE_DL_LIBS} RESOURCE_LOCK native_loader COMPILE_DEFINITIONS ${_native_defs})

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
pops_add_gtest_suite(NAME test_amr_weno5_native SOURCES "${_src}")

pops_test_source(_src test_amr_riemann_native)
pops_native_loader_definitions(_native_defs)
pops_add_gtest_suite(NAME test_amr_riemann_native SOURCES "${_src}" EXTRA_LIBS ${CMAKE_DL_LIBS} pops_runtime_amr ENABLE_EXPORTS RESOURCE_LOCK native_loader COMPILE_DEFINITIONS ${_native_defs})

pops_test_source(_src test_amr_imex_native)
pops_native_loader_definitions(_native_defs)
pops_add_gtest_suite(NAME test_amr_imex_native SOURCES "${_src}" EXTRA_LIBS ${CMAKE_DL_LIBS} pops_runtime_amr ENABLE_EXPORTS RESOURCE_LOCK native_loader COMPILE_DEFINITIONS ${_native_defs})

pops_test_source(_src test_amr_system_contract)
pops_add_gtest_suite(NAME test_amr_system_contract SOURCES "${_src}" EXTRA_LIBS ${CMAKE_DL_LIBS} pops_runtime_amr)

pops_test_source(_src test_amr_ssprk2)
pops_add_gtest_suite(NAME test_amr_ssprk2 SOURCES "${_src}" EXTRA_LIBS ${CMAKE_DL_LIBS} pops_runtime_amr)

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

# Link-only ABI fence: the source itself references the out-of-line StepAttemptRejected key
# function/typeinfo, while the target requests only the AMR facade. The link therefore proves that
# pops_runtime_amr carries the one runtime-core authority transitively (no System dependency).
pops_test_source(_src test_step_attempt_rejected_amr_link)
pops_add_gtest_suite(
  NAME test_step_attempt_rejected_amr_link
  SOURCES "${_src}"
  EXTRA_LIBS pops_runtime_amr
  LABELS unit runtime amr fast)

pops_test_source(_src test_program_reflux_ledger)
pops_add_gtest_suite(NAME test_program_reflux_ledger SOURCES "${_src}" EXTRA_LIBS ${CMAKE_DL_LIBS} pops_runtime_amr)

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
  pops_add_mpi_standalone_suite(test_mpi_external_lifecycle RANKS 1 2)

  set(POPS_MPI_RANKS_test_mpi_polar_schur 1 2 4)
  set(POPS_MPI_RANKS_test_mpi_mbox_parity 1 2 4)
  set(POPS_MPI_RANKS_test_mpi_hybrid_mbox_parity 1 2 4)
  set(POPS_MPI_RANKS_test_mpi_cutcell_multibox 1 2 4)
  set(POPS_MPI_RANK_PARITY_test_mpi_amr_compiled_parity 1 2 4)
  set(POPS_MPI_RANKS_test_mpi_amr_twoblock_parity 1 2 4)
  set(POPS_MPI_RANK_PARITY_test_mpi_amr_distributed_coarse 1 2 4)
  set(POPS_MPI_RANKS_test_mpi_amr_program_reflux 2 4)
  set(POPS_MPI_RANKS_test_mpi_composite_fac 1 2 4)
  set(POPS_MPI_RANKS_test_amr_regrid_mpi_parity 1 2 4)
  set(POPS_MPI_RANKS_test_mpi_system_solve_fields 1 2 4)
  set(POPS_MPI_RANKS_test_mpi_system_fft 1 2 4)
  set(POPS_MPI_RANKS_test_mpi_coupled_source 1 2 4)
  set(POPS_MPI_RANKS_test_mpi_field_nullspace_preflight 2)
  set(POPS_MPI_RANKS_test_krylov_collective_contract 2)
  set(POPS_MPI_RANKS_test_mpi_field_plan_consensus 2)
  set(POPS_MPI_RANKS_test_mpi_system_gather_scatter 1 2 4)
  set(POPS_MPI_RANKS_test_mpi_system_io_gather 1 2 4)
  set(POPS_MPI_RANKS_test_mpi_array_reduce 4)
  set(POPS_MPI_RANKS_test_mpi_multiblock_interface_scheduler 2)
  set(POPS_MPI_RANKS_test_mpi_coupler_inject 4)
  set(POPS_MPI_RANKS_test_mpi_fft_distributed 4)
  set(POPS_MPI_RANKS_test_mpi_fillboundary 4)
  set(POPS_MPI_RANKS_test_mpi_poisson 4)
  set(POPS_MPI_RANKS_test_mpi_redistribute 4)
  set(POPS_MPI_RANKS_test_mpi_smoke 4)
  set(POPS_MPI_RANKS_test_generic_krylov 1 2 4)
  set(POPS_MPI_RANKS_test_krylov_workspace_reentrancy 2)
  set(POPS_MPI_RANKS_test_pure_field_algebra_extreme_dot 2)
  set(POPS_MPI_RANKS_test_geometric_mg 2)
  set(POPS_MPI_RANKS_test_fill_boundary_cache 1 2 4)
  set(POPS_MPI_RANKS_test_copy_schedule_cache 1 2 4)
  set(POPS_MPI_RANKS_test_amr_system_bz_multibox 2 4)
  set(POPS_MPI_RANKS_test_world_communicator 1 2)

  set(POPS_CPP_MPI_ONLY_TESTS
    test_mpi_polar_schur
    test_mpi_mbox_parity
    test_mpi_hybrid_mbox_parity
    test_mpi_cutcell_multibox
    test_mpi_amr_compiled_parity
    test_mpi_amr_twoblock_parity
    test_mpi_amr_distributed_coarse
    test_mpi_amr_program_reflux
    test_mpi_composite_fac
    test_amr_regrid_mpi_parity
    test_mpi_system_solve_fields
    test_mpi_system_fft
    test_mpi_coupled_source
    test_mpi_field_nullspace_preflight
    test_krylov_collective_contract
    test_mpi_field_plan_consensus
    test_mpi_system_gather_scatter
    test_mpi_system_io_gather
    test_mpi_array_reduce
    test_mpi_multiblock_interface_scheduler
    test_mpi_coupler_inject
    test_mpi_fft_distributed
    test_mpi_fillboundary
    test_mpi_poisson
    test_mpi_redistribute
    test_mpi_smoke
  )

  # The HDF5 proof exists only when the collective native backend was actually configured.  The
  # standard `mpi` and `ci-mpi` presets request that backend strictly; a custom MPI-only consumer
  # does not receive a green skipped target that would pretend to validate absent HDF5 code.
  if(POPS_HAS_PARALLEL_HDF5)
    set(POPS_MPI_RANKS_test_mpi_hdf5_collective 1 2)
    list(APPEND POPS_CPP_MPI_ONLY_TESTS test_mpi_hdf5_collective)
  endif()

  foreach(_test IN LISTS POPS_CPP_MPI_ONLY_TESTS)
    set(_extra_libs)
    # test_mpi_system_* + test_mpi_coupled_source need the System runtime; test_mpi_amr_* +
    # test_amr_regrid_mpi_parity need the AMR runtime (AmrSystem). These are PREFIX matches --
    # a `$`-anchored pattern would only match the bare prefix and leave e.g.
    # test_mpi_amr_twoblock_parity / test_mpi_system_solve_fields without their runtime object
    # lib, so the link fails with undefined AmrSystem/System symbols.
    if(_test STREQUAL "test_mpi_hdf5_collective")
      list(APPEND _extra_libs pops_runtime_output)
    elseif(_test STREQUAL "test_mpi_field_plan_consensus")
      list(APPEND _extra_libs ${CMAKE_DL_LIBS} pops_runtime_system pops_runtime_amr)
    elseif(_test MATCHES "^test_mpi_system_" OR _test STREQUAL "test_mpi_coupled_source")
      list(APPEND _extra_libs ${CMAKE_DL_LIBS} pops_runtime_system)
    elseif(_test MATCHES "^test_mpi_amr_" OR _test STREQUAL "test_amr_regrid_mpi_parity")
      list(APPEND _extra_libs ${CMAKE_DL_LIBS} pops_runtime_amr)
    endif()
    if(DEFINED POPS_MPI_RANK_PARITY_${_test})
      if(DEFINED POPS_MPI_RANKS_${_test})
        message(FATAL_ERROR "MPI suite ${_test} declares both rank and rank-parity launches")
      endif()
      pops_add_mpi_gtest_suite(${_test}
        EXTRA_LIBS ${_extra_libs}
        RANK_PARITY ${POPS_MPI_RANK_PARITY_${_test}})
    else()
      pops_add_mpi_gtest_suite(${_test}
        EXTRA_LIBS ${_extra_libs}
        RANKS ${POPS_MPI_RANKS_${_test}})
    endif()
  endforeach()
  set_tests_properties(test_mpi_field_plan_consensus_np2 PROPERTIES TIMEOUT 30)
  set_tests_properties(test_mpi_field_nullspace_preflight_np2 PROPERTIES
    ENVIRONMENT "POPS_TEST_EXPECT_RANKS=2;OMP_NUM_THREADS=1"
    TIMEOUT 30)
  set_tests_properties(test_krylov_collective_contract_np2 PROPERTIES
    ENVIRONMENT "POPS_TEST_EXPECT_RANKS=2;OMP_NUM_THREADS=1"
    TIMEOUT 30)
  # This regression consumes two MPI processor slots and exercises transport, not on-node
  # threading. Keep one Kokkos worker per rank so the CTest PROCESSORS=2 accounting is honest.
  set_tests_properties(test_mpi_amr_program_reflux_np2 test_mpi_amr_program_reflux_np4 PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1")
  # This regression proves MPI trace transport and intentionally does not exercise on-node
  # threading.  One Kokkos worker per rank avoids OpenMP oversubscription and matches CTest's
  # PROCESSORS=2 accounting.
  set_tests_properties(test_mpi_multiblock_interface_scheduler_np2 PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1")

  # These are ordinary serial suites with additional MPI launch variants. Their rank sets are
  # declared authoritatively as `mpi_variants` in tests/test_manifest.toml; the source-only
  # architecture gate compares this projection and these CMake registrations exactly.
  set(POPS_CPP_MPI_VARIANT_TESTS
    test_generic_krylov
    test_krylov_workspace_reentrancy
    test_pure_field_algebra_extreme_dot
    test_geometric_mg
    test_fill_boundary_cache
    test_copy_schedule_cache
    test_amr_system_bz_multibox
    test_world_communicator
  )
  string(CONCAT POPS_GENERIC_KRYLOV_MPI_FILTER
    "GenericKrylov.footprint_authenticates_layout_and_preconditioner_presence:"
    "GenericKrylov.cg_converges_on_spd_operator:"
    "GenericKrylov.bicgstab_converges_with_identity_preconditioner:"
    "GenericKrylov.gmres_converges_on_spd_operator:"
    "GenericKrylov.cg_refuses_unproven_operator_and_gmres_solves_nonsymmetric_operator:"
    "GenericKrylov.arbitrary_halo_depth_is_a_first_class_prepared_footprint:"
    "GenericKrylov.mpi_variant_distributes_real_work_to_every_rank:"
    "GenericKrylov.geometric_preconditioner_preserves_a_custom_distribution:"
    "GenericKrylov.every_method_solves_two_components_and_preconditioned_routes:"
    "GenericKrylov.every_method_is_invariant_to_extreme_finite_equation_scaling:"
    "GenericKrylov.every_method_solves_a_compatible_periodic_nullspace_problem:"
    "GenericKrylov.every_method_reports_an_incompatible_rhs_before_gauge_or_iteration:"
    "GenericKrylov.nonfinite_nullspace_rhs_is_invalid_not_incompatible:"
    "GenericKrylov.periodic_nullspace_solve_is_collective_safe_with_an_empty_rank:"
    "GenericKrylov.mpi_empty_rank_runs_restarted_gmres_and_preconditioned_bicgstab:"
    "GenericKrylov.generic_methods_linearize_real_dirichlet_and_robin_preconditioners:"
    "GenericKrylov.prepared_full_tensor_bicgstab_converges_where_the_diagonal_vcycle_stalls:"
    "GenericKrylov.prepared_tensor_identity_matches_the_geometric_mg_reference:"
    "GenericKrylov.prepared_physical_boundary_offsets_use_the_effective_rhs_and_a_linear_preconditioner:"
    "GenericKrylov.prepared_nonzero_dirichlet_problem_matches_mg_and_the_mms:"
    "GenericKrylov.failed_solves_report_no_solved_value:"
    "GenericKrylov.bicgstab_omega_breakdown_commits_alpha_then_fails_honestly:"
    "GenericKrylov.snapshot_mutation_is_refused_and_workspace_is_reused:"
    "GenericKrylov.prepared_hot_solves_allocate_no_fab_or_communication_storage:"
    "GenericKrylov.rank_local_snapshot_drift_is_refused_collectively_at_all_safe_boundaries:"
    "test_krylov_method_provider.*:"
    "test_vector_distribution_provider.*"
  )
  string(CONCAT POPS_KRYLOV_WORKSPACE_REENTRANCY_MPI_FILTER
    "test_krylov_workspace_reentrancy.distinct_workspaces_run_fresh_operator_and_preconditioner_sessions_concurrently:"
    "test_krylov_workspace_reentrancy.exclusive_operator_rejects_overlapping_materialized_invocations_uniformly:"
    "test_krylov_workspace_reentrancy.exclusive_operator_lease_is_shared_by_provider_copies_across_prepared_problems:"
    "test_krylov_workspace_reentrancy.public_problem_sessions_reject_overlap_with_independent_workspace_invocation:"
    "test_krylov_workspace_reentrancy.exclusive_source_lease_is_released_after_prepare_exception_across_problems:"
    "test_krylov_workspace_reentrancy.workspace_bind_rejects_affine_zero_response_drift_between_provider_sessions:"
    "test_krylov_workspace_reentrancy.workspace_bind_publishes_rank_local_zero_response_comparison_failure_uniformly:"
    "test_krylov_workspace_reentrancy.prepared_problem_and_workspace_execute_on_an_embedding_owned_congruent_communicator:"
    "test_krylov_workspace_reentrancy.noncongruent_split_communicator_is_rejected_explicitly:"
    "test_krylov_workspace_reentrancy.rank_local_problem_construction_failure_is_published_before_lane_unwind:"
    "test_krylov_workspace_reentrancy.workspace_rebind_reserves_mutation_during_blocking_operator_prepare:"
    "test_krylov_workspace_reentrancy.problem_prepare_reserves_mutation_during_blocking_resource_freeze"
  )
  foreach(_test IN LISTS POPS_CPP_MPI_VARIANT_TESTS)
    foreach(_np IN LISTS POPS_MPI_RANKS_${_test})
      if(_test STREQUAL "test_generic_krylov")
        pops_add_mpi_rank_test(
          ${_test} ${_np}
          GTEST_FILTER
            "${POPS_GENERIC_KRYLOV_MPI_FILTER}")
      elseif(_test STREQUAL "test_krylov_workspace_reentrancy")
        pops_add_mpi_rank_test(
          ${_test} ${_np}
          GTEST_FILTER
            "${POPS_KRYLOV_WORKSPACE_REENTRANCY_MPI_FILTER}")
      elseif(_test STREQUAL "test_geometric_mg")
        pops_add_mpi_rank_test(
          ${_test} ${_np}
          GTEST_FILTER "GeometricMgCollectiveContract.*")
      else()
        pops_add_mpi_rank_test(${_test} ${_np})
      endif()
    endforeach()
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
message(STATUS "pops tests: ${_pops_capped} heavy runtime targets capped (-O0 + pops_heavy_test_tu pool)")
