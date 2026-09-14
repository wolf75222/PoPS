# Hints libomp pour AppleClang (qui ne trouve pas libomp seul). Sous Kokkos-only il reste UN
# consommateur : une install Kokkos batie avec Kokkos_ENABLE_OPENMP propage
# find_dependency(OpenMP REQUIRED) via son KokkosConfig.cmake -- sans ces hints,
# find_package(Kokkos) echouerait sur macOS. Ce N'EST PAS le backend OpenMP autonome (retire) :
# uniquement la plomberie de DECOUVERTE de libomp.
#
# Important macOS: ne jamais melanger deux libomp dans le meme binaire. Si un env conda est actif
# mais qu'aucun Kokkos conda n'est explicitement demande, find_package(Kokkos) peut trouver le Kokkos
# Homebrew; forcer alors OpenMP sur conda charge a la fois conda/libomp et Homebrew/libomp via
# libkokkoscore, puis les tests crashent dans libomp (__kmp_suspend_64 / mutex lock failed).
# Idempotent and inactive outside APPLE; called before and after Kokkos package discovery.
macro(pops_apple_libomp_hints)
  # Recompute these cache entries on every configure.  A build tree may be reused after switching
  # Kokkos prefixes; retaining the previous libomp would load two OpenMP runtimes in one process.
  if(APPLE)
    set(_pops_conda_libomp_prefix "")
    if(DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}/lib/libomp.dylib")
      set(_pops_conda_libomp_prefix "$ENV{CONDA_PREFIX}")
    endif()

    # Prefer the libomp colocated with an explicitly selected Kokkos prefix.  Do not require an
    # activated conda shell: CI, CTest and installed-package consumers commonly pass Kokkos_ROOT
    # directly.  Kokkos_DIR is the config directory (<prefix>/lib/cmake/Kokkos), so recover its
    # prefix before testing for the runtime.
    set(_pops_kokkos_libomp_prefix "")
    set(_pops_kokkos_config_dir "")
    if(DEFINED Kokkos_DIR)
      set(_pops_kokkos_config_dir "${Kokkos_DIR}")
    elseif(DEFINED ENV{Kokkos_DIR})
      set(_pops_kokkos_config_dir "$ENV{Kokkos_DIR}")
    endif()
    if(_pops_kokkos_config_dir)
      get_filename_component(_pops_kokkos_config_prefix
        "${_pops_kokkos_config_dir}/../../.." ABSOLUTE)
      if(EXISTS "${_pops_kokkos_config_prefix}/lib/libomp.dylib")
        set(_pops_kokkos_libomp_prefix "${_pops_kokkos_config_prefix}")
      endif()
    endif()
    if(NOT _pops_kokkos_libomp_prefix)
      if(DEFINED Kokkos_ROOT AND EXISTS "${Kokkos_ROOT}/lib/libomp.dylib")
        set(_pops_kokkos_libomp_prefix "${Kokkos_ROOT}")
      elseif(DEFINED ENV{Kokkos_ROOT} AND EXISTS "$ENV{Kokkos_ROOT}/lib/libomp.dylib")
        set(_pops_kokkos_libomp_prefix "$ENV{Kokkos_ROOT}")
      elseif(DEFINED POPS_KOKKOS_ROOT AND EXISTS "${POPS_KOKKOS_ROOT}/lib/libomp.dylib")
        set(_pops_kokkos_libomp_prefix "${POPS_KOKKOS_ROOT}")
      elseif(DEFINED ENV{POPS_KOKKOS_ROOT}
             AND EXISTS "$ENV{POPS_KOKKOS_ROOT}/lib/libomp.dylib")
        set(_pops_kokkos_libomp_prefix "$ENV{POPS_KOKKOS_ROOT}")
      endif()
    endif()

    set(LIBOMP_PREFIX "")
    if(_pops_kokkos_libomp_prefix)
      set(LIBOMP_PREFIX "${_pops_kokkos_libomp_prefix}")
    else()
      execute_process(COMMAND brew --prefix libomp
                      OUTPUT_VARIABLE LIBOMP_PREFIX
                      OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
      if((NOT LIBOMP_PREFIX OR NOT EXISTS "${LIBOMP_PREFIX}/lib/libomp.dylib")
         AND _pops_conda_libomp_prefix)
        set(LIBOMP_PREFIX "${_pops_conda_libomp_prefix}")
      endif()
    endif()
    if(LIBOMP_PREFIX AND EXISTS "${LIBOMP_PREFIX}/lib/libomp.dylib")
      # The first package search can discover Kokkos through CMAKE_PREFIX_PATH or the active
      # environment after these hints have run. FindOpenMP has already created its imported
      # target by then: updating cache variables alone does not change that target's link line.
      # Rebind the existing target to the runtime colocated with the resolved Kokkos package.
      if(TARGET OpenMP::OpenMP_CXX AND OpenMP_omp_LIBRARY
         AND NOT OpenMP_omp_LIBRARY STREQUAL "${LIBOMP_PREFIX}/lib/libomp.dylib")
        get_filename_component(_pops_previous_omp_libdir "${OpenMP_omp_LIBRARY}" DIRECTORY)
        get_filename_component(_pops_previous_omp_prefix "${_pops_previous_omp_libdir}" DIRECTORY)
        foreach(_pops_omp_property IN ITEMS
            INTERFACE_COMPILE_OPTIONS INTERFACE_INCLUDE_DIRECTORIES INTERFACE_LINK_LIBRARIES)
          get_target_property(_pops_omp_value OpenMP::OpenMP_CXX "${_pops_omp_property}")
          if(_pops_omp_value)
            string(REPLACE "${_pops_previous_omp_prefix}/" "${LIBOMP_PREFIX}/"
              _pops_omp_value "${_pops_omp_value}")
            set_property(TARGET OpenMP::OpenMP_CXX PROPERTY
              "${_pops_omp_property}" "${_pops_omp_value}")
          endif()
        endforeach()
      endif()
      set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include"
          CACHE STRING "" FORCE)
      set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING "" FORCE)
      set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib"
          CACHE FILEPATH "" FORCE)
    endif()
  endif()
endmacro()
