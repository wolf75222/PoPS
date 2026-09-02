# Module Python pybind11 de la lib PoPS. Chaque build produit une leaf `_pops`
# dans `pops/_native/dimN/`; le selecteur Python la charge sous le nom logique
# `pops._pops` apres resolution du domaine.
# Spatial/Explicit/IMEX/System + pops.integrate). Avec PYTHONPATH = <build>/python,
# `import pops` reste pur; `pops.compile(plan)` choisit ensuite la leaf correspondant au domaine.
#
#   cmake -B build-py -DPOPS_BUILD_PYTHON=ON && cmake --build build-py --target _pops
#   PYTHONPATH=build-py/python python3 -c "from pops.runtime.doctor import doctor; doctor()"

# LTO du module _pops. OFF par defaut : le LTO (re)optimise au LINK et rallongerait fortement le build
# -- le backend -O3 est deja le goulot (cf. docs/BUILD_PROFILING.md) -- pour un gain runtime marginal sur ce
# code de dispatch type-erased (la frontiere de bloc passe par std::function, opaque au LTO). Il est par
# ailleurs numeriquement neutre (pas -ffast-math). OFF => NO_EXTRAS : pybind11 N'ESSAIE PAS le LTO, donc
# plus de test "Performing Test HAS_FLTO - Failed" parasite, et le build est rapide/previsible. ON =>
# THIN_LTO (clang) pour un binaire "release perf" PONCTUEL ; exige une toolchain dont le linker accepte
# -flto=thin (sur macOS, plutot la chaine clang conda complete que le melange AppleClang + linker conda).
# The Python subtree owns only pybind module/init adapters.  System, AmrSystem, native builders and
# generated seam leaves are compiled once by the central targets in src/CMakeLists.txt.
set(POPS_MODULE_BINDING_SOURCES
    bindings/core/bindings.cpp bindings/core/init/init_core.cpp bindings/core/init/init_identity.cpp
    bindings/core/init/init_component_loader.cpp
    bindings/core/init/init_parallel_hdf5.cpp
    bindings/core/init/init_system.cpp
    bindings/core/init/init_amr.cpp)

option(POPS_PY_LTO "LTO (ThinLTO) sur le module _pops (OFF = build rapide, recommande)" OFF)
if(POPS_PY_LTO)
  pybind11_add_module(_pops THIN_LTO ${POPS_MODULE_BINDING_SOURCES})
  pops_runtime_enable_python_thin_lto()
else()
  # NO_EXTRAS empeche pybind11 d'ACTIVER le LTO sur _pops. C'est NECESSAIRE, pas cosmetique : des que
  # le test HAS_FLTO passe (cas du pybind11 de l'env, p.ex. conda), pybind11_add_module activerait le
  # ThinLTO par defaut -> link/codegen au LINK -> build nettement plus long (le backend -O3 est deja le
  # goulot). On re-applique la visibilite cachee (seule "extra" qui compte ici) pour rester identique au
  # binaire produit jusqu'ici ; le chemin DSL "production" exporte quand meme ses symboles via POPS_EXPORT.
  pybind11_add_module(_pops NO_EXTRAS ${POPS_MODULE_BINDING_SOURCES})
  set_target_properties(_pops PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
endif()
# pops_runtime_core_objects is intentionally direct: it is the one exported shared-exception ABI
# authority consumed by generated components.  The facade targets still carry the archive for C++
# executables, but its member is not extracted after this direct object has satisfied the symbols.
target_link_libraries(_pops PRIVATE pops_runtime_core_objects pops_runtime_system pops_runtime_amr pops_runtime_output pops::pops pops_dev_options ${CMAKE_DL_LIBS})  # dl : authenticated native component loading ; pops_dev_options : warnings/sanitizers opt-in (vide par defaut, cf. cmake/PopsDevTooling.cmake)
target_compile_features(_pops PRIVATE cxx_std_${POPS_CXX_STD})
# The pybind adapter is the final runtime module producer too. Its direct binding TUs use the same
# declaration-only exception ABI as the central runtime and select dllexport; generated native
# components replay only POPS_RUNTIME_SHARED_EXCEPTION_ABI and therefore select dllimport.
target_compile_definitions(_pops PRIVATE
  POPS_RUNTIME_SHARED_EXCEPTION_ABI
  POPS_EXPORT_BUILDING_MODULE)

# RAM CI : sous Kokkos, cc1plus a -O3 sur les TU geantes du runtime (system.cpp / amr_system.cpp =
# facades completes, briques + AMR instanciees) pese plusieurs GB CHACUNE ; des que DEUX compilent
# en parallele (preset ci-kokkos-python, jobs: 2), le runner GitHub est tue par famine memoire
# ("runner received a shutdown signal", exit 143). Cas reel post vague ADC-63..71 (job "Kokkos
# (OpenMP)", step build du module) : bindings/core/bindings.cpp sort du ccache en ~20 s, puis les
# cibles src/runtime System et AMR partent ENSEMBLE (jobs: 2, miss ccache) --
# la veille, la meme paire concurrente passait encore (runs verts a 15-23 min) : la vague a fait
# deborder le pic cumule. On serialise donc les compiles DU MODULE par le pool Ninja production
# pops_heavy_module_tu (declare dans le CMakeLists racine), distinct du pool -O0 des tests.
# -O3 est CONSERVE : c'est le module de production, seule la CONCURRENCE change (binaire
# identique). Scope POPS_HAS_KOKKOS : hors Kokkos les TU sont bien plus legeres, on garde le build
# local parallele. Sans effet hors generateur Ninja (JOB_POOL_COMPILE y est ignore).
if(POPS_HAS_KOKKOS)
  set_target_properties(_pops PROPERTIES JOB_POOL_COMPILE pops_heavy_module_tu)
endif()

# Norme C++ du LOADER baked dans le module : le DSL backend="production" doit compiler le modele
# natif avec EXACTEMENT cette norme, sinon __cplusplus diverge -> cle d'ABI differente -> rejet par
# add_native_block. POPS_CXX_STD vaut 20 sous Kokkos (CUDA 12.x sans -std=c++23) et 23 sinon (cf.
# CMakeLists.txt racine). On expose la valeur (entier 20/23) au binding, qui la publie en
# pops._pops.__cxx_std__ ; dsl.compile/compile_native en derivent le std natif au lieu de figer c++23.
target_compile_definitions(_pops PRIVATE POPS_CXX_STD=${POPS_CXX_STD})

# Version du projet (source unique : project(VERSION) racine), exposee en pops.__version__.
target_compile_definitions(_pops PRIVATE POPS_VERSION="${PROJECT_VERSION}")

# Chemin du COMPILATEUR du build, bake dans le module (expose en pops._pops.__cxx_compiler__). Le DSL
# backend="production" compile a l'execution un loader .so dont la cle d'ABI encode __VERSION__ du
# compilateur : le SEUL compilateur garanti compatible est CELUI-CI. Sans ce baking, dsl.py retombait
# sur `which c++` du PATH -- dans un env conda active, c'est souvent un autre compilateur (vieux gcc/
# clang) qui rejette -std=c++23 (bug reel : "error: invalid value 'c++23'") ou, pire, compile avec un
# __VERSION__ different -> rejet ABI. dsl.py prefere ce chemin s'il existe encore sur la machine.
target_compile_definitions(_pops PRIVATE POPS_CXX_COMPILER="${CMAKE_CXX_COMPILER}")

# Closed Kokkos development manifest selected by the host build.  Runtime codegen reauthenticates
# these files before replaying their include paths; it never guesses from sys.prefix/conda.
target_compile_definitions(_pops PRIVATE
  POPS_KOKKOS_ABI="${POPS_NATIVE_KOKKOS_ABI}"
  POPS_KOKKOS_INCLUDE="${POPS_NATIVE_KOKKOS_INCLUDE}"
  POPS_KOKKOS_HEADER_PATHS="${POPS_NATIVE_KOKKOS_HEADER_PATHS}"
  POPS_KOKKOS_HEADER_HASHES="${POPS_NATIVE_KOKKOS_HEADER_HASHES}")

# Contrat MPI::MPI_CXX exact du build, bake dans le module : includes, options/definitions de
# compilation, options/libraries de lien, chemins et hashes de mpi.h/libraries, plus l'empreinte ABI
# globale. Les composants natifs hors CMake rejouent puis reauthentifient ce manifeste integral.
# Le module herite deja -DPOPS_HAS_MPI via l'INTERFACE pops (cible MPI::MPI_CXX, racine), mais le DSL
# backend production compile ses loaders HORS CMake et n'en herite rien : sans -DPOPS_HAS_MPI +
# l'include MPI, comm.hpp y retombe sur ses stubs serie (n_ranks()=1, my_rank()=0) et toute disposition
# distribuee construite DANS le loader (p.ex. AmrSystem(distribute_coarse=True)) se replique sur chaque
# rang (pas de strong-scaling). Le helper codegen _native_mpi lit ces attributs et refuse un contrat
# MPI incomplet au lieu de produire un artefact faussement serie. Sans MPI : attributs vides.
if(POPS_NATIVE_MPI_INCLUDE)
  target_compile_definitions(_pops PRIVATE
    POPS_MPI_INCLUDE="${POPS_NATIVE_MPI_INCLUDE}"
    POPS_MPI_ABI="${POPS_NATIVE_MPI_ABI}"
    POPS_MPI_COMPILER="${POPS_NATIVE_MPI_COMPILER}"
    POPS_MPI_STANDARD="${POPS_NATIVE_MPI_STANDARD}"
    POPS_MPI_COMPILE_OPTIONS="${POPS_NATIVE_MPI_COMPILE_OPTIONS}"
    POPS_MPI_COMPILE_DEFINITIONS="${POPS_NATIVE_MPI_COMPILE_DEFINITIONS}"
    POPS_MPI_LINK_OPTIONS="${POPS_NATIVE_MPI_LINK_OPTIONS}"
    POPS_MPI_LINK_LIBRARIES="${POPS_NATIVE_MPI_LINK_LIBRARIES}"
    POPS_MPI_HEADER_PATHS="${POPS_NATIVE_MPI_HEADER_PATHS}"
    POPS_MPI_HEADER_HASHES="${POPS_NATIVE_MPI_HEADER_HASHES}"
    POPS_MPI_LIBRARY_PATHS="${POPS_NATIVE_MPI_LIBRARY_PATHS}"
    POPS_MPI_LIBRARY_HASHES="${POPS_NATIVE_MPI_LIBRARY_HASHES}")
endif()

# Signature du contrat exact des en-tetes installes, BAKEE dans la cle d'ABI du module (pops/runtime/abi_key.hpp
# via -DPOPS_HEADER_SIG). Le backend DSL "production" (dsl.compile_native) recalcule la MEME signature
# (pops.dsl.pops_header_signature) et la bake dans le loader .so : les cles concordent quand les en-tetes
# concordent, divergent sinon -> add_native_block leve une erreur explicite (pas d'UB silencieux).
# CALCUL (miroir bit-pour-bit de pops_header_signature) : sha256 du concatene TRIE, pour chaque ligne
# installee, de "<categorie> <chemin relatif>\n<sha256 du contenu>\n". Les en-tetes test-only
# n'entrent ni dans l'installation ni dans l'identite ABI publiee.
target_compile_definitions(_pops PRIVATE POPS_HEADER_SIG="${POPS_NATIVE_HEADER_SIGNATURE}")
message(STATUS "pops Python module: signature en-tetes POPS_HEADER_SIG=${POPS_NATIVE_HEADER_SIGNATURE}")

# L'extension va dans le paquet pops/ ; on y copie aussi les sources Python du paquet.
set(POPS_PY_PKG ${CMAKE_CURRENT_BINARY_DIR}/pops)
set(POPS_PY_NATIVE_PKG ${POPS_PY_PKG}/_native/dim${POPS_NATIVE_DIM})
set(POPS_PY_NATIVE_MANIFEST ${POPS_PY_PKG}/_native/variants.json)
set_target_properties(_pops PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${POPS_PY_NATIVE_PKG})

# The manifest authenticates the linked leaf before it is copied into a wheel.  Keep CMake's
# install image byte-identical to that leaf: otherwise CMake rewrites automatically discovered
# runtime paths at install time (``install_name_tool`` on Darwin) and invalidates both the digest
# and the linker ad-hoc signature.  Let CMake carry its normal dependency search paths into the
# install RPATH rather than hard-coding a host-specific prefix.
set_target_properties(_pops PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE)

# Publish the leaf only after interrogating the exact linked image.  The helper authenticates its
# compile-time dimension, ABI, backend and final bytes, then atomically merges the row into the
# build-tree inventory.  A distribution assembler may invoke the same helper for three isolated
# sub-builds; the runtime never guesses filenames or scans for a fallback extension.
add_custom_command(TARGET _pops POST_BUILD
  COMMAND "${Python_EXECUTABLE}"
          "${CMAKE_SOURCE_DIR}/scripts/write_native_variant_manifest.py"
          --extension "$<TARGET_FILE:_pops>"
          --manifest "${POPS_PY_NATIVE_MANIFEST}"
          --dimension "${POPS_NATIVE_DIM}"
          --version "${PROJECT_VERSION}"
  COMMENT "Authenticating PoPS native Dim=${POPS_NATIVE_DIM} variant manifest"
  VERBATIM)

# Build PIP (scikit-build-core definit SKBUILD) : la wheel embarque le paquet pur python/pops/
# (wheel.packages du pyproject) ; ICI on n'installe QUE l'extension compilee a cote, dans
# site-packages/pops/. Resultat : `pip install .` -> `import pops` SANS PYTHONPATH (pattern
# pyAMReX/WarpX). Le flux dev presets+PYTHONPATH reste inchange (SKBUILD absent).
if(SKBUILD)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/scripts/check_packaging_manifest.py"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    RESULT_VARIABLE _pops_packaging_preflight_result
    OUTPUT_VARIABLE _pops_packaging_preflight_stdout
    ERROR_VARIABLE _pops_packaging_preflight_stderr)
  if(NOT _pops_packaging_preflight_result EQUAL 0)
    message(FATAL_ERROR
      "PoPS packaging preflight failed before wheel assembly:\n"
      "${_pops_packaging_preflight_stdout}${_pops_packaging_preflight_stderr}")
  endif()
  install(TARGETS _pops LIBRARY DESTINATION pops/_native/dim${POPS_NATIVE_DIM})
  install(FILES "${POPS_PY_NATIVE_MANIFEST}" DESTINATION pops/_native)
  # External AOT packages compile against the exact headers whose signature is baked into _pops.
  # Keep them inside the wheel so installed-package compilation never reaches back into a checkout.
  pops_install_headers("pops/include")
endif()
# Stage the pure-Python package (pops/ and every sub-package) in the build tree next to _pops.
# GLOB_RECURSE + CONFIGURE_DEPENDS so new modules / sub-packages are picked up automatically (a flat
# explicit list silently dropped a new module from the build tree -- "partially initialized module"
# import errors that the source tree did not show). Type stubs and the PEP 561 marker are package
# sources too: cold builds and CI's cached-module fast path must expose the same tree.
# Build-tree imports need the complete pure-Python package beside ``_pops``.  Do not copy every file
# through ``configure_file(COPYONLY)``: that command reads both source and destination to compare
# contents on every configure, which can block for minutes when the checkout is backed by a macOS File
# Provider.  Per-file symbolic links preserve live source edits without content I/O.  Windows hosts
# that cannot create symlinks use CMake's explicit COPY_ON_ERROR fallback.
file(GLOB_RECURSE _pops_py_sources CONFIGURE_DEPENDS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
     pops/*.py pops/*.pyi)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/pops/py.typed")
  list(APPEND _pops_py_sources pops/py.typed)
endif()
foreach(pyf ${_pops_py_sources})
  get_filename_component(_pops_py_parent "${pyf}" DIRECTORY)
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${_pops_py_parent}")
  file(REMOVE "${CMAKE_CURRENT_BINARY_DIR}/${pyf}")
  file(CREATE_LINK
       "${CMAKE_CURRENT_SOURCE_DIR}/${pyf}"
       "${CMAKE_CURRENT_BINARY_DIR}/${pyf}"
       SYMBOLIC COPY_ON_ERROR RESULT _pops_py_link_result)
  if(NOT _pops_py_link_result STREQUAL "0")
    message(FATAL_ERROR "cannot stage Python package file ${pyf}: ${_pops_py_link_result}")
  endif()
endforeach()


# Python tests are pytest suites under tests/python/**.  The authoring/runtime
# surface is native Dim2 only, so Dim1 and Dim3 intentionally register no
# pops_python_* CTests.  CTest owns the environment; pytest owns collection,
# parametrization, fixtures, and markers.
if(POPS_NATIVE_DIM STREQUAL "2")
  set(_pops_py_path_sep ":")
  set(_pops_py_test_pythonpath
      "${CMAKE_CURRENT_BINARY_DIR}${_pops_py_path_sep}${CMAKE_CURRENT_SOURCE_DIR}${_pops_py_path_sep}${CMAKE_SOURCE_DIR}")
  if(DEFINED ENV{PYTHONPATH} AND NOT "$ENV{PYTHONPATH}" STREQUAL "")
    string(APPEND _pops_py_test_pythonpath "${_pops_py_path_sep}$ENV{PYTHONPATH}")
  endif()
  set(_pops_py_test_env
      "PYTHONPATH=${_pops_py_test_pythonpath}"
      "POPS_NATIVE_DIM=${POPS_NATIVE_DIM}"
      "POPS_NATIVE_VARIANTS_ROOT=${CMAKE_CURRENT_BINARY_DIR}/pops/_native"
      "POPS_INCLUDE=${CMAKE_SOURCE_DIR}/include"
      "POPS_TEST_BUILD_DIR=${CMAKE_BINARY_DIR}"
      "POPS_TEST_SOURCE_DIR=${CMAKE_SOURCE_DIR}")

  function(pops_add_pytest_suite name path labels)
    add_test(NAME ${name}
             COMMAND ${Python_EXECUTABLE} -m pytest ${CMAKE_SOURCE_DIR}/${path} -q ${ARGN})
    set_tests_properties(${name} PROPERTIES
      ENVIRONMENT "${_pops_py_test_env}"
      LABELS "${labels}"
      TIMEOUT 900)
  endfunction()

  set(_pops_py_io_mpi_entrypoints
      tests/python/integration/io/test_hdf5_parallel.py)
  set(_pops_py_mpi_entrypoints
      tests/python/integration/mpi/test_amr_clean_route_program_mpi.py
      tests/python/integration/mpi/test_amr_history_mpi.py
      tests/python/integration/mpi/test_amr_nonlinear_collective_mpi.py
      tests/python/integration/mpi/test_amr_regrid_on_restart_mpi.py
      tests/python/integration/mpi/test_async_balance_cadence_mpi.py
      tests/python/integration/mpi/test_external_amr_field_solver_mpi.py
      tests/python/integration/mpi/test_scientific_output_mpi.py
      tests/python/integration/mpi/test_uniform_history_checkpoint_mpi.py)
  set(_pops_py_rank_change_orchestrator
      tests/python/integration/mpi/test_amr_rank_change_restart.py)

  set(_pops_py_io_ignore_args)
  foreach(_pops_py_mpi_entrypoint IN LISTS _pops_py_io_mpi_entrypoints)
    list(APPEND _pops_py_io_ignore_args
         "--ignore=${CMAKE_SOURCE_DIR}/${_pops_py_mpi_entrypoint}")
  endforeach()
  set(_pops_py_mpi_ignore_args)
  foreach(_pops_py_mpi_entrypoint IN LISTS _pops_py_mpi_entrypoints)
    list(APPEND _pops_py_mpi_ignore_args
         "--ignore=${CMAKE_SOURCE_DIR}/${_pops_py_mpi_entrypoint}")
  endforeach()
  list(APPEND _pops_py_mpi_ignore_args
       "--ignore=${CMAKE_SOURCE_DIR}/${_pops_py_rank_change_orchestrator}")

  pops_add_pytest_suite(pops_python_unit_descriptors tests/python/unit/descriptors "unit;python;descriptors;fast")
  pops_add_pytest_suite(pops_python_unit_codegen tests/python/unit/codegen "unit;python;codegen;fast")
  pops_add_pytest_suite(pops_python_unit_time tests/python/unit/time "unit;python;time;fast")
  pops_add_pytest_suite(pops_python_unit_mesh tests/python/unit/mesh "unit;python;mesh;fast")
  pops_add_pytest_suite(pops_python_unit_solvers tests/python/unit/solvers "unit;python;solvers;fast")
  pops_add_pytest_suite(pops_python_unit_physics tests/python/unit/physics "unit;python;physics;fast")
  pops_add_pytest_suite(pops_python_unit_runtime tests/python/unit/runtime "unit;python;runtime;fast")

  pops_add_pytest_suite(pops_python_integration_bindings tests/python/integration/bindings "integration;python;bindings;medium")
  pops_add_pytest_suite(pops_python_integration_runtime tests/python/integration/runtime "integration;python;runtime;medium")
  pops_add_pytest_suite(pops_python_integration_amr tests/python/integration/amr "integration;python;amr;medium")
  pops_add_pytest_suite(pops_python_integration_native_loader tests/python/integration/native_loader "integration;python;native_loader;compiler;medium")
  pops_add_pytest_suite(pops_python_integration_io tests/python/integration/io "integration;python;io;hdf5;medium" ${_pops_py_io_ignore_args})
  pops_add_pytest_suite(pops_python_integration_mpi tests/python/integration/mpi "integration;python;mpi;medium" ${_pops_py_mpi_ignore_args})

  if(POPS_USE_MPI)
    set(_pops_py_mpi_bootstrap
        "import os, runpy, sys; from pops._native_selector import select_native_dimension; select_native_dimension(int(os.environ[\"POPS_NATIVE_DIM\"])); runpy.run_path(sys.argv[1], run_name=\"__main__\")")
    function(pops_add_mpi_pytest_entrypoint path)
      get_filename_component(_pops_py_mpi_stem "${path}" NAME_WE)
      add_test(
        NAME "pops_python_mpi_${_pops_py_mpi_stem}_np2"
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                ${Python_EXECUTABLE} ${MPIEXEC_POSTFLAGS} -u -c "${_pops_py_mpi_bootstrap}"
                "${CMAKE_SOURCE_DIR}/${path}")
      set_tests_properties("pops_python_mpi_${_pops_py_mpi_stem}_np2" PROPERTIES
        ENVIRONMENT "${_pops_py_test_env};POPS_REQUIRE_MPI_TESTS=1"
        LABELS "integration;python;mpi;np2"
        PROCESSORS 2
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        TIMEOUT 900)
    endfunction()

    foreach(_pops_py_mpi_entrypoint IN LISTS _pops_py_io_mpi_entrypoints _pops_py_mpi_entrypoints)
      pops_add_mpi_pytest_entrypoint("${_pops_py_mpi_entrypoint}")
    endforeach()

    add_test(
      NAME pops_python_mpi_test_amr_rank_change_restart_orchestrator
      COMMAND ${Python_EXECUTABLE} -m pytest "${CMAKE_SOURCE_DIR}/${_pops_py_rank_change_orchestrator}" -q)
    set_tests_properties(pops_python_mpi_test_amr_rank_change_restart_orchestrator PROPERTIES
      ENVIRONMENT "${_pops_py_test_env};POPS_REQUIRE_MPI_TESTS=1"
      LABELS "integration;python;mpi;orchestrator"
      PROCESSORS 2
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
      TIMEOUT 900)
  endif()
endif()
