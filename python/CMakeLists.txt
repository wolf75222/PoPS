# Module Python pybind11 de la lib PoPS. L'extension compilee s'appelle `_pops`
# et est deposee dans un PAQUET `pops/` (avec __init__.py : le sucre lisible
# Spatial/Explicit/IMEX/System + pops.integrate). Avec PYTHONPATH = <build>/python,
# `import pops` charge le paquet, qui importe l'extension _pops.
#
#   cmake -B build-py -DPOPS_BUILD_PYTHON=ON && cmake --build build-py --target _pops
#   PYTHONPATH=build-py/python python3 -c "import pops; print(pops.System)"

# LTO du module _pops. OFF par defaut : le LTO (re)optimise au LINK et rallongerait fortement le build
# -- le backend -O3 est deja le goulot (cf. docs/BUILD_PROFILING.md) -- pour un gain runtime marginal sur ce
# code de dispatch type-erased (la frontiere de bloc passe par std::function, opaque au LTO). Il est par
# ailleurs numeriquement neutre (pas -ffast-math). OFF => NO_EXTRAS : pybind11 N'ESSAIE PAS le LTO, donc
# plus de test "Performing Test HAS_FLTO - Failed" parasite, et le build est rapide/previsible. ON =>
# THIN_LTO (clang) pour un binaire "release perf" PONCTUEL ; exige une toolchain dont le linker accepte
# -flto=thin (sur macOS, plutot la chaine clang conda complete que le melange AppleClang + linker conda).
# ADC-593: the per-route block-build SEAM translation units (system/{isothermal,compressible}, amr/block,
# amr/compiled) are no longer hand-written one .cpp per (transport, flux). They are GENERATED from ONE
# declarative manifest (bindings/seam_combinations.cmake) into ${CMAKE_CURRENT_BINARY_DIR}/generated_seams
# by configure_file per row -- byte-equivalent in symbols/semantics to the deleted files. Adding a Riemann
# is now one manifest row, never a new pybind file (the acceptance criterion). The build-memory mitigation
# is UNCHANGED: the generated TUs are still one leaf each, still serialized by the JOB_POOL below.
include(${CMAKE_CURRENT_SOURCE_DIR}/bindings/seam_combinations.cmake)
set(POPS_GENERATED_SEAMS_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated_seams)
set(POPS_GENERATED_SEAM_SOURCES "")
foreach(_seam_row ${POPS_SEAM_COMBINATIONS})
  pops_generate_seam("${_seam_row}" "${POPS_GENERATED_SEAMS_DIR}" POPS_GENERATED_SEAM_SOURCES)
endforeach()

# The HAND-WRITTEN binding TUs: the module entry + init split, the two heavy facades (system.cpp /
# amr_system.cpp), the unique polar visitor body, and the two thin riemann DISPATCHERS
# (amr_block_compressible.cpp / amr_compiled_compressible.cpp -- one per transport, genuinely distinct
# control flow, classified in docs/design/pybind-binding-audit.md). The per-(transport, flux) seam leaves
# come from POPS_GENERATED_SEAM_SOURCES.
set(POPS_MODULE_SOURCES
    bindings/core/bindings.cpp bindings/core/init/init_core.cpp bindings/core/init/init_identity.cpp
    bindings/core/init/init_system.cpp
    bindings/core/init/init_amr.cpp bindings/system/base/system.cpp bindings/system/base/system_polar.cpp
    # ADC-632: system.cpp split by responsibility into sibling TUs sharing system_impl.hpp (install /
    # fields / io / profiling / program). Smaller TUs also cut the per-TU compile peak (the JOB_POOL
    # memory serialization below still applies, now over lighter units).
    bindings/system/base/system_install.cpp bindings/system/base/system_fields.cpp
    bindings/system/base/system_io.cpp bindings/system/base/system_profiling.cpp
    bindings/system/base/system_program.cpp
    bindings/amr/amr_system.cpp bindings/amr/block/compressible/amr_block_compressible.cpp
    bindings/amr/compiled/compressible/amr_compiled_compressible.cpp
    ${POPS_GENERATED_SEAM_SOURCES})

option(POPS_PY_LTO "LTO (ThinLTO) sur le module _pops (OFF = build rapide, recommande)" OFF)
if(POPS_PY_LTO)
  pybind11_add_module(_pops THIN_LTO ${POPS_MODULE_SOURCES})
else()
  # NO_EXTRAS empeche pybind11 d'ACTIVER le LTO sur _pops. C'est NECESSAIRE, pas cosmetique : des que
  # le test HAS_FLTO passe (cas du pybind11 de l'env, p.ex. conda), pybind11_add_module activerait le
  # ThinLTO par defaut -> link/codegen au LINK -> build nettement plus long (le backend -O3 est deja le
  # goulot). On re-applique la visibilite cachee (seule "extra" qui compte ici) pour rester identique au
  # binaire produit jusqu'ici ; le chemin DSL "production" exporte quand meme ses symboles via POPS_EXPORT.
  pybind11_add_module(_pops NO_EXTRAS ${POPS_MODULE_SOURCES})
  set_target_properties(_pops PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
endif()
target_link_libraries(_pops PRIVATE pops::pops pops_dev_options ${CMAKE_DL_LIBS})  # dl : add_dynamic_block (dlopen) ; pops_dev_options : warnings/sanitizers opt-in (vide par defaut, cf. cmake/PopsDevTooling.cmake)
target_compile_features(_pops PRIVATE cxx_std_${POPS_CXX_STD})

# RAM CI : sous Kokkos, cc1plus a -O3 sur les TU geantes du module (system.cpp / bindings/amr/amr_system.cpp =
# facades completes, briques + AMR instanciees) pese plusieurs GB CHACUNE ; des que DEUX compilent
# en parallele (preset ci-kokkos-python, jobs: 2), le runner GitHub est tue par famine memoire
# ("runner received a shutdown signal", exit 143). Cas reel post vague ADC-63..71 (job "Kokkos
# (OpenMP)", step build du module) : bindings/core/bindings.cpp sort du ccache en ~20 s, puis bindings/system/base/system.cpp et
# bindings/amr/amr_system.cpp partent ENSEMBLE (jobs: 2, miss ccache) et le runner meurt ~9 min plus tard --
# la veille, la meme paire concurrente passait encore (runs verts a 15-23 min) : la vague a fait
# deborder le pic cumule. On serialise donc les compiles DU MODULE par un pool Ninja taille 1
# (declare dans le CMakeLists racine, bloc Kokkos), pattern pops_heavy_tu de tests/CMakeLists.txt.
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

# ADC-319 : dossier(s) d'include MPI du build, bake dans le module (expose en pops._pops.__mpi_include__).
# Le module herite deja -DPOPS_HAS_MPI via l'INTERFACE pops (cible MPI::MPI_CXX, racine), mais le DSL
# backend "production"/"aot" compile ses loaders HORS CMake et n'en herite rien : sans -DPOPS_HAS_MPI +
# l'include MPI, comm.hpp y retombe sur ses stubs serie (n_ranks()=1, my_rank()=0) et toute disposition
# distribuee construite DANS le loader (p.ex. AmrSystem(distribute_coarse=True)) se replique sur chaque
# rang (pas de strong-scaling). dsl.py (_native_mpi_flags) lit cet attribut pour recompiler le loader
# avec le MEME MPI que le module. Sans MPI (build serie) : attribut absent -> loader serie inchange.
if(TARGET MPI::MPI_CXX)
  get_target_property(_pops_mpi_inc MPI::MPI_CXX INTERFACE_INCLUDE_DIRECTORIES)
  if(_pops_mpi_inc)
    string(REPLACE ";" "|" _pops_mpi_inc "${_pops_mpi_inc}")
    target_compile_definitions(_pops PRIVATE POPS_MPI_INCLUDE="${_pops_mpi_inc}")
  endif()
endif()

# Signature de l'arbre d'en-tetes du coeur, BAKED dans la cle d'ABI du module (pops/runtime/abi_key.hpp
# via -DPOPS_HEADER_SIG). Le backend DSL "production" (dsl.compile_native) recalcule la MEME signature
# (pops.dsl.pops_header_signature) et la bake dans le loader .so : les cles concordent quand les en-tetes
# concordent, divergent sinon -> add_native_block leve une erreur explicite (pas d'UB silencieux).
# CALCUL (miroir bit-pour-bit de pops_header_signature) : sha256 du concatene TRIE, pour chaque .hpp/.h
# sous include/, de "<chemin relatif>\n<sha256 du contenu>\n".
# CONFIGURE_DEPENDS : CMake re-genere (donc recalcule la signature) si un en-tete est AJOUTE/SUPPRIME.
# Une EDITION de contenu d'en-tete n'invalide pas le glob ; un simple `cmake --build` peut alors garder
# une signature perimee -> compile_native rejetterait les loaders (erreur EXPLICITE, jamais d'UB). Le
# flux standard `cmake -S . -B build-py ... && cmake --build` reconfigure et recalcule : c'est le mode
# nominal. Apres une edition d'en-tete sans reconfigure, relancer la configuration (re-`cmake -S`).
set(_pops_include "${CMAKE_CURRENT_SOURCE_DIR}/../include")
file(GLOB_RECURSE _pops_headers CONFIGURE_DEPENDS RELATIVE "${_pops_include}"
     "${_pops_include}/*.hpp" "${_pops_include}/*.h")
list(SORT _pops_headers)
set(_pops_hdr_blob "")
foreach(_h ${_pops_headers})
  file(SHA256 "${_pops_include}/${_h}" _h_digest)
  string(APPEND _pops_hdr_blob "${_h}\n${_h_digest}\n")
endforeach()
string(SHA256 _pops_header_sig "${_pops_hdr_blob}")
target_compile_definitions(_pops PRIVATE POPS_HEADER_SIG="${_pops_header_sig}")
message(STATUS "pops Python module: signature en-tetes POPS_HEADER_SIG=${_pops_header_sig}")

# L'extension va dans le paquet pops/ ; on y copie aussi les sources Python du paquet.
set(POPS_PY_PKG ${CMAKE_CURRENT_BINARY_DIR}/pops)
set_target_properties(_pops PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${POPS_PY_PKG})

# Build PIP (scikit-build-core definit SKBUILD) : la wheel embarque le paquet pur python/pops/
# (wheel.packages du pyproject) ; ICI on n'installe QUE l'extension compilee a cote, dans
# site-packages/pops/. Resultat : `pip install .` -> `import pops` SANS PYTHONPATH (pattern
# pyAMReX/WarpX). Le flux dev presets+PYTHONPATH reste inchange (SKBUILD absent).
if(SKBUILD)
  install(TARGETS _pops LIBRARY DESTINATION pops)
  # External AOT packages compile against the exact headers whose signature is baked into _pops.
  # Keep them inside the wheel so installed-package compilation never reaches back into a checkout.
  install(DIRECTORY "${_pops_include}/pops" DESTINATION pops/include)
endif()
# Copy the pure-Python package (pops/ and every sub-package) into the build tree next to _pops.
# GLOB_RECURSE + CONFIGURE_DEPENDS so new modules / sub-packages are picked up automatically (a flat
# explicit list silently dropped a new module from the build tree -- "partially initialized module"
# import errors that the source tree did not show). configure_file creates the destination sub-dirs.
file(GLOB_RECURSE _pops_py_sources CONFIGURE_DEPENDS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} pops/*.py)
foreach(pyf ${_pops_py_sources})
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${pyf} ${CMAKE_CURRENT_BINARY_DIR}/${pyf} COPYONLY)
endforeach()


# Python tests are pytest suites under tests/python/**. CTest owns the environment
# for the built package; pytest owns collection, parametrization, fixtures, and markers.
set(_pops_py_path_sep ":")
set(_pops_py_test_pythonpath
    "${CMAKE_CURRENT_BINARY_DIR}${_pops_py_path_sep}${CMAKE_CURRENT_SOURCE_DIR}${_pops_py_path_sep}${CMAKE_SOURCE_DIR}")
if(DEFINED ENV{PYTHONPATH} AND NOT "$ENV{PYTHONPATH}" STREQUAL "")
  string(APPEND _pops_py_test_pythonpath "${_pops_py_path_sep}$ENV{PYTHONPATH}")
endif()
set(_pops_py_test_env
    "PYTHONPATH=${_pops_py_test_pythonpath}"
    "POPS_INCLUDE=${CMAKE_SOURCE_DIR}/include"
    "POPS_TEST_BUILD_DIR=${CMAKE_BINARY_DIR}"
    "POPS_TEST_SOURCE_DIR=${CMAKE_SOURCE_DIR}")

function(pops_add_pytest_suite name path labels)
  add_test(NAME ${name}
           COMMAND ${Python_EXECUTABLE} -m pytest ${CMAKE_SOURCE_DIR}/${path} -q)
  set_tests_properties(${name} PROPERTIES
    ENVIRONMENT "${_pops_py_test_env}"
    LABELS "${labels}"
    TIMEOUT 900)
endfunction()

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
pops_add_pytest_suite(pops_python_integration_io tests/python/integration/io "integration;python;io;hdf5;medium")
pops_add_pytest_suite(pops_python_integration_mpi tests/python/integration/mpi "integration;python;mpi;medium")
