# 3.21 : aligne sur CMakePresets.json (schema v3 = CMake >= 3.21) et debloque
# PROJECT_IS_TOP_LEVEL (garde consommateur ci-dessous). L'env conda sert >= 3.27.
cmake_minimum_required(VERSION 3.21)

project(PoPS
  VERSION 1.0.0  # single source of the version: propagated to the Python module (__version__), the
                 # pip wheel, Doxygen and Sphinx (see docs/VERSIONING.md). Never duplicate elsewhere.
  DESCRIPTION "Solveur advection-diffusion-couplage hyperbolique-elliptique sur AMR, cible cluster"
  LANGUAGES CXX
)

# Shared, exhaustive API/ABI/SDK/test-only header contract. Both the source package install and the
# SKBUILD wheel consume it; every installed category enters the native header signature.
include(cmake/PopsPublicHeaders.cmake)

# Backends par VARIABLE D'ENVIRONNEMENT (pattern WarpX/pyAMReX) : permet `POPS_USE_KOKKOS=ON
# Kokkos_ROOT=$CONDA_PREFIX pip install .` sans toucher aux flags CMake (pip ne propage pas de
# -D, mais propage l'environnement ; <Pkg>_ROOT est lu nativement par find_package). La graine
# d'env ne joue QUE si l'option n'est pas posee en cache/-D (CMP0077 NEW : la variable normale
# prime sur option()) -> -DPOPS_USE_X=... reste prioritaire, rien d'hardcode.
foreach(_pops_opt POPS_BUILD_TESTS POPS_BUILD_PYTHON POPS_USE_KOKKOS POPS_USE_MPI POPS_USE_HDF5)
  if(DEFINED ENV{${_pops_opt}} AND NOT DEFINED ${_pops_opt} AND NOT DEFINED CACHE{${_pops_opt}})
    set(${_pops_opt} "$ENV{${_pops_opt}}")
    message(STATUS "pops: option ${_pops_opt}=$ENV{${_pops_opt}} (heritee de l'environnement)")
  endif()
endforeach()

# Defaut TOP-LEVEL SEULEMENT : un consommateur qui tire PoPS via FetchContent/add_subdirectory
# (cf. README "Utiliser le coeur") ne doit PAS compiler les ~140 executables de tests/ ni voir
# enable_testing() pollue. Surchargenable (-DPOPS_BUILD_TESTS=ON) dans les deux sens.
option(POPS_BUILD_TESTS    "Build tests"    ${PROJECT_IS_TOP_LEVEL})
# Module Python `pops` (pybind11) : c'est LES BINDINGS DE LA LIB, donc ils vivent ici.
# Le coeur public reste header-only ; le module lie les objets runtime centraux (src/runtime)
# + les bindings. adc_cases ne contient QUE du code Python qui importe ce module.
option(POPS_BUILD_PYTHON   "Build the pybind11 Python module `pops`" OFF)
# Backend de dispatch on-node : KOKKOS EST LE SEUL backend on-node, et il est OBLIGATOIRE.
# Un meme code source Kokkos couvre les trois cibles on-node via l'espace d'execution choisi
# A L'INSTALLATION DE KOKKOS (pas par un drapeau pops) :
#   - sequentiel       : Kokkos Serial  (Kokkos_ENABLE_SERIAL=ON)
#   - CPU multi-thread : Kokkos OpenMP  (Kokkos_ENABLE_OPENMP=ON)
#   - GPU              : Kokkos Cuda/HIP (Kokkos_ENABLE_CUDA=ON / Kokkos_ENABLE_HIP=ON)
# Il n'y a PLUS de backend OpenMP autonome (POPS_USE_OPENMP), plus de `#pragma omp`, plus de
# boucle hote manuelle comme chemin de production : tout le parallelisme on-node passe par le
# seam for_each_cell -> Kokkos (cf. include/pops/mesh/execution/for_each.hpp).
#
# Kokkos n'a PAS besoin d'etre pre-installe : on utilise une install existante si elle est trouvee
# (-DKokkos_ROOT=..., p.ex. cache CI / module ROMEO), SINON on le RECUPERE + construit a la volee
# (FetchContent). Un simple `cmake -B build` marche donc sur une machine sans Kokkos.
option(POPS_USE_KOKKOS     "Enable Kokkos backend (the only on-node backend, mandatory)" ON)
option(POPS_USE_MPI        "Enable the MPI backend (distributed comm seam)" OFF)
option(POPS_USE_HDF5       "Enable the native collective HDF5 writer (requires MPI)" OFF)

# HDF5 is not a second, serial output backend.  This option owns the native collective writer,
# whose process contract is MPI_COMM_WORLD and whose implementation calls the parallel HDF5 C API.
# Reject the impossible half-capability before discovering Kokkos or enabling the C language: a
# serial build must remain a genuinely CXX-only package, not a green package exporting an unused
# HDF5 dependency or a runtime stub.
if(POPS_USE_HDF5 AND NOT POPS_USE_MPI)
  message(FATAL_ERROR
    "POPS_USE_HDF5=ON enables the native collective HDF5 writer and therefore requires "
    "POPS_USE_MPI=ON plus a parallel HDF5 C library. For serial output, keep POPS_USE_HDF5=OFF.")
endif()
# Version de Kokkos recuperee par FetchContent quand aucune install n'est trouvee (release tarball +
# verification SHA256). Surchargeables ensemble (changer la version impose de changer le SHA256, sinon
# la verification d'integrite echoue). Le SHA256 est publie sur la page des releases Kokkos.
set(POPS_KOKKOS_FETCH_VERSION "4.4.01" CACHE STRING "Kokkos release version fetched when no install is found")
set(POPS_KOKKOS_FETCH_SHA256
    "3413f0cb39912128d91424ebd92e8832009e7eeaf6fa8da58e99b0d37860d972"
    CACHE STRING "SHA256 of kokkos-<POPS_KOKKOS_FETCH_VERSION>.tar.gz (empty = skip integrity check)")

# Kokkos-only : configurer sans Kokkos est une ERREUR FATALE (le seam for_each_cell ne compile
# pas sans POPS_HAS_KOKKOS). Le serie passe par Kokkos Serial, pas par une boucle C++ manuelle.
# (Pas besoin d'installer Kokkos a la main : il est recupere via FetchContent si introuvable.)
if(NOT POPS_USE_KOKKOS)
  message(FATAL_ERROR
    "PoPS is Kokkos-only: configure with -DPOPS_USE_KOKKOS=ON. Kokkos est trouve s'il est installe "
    "(-DKokkos_ROOT=...) ou recupere + construit automatiquement (FetchContent). Le backend OpenMP "
    "autonome a ete retire.")
endif()

# Norme C++ : 20. Kokkos etant obligatoire et compile sous nvcc pour la cible Cuda (CUDA 12.x
# ne propose pas -std=c++23), tout le coeur compile en C++20 (le seul usage >20 etait cote hote
# et ne sert plus depuis le retrait du chemin non-Kokkos).
set(POPS_CXX_STD 20)
set(CMAKE_CXX_STANDARD ${POPS_CXX_STD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# macOS : si le compilateur retenu est un clang LLVM (Homebrew) et non AppleClang, on INFORME --
# sans forcer quoi que ce soit (le choix reste a l'utilisateur via CC/CXX). Cas reel mesure : un
# `.zshrc` qui met /opt/homebrew/opt/llvm/bin en tete de PATH fait compiler les grosses TU du
# module Python 4-5x plus lentement qu'AppleClang (system.cpp : >1h au lieu de ~8 min), sans
# aucun message expliquant pourquoi. AppleClang s'identifie "AppleClang" ; le LLVM brut "Clang".
if(APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  message(STATUS "pops: compilateur = clang LLVM (${CMAKE_CXX_COMPILER}). Sur macOS, AppleClang "
                 "compile les grosses TU du module nettement plus vite ; pour le forcer : "
                 "CXX=/usr/bin/clang++ cmake ... (ou pip install .). Cause frequente : "
                 "/opt/homebrew/opt/llvm/bin en tete de PATH.")
endif()

# ccache : cache de compilation. ZERO impact sur le code genere (meme objets) ; accelere
# fortement les RE-builds (ex. quand seul un .cpp change). Auto-detecte s'il est installe
# (l'env conda le fournit, cf. environment.yml). Inactif si absent, surchargeable
# (-DPOPS_USE_CCACHE=OFF ou -DCMAKE_CXX_COMPILER_LAUNCHER=...). On l'EVITE sous nvcc_wrapper
# (chemin GPU Kokkos/CUDA) ou ccache peut mal interagir avec le wrapper.
option(POPS_USE_CCACHE "Utiliser ccache comme launcher de compilation s'il est present" ON)
if(POPS_USE_CCACHE AND NOT CMAKE_CXX_COMPILER_LAUNCHER AND NOT CMAKE_CXX_COMPILER MATCHES "nvcc")
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    message(STATUS "ccache actif : ${CCACHE_PROGRAM} (cache de compilation ; -DPOPS_USE_CCACHE=OFF pour desactiver)")
  endif()
endif()

# Couche physique + maillage : bibliotheque d'interface (header-only) pour
# l'instant. L'operateur spatial, l'elliptique et l'integrateur viendront en
# couches separees ; le backend parallele (AMR/MPI/Kokkos) reste confine au
# seam de dispatch et ne touche jamais a la couche physique.
add_library(pops INTERFACE)
add_library(pops::pops ALIAS pops)
target_include_directories(pops INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>  # consommation via find_package(pops) installe (GNUInstallDirs vient plus tard)
)
target_compile_features(pops INTERFACE cxx_std_${POPS_CXX_STD})

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
# Macro idempotente, no-op hors APPLE ; invoquee AVANT la resolution de Kokkos ci-dessous.
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
      set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include"
          CACHE STRING "" FORCE)
      set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING "" FORCE)
      set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib"
          CACHE FILEPATH "" FORCE)
    endif()
  endif()
endmacro()

# Resolution de Kokkos (cf. wiki Kokkos "Integrating Kokkos into your CMake project", methode 4 :
# supporter a la fois un Kokkos externe ET embarque). find_package D'ABORD (CONFIG, version mini 4.2,
# QUIET) pour reutiliser une install existante (cache CI, module ROMEO, Homebrew, -DKokkos_ROOT=...).
# A DEFAUT, FetchContent recupere le tarball de release (verifie par SHA256) et le construit dans
# l'arbre de build -> aucune install manuelle requise. La cible on-node se choisit par les options
# Kokkos (Kokkos_ENABLE_SERIAL / _OPENMP / _CUDA / _HIP) -- defaut Serial pour un CPU mono-thread
# universel. Quel que soit le chemin, la cible importee `Kokkos::kokkos` est creee et attachee a pops.
# (On garde la forme explicite if/else plutot que FetchContent FIND_PACKAGE_ARGS, qui exige CMake>=3.24
# alors que ce projet cible 3.20. CMAKE_DISABLE_FIND_PACKAGE_Kokkos=ON force le chemin FetchContent.)
# La version mini 4.2 accepte les versions plus recentes, y compris 5.x (Homebrew 5.1.1, p.ex.).
#
# ADC-168 NO-GO : Kokkos CUDA (GPU) n'est PAS supporte en natif Windows. Kokkos 4.x n'a aucune
# toolchain CUDA-sous-Windows qui marche -- le generateur Visual Studio avec Kokkos_ENABLE_CUDA
# suppose CXX=nvcc (modele Linux), et nvcc-comme-CXX sous Ninja recoit des flags MSVC non traduits
# (/MDd, /RTC1, /showIncludes ...) faute d'un nvcc_wrapper Windows. Le chemin GPU supporte est WSL2
# (ADC-96) ; le natif Windows ne construit que le CPU Serial/OpenMP. On refuse TOT (avant toute
# resolution/recuperation de Kokkos) une configuration Windows native demandant CUDA.
if(WIN32 AND Kokkos_ENABLE_CUDA)
  message(FATAL_ERROR
    "Kokkos CUDA native Windows is unsupported in PoPS. "
    "Use WSL2 for CUDA/Kokkos GPU. "
    "Native Windows supports CPU Serial/OpenMP only. "
    "Refs ADC-168 / ADC-96.")
endif()
pops_apple_libomp_hints()
find_package(Kokkos 4.2 QUIET CONFIG)
if(Kokkos_FOUND)
  message(STATUS "Kokkos trouve (install ${Kokkos_VERSION}) : for_each_cell backend = Kokkos (${Kokkos_DEVICES})")
else()
  message(STATUS "Kokkos introuvable -> FetchContent (release ${POPS_KOKKOS_FETCH_VERSION}). "
                 "Cible on-node via -DKokkos_ENABLE_SERIAL/_OPENMP/_CUDA (defaut : Serial).")
  include(FetchContent)
  # Defaut hors-config : au moins Serial, pour un CPU mono-thread fonctionnel sur toute machine.
  # (Si l'utilisateur a deja passe une option Kokkos_ENABLE_*, on la respecte.)
  if(NOT DEFINED Kokkos_ENABLE_SERIAL AND NOT DEFINED Kokkos_ENABLE_OPENMP
     AND NOT DEFINED Kokkos_ENABLE_CUDA AND NOT DEFINED Kokkos_ENABLE_HIP)
    set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "Kokkos Serial backend (defaut pops FetchContent)" FORCE)
  endif()
  # Le Kokkos recupere se compile en bibliotheque statique ; pour le lier dans l'extension
  # PARTAGEE _pops (pybind11_add_module) sous Linux il faut du code position-independant, sinon le
  # link echoue ("relocation R_X86_64_PC32 ... recompile with -fPIC"). On le pose AVANT
  # FetchContent_MakeAvailable pour que les cibles Kokkos en heritent ; sans ca le repli
  # FetchContent ne marchait que parce que build_docs.sh / wheels.yml passaient le flag a la main
  # (ADC-387).
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  set(_pops_kokkos_url
      "https://github.com/kokkos/kokkos/releases/download/${POPS_KOKKOS_FETCH_VERSION}/kokkos-${POPS_KOKKOS_FETCH_VERSION}.tar.gz")
  if(POPS_KOKKOS_FETCH_SHA256)
    FetchContent_Declare(Kokkos URL "${_pops_kokkos_url}" URL_HASH SHA256=${POPS_KOKKOS_FETCH_SHA256})
  else()
    FetchContent_Declare(Kokkos URL "${_pops_kokkos_url}")  # pas de SHA256 : verification d'integrite desactivee
  endif()
  FetchContent_MakeAvailable(Kokkos)
  message(STATUS "Kokkos recupere (FetchContent ${POPS_KOKKOS_FETCH_VERSION}) : devices = ${Kokkos_DEVICES}")
endif()
target_link_libraries(pops INTERFACE Kokkos::kokkos)
target_compile_definitions(pops INTERFACE POPS_HAS_KOKKOS)
set(POPS_HAS_KOKKOS ON)
# Pool Ninja taille 1 des TU de production du module Python (_pops) : serialisation anti-OOM de
# leurs compiles -O3 sous Kokkos, cf. python/CMakeLists.txt. Declare ICI et UNE SEULE fois : un nom
# de pool duplique dans la propriete GLOBALE JOB_POOLS rend build.ninja invalide
# ("ninja: error: duplicate pool"). Les consommateurs ne posent que JOB_POOL_COMPILE.
#
# La profondeur production reste independante du pool -O0 des tests/runtime declare par src/ quand
# POPS_BUILD_TESTS=ON. Elle ne doit etre relevee qu'apres mesure de la RSS par TU sur le runner vise.
# Les builds Python locaux et les wheels peuvent fournir explicitement une valeur mesuree plus haute.
set(POPS_HEAVY_MODULE_TU_POOL 1 CACHE STRING
    "Profondeur du pool Ninja des TU lourdes de production _pops (-O3) ; defaut 1 (anti-OOM)")
set_property(GLOBAL APPEND PROPERTY
  JOB_POOLS pops_heavy_module_tu=${POPS_HEAVY_MODULE_TU_POOL})

# Pool distinct pour les TU runtime/tests compilees a -O0. Il n'est declare ni consomme lorsque les
# tests sont desactives, et ne peut donc pas relever implicitement la concurrence du module -O3.
set(POPS_HEAVY_TEST_TU_POOL 1 CACHE STRING
    "Profondeur du pool Ninja des TU lourdes runtime/tests (-O0) ; defaut 1 (anti-OOM)")

# MPI : backend distribue, confine au seam comm (parallel/comm.hpp). Optionnel :
# par defaut OFF, le build et les tests restent serie (comm.hpp = rang unique).
# Quand ON, comm.hpp passe par MPI_Comm_rank/size + collectives, et le test
# test_mpi_smoke tourne via mpirun. POPS_HAS_MPI est propage a tout ce qui lie pops.
set(POPS_HAS_MPI OFF)
if(POPS_USE_MPI)
  # The core consumes only the C++ MPI target.  Do not make a C compiler or MPI_C an incidental
  # requirement of ordinary C++/FetchContent consumers.
  find_package(MPI REQUIRED COMPONENTS CXX)
  target_link_libraries(pops INTERFACE MPI::MPI_CXX)
  target_compile_definitions(pops INTERFACE POPS_HAS_MPI)
  set(POPS_HAS_MPI ON)
  message(STATUS "MPI found (${MPIEXEC_EXECUTABLE}) : enabling distributed comm seam")
endif()

# Sortie HDF5 native. La route collective est une implementation C++/HDF5 C qui utilise
# MPI_COMM_WORLD directement : h5py reste uniquement le lecteur/ecrivain serie Python. Aucune
# liaison MPI Python n'est acceptee comme backend de calcul ou de publication.
set(POPS_HAS_HDF5 OFF)
set(POPS_HAS_PARALLEL_HDF5 OFF)
if(POPS_USE_HDF5)
  # The native HDF5 writer consumes the C API.  Enable C only for this opt-in route so the
  # default and MPI-only header packages remain consumable by a C++-only toolchain.
  enable_language(C)
  # MODULE (wrappers h5cc/h5pcc + variables) plutot que CONFIG : le hdf5-config.cmake
  # de spack exige une cible importee MPI::MPI_C ; le projet active donc explicitement le langage C.
  # Some parallel-HDF5 package configurations reference MPI::MPI_C from their imported targets.
  # Materialize that component only on the collective HDF5 route; MPI without HDF5 stays CXX-only.
  find_package(MPI REQUIRED COMPONENTS C CXX)
  set(HDF5_PREFER_PARALLEL TRUE)
  find_package(HDF5 MODULE REQUIRED COMPONENTS C)
  # FindHDF5's imported target carries platform-specific compile definitions and transitive
  # dependencies (notably H5_BUILT_AS_DYNAMIC_LIB on Windows); raw library/include variables do not.
  target_link_libraries(pops INTERFACE HDF5::HDF5)
  target_compile_definitions(pops INTERFACE POPS_HAS_HDF5)
  set(POPS_HAS_HDF5 ON)
  if(NOT HDF5_IS_PARALLEL)
    message(FATAL_ERROR
      "POPS_USE_HDF5=ON requires a parallel HDF5 C library; "
      "a serial HDF5 discovery cannot implement the requested collective writer")
  endif()
  target_compile_definitions(pops INTERFACE POPS_HAS_PARALLEL_HDF5)
  set(POPS_HAS_PARALLEL_HDF5 ON)
  message(STATUS
    "Parallel HDF5 found (${HDF5_VERSION}) : enabling native collective DataWriter")
endif()

# One concrete MPI ABI identity for exported C++ consumers and runtime-compiled components. A
# parent project may already own an unrelated MPI::MPI_CXX target; only PoPS' explicit feature state
# authorizes consuming it or exporting POPS_MPI_ABI.
set(POPS_NATIVE_MPI_ABI "")
set(POPS_NATIVE_HDF5_MPI_ABI "")
if(POPS_HAS_MPI)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/PopsMpiContract.cmake)
  pops_configure_mpi_contract(pops)
  if(POPS_HAS_HDF5)
    pops_configure_parallel_hdf5_mpi_contract(POPS_NATIVE_HDF5_MPI_ABI)
  endif()
endif()

# Outillage QUALITE opt-in (warnings stricts, sanitizers) porte par la cible INTERFACE
# `pops_dev_options`, lie en PRIVATE par les SEULES cibles internes (tests/, module _pops). Inclus ICI :
# apres la resolution de Kokkos, AVANT add_subdirectory(tests)/add_subdirectory(python) qui le lient.
# Tout est OFF par defaut (cible vide) -> aucun impact sur ci.yml, les builds locaux ni `adc_cases`.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/PopsDevTooling.cmake)

# PoPS = COEUR de la bibliotheque (generique, header-only). Les modeles
# physiques, les facades compilees (libadc), les bindings Python et les exemples
# vivent dans le depot applicatif separe adc_cases, qui consomme pops::pops via
# FetchContent. Le coeur ne connait aucun modele : aucune dependance coeur->cas.

# enable_testing() des que des tests existent : le coeur (tests/) OU le module Python.
# Sans ca, `ctest --test-dir build-py` avec POPS_BUILD_TESTS=OFF ne voit aucun test.
if(POPS_BUILD_TESTS OR POPS_BUILD_PYTHON)
  enable_testing()
  # One compiled-runtime source manifest for both mutually-exclusive consumers (tests and Python).
  add_subdirectory(src)
endif()
if(POPS_BUILD_TESTS)
  add_subdirectory(tests)
  add_test(
    NAME test_serial_add_subdirectory_ignores_parent_mpi
    COMMAND "${CMAKE_COMMAND}"
      "-DPOPS_SOURCE_DIR=${CMAKE_SOURCE_DIR}"
      "-DPOPS_FIXTURE_SOURCE_DIR=${CMAKE_SOURCE_DIR}/tests/cmake/serial_parent_mpi_target"
      "-DPOPS_KOKKOS_FIXTURE_DIR=${CMAKE_SOURCE_DIR}/tests/cmake/mpi_abi_mismatch"
      "-DPOPS_TEST_BUILD_DIR=${CMAKE_BINARY_DIR}/serial-parent-mpi-smoke"
      "-DPOPS_GENERATOR=${CMAKE_GENERATOR}"
      -P "${CMAKE_SOURCE_DIR}/tests/cmake/run_serial_parent_mpi_target.cmake")
  set_tests_properties(test_serial_add_subdirectory_ignores_parent_mpi PROPERTIES
    LABELS "cmake;integration;serial;mpi"
    TIMEOUT 30)
  add_test(
    NAME test_hdf5_mpi_binary_identity
    COMMAND "${CMAKE_COMMAND}"
      "-DPOPS_FIXTURE_SOURCE_DIR=${CMAKE_SOURCE_DIR}/tests/cmake/hdf5_mpi_binary_identity"
      "-DPOPS_TEST_BUILD_DIR=${CMAKE_BINARY_DIR}/hdf5-mpi-binary-identity-smoke"
      "-DPOPS_RUNTIME_SCAN_SCRIPT=${CMAKE_SOURCE_DIR}/cmake/PopsRuntimeDependencyScan.cmake"
      "-DPOPS_GENERATOR=${CMAKE_GENERATOR}"
      -P "${CMAKE_SOURCE_DIR}/tests/cmake/run_hdf5_mpi_binary_identity.cmake")
  set_tests_properties(test_hdf5_mpi_binary_identity PROPERTIES
    LABELS "cmake;integration;hdf5;mpi"
    TIMEOUT 30)
endif()

# Module Python `pops` : adaptateurs pybind11 lies aux facades runtime centrales.
if(POPS_BUILD_PYTHON)
  include(FetchContent)
  # macOS expose souvent plusieurs Python (framework Python.org, conda, brew). Sans garde,
  # FindPython prend le framework le plus recent comme Interpreter mais un autre pour
  # Development.Module : l'interpreteur ne peut alors pas importer l'extension compilee
  # (suffixe ABI cpython-3XY different). On deprioritise les frameworks pour que les deux
  # concordent. Surchargeable via -DPython_EXECUTABLE=/chemin/python pour pinner.
  if(APPLE)
    set(Python_FIND_FRAMEWORK LAST)
  endif()
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  message(STATUS "pops Python module: interpreteur ${Python_EXECUTABLE} (${Python_VERSION})")
  # pybind11 : on PREFERE celui de l'environnement (conda/pip) plutot que de le retelecharger a
  # chaque configure. L'interpreteur retenu expose son dossier de config CMake via
  # `python -m pybind11 --cmakedir` ; on l'ajoute en tete du chemin de recherche pour que le
  # find_package CONFIG ci-dessous le trouve. Benefice double : pas de FetchContent reseau, et
  # plus de "CMake Deprecation Warning" du CMakeLists amont de pybind11 (son
  # cmake_minimum_required(VERSION 3.5) est deprecie par CMake >= 3.31).
  execute_process(
    COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE _pops_pybind11_cmakedir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _pops_pybind11_probe
    ERROR_QUIET)
  if(_pops_pybind11_probe EQUAL 0 AND IS_DIRECTORY "${_pops_pybind11_cmakedir}")
    list(PREPEND CMAKE_PREFIX_PATH "${_pops_pybind11_cmakedir}")
  endif()
  find_package(pybind11 2.13 CONFIG QUIET)  # floor aligne sur environment.yml + le tag FetchContent
  if(pybind11_FOUND)
    message(STATUS "pops Python module: pybind11 ${pybind11_VERSION} trouve dans l'environnement (pas de FetchContent)")
  else()
    # Repli reseau : aucun pybind11 dans l'environnement. Le CMakeLists de pybind11 v2.13.6 emet un
    # warning de depreciation (cmake_minimum_required 3.5) qui n'est PAS de notre ressort. On le
    # neutralise UNIQUEMENT autour de FetchContent_MakeAvailable, puis on restaure l'etat precedent :
    # les depreciations de notre propre code restent signalees.
    message(STATUS "pops Python module: pybind11 absent de l'environnement -> repli FetchContent (v2.13.6)")
    # Python a deja ete trouve via find_package(... Development.Module) plus haut : on force le
    # pybind11 telecharge a reutiliser ce Python moderne. Sinon il retombe sur sa detection classique,
    # qui ne consomme PAS l'include de Development.Module -> "Python.h: No such file or directory" au
    # build sur les machines sans pybind11 dans l'environnement (spack/HPC). Le chemin
    # find_package(pybind11) ci-dessus n'est pas concerne (ADC-386).
    set(PYBIND11_FINDPYTHON ON)
    set(_pops_warn_dep_saved "${CMAKE_WARN_DEPRECATED}")
    set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11.git
      GIT_TAG        v2.13.6)
    FetchContent_MakeAvailable(pybind11)
    set(CMAKE_WARN_DEPRECATED "${_pops_warn_dep_saved}" CACHE BOOL "" FORCE)
  endif()
  add_subdirectory(python)
endif()

# --- Installation / export du COEUR (find_package(pops)) -------------------------------------
# Rend la lib header-only consommable HORS FetchContent : `cmake --install` depose les en-tetes
# + popsConfig.cmake, et un consommateur fait find_package(pops CONFIG) + pops::pops. Prerequis d'un
# eventuel paquet (conda-forge feedstock, Spack). Defaut : ON seulement en top-level ET hors
# build pip (SKBUILD : la wheel ne doit contenir QUE le paquet Python, pas include/ ni cmake/).
if(PROJECT_IS_TOP_LEVEL AND NOT SKBUILD)
  set(_pops_install_default ON)
else()
  set(_pops_install_default OFF)
endif()
option(POPS_INSTALL "Regles d'installation du coeur header-only (find_package(pops))" ${_pops_install_default})
if(POPS_INSTALL)
  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)
  install(TARGETS pops EXPORT popsTargets)
  pops_install_headers("${CMAKE_INSTALL_INCLUDEDIR}")
  install(EXPORT popsTargets NAMESPACE pops:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pops)
  configure_package_config_file(cmake/popsConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/popsConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pops)
  # SemVer before 1.0: 0.y is the breaking-version boundary. SameMajorVersion would incorrectly
  # accept every 0.x package for every other 0.x consumer. At 1.0+, the major is the boundary.
  if(PROJECT_VERSION_MAJOR EQUAL 0)
    set(_pops_cmake_compatibility SameMinorVersion)
  else()
    set(_pops_cmake_compatibility SameMajorVersion)
  endif()
  write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/popsConfigVersion.cmake
    COMPATIBILITY ${_pops_cmake_compatibility} ARCH_INDEPENDENT)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/popsConfig.cmake
                ${CMAKE_CURRENT_BINARY_DIR}/popsConfigVersion.cmake
                ${CMAKE_CURRENT_SOURCE_DIR}/cmake/PopsMpiContract.cmake
                ${CMAKE_CURRENT_SOURCE_DIR}/cmake/PopsRuntimeDependencyScan.cmake
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pops)

  # Exercise the actual installed package, not the build-tree alias.  The same CTest is run by the
  # serial and MPI/HDF5 CI configurations, so it proves both CXX-only consumption and the exact
  # C/MPI_C/HDF5 dependency rehydration of the collective writer.
  if(POPS_BUILD_TESTS)
    add_test(
      NAME test_installed_package_consumer
      COMMAND "${CMAKE_COMMAND}"
        "-DPOPS_BINARY_DIR=${CMAKE_BINARY_DIR}"
        "-DPOPS_CONSUMER_SOURCE_DIR=${CMAKE_SOURCE_DIR}/tests/cmake/installed_package_consumer"
        "-DPOPS_SMOKE_ROOT=${CMAKE_BINARY_DIR}/installed-package-smoke"
        "-DPOPS_BUILD_CONFIG=$<CONFIG>"
        "-DPOPS_GENERATOR=${CMAKE_GENERATOR}"
        "-DPOPS_DEPENDENCY_PREFIX_PATH=${CMAKE_PREFIX_PATH}"
        "-DPOPS_CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
        "-DPOPS_CMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
        "-DPOPS_Kokkos_DIR=${Kokkos_DIR}"
        "-DPOPS_MPI_CXX_COMPILER=${MPI_CXX_COMPILER}"
        "-DPOPS_MPI_C_COMPILER=${MPI_C_COMPILER}"
        "-DPOPS_HDF5_ROOT=${HDF5_ROOT}"
        "-DPOPS_OpenMP_CXX_FLAGS=${OpenMP_CXX_FLAGS}"
        "-DPOPS_OpenMP_omp_LIBRARY=${OpenMP_omp_LIBRARY}"
        "-DPOPS_EXPECT_MPI=${POPS_HAS_MPI}"
        "-DPOPS_EXPECT_HDF5=${POPS_HAS_HDF5}"
        "-DPOPS_EXPECT_PARALLEL_HDF5=${POPS_HAS_PARALLEL_HDF5}"
        "-DPOPS_EXPECT_C_LOADED=${POPS_HAS_HDF5}"
        -P "${CMAKE_SOURCE_DIR}/tests/cmake/run_installed_package_smoke.cmake")
    set_tests_properties(test_installed_package_consumer PROPERTIES
      LABELS "cmake;integration;packaging"
      TIMEOUT 120)

    if(POPS_HAS_MPI)
      add_test(
        NAME test_installed_package_mpi_abi_mismatch
        COMMAND "${CMAKE_COMMAND}"
          "-DPOPS_BINARY_DIR=${CMAKE_BINARY_DIR}"
          "-DPOPS_CONSUMER_SOURCE_DIR=${CMAKE_SOURCE_DIR}/tests/cmake/installed_package_consumer"
          "-DPOPS_FAKE_MPI_MODULE_DIR=${CMAKE_SOURCE_DIR}/tests/cmake/mpi_abi_mismatch"
          "-DPOPS_SMOKE_ROOT=${CMAKE_BINARY_DIR}/installed-package-mpi-mismatch"
          "-DPOPS_BUILD_CONFIG=$<CONFIG>"
          "-DPOPS_GENERATOR=${CMAKE_GENERATOR}"
          "-DPOPS_DEPENDENCY_PREFIX_PATH=${CMAKE_PREFIX_PATH}"
          "-DPOPS_CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
          "-DPOPS_CMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
          "-DPOPS_Kokkos_DIR=${CMAKE_SOURCE_DIR}/tests/cmake/mpi_abi_mismatch"
          -P "${CMAKE_SOURCE_DIR}/tests/cmake/run_installed_package_mpi_mismatch.cmake")
      set_tests_properties(test_installed_package_mpi_abi_mismatch PROPERTIES
        LABELS "cmake;integration;packaging;mpi"
        TIMEOUT 60)
    endif()

    if(NOT POPS_HAS_MPI)
      add_test(
        NAME test_hdf5_without_mpi_rejected
        COMMAND "${CMAKE_COMMAND}"
          "-DPOPS_SOURCE_DIR=${CMAKE_SOURCE_DIR}"
          "-DPOPS_REJECT_BUILD_DIR=${CMAKE_BINARY_DIR}/hdf5-without-mpi-rejected"
          "-DPOPS_GENERATOR=${CMAKE_GENERATOR}"
          -P "${CMAKE_SOURCE_DIR}/tests/cmake/expect_hdf5_without_mpi_rejected.cmake")
      set_tests_properties(test_hdf5_without_mpi_rejected PROPERTIES
        LABELS "cmake;integration;packaging"
        TIMEOUT 30)
    endif()
  endif()
endif()

# --- Cible documentation (top-level UNIQUEMENT) ---------------------------------------------
# `cmake --build <dir> --target docs` lance scripts/build_docs.sh (lint + Sphinx + Doxygen + site
# combine). HORS de la cible ALL : jamais construite par un build standard ni par le CI de code ;
# c'est un raccourci de confort, equivalent a `bash scripts/build_docs.sh` depuis la racine. Le
# script verifie LUI-MEME ses dependances (sphinx, doxygen, graphviz) et sort proprement si l'une
# manque -> aucune logique CMake en plus ici. Top-level seulement : un consommateur (FetchContent /
# add_subdirectory) n'herite pas de cette cible de commodite.
if(PROJECT_IS_TOP_LEVEL)
  add_custom_target(docs
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/scripts/build_docs.sh
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    USES_TERMINAL
    COMMENT "Build de la documentation (Sphinx + Doxygen) via scripts/build_docs.sh")
endif()
