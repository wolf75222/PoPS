cmake_minimum_required(VERSION 3.20)

project(adc_cpp
  VERSION 0.0.1
  DESCRIPTION "Solveur advection-diffusion-couplage hyperbolique-elliptique sur AMR, cible cluster"
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option(ADC_BUILD_TESTS "Build tests" ON)

# Couche physique : bibliotheque d'interface (header-only) pour l'instant.
# Le maillage, l'operateur spatial, l'elliptique et l'integrateur viendront
# en couches separees ; le backend parallele (AMR/MPI/Kokkos) ne touchera
# jamais a cette cible.
add_library(adc INTERFACE)
add_library(adc::adc ALIAS adc)
target_include_directories(adc INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_compile_features(adc INTERFACE cxx_std_23)

if(ADC_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
