# Driver diocotron HyQMOM GPU/CPU (briques DSL emises + System header-only).
# Local (shakeout, Kokkos OpenMP) :
#   cmake -S . -B build -DADC_INCLUDE=$ADC/include -DKokkos_ROOT=/opt/homebrew/anaconda3 \
#         -DCMAKE_BUILD_TYPE=Debug && cmake --build build
# ROMEO armgpu (CUDA H100) -- cf. diocotron_gpu.sbatch :
#   cmake -S . -B build -DCMAKE_CXX_COMPILER=$HOME/adc_gpu_p1/kinstall/bin/nvcc_wrapper \
#         -DKokkos_ROOT=$HOME/adc_gpu_p1/kinstall -DADC_INCLUDE=... -DCMAKE_BUILD_TYPE=Release
cmake_minimum_required(VERSION 3.20)
project(diocotron_gpu CXX)
find_package(Kokkos REQUIRED)
if(NOT ADC_INCLUDE)
  message(FATAL_ERROR "Passer -DADC_INCLUDE=/chemin/vers/adc_cpp/include")
endif()
# adc::System n'est pas header-only : ses definitions vivent dans python/system.cpp
# (TU pur C++, sans pybind) -- on le compile DANS le driver (grosse TU, RAM en consequence).
add_executable(diocotron_gpu diocotron_gpu.cpp ${ADC_INCLUDE}/../python/system.cpp)
target_include_directories(diocotron_gpu PRIVATE ${ADC_INCLUDE} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(diocotron_gpu PRIVATE Kokkos::kokkos)
target_compile_definitions(diocotron_gpu PRIVATE ADC_HAS_KOKKOS)  # coeur Kokkos-only (#263)
target_compile_features(diocotron_gpu PRIVATE cxx_std_20)
