/// @file
/// @brief Deep-owned moving geometry storage for the existing Program transaction.
#pragma once

#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/spatial/nd/face_field.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace pops::runtime::program {

/// A fixed-topology 1D moving interval representation. Faces and measures live
/// in Kokkos storage, with exactly the same partition as the physical state.
/// This is accepted/provisional Program data, never an independent commit owner.
/// The rank template lets the shared runtime aggregate own it; the first update
/// provider serves Dim=1 only and refuses higher-dimensional geometry explicitly.
template <int Dim>
struct MovingIntervalGeometry {
  int runtime_block = -1;
  MultiFab<Dim> measures;
  std::vector<nd::FaceField<Dim>> coordinates;
  std::vector<nd::FaceField<Dim>> swept_volumes;
  std::uint64_t generation = 0;
  std::string last_interval;
};

}  // namespace pops::runtime::program
