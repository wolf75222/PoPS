/// @file
/// @brief Deep-owned moving geometry storage for the existing Program transaction.
#pragma once

#include <pops/core/state/state.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/spatial/nd/face_field.hpp>
#include <pops/runtime/multiblock/evaluation_point.hpp>
#include <pops/runtime/program/accepted_exchange.hpp>
#include <pops/runtime/program/prepared_resource_lifetime.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace pops::runtime::program {

template <int Dim> class ProgramContext;

/// Device-copyable quadrature coordinates; the exact logical/attempt authority
/// stays on the host RuntimeIntervalEvaluation rather than in a kernel capture.
struct MovingIntervalTime {
  Real begin = 0, end = 0, duration = 0;
};
struct MovingFaceGeometry {
  Real reference = 0, previous_position = 0, position = 0, swept_volume = 0;
};
struct MovingCellGeometry {
  Real reference = 0, previous_center = 0, center = 0, previous_measure = 0, measure = 0;
};
template <int Components>
struct MovingFaceEvaluation {
  StateVec<Components> physical_amount{}, density{};
};

/// Values produced by one generic native interval evaluator. They are already
/// integrated quantities; the runtime never inserts a physical constitutive law.
template <int Dim>
struct MovingIntervalInputs {
  std::vector<nd::FaceField<Dim>> coordinates, swept_volumes, physical_flux, face_density;
  MultiFab<Dim> source;
};

/// An owned evaluation issued by ProgramContext while executing its producer.
/// Raw arrays cannot be retroactively tagged at publication time. The producer
/// receives the exact runtime point, including the effective interval duration.
template <int Dim>
class RuntimeIntervalEvaluation {
  friend class ProgramContext<Dim>;
 public:
  const runtime::multiblock::BoundaryEvaluationPoint& point() const noexcept { return point_; }
  const std::string& physical_frame() const noexcept { return physical_frame_; }
  const std::string& quadrature_identity() const noexcept { return quadrature_; }
  double interval_begin() const noexcept { return point_.physical_time; }
  double interval_end() const noexcept { return point_.physical_time + point_.dt; }
 private:
  RuntimeIntervalEvaluation() = default;
  const ProgramContext<Dim>* owner_ = nullptr;
  PreparedResourceAttempt attempt_;
  std::string identity_, physical_frame_, quadrature_;
  int program_block_ = -1;
  std::uint64_t generation_ = 0;
  runtime::multiblock::BoundaryEvaluationPoint point_;
  MovingIntervalInputs<Dim> inputs_;
  MultiFab<Dim> initial_state_;
};

/// Independent previous accepted inputs for a saved Reynolds/GCL receipt.
/// Old measures are stored explicitly, never reconstructed from the very sweep
/// whose consistency a verifier is supposed to assess.
template <int Dim>
struct MovingIntervalReceipt {
  runtime::multiblock::BoundaryEvaluationPoint point;
  std::string physical_frame, quadrature_identity;
  Real geometry_tolerance = 0;
  MultiFab<Dim> previous_state, previous_measures, integrated_source;
  std::vector<nd::FaceField<Dim>> previous_coordinates, physical_flux, face_density;
};

/// A fixed-topology 1D moving interval representation. Faces and measures live
/// in Kokkos storage, with exactly the same partition as the physical state.
/// This is accepted/provisional Program data, never an independent commit owner.
/// The rank template lets the shared runtime aggregate own it; the first update
/// provider serves Dim=1 only and refuses higher-dimensional geometry explicitly.
template <int Dim>
struct MovingIntervalGeometry {
  int runtime_block = -1;
  std::string physical_frame;
  MultiFab<Dim> measures;
  std::vector<nd::FaceField<Dim>> coordinates;
  std::vector<nd::FaceField<Dim>> swept_volumes;
  std::uint64_t generation = 0;
  std::string last_interval;
  std::optional<MovingIntervalReceipt<Dim>> last_receipt;
};

/// A coupled SSA candidate, with its previous state and independent geometry
/// retained for accepted-state receipts. Preparation never publishes its ledger.
template <int Dim>
class PreparedMovingIntervalUpdate {
  friend class ProgramContext<Dim>;
 public:
  PreparedMovingIntervalUpdate(PreparedMovingIntervalUpdate&&) = default;
  PreparedMovingIntervalUpdate& operator=(PreparedMovingIntervalUpdate&&) = default;
  PreparedMovingIntervalUpdate(const PreparedMovingIntervalUpdate&) = delete;
  const MultiFab<Dim>& state() const noexcept { return state_; }
  const MovingIntervalGeometry<Dim>& geometry() const noexcept { return geometry_; }
 private:
  PreparedMovingIntervalUpdate() = default;
  const ProgramContext<Dim>* owner_ = nullptr;
  PreparedResourceAttempt attempt_;
  std::string identity_, physical_frame_, quadrature_, previous_interval_;
  int program_block_ = -1;
  runtime::multiblock::BoundaryEvaluationPoint point_;
  MovingIntervalGeometry<Dim> geometry_;
  MultiFab<Dim> state_, initial_state_;
  std::vector<ExchangeRecord> records_;
  bool published_ = false;
};

}  // namespace pops::runtime::program
