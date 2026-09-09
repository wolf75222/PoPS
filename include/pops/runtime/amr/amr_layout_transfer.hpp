/// Prepared composite physical transport across independently decomposed AMR layouts.
#pragma once

#include <pops/runtime/export.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/parallel/execution_lane.hpp>
#include <pops/runtime/system.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace pops {

/// Borrowed for one call only. State may be the current hierarchy or an explicitly qualified
/// Program stage. Coverage is one on finest-owner cells and zero on covered coarse cells.
template <int Dim>
struct AmrTransferLevelView {
  int level = 0;
  Geometry<Dim> geometry;
  const MultiFab<Dim>* state = nullptr;
  const MultiFab<Dim>* coverage = nullptr;
  const MultiFab<Dim>* activity = nullptr;
  const MultiFab<Dim>* relative_measure = nullptr;
};

template <int Dim>
struct AmrTransferEndpoint {
  std::string layout_identity;
  std::string state_identity;
  std::string hierarchy_identity;
  std::uint64_t hierarchy_generation = 0;
  std::array<bool, Dim> periodicity{};
  std::string stage_identity = "accepted-current";
  std::uint64_t stage_generation = 0;
  // Cartesian cells, or an explicitly declared piecewise-constant relative volume density.
  std::string measure_identity = "pops://measure/cartesian-cells@1";
  std::vector<AmrTransferLevelView<Dim>> levels;
};

struct AmrLayoutTransferBudget {
  std::size_t destination_cells = 0;
  std::size_t intersection_probes = 0;
  std::size_t canonical_jobs = 0;
  std::size_t transported_elements = 0;
  std::size_t prepared_bytes = 0;
};

template <int Dim>
struct AmrPhysicalTransferSpec {
  SystemLayoutTransferSpec<Dim> authentication;
  std::string physical_contract_identity;
  std::array<std::vector<double>, Dim> base_bin_weights;
  std::array<double, Dim> base_bin_lower{};
  std::array<double, Dim> base_bin_upper{};
  // Each weight is the complete signed measure of one uniform base bin. Its density is
  // constant within that bin; refined/coarsened source cells integrate the exact bin overlap.
  std::string quadrature_identity = "pops://measure/piecewise-constant-base-bins@1";
  AmrLayoutTransferBudget budget;
};

struct AmrLayoutTransferReceipt {
  SystemLayoutTransferReceipt transfer;
  std::string physical_contract_identity;
  std::string source_hierarchy_identity;
  std::string target_hierarchy_identity;
  std::uint64_t source_hierarchy_generation = 0;
  std::uint64_t target_hierarchy_generation = 0;
  std::string source_stage_identity;
  std::uint64_t source_stage_generation = 0;
  std::string target_stage_identity;
  std::uint64_t target_stage_generation = 0;
  std::uint64_t source_active_elements = 0;
  std::uint64_t destination_active_elements = 0;
  std::size_t canonical_jobs = 0;
  std::size_t transported_elements = 0;
  // Deterministic upper bound across ranks for prepared numerical storage and transport buffers.
  std::size_t prepared_bytes = 0;
};

/// Keeps topology, intersections, snapshots and a private communication lane; never retains
/// borrowed state/stage pointers. A changed hierarchy identity/generation requires preparation
/// again, including after restart. Publication and covered-coarse synchronization belong to the
/// enclosing native System transaction; apply writes only detached active-cell candidates.
template <int Dim>
class POPS_EXPORT PreparedAmrLayoutTransfer final {
 public:
  static std::shared_ptr<PreparedAmrLayoutTransfer> prepare(
      const AmrTransferEndpoint<Dim>& source, const AmrTransferEndpoint<Dim>& target,
      AmrPhysicalTransferSpec<Dim> spec,
      std::shared_ptr<component::LoadedComponent> provider_component,
      SystemLayoutTransferExecution execution, const ExecutionLane& authority);
  ~PreparedAmrLayoutTransfer();
  PreparedAmrLayoutTransfer(const PreparedAmrLayoutTransfer&) = delete;
  PreparedAmrLayoutTransfer& operator=(const PreparedAmrLayoutTransfer&) = delete;

  /// Independent endpoint-mask census and prepared provenance; never reads a computed receipt.
  AmrLayoutTransferReceipt expected_receipt_contract(const AmrTransferEndpoint<Dim>& source,
                                                     const AmrTransferEndpoint<Dim>& target) const;
  void begin_transaction(std::uint64_t generation);
  void capture(const AmrTransferEndpoint<Dim>& source, std::uint64_t generation,
               std::uint64_t attempt);
  AmrLayoutTransferReceipt apply(const AmrTransferEndpoint<Dim>& target,
                                 const std::vector<MultiFab<Dim>*>& candidates,
                                 std::uint64_t generation, std::uint64_t attempt);
  void reject_attempt(std::uint64_t generation, std::uint64_t attempt);
  void finalize_transaction(std::uint64_t generation) noexcept;
  void rollback_transaction(std::uint64_t generation) noexcept;
  std::size_t canonical_jobs() const noexcept;
  std::size_t transported_elements() const noexcept;
  std::size_t prepared_bytes() const noexcept;

 private:
  struct Impl;
  explicit PreparedAmrLayoutTransfer(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> p_;
};

}  // namespace pops
