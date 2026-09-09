/// Transactional AmrSystem adapter for the generic native composite physical transport.
#pragma once

#include <pops/runtime/amr/amr_layout_transfer.hpp>

namespace pops {
template <int Dim>
class AmrSystem;

template <int Dim>
class POPS_EXPORT PreparedAmrSystemLayoutTransfer final {
 public:
  static AmrLayoutTransferBudget capacity_budget(AmrSystem<Dim>& source, AmrSystem<Dim>& target,
                                                 const std::string& source_block,
                                                 const std::string& target_block,
                                                 std::size_t source_cells,
                                                 std::size_t target_cells);
  static std::shared_ptr<PreparedAmrSystemLayoutTransfer> prepare(
      AmrSystem<Dim>& source, AmrSystem<Dim>& target,
      std::shared_ptr<component::LoadedComponent> provider, AmrPhysicalTransferSpec<Dim> spec,
      SystemLayoutTransferExecution execution);
  ~PreparedAmrSystemLayoutTransfer();
  PreparedAmrSystemLayoutTransfer(const PreparedAmrSystemLayoutTransfer&) = delete;
  PreparedAmrSystemLayoutTransfer& operator=(const PreparedAmrSystemLayoutTransfer&) = delete;
  AmrLayoutTransferReceipt expected_receipt_contract() const;
  void begin_transaction(std::uint64_t generation);
  void capture(std::uint64_t generation, std::uint64_t attempt);
  AmrLayoutTransferReceipt apply(std::uint64_t generation, std::uint64_t attempt);
  void reject_attempt(std::uint64_t generation, std::uint64_t attempt);
  void finalize_transaction(std::uint64_t generation) noexcept;
  void rollback_transaction(std::uint64_t generation) noexcept;

 private:
  struct Impl;
  explicit PreparedAmrSystemLayoutTransfer(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> p_;
};
}  // namespace pops
