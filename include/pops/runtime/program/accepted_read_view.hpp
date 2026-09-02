#pragma once

#include <pops/runtime/program/program_transaction.hpp>

#include <utility>

namespace pops {

template <int Dim>
class System;
template <int Dim>
class AmrSystem;

namespace runtime::program {
template <int Dim>
class ProgramExecutionServices;
}

/// Move-only view of an object in the last sealed accepted generation.
///
/// The view owns the read lease which protects the pointed-to object.  It deliberately has no
/// implicit conversion to a reference and its pointer/dereference operators are lvalue-qualified:
/// callers must name the view before borrowing the object, so a temporary cannot leak a pointer
/// beyond the lease's lifetime.
template <class T>
class AcceptedReadView final {
 public:
  AcceptedReadView() noexcept = default;
  AcceptedReadView(const AcceptedReadView&) = delete;
  AcceptedReadView& operator=(const AcceptedReadView&) = delete;
  AcceptedReadView(AcceptedReadView&& other) noexcept
      : lease_(std::move(other.lease_)), object_(std::exchange(other.object_, nullptr)) {}
  AcceptedReadView& operator=(AcceptedReadView&& other) noexcept {
    if (this != &other) {
      lease_ = std::move(other.lease_);
      object_ = std::exchange(other.object_, nullptr);
    }
    return *this;
  }

  [[nodiscard]] bool valid() const noexcept { return object_ != nullptr && lease_.valid(); }
  [[nodiscard]] explicit operator bool() const noexcept { return valid(); }
  [[nodiscard]] T* get() & noexcept { return valid() ? object_ : nullptr; }
  [[nodiscard]] T* get() const& noexcept { return valid() ? object_ : nullptr; }
  T* get() && = delete;
  T* get() const&& = delete;
  [[nodiscard]] T& operator*() & noexcept { return *get(); }
  [[nodiscard]] const T& operator*() const& noexcept { return *get(); }
  T& operator*() && = delete;
  const T& operator*() const&& = delete;
  [[nodiscard]] T* operator->() & noexcept { return get(); }
  [[nodiscard]] T* operator->() const& noexcept { return get(); }
  T* operator->() && = delete;
  T* operator->() const&& = delete;
  [[nodiscard]] runtime::program::AcceptedGeneration generation() const noexcept {
    return lease_.generation();
  }

 private:
  template <int>
  friend class System;
  template <int>
  friend class AmrSystem;
  template <int>
  friend class runtime::program::ProgramExecutionServices;

  AcceptedReadView(runtime::program::AcceptedReadLease&& lease, T* object) noexcept
      : lease_(std::move(lease)), object_(object) {}

  runtime::program::AcceptedReadLease lease_;
  T* object_ = nullptr;
};

}  // namespace pops
