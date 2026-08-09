/// @file
/// @brief Owning compile-time-ranked field storage in a selected Kokkos memory space.

#pragma once

#include <pops/core/foundation/kokkos_env.hpp>
#include <pops/core/foundation/types.hpp>
#include <pops/mesh/index/box.hpp>
#include <pops/mesh/storage/field_view.hpp>

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace pops {

/// Component-slowest field storage.  Host access is explicit through a Kokkos host mirror so a
/// device-only MemorySpace is never presented as directly host-accessible.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class Fab {
 public:
  static_assert(Dim >= 1 && Dim <= 3, "pops::Fab only supports dimensions 1, 2, and 3");

  using value_type = Real;
  using memory_space = MemorySpace;
  using storage_type = Kokkos::View<Real*, MemorySpace>;
  using raw_host_mirror_type = typename storage_type::host_mirror_type;

  /// Host view coupled to the Fab layout that created it.  It deliberately does not expose a
  /// rebindable raw view at the copy boundary.
  class HostMirror {
   public:
    Real& operator()(std::size_t index) { return values_(index); }
    const Real& operator()(std::size_t index) const { return values_(index); }
    std::size_t size() const noexcept { return size_; }

   private:
    friend class Fab;

    HostMirror(raw_host_mirror_type values, const Fab* source, std::size_t size,
               std::uint64_t generation)
        : values_(std::move(values)), source_(source), size_(size), generation_(generation) {}

    raw_host_mirror_type values_{};
    const Fab* source_ = nullptr;
    std::size_t size_ = 0;
    std::uint64_t generation_ = 0;
  };

  using host_mirror_type = HostMirror;

  Fab() = default;

  Fab(const Box<Dim>& valid, int ncomp, Extent<Dim> ghosts = {})
      : valid_(valid), ncomp_(ncomp), ghosts_(ghosts) {
    if (ncomp < 1)
      throw std::invalid_argument("pops::Fab: ncomp must be positive");
    grown_ = grown_with_ghosts(valid_, ghosts_);
    initialize_layout();
    if (size_ == 0)
      return;
    detail::ensure_kokkos_initialized();
    data_ = storage_type("pops_fab", size_);
    Kokkos::deep_copy(data_, Real{0});
  }

  Fab(const Fab& other)
      : valid_(other.valid_),
        grown_(other.grown_),
        ncomp_(other.ncomp_),
        ghosts_(other.ghosts_),
        component_stride_(other.component_stride_),
        size_(other.size_) {
    for (int axis = 0; axis < Dim; ++axis)
      strides_[axis] = other.strides_[axis];
    if (size_ == 0)
      return;
    detail::ensure_kokkos_initialized();
    data_ = storage_type("pops_fab_copy", size_);
    Kokkos::deep_copy(data_, other.data_);
  }

  Fab& operator=(const Fab& other) {
    if (this != &other) {
      Fab copy(other);
      *this = std::move(copy);
    }
    return *this;
  }

  Fab(Fab&& other) noexcept { move_from(std::move(other)); }

  Fab& operator=(Fab&& other) noexcept {
    if (this != &other) {
      reset_moved_from();
      move_from(std::move(other));
    }
    return *this;
  }

  const Box<Dim>& box() const { return valid_; }
  const Box<Dim>& grown_box() const { return grown_; }
  int ncomp() const { return ncomp_; }
  const Extent<Dim>& ghosts() const { return ghosts_; }
  std::size_t size() const { return size_; }

  FieldView<Real, Dim> view() {
    FieldView<Real, Dim> result{};
    result.data = data_.data();
    result.origin = grown_.lo;
    result.extents = grown_.extent();
    for (int axis = 0; axis < Dim; ++axis)
      result.strides[axis] = strides_[axis];
    result.ncomp = ncomp_;
    result.component_stride = component_stride_;
    return result;
  }

  FieldView<const Real, Dim> view() const {
    FieldView<const Real, Dim> result{};
    result.data = data_.data();
    result.origin = grown_.lo;
    result.extents = grown_.extent();
    for (int axis = 0; axis < Dim; ++axis)
      result.strides[axis] = strides_[axis];
    result.ncomp = ncomp_;
    result.component_stride = component_stride_;
    return result;
  }

  const storage_type& storage() const { return data_; }

  host_mirror_type create_host_mirror() const {
    return host_mirror_type(size_ == 0 ? raw_host_mirror_type{} : Kokkos::create_mirror_view(data_),
                            this, size_, generation_);
  }
  void copy_to_host(const host_mirror_type& host) const {
    validate_mirror(host);
    if (size_ != 0)
      Kokkos::deep_copy(host.values_, data_);
  }
  void copy_from_host(const host_mirror_type& host) {
    validate_mirror(host);
    if (size_ != 0)
      Kokkos::deep_copy(data_, host.values_);
  }
  void set_val(Real value) {
    if (size_ != 0)
      Kokkos::deep_copy(data_, value);
  }

 private:
  void validate_mirror(const host_mirror_type& host) const {
    if (host.source_ != this || host.generation_ != generation_ || host.size_ != size_ ||
        host.values_.extent(0) != size_)
      throw std::invalid_argument("pops::Fab host mirror does not match this Fab association");
  }

  void reset_moved_from() noexcept {
    valid_ = Box<Dim>{};
    grown_ = Box<Dim>{};
    ncomp_ = 0;
    ghosts_ = Extent<Dim>{};
    for (int axis = 0; axis < Dim; ++axis)
      strides_[axis] = 0;
    component_stride_ = 0;
    size_ = 0;
    data_ = storage_type{};
    ++generation_;
  }

  void move_from(Fab&& other) noexcept {
    valid_ = other.valid_;
    grown_ = other.grown_;
    ncomp_ = other.ncomp_;
    ghosts_ = other.ghosts_;
    for (int axis = 0; axis < Dim; ++axis)
      strides_[axis] = other.strides_[axis];
    component_stride_ = other.component_stride_;
    size_ = other.size_;
    data_ = std::move(other.data_);
    ++generation_;
    other.reset_moved_from();
  }

  void initialize_layout() {
    const Extent<Dim> extents = grown_.extent();
    std::int64_t cells = 1;
    strides_[0] = 1;
    for (int axis = 0; axis < Dim; ++axis) {
      if (extents[axis] == 0) {
        size_ = 0;
        component_stride_ = 0;
        return;
      }
      if (axis > 0)
        strides_[axis] = cells;
      if (cells > std::numeric_limits<std::int64_t>::max() / extents[axis])
        throw std::overflow_error("pops::Fab: cell count exceeds int64_t");
      cells *= extents[axis];
    }
    component_stride_ = cells;
    if (cells > std::numeric_limits<std::int64_t>::max() / ncomp_)
      throw std::overflow_error("pops::Fab: element count exceeds int64_t");
    const std::int64_t elements = cells * ncomp_;
    if (static_cast<std::uint64_t>(elements) > std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("pops::Fab: element count exceeds size_t");
    size_ = static_cast<std::size_t>(elements);
  }

  static Box<Dim> grown_with_ghosts(const Box<Dim>& valid, const Extent<Dim>& ghosts) {
    for (int axis = 0; axis < Dim; ++axis)
      if (ghosts[axis] < 0)
        throw std::invalid_argument("pops::Fab: ghost extents must be non-negative");
    if (valid.empty())
      return valid;

    Box<Dim> result = valid;
    for (int axis = 0; axis < Dim; ++axis) {
      if (ghosts[axis] >
          static_cast<std::int64_t>(std::numeric_limits<int>::max()) - valid.hi[axis])
        throw std::overflow_error("pops::Fab: ghost growth upper bound overflow");
      if (ghosts[axis] >
          static_cast<std::int64_t>(valid.lo[axis]) - std::numeric_limits<int>::min())
        throw std::overflow_error("pops::Fab: ghost growth lower bound overflow");
      result.lo[axis] =
          detail::checked_box_index(static_cast<std::int64_t>(valid.lo[axis]) - ghosts[axis],
                                    "pops::Fab: ghost growth lower bound overflow");
      result.hi[axis] =
          detail::checked_box_index(static_cast<std::int64_t>(valid.hi[axis]) + ghosts[axis],
                                    "pops::Fab: ghost growth upper bound overflow");
    }
    return result;
  }

  Box<Dim> valid_{};
  Box<Dim> grown_{};
  int ncomp_{0};
  Extent<Dim> ghosts_{};
  std::int64_t strides_[Dim]{};
  std::int64_t component_stride_{0};
  std::size_t size_{0};
  storage_type data_{};
  std::uint64_t generation_{1};
};

}  // namespace pops
