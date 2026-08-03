/// @file
/// @brief Axis-indexed owning and non-owning face fields for compile-time dimensions.

#pragma once

#include <pops/mesh/storage/fab.hpp>

#include <array>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace pops::nd {

template <int Dim>
Box<Dim> face_box(const Box<Dim>& cells, int axis) {
  static_assert(Dim >= 1 && Dim <= 3, "ND face boxes support dimensions 1..3");
  if (axis < 0 || axis >= Dim)
    throw std::invalid_argument("ND face-box axis is outside the compile-time dimension");
  if (cells.empty())
    return cells;
  if (cells.hi[axis] == std::numeric_limits<int>::max())
    throw std::overflow_error("ND face-box upper bound exceeds the signed index range");
  Box<Dim> result = cells;
  ++result.hi[axis];
  return result;
}

template <int Axis, int Dim>
Box<Dim> face_box(const Box<Dim>& cells) {
  static_assert(Axis >= 0 && Axis < Dim, "ND face-box axis is outside the compile-time dimension");
  return face_box(cells, Axis);
}

template <class T, int Dim>
struct FaceFieldView {
  static_assert(Dim >= 1 && Dim <= 3, "ND face fields support dimensions 1..3");

  static constexpr int dimension = Dim;
  FieldView<T, Dim> axes[Dim]{};
  Box<Dim> cells{};
  int ncomp = 0;

  template <int Axis>
  POPS_HD T& operator()(const Index<Dim>& face, int component = 0) const {
    static_assert(Axis >= 0 && Axis < Dim,
                  "ND face-field axis is outside the compile-time dimension");
    return axes[Axis](face, component);
  }

  template <int Axis>
  POPS_HD const FieldView<T, Dim>& axis() const {
    static_assert(Axis >= 0 && Axis < Dim,
                  "ND face-field axis is outside the compile-time dimension");
    return axes[Axis];
  }
};

/// One component-slowest Fab per logical face direction.  The field is an owning preparation
/// object; kernels capture only the trivially copyable FaceFieldView returned by view().
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class FaceField {
 public:
  static_assert(Dim >= 1 && Dim <= 3, "ND face fields support dimensions 1..3");

  using memory_space = MemorySpace;
  using FabType = Fab<Dim, MemorySpace>;

  FaceField() = default;

  FaceField(const Box<Dim>& cells, int ncomp) : cells_(cells), ncomp_(ncomp) {
    if (ncomp < 1)
      throw std::invalid_argument("ND face fields require a positive component count");
    for (int axis = 0; axis < Dim; ++axis)
      faces_[static_cast<std::size_t>(axis)] = FabType(face_box(cells, axis), ncomp);
  }

  const Box<Dim>& cell_box() const { return cells_; }
  int ncomp() const { return ncomp_; }

  template <int Axis>
  FabType& field() {
    static_assert(Axis >= 0 && Axis < Dim,
                  "ND face-field axis is outside the compile-time dimension");
    return faces_[static_cast<std::size_t>(Axis)];
  }

  template <int Axis>
  const FabType& field() const {
    static_assert(Axis >= 0 && Axis < Dim,
                  "ND face-field axis is outside the compile-time dimension");
    return faces_[static_cast<std::size_t>(Axis)];
  }

  FaceFieldView<Real, Dim> view() {
    FaceFieldView<Real, Dim> result{};
    result.cells = cells_;
    result.ncomp = ncomp_;
    for (int axis = 0; axis < Dim; ++axis)
      result.axes[axis] = faces_[static_cast<std::size_t>(axis)].view();
    return result;
  }

  FaceFieldView<const Real, Dim> view() const {
    FaceFieldView<const Real, Dim> result{};
    result.cells = cells_;
    result.ncomp = ncomp_;
    for (int axis = 0; axis < Dim; ++axis)
      result.axes[axis] = faces_[static_cast<std::size_t>(axis)].view();
    return result;
  }

  void set_val(Real value) {
    for (auto& face : faces_)
      face.set_val(value);
  }

 private:
  Box<Dim> cells_{};
  int ncomp_ = 0;
  std::array<FabType, Dim> faces_{};
};

static_assert(std::is_trivially_copyable_v<FaceFieldView<Real, 1>>);
static_assert(std::is_trivially_copyable_v<FaceFieldView<Real, 2>>);
static_assert(std::is_trivially_copyable_v<FaceFieldView<Real, 3>>);

}  // namespace pops::nd
