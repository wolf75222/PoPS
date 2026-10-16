/// Explicit host publication of a completed native field/face product.
#pragma once

#include <pops/numerics/spatial/nd/face_field.hpp>
#include <optional>
#include <utility>

namespace pops::runtime::program {
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class HostFabPublication {
  using FabType = Fab<Dim, MemorySpace>;
  using HostStorage = typename FabType::raw_host_mirror_type;
  HostStorage storage_;
  FieldView<const Real, Dim> view_{};

 public:
  explicit HostFabPublication(const FabType& field)
      : storage_(Kokkos::create_mirror(field.storage())), view_(field.view()) {
    Kokkos::deep_copy(storage_, field.storage());
    view_.data = storage_.data();
  }
  auto view() const { return view_; }
};

template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class HostFacePublication {
  std::array<std::optional<HostFabPublication<Dim, MemorySpace>>, Dim> axes_;
  nd::FaceFieldView<const Real, Dim> view_{};
  template <std::size_t... Axis>
  void copy_(const nd::FaceField<Dim, MemorySpace>& faces, std::index_sequence<Axis...>) {
    (axes_[Axis].emplace(faces.template field<static_cast<int>(Axis)>()), ...);
    ((view_.axes[Axis] = axes_[Axis]->view()), ...);
  }

 public:
  explicit HostFacePublication(const nd::FaceField<Dim, MemorySpace>& faces) {
    view_.cells = faces.cell_box();
    view_.ncomp = faces.ncomp();
    copy_(faces, std::make_index_sequence<Dim>{});
  }
  auto view() const { return view_; }
};
}  // namespace pops::runtime::program
