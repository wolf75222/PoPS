#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "pops/mesh/geometry/geometry.hpp"
#include "pops/mesh/storage/multifab.hpp"
#include "pops/runtime/program/prepared_resource_lifetime.hpp"

namespace pops::runtime::program {

template <int Dim, class MemorySpace>
class AmrProgramContext;

/// A borrowed, revocable representation view. Geometry belongs to the retained
/// source/candidate snapshot, never to whichever level happens to be active.
template <int Dim>
class FieldTopologyView {
 public:
  using Field = MultiFab<Dim>;
  enum class Side : std::uint8_t { Source, Target };

  FieldTopologyView() = default;
  void require() const {
    if (!data_ || !data_->attempt.visible())
      throw std::logic_error("Field topology view is absent or its regrid attempt was revoked");
  }
  void require_field(const Field& field) const {
    require();
    if (field.layout() != data_->layout || field.distribution() != data_->distribution ||
        field.local_rank() != data_->local_rank || field.ghosts() != data_->ghosts ||
        field.ncomp() != data_->components)
      throw std::invalid_argument("Field differs from its exact topology representation");
  }
  bool same_view(const FieldTopologyView& other) const noexcept {
    return data_ && data_ == other.data_;
  }
  const Geometry<Dim>& geometry() const {
    require();
    return data_->geometry;
  }
  const typename Field::layout_type& layout() const {
    require();
    return data_->layout;
  }
  const typename Field::distribution_type& distribution() const {
    require();
    return data_->distribution;
  }
  const typename Field::rank_type& local_rank() const {
    require();
    return data_->local_rank;
  }
  const typename Field::ghost_type& ghosts() const {
    require();
    return data_->ghosts;
  }
  int components() const {
    require();
    return data_->components;
  }
  int level() const {
    require();
    return data_->level;
  }
  int program_block() const {
    require();
    return data_->program_block;
  }
  std::string_view field_provider() const {
    require();
    return data_->field_provider;
  }
  std::uint64_t topology_epoch() const {
    require();
    return data_->topology_epoch;
  }
  std::uint64_t materialization_generation() const {
    require();
    return data_->materialization_generation;
  }
  Side side() const {
    require();
    return data_->side;
  }
  const Field* active_mask() const {
    require();
    return data_->active.get();
  }
  const Field* volume_fraction() const {
    require();
    return data_->fraction.get();
  }
  const Field& coverage_mask() const {
    require();
    return *data_->coverage;
  }

 private:
  template <int, class>
  friend class AmrProgramContext;
  struct Data {
    PreparedResourceAttempt attempt;
    int level, program_block;
    std::uint64_t topology_epoch, materialization_generation;
    Side side;
    Geometry<Dim> geometry;
    typename Field::layout_type layout;
    typename Field::distribution_type distribution;
    typename Field::rank_type local_rank;
    typename Field::ghost_type ghosts;
    int components;
    std::shared_ptr<const Field> active, fraction, coverage;
    std::string field_provider;
  };
  explicit FieldTopologyView(std::shared_ptr<const Data> data) : data_(std::move(data)) {}
  std::shared_ptr<const Data> data_;
};

}  // namespace pops::runtime::program
