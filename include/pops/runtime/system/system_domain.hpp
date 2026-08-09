/// @file
/// @brief Exact compile-time-ranked Cartesian geometry and ownership of one System.

#pragma once

#include <pops/core/state/state.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/rank_space.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/parallel/prepared_load_balance.hpp>
#include <pops/runtime/system.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace pops::runtime::system {

/// Immutable Cartesian spatial authority shared by every block of one System.
///
/// The resolved Python layout supplies every ranked value. Ownership is materialized once through
/// the prepared load-balance authority and then retained with the fields; no consumer reconstructs
/// owners from process count or array shape. Polar and embedded-boundary implementations are
/// capability-qualified providers and therefore deliberately absent from this core type.
template <int Dim>
struct SystemDomain {
  static_assert(Dim >= 1 && Dim <= 3, "SystemDomain only supports dimensions 1, 2, and 3");

  using field_type = MultiFab<Dim>;
  using layout_type = mesh::BoxArray<Dim>;
  using distribution_type = mesh::Distribution<Dim>;
  using rank_space_type = mesh::RankSpace<Dim>;

  SystemConfig<Dim> cfg;
  Box<Dim> dom;
  Geometry<Dim> geom;
  layout_type ba;
  rank_space_type rank_space;
  std::shared_ptr<const PreparedLoadBalanceAuthority<Dim>> load_balance;
  distribution_type dm;
  Index<Dim> local_rank;
  std::array<bool, Dim> periodicity;
  /// Legacy field/embedded-boundary workspace is intentionally unallocated.  Provider values are
  /// owned by the exact auxiliary storage groups; a feature that still reaches this obsolete route
  /// fails on its own typed migration rather than silently reserving a physical component.
  field_type aux;
  int aux_ncomp = 0;

  explicit SystemDomain(const SystemConfig<Dim>& config)
      : cfg(validated_config_(config)),
        dom(cfg.index_domain()),
        geom(Geometry<Dim>::from_bounds(dom, cfg.lower, cfg.upper)),
        ba(cfg.materialized_boxes()),
        rank_space(process_rank_space_()),
        load_balance(prepare_authority_(cfg)),
        dm(prepare_distribution_(*load_balance, ba, rank_space)),
        local_rank(rank_space.coordinate(static_cast<std::size_t>(my_rank()))),
        periodicity(cfg.periodicity) {}

  struct LayoutReport {
    Extent<Dim> shape{};
    RealVector<Dim> lower{};
    RealVector<Dim> upper{};
    std::array<bool, Dim> periodicity{};
    std::size_t boxes = 0;
    int aux_components = 0;
    std::string coordinate_system;
  };

  LayoutReport layout_report() const {
    return {
        cfg.shape, cfg.lower, cfg.upper, periodicity, ba.size(), aux_ncomp, cfg.coordinate_system};
  }

 private:
  static SystemConfig<Dim> validated_config_(SystemConfig<Dim> config) {
    config.validate_spatial_domain();
    if (config.coordinate_system != runtime_config_detail::cartesian_coordinate_system<Dim>())
      throw std::invalid_argument(
          "System Cartesian core requires the exact ranked Cartesian coordinate provider");
    return config;
  }

  static rank_space_type process_rank_space_() {
    Extent<Dim> process_shape = runtime_config_detail::filled_extent<Dim>(1);
    process_shape[0] = n_ranks();
    return rank_space_type(Index<Dim>{}, process_shape);
  }

  static std::shared_ptr<const PreparedLoadBalanceAuthority<Dim>> prepare_authority_(
      const SystemConfig<Dim>& config) {
    return std::make_shared<const PreparedLoadBalanceAuthority<Dim>>(
        prepare_load_balance_authority<Dim>(config.load_balance_route, config.load_balance_identity,
                                            config.load_balance_options));
  }

  static distribution_type prepare_distribution_(const PreparedLoadBalanceAuthority<Dim>& authority,
                                                 const layout_type& layout,
                                                 const rank_space_type& processes) {
    std::int64_t cells = 0;
    for (const Box<Dim>& box : layout.boxes()) {
      const std::int64_t box_cells = box.numPts();
      if (box_cells < 1 || box_cells > std::numeric_limits<std::int64_t>::max() - cells)
        throw std::overflow_error("System ownership preparation cell budget exceeds int64_t");
      cells += box_cells;
    }
    const parallel::LoadBalancePreparationBudget budget{layout.size(), processes.size(), cells};
    return authority.prepare(layout, processes, budget).plan().distribution();
  }
};

}  // namespace pops::runtime::system
