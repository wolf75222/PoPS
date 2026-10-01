#pragma once

#include <pops/runtime/program/accepted_exchange.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <Kokkos_Core.hpp>
#include <bit>
#include <algorithm>
#include <cstdint>
#include <exception>
#include <limits>
#include <span>
#include <string>
#include <utility>

namespace pops::runtime::program {

inline std::size_t interaction_add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::overflow_error("direct spatial interaction workspace overflow");
  return a + b;
}
inline std::size_t interaction_product(std::size_t a, std::size_t b) {
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
    throw std::overflow_error("direct spatial interaction workspace overflow");
  return a * b;
}
template <class Function>
void interaction_phase(const ExecutionLane& lane, Function&& function) {
  std::exception_ptr error;
  try { function(); } catch (...) { error = std::current_exception(); }
  if (all_reduce_max(error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && error) std::rethrow_exception(error);
    throw std::runtime_error("direct spatial interaction failed collectively");
  }
}

/// One physical contributor; transport exact binary64 words (including -0).
/// No floating SUM normalization or dynamic rank buffer is introduced.
inline Real interaction_owner_value(Real value, bool owner, const ExecutionLane& lane) {
  const auto bits = owner ? std::bit_cast<std::uint64_t>(value) : std::uint64_t{0};
  std::uint64_t received = 0;
  for (unsigned shift = 0; shift < 64; shift += 16)
    received |= static_cast<std::uint64_t>(all_reduce_sum(static_cast<long>((bits >> shift) & 0xffffU), lane)) << shift;
  return std::bit_cast<Real>(received);
}

/// Read a binary64 cell without a floating reduction's +0 normalization.
template <int Dim, class View>
Real interaction_cell_value(View values, const Index<Dim>& cell, int component) {
  std::uint64_t bits = 0;
  for (unsigned shift = 0; shift < 64; shift += 16) {
    const Real word = for_each_cell_reduce_sum(Box<Dim>(cell, cell), [=] POPS_HD(const Index<Dim>& index) {
      return static_cast<Real>((std::bit_cast<std::uint64_t>(values(index, component)) >> shift) & 0xffffU);
    });
    bits |= static_cast<std::uint64_t>(word) << shift;
  }
  return std::bit_cast<Real>(bits);
}

inline void interaction_append_point(ExactContractBuilder& exact,
                                     const runtime::multiblock::BoundaryEvaluationPoint& point) {
  if (point.clock.empty() || point.tick < 0 || point.level < 0 || point.substep < 0 ||
      point.stage < 0 || !std::isfinite(point.dt) || !(point.dt > 0) ||
      !std::isfinite(point.physical_time))
    throw std::invalid_argument("direct interaction has no issued runtime frame");
  exact.text(point.clock).scalar(point.tick).scalar(point.level).scalar(point.substep)
      .scalar(point.stage).scalar(point.stage_fraction.numerator).scalar(point.stage_fraction.denominator)
      .scalar(point.dt).scalar(point.physical_time).text(point.graph_identity)
      .text(point.rate_identity).text(point.application_identity);
}

template <int Dim, class MemorySpace>
struct InteractionLevelView {
  const MultiFab<Dim, MemorySpace>* field;
  const MultiFab<Dim, MemorySpace>* active;
  const MultiFab<Dim, MemorySpace>* coverage;
  const MultiFab<Dim, MemorySpace>* kappa;
  Geometry<Dim> geometry;
};

/// Direct midpoint quadrature over the exact physical owner quotient. There is
/// no N-by-N matrix, implicit periodic image, symmetry or positivity assumption.
/// Each source cell is broadcast as scalar words after its owner vote; hence
/// transport has no MPI int-count ceiling and empty ranks follow every phase.
/// Kernel and output are entirely provisional; this function never publishes.
template <int Dim, class MemorySpace, class Kernel>
MultiFab<Dim, MemorySpace> direct_spatial_interaction(
    std::span<const InteractionLevelView<Dim, MemorySpace>> levels, std::size_t target_level,
    std::span<const int> components, std::uint64_t max_bytes, std::string_view identity,
    const ExecutionLane& lane, Kernel kernel) {
  using Field = MultiFab<Dim, MemorySpace>;
  using Snapshot = Kokkos::View<Real**, Kokkos::LayoutRight, MemorySpace>;
  std::string contract;
  interaction_phase(lane, [&] {
    if (levels.empty() || target_level >= levels.size() || components.empty() ||
        components.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        max_bytes == 0 || identity.empty())
      throw std::invalid_argument("direct spatial interaction has incomplete authority");
    ExactContractBuilder exact;
    exact.text("pops.direct-spatial-interaction@1").text(identity).scalar(max_bytes)
        .scalar(levels.size()).scalar(target_level).scalar(components.size());
    for (int component : components) exact.scalar(component);
    for (std::size_t i = 0; i < components.size(); ++i)
      for (std::size_t j = 0; j < i; ++j)
        if (components[i] == components[j]) throw std::invalid_argument("direct interaction selects a component twice");
    for (const auto& level : levels) {
      if (!level.field) throw std::invalid_argument("direct interaction has a missing source level");
      const auto& field = *level.field;
      exact.scalar(field.ncomp()).scalar(field.layout().size()).scalar(field.distribution().replicated());
      const auto& ranks = field.distribution().rank_space();
      if (ranks.size() != static_cast<std::size_t>(lane.size()) ||
          ranks.linear_rank(field.local_rank()) != static_cast<std::size_t>(lane.rank()))
        throw std::invalid_argument("direct interaction distribution uses a foreign execution lane");
      for (int axis = 0; axis < Dim; ++axis)
        exact.scalar(level.geometry.domain().lo[axis]).scalar(level.geometry.domain().hi[axis])
            .scalar(level.geometry.lower()[axis]).scalar(level.geometry.upper()[axis])
            .scalar(ranks.origin()[axis]).scalar(ranks.extent()[axis]);
      for (std::size_t global = 0; global < field.layout().size(); ++global) {
        const auto box = field.layout()[global];
        if (box.empty() || box.intersect(level.geometry.domain()) != box)
          throw std::invalid_argument("direct interaction valid box lies outside its physical domain");
        for (std::size_t prior = 0; prior < global; ++prior)
          if (!box.intersect(field.layout()[prior]).empty())
            throw std::invalid_argument("direct interaction source boxes overlap");
        for (int axis = 0; axis < Dim; ++axis) {
          exact.scalar(box.lo[axis]).scalar(box.hi[axis]);
          if (!field.distribution().replicated()) exact.scalar(field.distribution().owner(global)[axis]);
        }
      }
      for (int component : components)
        if (component < 0 || component >= field.ncomp())
          throw std::invalid_argument("direct interaction source component is out of range");
      for (const Field* mask : {level.active, level.coverage, level.kappa}) {
        exact.presence(mask != nullptr);
        if (mask && (mask->ncomp() != 1 || mask->layout() != field.layout() ||
                     mask->distribution() != field.distribution() || mask->local_rank() != field.local_rank()))
          throw std::invalid_argument("direct interaction mask has foreign topology/ownership");
      }
    }
    contract = std::move(exact).release();
  });
  if (!all_ranks_agree_exact_ordered_byte_pairs({{"spatial-interaction", contract}}, lane))
    throw std::invalid_argument("direct spatial interaction authorities differ across ranks");

  // All ranks walk the same sealed boxes. Ownership is proven even for masked
  // cells; a replicated carrier contributes on lane rank zero only.
  auto visit = [&](auto&& consumer) {
    for (const auto& level : levels) {
      const auto& field = *level.field;
      Real volume = 1;
      interaction_phase(lane, [&] {
        for (int axis = 0; axis < Dim; ++axis) {
          const Real spacing = level.geometry.spacing(axis);
          if (!std::isfinite(spacing) || !(spacing > 0))
            throw std::invalid_argument("direct interaction has invalid physical cell spacing");
          volume *= spacing;
        }
        if (!std::isfinite(volume) || !(volume > 0))
          throw std::invalid_argument("direct interaction has invalid physical cell volume");
      });
      for (std::size_t global = 0; global < field.layout().size(); ++global) {
        const auto box = field.layout()[global];
        for (std::size_t linear = 0; linear < static_cast<std::size_t>(box.numPts()); ++linear) {
          Index<Dim> cell{};
          auto remainder = linear;
          for (int axis = 0; axis < Dim; ++axis) {
            const auto extent = static_cast<std::size_t>(box.length(axis));
            cell[axis] = static_cast<int>(static_cast<std::int64_t>(box.lo[axis]) + remainder % extent);
            remainder /= extent;
          }
          const bool resident = field.contains_local(global);
          const bool owner = resident && (!field.distribution().replicated() || lane.rank() == 0);
          Real measure = 0;
          interaction_phase(lane, [&] {
            if (!resident) return;
            const auto active = level.active ? level.active->fab_global(global).view() : FieldView<const Real, Dim>{};
            const auto coverage = level.coverage ? level.coverage->fab_global(global).view() : FieldView<const Real, Dim>{};
            const auto kappa = level.kappa ? level.kappa->fab_global(global).view() : FieldView<const Real, Dim>{};
            const bool has_active = level.active, has_coverage = level.coverage, has_kappa = level.kappa;
            measure = for_each_cell_reduce_sum(Box<Dim>(cell, cell), [=] POPS_HD(const Index<Dim>& index) {
              const Real a = has_active ? active(index, 0) : Real(1);
              const Real c = has_coverage ? coverage(index, 0) : Real(1);
              const Real k = has_kappa ? kappa(index, 0) : Real(1);
              if (!Kokkos::isfinite(a) || !Kokkos::isfinite(c) || !Kokkos::isfinite(k) ||
                  (a != 0 && a != 1) || (c != 0 && c != 1) || k < 0 || k > 1)
                return std::numeric_limits<Real>::quiet_NaN();
              return a == 1 && c == 1 ? volume * k : Real(0);
            });
            if (!std::isfinite(measure)) throw std::invalid_argument("direct interaction has invalid active/coverage/kappa geometry");
          });
          const long owners = all_reduce_sum(owner ? 1L : 0L, lane);
          interaction_phase(lane, [&] {
            if (owners != 1) throw std::invalid_argument("direct interaction physical cell has missing/duplicate owner");
          });
          // Authenticate every replicated geometry mask independently. Equal
          // products (especially an excluded cell's zero measure) do not prove
          // equal kappa/coverage/active carriers.
          if (field.distribution().replicated()) {
            for (const Field* mask : {level.active, level.coverage, level.kappa}) {
              if (!mask) continue;
              Real local_mask = 0;
              interaction_phase(lane, [&] {
                local_mask = interaction_cell_value<Dim>(mask->fab_global(global).view(), cell, 0);
              });
              const Real owner_mask = interaction_owner_value(local_mask, owner, lane);
              interaction_phase(lane, [&] {
                if (std::bit_cast<std::uint64_t>(local_mask) != std::bit_cast<std::uint64_t>(owner_mask))
                  throw std::invalid_argument("direct interaction replicated geometry masks differ");
              });
            }
          }
          const Real local_measure = measure;
          measure = interaction_owner_value(measure, owner, lane);
          interaction_phase(lane, [&] {
            if (!std::isfinite(measure) || measure < 0)
              throw std::invalid_argument("direct interaction physical cell has invalid measure");
            if (field.distribution().replicated() && std::bit_cast<std::uint64_t>(local_measure) != std::bit_cast<std::uint64_t>(measure))
              throw std::invalid_argument("direct interaction replicated physical measures differ");
          });
          if (measure > 0) consumer(level, global, cell, owner, measure);
        }
      }
    }
  };
  std::size_t count = 0;
  visit([&](const auto&, auto, const auto&, bool, Real) { interaction_phase(lane, [&] { count = interaction_add(count, 1); }); });
  const std::size_t columns = interaction_add(Dim + 1, components.size());
  const auto& target = levels[target_level];
  const auto& prototype = *target.field;
  interaction_phase(lane, [&] {
    // Count explicit host+device snapshot, target values, Fab/index vectors,
    // level-view descriptors and component vector. Contract/control metadata,
    // borrowed source/masks and pre-existing runtime caches are not capped here.
    std::size_t bytes = interaction_product(interaction_product(count, columns), 2 * sizeof(Real));
    bytes = interaction_add(bytes, interaction_product(levels.size(), sizeof(InteractionLevelView<Dim, MemorySpace>)));
    bytes = interaction_add(bytes, interaction_product(components.size(), sizeof(int)));
    bytes = interaction_add(bytes, interaction_product(prototype.layout().size(), sizeof(std::size_t) + 2 * sizeof(Box<Dim>) + sizeof(Index<Dim>)));
    bytes = interaction_add(bytes, interaction_product(prototype.local_size(), sizeof(typename Field::fab_type) + sizeof(std::size_t)));
    for (std::size_t local = 0; local < prototype.local_size(); ++local)
      bytes = interaction_add(bytes, interaction_product(interaction_product(static_cast<std::size_t>(prototype.box(local).numPts()), components.size()), sizeof(Real)));
    if (bytes > max_bytes) throw std::length_error("direct spatial interaction workspace budget exceeded");
  });
  Snapshot snapshot;
  decltype(Kokkos::create_mirror(std::declval<Snapshot>())) host;
  Field output;
  interaction_phase(lane, [&] {
    snapshot = Snapshot(Kokkos::view_alloc(Kokkos::WithoutInitializing, "spatial interaction source"), count, columns);
    host = Kokkos::create_mirror(snapshot);
    output = Field(prototype.layout(), prototype.distribution(), prototype.local_rank(), static_cast<int>(components.size()), Extent<Dim>{});
  });
  std::size_t row = 0;
  visit([&](const auto& level, std::size_t global, const auto& cell, bool owner, Real measure) {
    interaction_phase(lane, [&] {
      if (row >= count) throw std::logic_error("direct interaction active geometry changed during snapshot");
      const auto center = level.geometry.cell_center(cell);
      for (int axis = 0; axis < Dim; ++axis) host(row, axis) = center[axis];
      host(row, Dim) = measure;
    });
    for (std::size_t selected = 0; selected < components.size(); ++selected) {
      Real value = 0;
      interaction_phase(lane, [&] {
        if (!level.field->contains_local(global)) return;
        const auto values = level.field->fab_global(global).view();
        const int component = components[selected];
        value = interaction_cell_value<Dim>(values, cell, component);
        if (!std::isfinite(value)) throw std::overflow_error("direct interaction has nonfinite active source");
      });
      const Real local_value = value;
      value = interaction_owner_value(value, owner, lane);
      interaction_phase(lane, [&] {
        if (!std::isfinite(value)) throw std::overflow_error("direct interaction source transport overflow");
        if (level.field->distribution().replicated() && std::bit_cast<std::uint64_t>(local_value) != std::bit_cast<std::uint64_t>(value))
          throw std::invalid_argument("direct interaction replicated source values differ");
        host(row, Dim + 1 + selected) = value;
      });
    }
    ++row;
  });
  interaction_phase(lane, [&] {
    if (row != count) throw std::logic_error("direct interaction active geometry changed during snapshot");
    Kokkos::deep_copy(snapshot, host);
  });
  Real invalid = 0;
  interaction_phase(lane, [&] {
    const auto geometry = target.geometry;
    const int width = static_cast<int>(components.size());
    for (std::size_t local = 0; local < output.local_size(); ++local) {
      const auto values = output.fab(local).view();
      const auto active = target.active ? target.active->fab(local).view() : FieldView<const Real, Dim>{};
      const auto coverage = target.coverage ? target.coverage->fab(local).view() : FieldView<const Real, Dim>{};
      const auto kappa = target.kappa ? target.kappa->fab(local).view() : FieldView<const Real, Dim>{};
      const bool has_active = target.active, has_coverage = target.coverage, has_kappa = target.kappa;
      invalid = std::max(invalid, for_each_cell_reduce_max(output.box(local), [=] POPS_HD(const Index<Dim>& cell) {
        const auto x = geometry.cell_center(cell);
        const bool included = (!has_active || active(cell, 0) == 1) && (!has_coverage || coverage(cell, 0) == 1) && (!has_kappa || kappa(cell, 0) > 0);
        Real failure = 0;
        for (int component = 0; component < width; ++component) {
          FiniteCompensatedSum sum;
          if (included) for (std::size_t source = 0; source < count; ++source) {
            RealVector<Dim> y{};
            for (int axis = 0; axis < Dim; ++axis) y[axis] = snapshot(source, axis);
            const Real weight = kernel(x, y);
            sum.add((weight * snapshot(source, Dim)) * snapshot(source, Dim + 1 + component));
          }
          values(cell, component) = sum.value();
          if (!sum.finite()) failure = 1;
        }
        return failure;
      }));
    }
    if (invalid != 0) throw std::overflow_error("direct interaction has nonfinite kernel/product/sum");
  });
  return output;
}
}  // namespace pops::runtime::program
