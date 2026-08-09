#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace_builtins.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace_prepare.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace_provider.hpp>
#include <pops/parallel/comm.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <source_location>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace pops;

namespace {

constexpr std::string_view kExactReplicaValidationFailure =
    "replicated vector values failed exact collective validation";

template <int Dim>
Extent<Dim> filled_extent(int value) {
  Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim>
Index<Dim> rank_coordinate(int rank) {
  Index<Dim> result{};
  result[0] = rank;
  return result;
}

template <int Dim>
mesh::RankSpace<Dim> two_rank_space() {
  Extent<Dim> extent = filled_extent<Dim>(1);
  extent[0] = 2;
  return mesh::RankSpace<Dim>{Index<Dim>{}, extent};
}

template <int Dim>
mesh::BoxArray<Dim> two_patch_layout(bool drift_second_patch = false) {
  Index<Dim> first_lo{};
  Index<Dim> first_hi{};
  Index<Dim> second_lo{};
  Index<Dim> second_hi{};
  first_hi[0] = 1;
  second_lo[0] = 2;
  second_hi[0] = drift_second_patch ? 4 : 3;
  for (int axis = 1; axis < Dim; ++axis) {
    first_hi[axis] = 1;
    second_hi[axis] = 1;
  }
  return mesh::BoxArray<Dim>(
      std::vector<Box<Dim>>{Box<Dim>{first_lo, first_hi}, Box<Dim>{second_lo, second_hi}});
}

template <int Dim>
std::shared_ptr<MultiFab<Dim>> make_distributed_field(bool drift_layout = false) {
  const mesh::BoxArray<Dim> layout = two_patch_layout<Dim>(drift_layout);
  const mesh::RankSpace<Dim> ranks = two_rank_space<Dim>();
  const mesh::Distribution<Dim> distribution = mesh::Distribution<Dim>::partitioned(
      layout, ranks, {rank_coordinate<Dim>(0), rank_coordinate<Dim>(1)});
  auto result = std::make_shared<MultiFab<Dim>>(layout, distribution,
                                                rank_coordinate<Dim>(my_rank()), 1, Extent<Dim>{});
  result->set_val(Real(0));
  return result;
}

template <int Dim>
std::shared_ptr<MultiFab<Dim>> make_replicated_field(Real value) {
  const mesh::BoxArray<Dim> layout = two_patch_layout<Dim>();
  const mesh::RankSpace<Dim> ranks = two_rank_space<Dim>();
  auto result =
      std::make_shared<MultiFab<Dim>>(layout, mesh::Distribution<Dim>::replicated(layout, ranks),
                                      rank_coordinate<Dim>(my_rank()), 1, Extent<Dim>{});
  result->set_val(value);
  return result;
}

template <int Dim>
std::shared_ptr<MultiFab<Dim>> make_rank_zero_distributed_field(Real value) {
  Index<Dim> lo{};
  Index<Dim> hi{};
  hi[0] = 3;
  for (int axis = 1; axis < Dim; ++axis)
    hi[axis] = 1;
  const mesh::BoxArray<Dim> layout(std::vector<Box<Dim>>{Box<Dim>{lo, hi}});
  const mesh::RankSpace<Dim> ranks = two_rank_space<Dim>();
  const mesh::Distribution<Dim> distribution =
      mesh::Distribution<Dim>::partitioned(layout, ranks, {rank_coordinate<Dim>(0)});
  auto result = std::make_shared<MultiFab<Dim>>(layout, distribution,
                                                rank_coordinate<Dim>(my_rank()), 1, Extent<Dim>{});
  result->set_val(value);
  return result;
}

template <int Dim>
struct MeanZeroByAxisZero {
  FieldView<Real, Dim> values;

  POPS_HD void operator()(const Index<Dim>& index) const {
    values(index, 0) = index[0] % 2 == 0 ? Real(1) : Real(-1);
  }
};

template <int Dim>
struct LabelByAxisZero {
  FieldView<Real, Dim> values;

  POPS_HD void operator()(const Index<Dim>& index) const {
    values(index, 0) = index[0] < 2 ? Real(1) : Real(2);
  }
};

template <int Dim>
struct MarkAxisZeroSlab {
  FieldView<Real, Dim> values;
  int slab;

  POPS_HD void operator()(const Index<Dim>& index) const {
    if (index[0] == slab)
      values(index, 0) = Real(1);
  }
};

template <int Dim>
struct AbsoluteError {
  FieldView<const Real, Dim> values;
  Real expected;

  POPS_HD Real operator()(const Index<Dim>& index) const {
    const Real error = values(index, 0) - expected;
    return error < Real(0) ? -error : error;
  }
};

template <int Dim>
void set_mean_zero_pattern(MultiFab<Dim>& field) {
  for (std::size_t local = 0; local < field.local_size(); ++local)
    for_each_cell(field.box(local), MeanZeroByAxisZero<Dim>{field.fab(local).view()});
}

template <int Dim>
void set_component_labels(MultiFab<Dim>& field) {
  for (std::size_t local = 0; local < field.local_size(); ++local)
    for_each_cell(field.box(local), LabelByAxisZero<Dim>{field.fab(local).view()});
}

template <int Dim>
void set_isometric_rank_permutation(MultiFab<Dim>& field) {
  field.set_val(Real(0));
  for (std::size_t local = 0; local < field.local_size(); ++local)
    for_each_cell(field.box(local), MarkAxisZeroSlab<Dim>{field.fab(local).view(), my_rank()});
}

template <int Dim>
double maximum_error(const MultiFab<Dim>& field, Real expected) {
  Real local_maximum = Real(0);
  for (std::size_t local = 0; local < field.local_size(); ++local)
    local_maximum =
        std::max(local_maximum,
                 for_each_cell_reduce_max(field.box(local),
                                          AbsoluteError<Dim>{field.fab(local).view(), expected}));
  return static_cast<double>(all_reduce_max(local_maximum));
}

template <class Operation>
bool uniformly_rejected(Operation&& operation,
                        std::string_view expected = "collective preflight rejected") {
  bool rejected = false;
  std::string message;
  try {
    operation();
  } catch (const std::runtime_error& error) {
    rejected = true;
    message = error.what();
  } catch (...) {
    message = "non-runtime exception";
  }
  const long rejected_ranks = all_reduce_sum(rejected ? 1L : 0L);
  const bool same_message = all_ranks_agree_exact_ordered_byte_pairs(
      {{std::string_view("field-nullspace-preflight-exception"), std::string_view(message)}});
  return rejected_ranks == n_ranks() && same_message && message.find(expected) != std::string::npos;
}

template <class Operation>
bool uniformly_accepted(Operation&& operation) {
  bool failed = false;
  std::string message;
  try {
    operation();
  } catch (const std::exception& error) {
    failed = true;
    message = error.what();
  } catch (...) {
    failed = true;
    message = "non-standard exception";
  }
  if (failed)
    std::cerr << "field-nullspace MPI accepted path failed: " << message << '\n';
  const long failed_ranks = all_reduce_sum(failed ? 1L : 0L);
  const bool same_message = all_ranks_agree_exact_ordered_byte_pairs(
      {{std::string_view("field-nullspace-accepted-exception"), std::string_view(message)}});
  return failed_ranks == 0 && same_message;
}

template <int Dim>
class OperatorFactsBlindProvider final : public FieldNullspaceProvider<Dim> {
 public:
  [[nodiscard]] std::string_view identity() const noexcept override {
    return "test.field-nullspace.operator-facts-blind";
  }
  [[nodiscard]] std::uint64_t interface_version() const noexcept override { return 1; }
  [[nodiscard]] std::string_view collective_contract() const noexcept override {
    return "test.field-nullspace.operator-facts-blind@1";
  }
  [[nodiscard]] PreparedProviderOptions default_options() const override {
    return {"test.field-nullspace.operator-facts-blind.options@1", {}};
  }
  [[nodiscard]] bool accepts_options(
      const PreparedProviderOptions& options) const noexcept override {
    return options.schema_identity == "test.field-nullspace.operator-facts-blind.options@1" &&
           options.values.empty();
  }
  [[nodiscard]] PreparedProviderSupport supports(
      const FieldNullspaceProviderRequest<Dim>&) const noexcept override {
    return PreparedProviderSupport::accept();
  }
  [[nodiscard]] std::string expected_prepared_contract(
      const FieldNullspaceProviderRequest<Dim>&) const override {
    ExactContractBuilder contract;
    contract.text("test.prepared-field-nullspace.operator-facts-blind")
        .scalar(std::uint32_t{1})
        .scalar(Dim);
    return std::move(contract).release();
  }
  [[nodiscard]] PreparedFieldNullspace<Dim> prepare(
      const FieldNullspaceProviderRequest<Dim>& request) const override {
    return {std::string(identity()), interface_version(), expected_prepared_contract(request), {}};
  }
};

template <int Dim, class Require>
void exercise_exact_ranked_preflight(Require&& require) {
  const int rank = my_rank();
  const auto field = make_distributed_field<Dim>();
  const std::array<PreparedVectorDistribution<Dim>, 1> distributed_level{
      PreparedVectorDistribution<Dim>::Distributed};
  const std::span<const PreparedVectorDistribution<Dim>> distributed_span(distributed_level);

  // Operator facts are authenticated before a provider callback, even when the provider ignores
  // those facts completely.
  {
    FieldNullspaceProviderRegistry<Dim> registry;
    registry.add(std::make_shared<OperatorFactsBlindProvider<Dim>>());
    const FieldNullspaceProviderSelection selection{
        "test.field-nullspace.operator-facts-blind",
        {"test.field-nullspace.operator-facts-blind.options@1", {}}};
    FieldNullspaceProviderRequest<Dim> request;
    request.plan_identity = "mpi-preflight:operator-facts-id-drift";
    request.operator_facts = make_field_nullspace_operator_facts(
        "mpi-preflight:boundary-set@1",
        {{rank == 0 ? "boundary:a" : "boundary:b",
          FieldBoundaryNullspaceBehavior::PreservesConstantMode}},
        false);
    require(uniformly_rejected(
        [&] { (void)prepare_field_nullspace_collectively<Dim>(registry, selection, request); },
        "operator facts differ across MPI ranks"));

    request.plan_identity = "mpi-preflight:operator-facts-behavior-drift";
    request.operator_facts = make_field_nullspace_operator_facts(
        "mpi-preflight:boundary-set@1",
        {{"boundary:a", rank == 0 ? FieldBoundaryNullspaceBehavior::PreservesConstantMode
                                  : FieldBoundaryNullspaceBehavior::ConstrainsConstantMode}},
        false);
    require(uniformly_rejected(
        [&] { (void)prepare_field_nullspace_collectively<Dim>(registry, selection, request); },
        "operator facts differ across MPI ranks"));
  }

  // A valid connected request is materialized through the same exact provider protocol in every
  // compile-time rank.
  {
    const auto registry = make_default_field_nullspace_provider_registry<Dim>();
    FieldNullspaceProviderRequest<Dim> request;
    request.plan_identity = "mpi-preflight:connected:" + std::to_string(Dim);
    request.operator_facts = make_field_nullspace_operator_facts(
        "mpi-preflight:boundaryless:" + std::to_string(Dim), {}, false);
    request.topology.identity = "mpi-preflight:layout:" + std::to_string(Dim);
    request.topology.exact_layout_contract =
        PreparedVectorDistribution<Dim>::Distributed.layout_contract(*field);
    request.topology.connected_component_contract =
        "mpi-preflight:connected-component:" + std::to_string(Dim);
    request.topology.layouts = {field.get()};
    request.topology.cell_measure = {Real(1)};
    request.topology.level_distributions = {PreparedVectorDistribution<Dim>::Distributed};
    PreparedFieldNullspace<Dim> prepared;
    require(uniformly_accepted([&] {
      prepared = prepare_field_nullspace_collectively<Dim>(
          *registry, operator_topology_zero_mean_nullspace(), request);
    }));
    require(prepared.plan.bases.size() == 1U && prepared.plan.gauges.size() == 1U);
  }

  // Replicated exact-rank fields and masks contribute once globally, independent of rank count.
  {
    const auto replica = make_replicated_field<Dim>(Real(0));
    set_mean_zero_pattern(*replica);
    const auto mask = make_replicated_field<Dim>(Real(1));
    FieldNullspacePlan<Dim> plan =
        constant_mean_zero_nullspace<Dim>("replicated-nullspace", "mpi-preflight", Real(1));
    plan.bases[0].masks = {mask};
    const std::vector<const MultiFab<Dim>*> const_levels{replica.get()};
    const std::vector<MultiFab<Dim>*> mutable_levels{replica.get()};
    const std::array<PreparedVectorDistribution<Dim>, 1> distributions{
        PreparedVectorDistribution<Dim>::Replicated};
    const std::span<const PreparedVectorDistribution<Dim>> distribution_span(distributions);
    require(uniformly_accepted([&] {
      validate_field_nullspace_basis<Dim>(const_levels, plan, distribution_span);
      const std::vector<double> witness =
          require_field_nullspace_compatible<Dim>(const_levels, plan, distribution_span);
      const double cell_count = static_cast<double>(std::size_t{1} << (Dim + 1));
      if (witness.size() != 2U || witness[0] != 0.0 || witness[1] != cell_count)
        throw std::runtime_error("replicated exact-rank witness was double-counted");
      apply_field_gauge<Dim>(mutable_levels, plan, distribution_span);
    }));
  }

  // Labelled factories use the same replicated ownership contract without double-counting labels
  // or their materialized masks.
  {
    const auto labels = make_replicated_field<Dim>(Real(0));
    const auto rhs = make_replicated_field<Dim>(Real(0));
    set_component_labels(*labels);
    set_mean_zero_pattern(*rhs);
    const std::array<PreparedVectorDistribution<Dim>, 1> distributions{
        PreparedVectorDistribution<Dim>::Replicated};
    const std::span<const PreparedVectorDistribution<Dim>> distribution_span(distributions);
    require(uniformly_accepted([&] {
      const FieldNullspacePlan<Dim> plan = labelled_mean_zero_nullspace<Dim>(
          "replicated-components", "replicated-components-layout",
          std::vector<std::shared_ptr<const MultiFab<Dim>>>{labels},
          {{1, "left", "mpi:label:1"}, {2, "right", "mpi:label:2"}}, {}, {Real(1)}, 0,
          distribution_span);
      const std::vector<const MultiFab<Dim>*> levels{rhs.get()};
      const std::vector<double> witness =
          require_field_nullspace_compatible<Dim>(levels, plan, distribution_span);
      const double component_cells = static_cast<double>(std::size_t{1} << Dim);
      if (witness.size() != 4U || witness[0] != 0.0 || witness[1] != component_cells ||
          witness[2] != 0.0 || witness[3] != component_cells)
        throw std::runtime_error("replicated labelled witness was double-counted");
    }));
  }

  // Rank-local metadata drift is rejected coherently before scientific reductions.
  {
    const FieldNullspacePlan<Dim> plan =
        constant_mean_zero_nullspace<Dim>("nullspace", "mpi-preflight");
    const std::vector<const MultiFab<Dim>*> layouts{rank == 0 ? nullptr : field.get()};
    require(uniformly_rejected(
        [&] { validate_field_nullspace_basis<Dim>(layouts, plan, distributed_span); }));
  }
  {
    FieldNullspacePlan<Dim> plan = constant_mean_zero_nullspace<Dim>(
        rank == 0 ? "nullspace-rank-0" : "nullspace-rank-1", "mpi-preflight");
    const std::vector<const MultiFab<Dim>*> levels{field.get()};
    require(uniformly_rejected(
        [&] { (void)require_field_nullspace_compatible<Dim>(levels, plan, distributed_span); }));
  }
  {
    FieldNullspacePlan<Dim> plan = constant_mean_zero_nullspace<Dim>("nullspace", "mpi-preflight");
    if (rank == 0)
      plan.bases[0].coverage = {field};
    const std::vector<const MultiFab<Dim>*> layouts{field.get()};
    require(uniformly_rejected(
        [&] { validate_field_nullspace_basis<Dim>(layouts, plan, distributed_span); }));
  }
  {
    FieldNullspacePlan<Dim> plan = constant_mean_zero_nullspace<Dim>("nullspace", "mpi-preflight");
    plan.bases[0].cell_measure = {Real(1), Real(1)};
    const std::vector<const MultiFab<Dim>*> levels{field.get()};
    const int first_level = rank == 0 ? 1 : 0;
    require(uniformly_rejected([&] {
      (void)require_field_nullspace_compatible<Dim>(levels, plan, distributed_span, first_level);
    }));
  }
  {
    FieldNullspacePlan<Dim> plan = constant_mean_zero_nullspace<Dim>("nullspace", "mpi-preflight");
    if (rank == 0) {
      plan.bases[0].identity = "rank-zero-basis";
      plan.gauges[0].basis_identity = "rank-zero-basis";
    }
    const std::vector<const MultiFab<Dim>*> layouts{field.get()};
    require(uniformly_rejected(
        [&] { validate_field_nullspace_basis<Dim>(layouts, plan, distributed_span); }));
  }
  {
    FieldNullspacePlan<Dim> plan = constant_mean_zero_nullspace<Dim>("nullspace", "mpi-preflight");
    plan.bases[0].cell_measure = {Real(1), Real(1)};
    std::vector<const MultiFab<Dim>*> levels{field.get()};
    if (rank == 0)
      levels.push_back(field.get());
    const std::vector<PreparedVectorDistribution<Dim>> distributions(
        levels.size(), PreparedVectorDistribution<Dim>::Distributed);
    require(uniformly_rejected([&] {
      (void)require_field_nullspace_compatible<Dim>(
          levels, plan, std::span<const PreparedVectorDistribution<Dim>>(distributions));
    }));
  }
  {
    FieldNullspacePlan<Dim> plan = constant_mean_zero_nullspace<Dim>("nullspace", "mpi-preflight");
    plan.gauges[0].value = rank == 0 ? Real(0) : Real(1);
    const std::vector<MultiFab<Dim>*> levels{field.get()};
    require(uniformly_rejected([&] { apply_field_gauge<Dim>(levels, plan, distributed_span); }));
  }
  {
    FieldNullspacePlan<Dim> plan = constant_mean_zero_nullspace<Dim>("nullspace", "mpi-preflight");
    if (rank == 0)
      plan.gauges.clear();
    const std::vector<MultiFab<Dim>*> levels{field.get()};
    require(uniformly_rejected([&] { apply_field_gauge<Dim>(levels, plan, distributed_span); }));
  }
  {
    const FieldNullspacePlan<Dim> plan =
        constant_mean_zero_nullspace<Dim>("nullspace", "mpi-preflight");
    const std::vector<const MultiFab<Dim>*> levels{field.get()};
    const std::array<PreparedVectorDistribution<Dim>, 1> distributions{
        rank == 0 ? PreparedVectorDistribution<Dim>::Distributed
                  : PreparedVectorDistribution<Dim>::Replicated};
    require(uniformly_rejected([&] {
      (void)require_field_nullspace_compatible<Dim>(
          levels, plan, std::span<const PreparedVectorDistribution<Dim>>(distributions));
    }));
  }
  {
    const auto drifted = make_distributed_field<Dim>(rank == 0);
    const FieldNullspacePlan<Dim> plan =
        constant_mean_zero_nullspace<Dim>("nullspace", "mpi-preflight");
    const std::vector<const MultiFab<Dim>*> levels{drifted.get()};
    require(uniformly_rejected(
        [&] { (void)require_field_nullspace_compatible<Dim>(levels, plan, distributed_span); }));
  }

  // Component vocabulary drift is caught before mask allocation or label-count reduction.
  {
    std::vector<FieldConnectedComponent> components{{1, "material-a", "mpi-preflight:label:1"}};
    if (rank == 0)
      components.push_back({2, "material-b", "mpi-preflight:label:2"});
    require(uniformly_rejected([&] {
      (void)labelled_mean_zero_nullspace<Dim>(
          "prepared-nullspace", "prepared-layout",
          std::vector<std::shared_ptr<const MultiFab<Dim>>>{field}, components, {}, {Real(1)}, 0,
          distributed_span);
    }));
  }

  // Exact replicated validation rejects both scalar drift and isometric value permutations.
  {
    const auto replica = make_replicated_field<Dim>(rank == 0 ? Real(0) : Real(1));
    const FieldNullspacePlan<Dim> plan =
        constant_mean_zero_nullspace<Dim>("replicated-nullspace", "mpi-preflight");
    const std::vector<const MultiFab<Dim>*> levels{replica.get()};
    const std::array<PreparedVectorDistribution<Dim>, 1> replicated_level{
        PreparedVectorDistribution<Dim>::Replicated};
    require(uniformly_rejected(
        [&] {
          (void)require_field_nullspace_compatible<Dim>(
              levels, plan, std::span<const PreparedVectorDistribution<Dim>>(replicated_level));
        },
        kExactReplicaValidationFailure));
  }
  {
    const auto replica = make_replicated_field<Dim>(Real(0));
    set_isometric_rank_permutation(*replica);
    const FieldNullspacePlan<Dim> plan =
        constant_mean_zero_nullspace<Dim>("replicated-nullspace", "mpi-preflight");
    const std::vector<MultiFab<Dim>*> levels{replica.get()};
    const std::array<PreparedVectorDistribution<Dim>, 1> replicated_level{
        PreparedVectorDistribution<Dim>::Replicated};
    require(uniformly_rejected(
        [&] {
          apply_field_gauge<Dim>(
              levels, plan, std::span<const PreparedVectorDistribution<Dim>>(replicated_level));
        },
        kExactReplicaValidationFailure));
  }
  {
    const auto stable_layout = make_replicated_field<Dim>(Real(0));
    const auto permuted_mask = make_replicated_field<Dim>(Real(0));
    set_isometric_rank_permutation(*permuted_mask);
    FieldNullspacePlan<Dim> plan =
        constant_mean_zero_nullspace<Dim>("replicated-mask", "mpi-preflight");
    plan.bases[0].masks = {permuted_mask};
    const std::vector<const MultiFab<Dim>*> levels{stable_layout.get()};
    const std::array<PreparedVectorDistribution<Dim>, 1> replicated_level{
        PreparedVectorDistribution<Dim>::Replicated};
    require(uniformly_rejected(
        [&] {
          validate_field_nullspace_basis<Dim>(
              levels, plan, std::span<const PreparedVectorDistribution<Dim>>(replicated_level));
        },
        kExactReplicaValidationFailure));
  }

  // Valid distributed and mixed replicated/distributed reductions retain their mathematical
  // semantics after the exact-rank cutover.
  {
    set_mean_zero_pattern(*field);
    const FieldNullspacePlan<Dim> plan =
        constant_mean_zero_nullspace<Dim>("distributed-nullspace", "mpi-preflight");
    const std::vector<const MultiFab<Dim>*> levels{field.get()};
    require(uniformly_accepted([&] {
      const std::vector<double> witness =
          require_field_nullspace_compatible<Dim>(levels, plan, distributed_span);
      if (witness.size() != 2U || std::abs(witness[0]) > 1e-13)
        throw std::runtime_error("distributed exact-rank witness is not mean-zero");
    }));
  }
  {
    const auto rank_zero_field = make_rank_zero_distributed_field<Dim>(Real(0));
    set_mean_zero_pattern(*rank_zero_field);
    const FieldNullspacePlan<Dim> plan =
        constant_mean_zero_nullspace<Dim>("empty-rank-nullspace", "mpi-preflight");
    const std::vector<const MultiFab<Dim>*> rhs_levels{rank_zero_field.get()};
    require(uniformly_accepted([&] {
      const std::vector<double> witness =
          require_field_nullspace_compatible<Dim>(rhs_levels, plan, distributed_span);
      if (witness.size() != 2U || std::abs(witness[0]) > 1e-13)
        throw std::runtime_error("empty-rank exact-rank witness is not mean-zero");
    }));
    rank_zero_field->set_val(Real(3));
    const std::vector<MultiFab<Dim>*> phi_levels{rank_zero_field.get()};
    require(
        uniformly_accepted([&] { apply_field_gauge<Dim>(phi_levels, plan, distributed_span); }));
    require(maximum_error(*rank_zero_field, Real(0)) < 1e-13);
  }
  {
    const auto replica = make_replicated_field<Dim>(Real(1));
    const auto distributed = make_distributed_field<Dim>();
    distributed->set_val(Real(0));
    FieldNullspacePlan<Dim> plan =
        constant_mean_zero_nullspace<Dim>("mixed-nullspace", "mpi-preflight");
    plan.bases[0].cell_measure.push_back(Real(1));
    const std::vector<const MultiFab<Dim>*> levels{replica.get(), distributed.get()};
    const std::array<PreparedVectorDistribution<Dim>, 2> distributions{
        PreparedVectorDistribution<Dim>::Replicated, PreparedVectorDistribution<Dim>::Distributed};
    require(uniformly_accepted([&] {
      const std::vector<MultiFab<Dim>*> mutable_levels{replica.get(), distributed.get()};
      const std::span<const PreparedVectorDistribution<Dim>> distribution_span(distributions);
      apply_field_gauge<Dim>(mutable_levels, plan, distribution_span);
      const std::vector<double> witness =
          require_field_nullspace_compatible<Dim>(levels, plan, distribution_span);
      // The replicated level contributes its physical cells once, while the distributed level
      // contributes the same global cell count partitioned across ranks.  After the shared gauge
      // shift both levels have magnitude one half, so their combined absolute moment equals one
      // global level's cell count, not twice that count.
      const double hierarchy_cells = static_cast<double>(std::size_t{1} << (Dim + 1));
      if (witness.size() != 2U || std::abs(witness[0]) > 1e-13 || witness[1] != hierarchy_cells)
        throw std::runtime_error(
            "mixed exact-rank witness is not mean-zero: size=" +
            std::to_string(witness.size()) + " moment=" +
            (witness.empty() ? std::string("missing") : std::to_string(witness[0])) + " mass=" +
            (witness.size() < 2U ? std::string("missing") : std::to_string(witness[1])) +
            " expected-mass=" + std::to_string(hierarchy_cells));
    }));
    require(maximum_error(*replica, Real(0.5)) < 1e-13);
    require(maximum_error(*distributed, Real(-0.5)) < 1e-13);
  }
}

int run_field_nullspace_preflight(int argc, char** argv) {
  comm_init(&argc, &argv);
  Kokkos::ScopeGuard guard(argc, argv);
  const int rank = my_rank();
  long failures = n_ranks() == 2 ? 0 : 1;
  const auto require = [&failures, rank](bool condition, const std::source_location where =
                                                             std::source_location::current()) {
    if (!condition) {
      std::cerr << "field-nullspace MPI preflight check failed on rank " << rank << " at "
                << where.file_name() << ':' << where.line() << '\n';
      ++failures;
    }
  };

  exercise_exact_ranked_preflight<1>(require);
  exercise_exact_ranked_preflight<2>(require);
  exercise_exact_ranked_preflight<3>(require);

  failures = all_reduce_sum(failures);
  comm_finalize();
  return failures == 0 ? 0 : 1;
}

}  // namespace

TEST(test_mpi_field_nullspace_preflight, ExactRankedDriftFailsCoherentlyIn1D2DAnd3D) {
  EXPECT_EQ(
      pops::test::RunTestBody(&run_field_nullspace_preflight, "test_mpi_field_nullspace_preflight"),
      0);
}
