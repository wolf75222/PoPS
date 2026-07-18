#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/numerics/elliptic/interface/field_nullspace.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace_prepare.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace_provider.hpp>
#include <pops/parallel/comm.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

namespace {

const std::array<PreparedVectorDistribution, 1> kDistributedLevel{
    PreparedVectorDistribution::Distributed};
const std::array<PreparedVectorDistribution, 1> kReplicatedLevel{
    PreparedVectorDistribution::Replicated};
constexpr std::string_view kExactReplicaValidationFailure =
    "replicated vector values failed exact collective validation";

class OperatorFactsBlindProvider final : public FieldNullspaceProvider {
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
    return options.schema_identity ==
               "test.field-nullspace.operator-facts-blind.options@1" &&
           options.values.empty();
  }
  [[nodiscard]] PreparedProviderSupport supports(
      const FieldNullspaceProviderRequest&) const noexcept override {
    return PreparedProviderSupport::accept();
  }
  [[nodiscard]] std::string expected_prepared_contract(
      const FieldNullspaceProviderRequest&) const override {
    ExactContractBuilder contract;
    contract.text("test.prepared-field-nullspace.operator-facts-blind")
        .scalar(std::uint32_t{1});
    return std::move(contract).release();
  }
  [[nodiscard]] PreparedFieldNullspace prepare(
      const FieldNullspaceProviderRequest& request) const override {
    return {std::string(identity()), interface_version(), expected_prepared_contract(request), {}};
  }
};

std::shared_ptr<MultiFab> make_field(bool drift_layout = false) {
  std::vector<Box2D> boxes{Box2D{{0, 0}, {1, 1}}, Box2D{{2, 0}, {drift_layout ? 4 : 3, 1}}};
  auto result = std::make_shared<MultiFab>(BoxArray(std::move(boxes)),
                                           DistributionMapping(std::vector<int>{0, 1}), 1, 0);
  result->set_val(Real(0));
  return result;
}

std::shared_ptr<MultiFab> make_rank_local_replica(Real value) {
  const std::vector<Box2D> boxes{Box2D{{0, 0}, {1, 1}}, Box2D{{2, 0}, {3, 1}}};
  auto result = std::make_shared<MultiFab>(
      BoxArray(boxes), DistributionMapping(std::vector<int>(boxes.size(), my_rank())), 1, 0);
  result->set_val(value);
  return result;
}

std::shared_ptr<MultiFab> make_rank_zero_distributed(Real value) {
  auto result = std::make_shared<MultiFab>(BoxArray(std::vector<Box2D>{Box2D{{0, 0}, {3, 1}}}),
                                           DistributionMapping(std::vector<int>{0}), 1, 0);
  result->set_val(value);
  return result;
}

void set_mean_zero_pattern(MultiFab& field) {
  for (int li = 0; li < field.local_size(); ++li) {
    Array4 values = field.fab(li).array();
    const Box2D box = field.box(li);
    for (int j = box.lo[1]; j <= box.hi[1]; ++j)
      for (int i = box.lo[0]; i <= box.hi[0]; ++i)
        values(i, j, 0) = i < 2 ? Real(1) : Real(-1);
  }
}

void set_isometric_rank_permutation(MultiFab& field) {
  field.set_val(Real(0));
  field.sync_host();
  Array4 values = field.fab(0).array();
  if (my_rank() == 0)
    values(0, 0, 0) = Real(1);
  else
    values(1, 0, 0) = Real(1);
}

double maximum_error(const MultiFab& field, Real expected) {
  double local = 0.0;
  for (int li = 0; li < field.local_size(); ++li) {
    const ConstArray4 values = field.fab(li).const_array();
    const Box2D box = field.box(li);
    for (int j = box.lo[1]; j <= box.hi[1]; ++j)
      for (int i = box.lo[0]; i <= box.hi[0]; ++i)
        local = std::max(local, std::abs(static_cast<double>(values(i, j, 0) - expected)));
  }
  return all_reduce_max(local);
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

int run_field_nullspace_preflight(int argc, char** argv) {
  comm_init(&argc, &argv);
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#endif
  const int rank = my_rank();
  long failures = n_ranks() == 2 ? 0 : 1;
  const auto require = [&failures, rank](
                           bool condition,
                           const std::source_location where =
                               std::source_location::current()) {
    if (!condition) {
      std::cerr << "field-nullspace MPI preflight check failed on rank " << rank << " at "
                << where.file_name() << ':' << where.line() << '\n';
      ++failures;
    }
  };

  const std::shared_ptr<MultiFab> field = make_field();

  // The collective core authenticates operator facts before invoking provider declarations.  Even
  // a conforming external provider that deliberately ignores those facts cannot mask rank-local
  // boundary identity or behavior drift.
  {
    FieldNullspaceProviderRegistry registry;
    registry.add(std::make_shared<OperatorFactsBlindProvider>());
    const FieldNullspaceProviderSelection selection{
        "test.field-nullspace.operator-facts-blind",
        {"test.field-nullspace.operator-facts-blind.options@1", {}}};

    FieldNullspaceProviderRequest request;
    request.plan_identity = "mpi-preflight:operator-facts-id-drift";
    request.operator_facts = make_field_nullspace_operator_facts(
        "mpi-preflight:boundary-set@1",
        {{rank == 0 ? "boundary:a" : "boundary:b",
          FieldBoundaryNullspaceBehavior::PreservesConstantMode}},
        false);
    require(uniformly_rejected(
        [&] {
          (void)prepare_field_nullspace_collectively(registry, selection, request);
        },
        "operator facts differ across MPI ranks"));

    request.plan_identity = "mpi-preflight:operator-facts-behavior-drift";
    request.operator_facts = make_field_nullspace_operator_facts(
        "mpi-preflight:boundary-set@1",
        {{"boundary:a", rank == 0
                            ? FieldBoundaryNullspaceBehavior::PreservesConstantMode
                            : FieldBoundaryNullspaceBehavior::ConstrainsConstantMode}},
        false);
    require(uniformly_rejected(
        [&] {
          (void)prepare_field_nullspace_collectively(registry, selection, request);
        },
        "operator facts differ across MPI ranks"));
  }

  // Preparation must agree component cardinality before allocating one mask per component or
  // entering the label-count reduction.
  {
    std::vector<FieldConnectedComponent> components{{1, "material-a", "mpi-preflight:label:1"}};
    if (rank == 0)
      components.push_back({2, "material-b", "mpi-preflight:label:2"});
    require(uniformly_rejected([&] {
      (void)labelled_mean_zero_nullspace(
          "prepared-nullspace", "prepared-layout", {std::shared_ptr<const MultiFab>(field)},
          components, {}, {Real(1)}, 0, kDistributedLevel);
    }));
  }

  // A rank-local null field is converted into the same exception on every rank before any Gram
  // work or dereference.
  {
    const FieldNullspacePlan plan = constant_mean_zero_nullspace("nullspace", "mpi-preflight");
    const std::vector<const MultiFab*> layouts{rank == 0 ? nullptr : field.get()};
    require(uniformly_rejected(
        [&] { validate_field_nullspace_basis(layouts, plan, kDistributedLevel); }));
  }

  // Exact plan tokens, scope, first level, basis identities and level cardinality are all part of
  // the native byte-consensus payload, independently of their string lengths.
  {
    FieldNullspacePlan plan = constant_mean_zero_nullspace(
        rank == 0 ? "nullspace-rank-0" : "nullspace-rank-1", "mpi-preflight");
    require(uniformly_rejected([&] { (void)require_field_nullspace_compatible(*field, plan); }));
  }
  {
    FieldNullspacePlan plan = constant_mean_zero_nullspace("nullspace", "mpi-preflight");
    if (rank == 0)
      plan.bases[0].coverage = {field};
    require(uniformly_rejected(
        [&] { validate_field_nullspace_basis({field.get()}, plan, kDistributedLevel); }));
  }
  {
    FieldNullspacePlan plan = constant_mean_zero_nullspace("nullspace", "mpi-preflight");
    plan.bases[0].cell_measure = {Real(1), Real(1)};
    const int first_level = rank == 0 ? 1 : 0;
    require(uniformly_rejected(
        [&] {
          (void)require_field_nullspace_compatible({field.get()}, plan, kDistributedLevel,
                                                   first_level);
        }));
  }
  {
    FieldNullspacePlan plan = constant_mean_zero_nullspace("nullspace", "mpi-preflight");
    if (rank == 0) {
      plan.bases[0].identity = "rank-zero-basis";
      plan.gauges[0].basis_identity = "rank-zero-basis";
    }
    require(uniformly_rejected(
        [&] { validate_field_nullspace_basis({field.get()}, plan, kDistributedLevel); }));
  }
  {
    FieldNullspacePlan plan = constant_mean_zero_nullspace("nullspace", "mpi-preflight");
    plan.bases[0].cell_measure = {Real(1), Real(1)};
    std::vector<const MultiFab*> levels{field.get()};
    if (rank == 0)
      levels.push_back(field.get());
    std::vector<PreparedVectorDistribution> distributions(
        levels.size(), PreparedVectorDistribution::Distributed);
    require(uniformly_rejected(
        [&] { (void)require_field_nullspace_compatible(levels, plan, distributions); }));
  }

  // Gauge identity and physical layout divergence must be rejected before the gauge-moment
  // collective.  Both cases previously let one rank throw while the other entered Allreduce.
  {
    FieldNullspacePlan plan = constant_mean_zero_nullspace("nullspace", "mpi-preflight");
    if (rank == 0)
      plan.gauges[0].basis_identity = "missing-basis";
    require(uniformly_rejected([&] { apply_field_gauge(*field, plan); }));
  }
  {
    const std::shared_ptr<MultiFab> local_field = make_field(rank == 0);
    const FieldNullspacePlan plan = constant_mean_zero_nullspace("nullspace", "mpi-preflight");
    require(uniformly_rejected([&] { apply_field_gauge(*local_field, plan); }));
  }

  // A matching contract still reaches all three scientific collectives and remains a transparent
  // success path.
  {
    const FieldNullspacePlan plan = constant_mean_zero_nullspace("nullspace", "mpi-preflight");
    try {
      validate_field_nullspace_basis({field.get()}, plan, kDistributedLevel);
      (void)require_field_nullspace_compatible(*field, plan);
      apply_field_gauge(*field, plan);
    } catch (...) {
      ++failures;
    }
  }

  // A replicated field owns every global box on every rank, so its concrete dmap contains the
  // current rank by construction.  Collective metadata agreement canonicalizes precisely this
  // ownership pattern while retaining exact comparison for ordinary distributed layouts.
  {
    const std::shared_ptr<MultiFab> replicated_field = make_rank_local_replica(Real(0));
    set_mean_zero_pattern(*replicated_field);
    const std::shared_ptr<MultiFab> replicated_mask = make_rank_local_replica(Real(1));
    FieldNullspacePlan plan =
        constant_mean_zero_nullspace("replicated-nullspace", "mpi-preflight", Real(1));
    plan.bases[0].masks = {replicated_mask};
    try {
      validate_field_nullspace_basis({replicated_field.get()}, plan, kReplicatedLevel);
      const std::vector<double> witness =
          require_field_nullspace_compatible(*replicated_field, plan,
                                             PreparedVectorDistribution::Replicated);
      require(witness.size() == 2 && witness[0] == 0.0 && witness[1] == 8.0);
      apply_field_gauge(*replicated_field, plan, PreparedVectorDistribution::Replicated);
    } catch (...) {
      ++failures;
    }
  }

  // The connected-component factory consumes the same generic ownership contract; it must not
  // double-count replicated labels or the masks materialized from them.
  {
    const std::shared_ptr<MultiFab> labels = make_rank_local_replica(Real(0));
    const std::shared_ptr<MultiFab> rhs = make_rank_local_replica(Real(0));
    for (int li = 0; li < labels->local_size(); ++li) {
      Array4 label = labels->fab(li).array();
      Array4 value = rhs->fab(li).array();
      const Box2D box = labels->box(li);
      for (int j = box.lo[1]; j <= box.hi[1]; ++j) {
        for (int i = box.lo[0]; i <= box.hi[0]; ++i) {
          label(i, j, 0) = i < 2 ? Real(1) : Real(2);
          value(i, j, 0) = j == 0 ? Real(1) : Real(-1);
        }
      }
    }
    try {
      const FieldNullspacePlan plan = labelled_mean_zero_nullspace(
          "replicated-components", "replicated-components-layout", {labels},
          {{1, "left", "mpi:label:1"}, {2, "right", "mpi:label:2"}}, {}, {Real(1)}, 0,
          kReplicatedLevel);
      const std::vector<double> witness = require_field_nullspace_compatible(
          *rhs, plan, PreparedVectorDistribution::Replicated);
      require(witness.size() == 4 && witness[0] == 0.0 && witness[1] == 4.0 && witness[2] == 0.0 &&
              witness[3] == 4.0);
    } catch (...) {
      ++failures;
    }
  }

  // A valid distributed mono-box layout is concentrated on rank zero and must pass through the
  // primary exact witness.  Rank zero alone satisfying the local-replica shape must not cause the
  // other ranks to disagree with it.
  {
    const std::shared_ptr<MultiFab> distributed_field = make_rank_zero_distributed(Real(0));
    set_mean_zero_pattern(*distributed_field);
    const FieldNullspacePlan plan =
        constant_mean_zero_nullspace("distributed-nullspace", "mpi-preflight");
    try {
      validate_field_nullspace_basis({distributed_field.get()}, plan, kDistributedLevel);
      const std::vector<double> witness =
          require_field_nullspace_compatible(*distributed_field, plan);
      require(witness.size() == 2 && witness[0] == 0.0 && witness[1] == 8.0);
      apply_field_gauge(*distributed_field, plan);
    } catch (...) {
      ++failures;
    }
  }

  // Ownership is authored, never inferred from a rank-local shape. A rank-zero distributed field
  // declared as replicated is malformed on every rank and must fail before a scientific reduction.
  {
    const std::shared_ptr<MultiFab> distributed_field = make_rank_zero_distributed(Real(0));
    const FieldNullspacePlan plan =
        constant_mean_zero_nullspace("misdeclared-replica", "mpi-preflight", Real(1));
    require(uniformly_rejected(
        [&] { validate_field_nullspace_basis({distributed_field.get()}, plan, kReplicatedLevel); }));
  }

  // Replicated scientific values are consensus checked, not silently averaged or selected locally.
  {
    const std::shared_ptr<MultiFab> divergent_replica =
        make_rank_local_replica(rank == 0 ? Real(0) : Real(1));
    const FieldNullspacePlan plan =
        constant_mean_zero_nullspace("divergent-replica", "mpi-preflight", Real(1));
    require(uniformly_rejected([&] {
      apply_field_gauge(*divergent_replica, plan, PreparedVectorDistribution::Replicated);
    },
                               kExactReplicaValidationFailure));
  }

  // Equal norms and moments are not an equality certificate. Exact chunked consensus rejects both
  // a permuted solved field and a permuted basis mask before Gram/gauge reductions can cancel them.
  {
    const std::shared_ptr<MultiFab> permuted_field = make_rank_local_replica(Real(0));
    set_isometric_rank_permutation(*permuted_field);
    const FieldNullspacePlan plan = constant_mean_zero_nullspace(
        "isometric-field-replica", "mpi-preflight", Real(1));
    require(uniformly_rejected([&] {
      apply_field_gauge(*permuted_field, plan, PreparedVectorDistribution::Replicated);
    },
                               kExactReplicaValidationFailure));

    const std::shared_ptr<MultiFab> stable_layout = make_rank_local_replica(Real(0));
    const std::shared_ptr<MultiFab> permuted_mask = make_rank_local_replica(Real(0));
    set_isometric_rank_permutation(*permuted_mask);
    FieldNullspacePlan masked = constant_mean_zero_nullspace(
        "isometric-mask-replica", "mpi-preflight", Real(1));
    masked.bases[0].masks = {permuted_mask};
    require(
        uniformly_rejected(
            [&] { validate_field_nullspace_basis({stable_layout.get()}, masked, kReplicatedLevel); },
                           kExactReplicaValidationFailure));
  }

  // Mixed hierarchies require exact-or-replicated agreement independently per layout slot: the
  // coarse slot is rank-local replicated while the second slot is a valid rank-zero distribution.
  {
    const std::shared_ptr<MultiFab> replicated_field = make_rank_local_replica(Real(1));
    const std::shared_ptr<MultiFab> distributed_field = make_rank_zero_distributed(Real(0));
    FieldNullspacePlan plan = constant_mean_zero_nullspace("mixed-nullspace", "mpi-preflight");
    plan.bases[0].cell_measure = {Real(1), Real(1)};
    const std::array<PreparedVectorDistribution, 2> distributions{
        PreparedVectorDistribution::Replicated, PreparedVectorDistribution::Distributed};
    try {
      validate_field_nullspace_basis({replicated_field.get(), distributed_field.get()}, plan,
                                     distributions);
      apply_field_gauge({replicated_field.get(), distributed_field.get()}, plan, distributions);
      require(maximum_error(*replicated_field, Real(0.5)) == 0.0);
      require(maximum_error(*distributed_field, Real(-0.5)) == 0.0);
      const std::vector<double> witness = require_field_nullspace_compatible(
          {replicated_field.get(), distributed_field.get()}, plan, distributions);
      require(witness.size() == 2 && witness[0] == 0.0 && witness[1] == 8.0);
    } catch (...) {
      ++failures;
    }
  }

  failures = all_reduce_sum(failures);
  comm_finalize();
  return failures == 0 ? 0 : 1;
}

}  // namespace

TEST(test_mpi_field_nullspace_preflight, RankLocalDriftFailsCoherentlyBeforeScientificCollectives) {
  EXPECT_EQ(
      pops::test::RunTestBody(&run_field_nullspace_preflight, "test_mpi_field_nullspace_preflight"),
      0);
}
