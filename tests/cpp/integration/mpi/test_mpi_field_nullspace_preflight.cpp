#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/numerics/elliptic/interface/field_nullspace.hpp>
#include <pops/parallel/comm.hpp>

#include <memory>
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

std::shared_ptr<MultiFab> make_field(bool drift_layout = false) {
  std::vector<Box2D> boxes{
      Box2D{{0, 0}, {1, 1}}, Box2D{{2, 0}, {drift_layout ? 4 : 3, 1}}};
  auto result = std::make_shared<MultiFab>(BoxArray(std::move(boxes)),
                                           DistributionMapping(std::vector<int>{0, 1}), 1, 0);
  result->set_val(Real(0));
  return result;
}

template <class Operation>
bool uniformly_rejected(Operation&& operation) {
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
  return rejected_ranks == n_ranks() && same_message &&
         message.find("collective preflight rejected") != std::string::npos;
}

int run_field_nullspace_preflight(int argc, char** argv) {
  comm_init(&argc, &argv);
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#endif
  const int rank = my_rank();
  long failures = n_ranks() == 2 ? 0 : 1;
  const auto require = [&failures](bool condition) {
    if (!condition)
      ++failures;
  };

  const std::shared_ptr<MultiFab> field = make_field();

  // Preparation must agree component cardinality before allocating one mask per component or
  // entering the label-count reduction.
  {
    std::vector<FieldConnectedComponent> components{
        {1, "material-a", "mpi-preflight:label:1"}};
    if (rank == 0)
      components.push_back({2, "material-b", "mpi-preflight:label:2"});
    require(uniformly_rejected([&] {
      (void)labelled_mean_zero_nullspace(
          "prepared-nullspace", "prepared-layout", FieldNullspaceScope::Uniform,
          {std::shared_ptr<const MultiFab>(field)}, components, {}, {Real(1)}, 0);
    }));
  }

  // A rank-local null field is converted into the same exception on every rank before any Gram
  // work or dereference.
  {
    const FieldNullspacePlan plan =
        constant_mean_zero_nullspace("nullspace", "mpi-preflight");
    const std::vector<const MultiFab*> layouts{rank == 0 ? nullptr : field.get()};
    require(uniformly_rejected([&] { validate_field_nullspace_basis(layouts, plan); }));
  }

  // Exact plan tokens, scope, first level, basis identities and level cardinality are all part of
  // the native byte-consensus payload, independently of their string lengths.
  {
    FieldNullspacePlan plan = constant_mean_zero_nullspace(
        rank == 0 ? "nullspace-rank-0" : "nullspace-rank-1", "mpi-preflight");
    require(uniformly_rejected(
        [&] { (void)require_field_nullspace_compatible(*field, plan); }));
  }
  {
    FieldNullspacePlan plan =
        constant_mean_zero_nullspace("nullspace", "mpi-preflight");
    if (rank == 0)
      plan.scope = FieldNullspaceScope::Composite;
    require(uniformly_rejected(
        [&] { validate_field_nullspace_basis({field.get()}, plan); }));
  }
  {
    FieldNullspacePlan plan =
        constant_mean_zero_nullspace("nullspace", "mpi-preflight");
    plan.bases[0].cell_measure = {Real(1), Real(1)};
    const int first_level = rank == 0 ? 1 : 0;
    require(uniformly_rejected([&] {
      (void)require_field_nullspace_compatible({field.get()}, plan, first_level);
    }));
  }
  {
    FieldNullspacePlan plan =
        constant_mean_zero_nullspace("nullspace", "mpi-preflight");
    if (rank == 0) {
      plan.bases[0].identity = "rank-zero-basis";
      plan.gauges[0].basis_identity = "rank-zero-basis";
    }
    require(uniformly_rejected(
        [&] { validate_field_nullspace_basis({field.get()}, plan); }));
  }
  {
    FieldNullspacePlan plan =
        constant_mean_zero_nullspace("nullspace", "mpi-preflight");
    plan.bases[0].cell_measure = {Real(1), Real(1)};
    std::vector<const MultiFab*> levels{field.get()};
    if (rank == 0)
      levels.push_back(field.get());
    require(uniformly_rejected(
        [&] { (void)require_field_nullspace_compatible(levels, plan); }));
  }

  // Gauge identity and physical layout divergence must be rejected before the gauge-moment
  // collective.  Both cases previously let one rank throw while the other entered Allreduce.
  {
    FieldNullspacePlan plan =
        constant_mean_zero_nullspace("nullspace", "mpi-preflight");
    if (rank == 0)
      plan.gauges[0].basis_identity = "missing-basis";
    require(uniformly_rejected([&] { apply_field_gauge(*field, plan); }));
  }
  {
    const std::shared_ptr<MultiFab> local_field = make_field(rank == 0);
    const FieldNullspacePlan plan =
        constant_mean_zero_nullspace("nullspace", "mpi-preflight");
    require(uniformly_rejected([&] { apply_field_gauge(*local_field, plan); }));
  }

  // A matching contract still reaches all three scientific collectives and remains a transparent
  // success path.
  {
    const FieldNullspacePlan plan =
        constant_mean_zero_nullspace("nullspace", "mpi-preflight");
    try {
      validate_field_nullspace_basis({field.get()}, plan);
      (void)require_field_nullspace_compatible(*field, plan);
      apply_field_gauge(*field, plan);
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
  EXPECT_EQ(pops::test::RunTestBody(&run_field_nullspace_preflight,
                                    "test_mpi_field_nullspace_preflight"),
            0);
}
