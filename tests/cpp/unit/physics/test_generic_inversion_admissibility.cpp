#include <pops/core/state/state.hpp>
#include <pops/numerics/nonlinear/prepared_variable_recovery.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/physics/admissibility/admissibility.hpp>
#include <pops/physics/inversion/inversion.hpp>
#include <pops/runtime/config/runtime_params.hpp>
#include <pops/runtime/recovery/device_candidate_recovery.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void ensure_kokkos() {
#if defined(POPS_HAS_KOKKOS)
  static Kokkos::ScopeGuard guard;
  (void)guard;
#endif
}

struct OwnedParameterPredicate {
  pops::RuntimeParams parameters;
  using pops_device_capture_members = pops::DeviceCaptureMembers<decltype(parameters)>;

  static constexpr pops::PreparedProviderIdentity provider_identity() noexcept {
    return {"test.owned-parameter-predicate", 1};
  }
  void serialize_exact_parameters(pops::ExactContractBuilder& contract) const {
    parameters.require_bound();
    contract.scalar(parameters.get(64));
  }
  POPS_HD bool operator()(const pops::StateVec<1>& candidate) const {
    return candidate[0] <= parameters.get(64);
  }
};

struct HostContainerPredicate {
  std::vector<double> values;
  using pops_device_capture_members = pops::DeviceCaptureMembers<decltype(values)>;
};

using UnmanagedPredicateView = Kokkos::View<double*, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
struct BorrowedPredicate {
  UnmanagedPredicateView values;
  using pops_device_capture_members = pops::DeviceCaptureMembers<decltype(values)>;
};

static_assert(!pops::device_capture_v<pops::CustomInequality<HostContainerPredicate>>);
static_assert(!pops::device_capture_v<pops::CustomInequality<BorrowedPredicate>>);

template <class Storage>
struct RecoveryAdmissibilityModel {
  using State = pops::StateVec<1>;
  using Prim = pops::StateVec<1>;
  static constexpr int dimension = pops::kNativeDimension;
  static constexpr int n_vars = 1;
  Storage storage;
  using pops_device_capture_members = pops::DeviceCaptureMembers<decltype(storage)>;

  POPS_HD Prim to_primitive(const State& state) const { return state; }
  POPS_HD State to_conservative(const Prim& primitive) const { return primitive; }
  POPS_HD bool recovery_admissible(const Prim&, int*) const { return true; }
};

// The recovery predicate owns exactly its model: bound runtime parameters are captured through
// the model's exhaustive contract, and a host container still never is.
using ParameterRecoveryModel = RecoveryAdmissibilityModel<pops::RuntimeParams>;
static_assert(
    pops::device_capture_v<pops::ModelRecoveryAdmissibilityProvider<ParameterRecoveryModel>>);
static_assert(
    !pops::device_capture_v<
        pops::ModelRecoveryAdmissibilityProvider<RecoveryAdmissibilityModel<std::vector<double>>>>);
static_assert(sizeof(decltype(pops::prepare_model_recovery_admissibility(
                  std::declval<const ParameterRecoveryModel&>()))) > 0);

enum class RecoveryFailure : std::uint8_t {
  kNone = 0,
  kInvalidState = 1,
  kNoConvergence = 2,
};

template <int Dim>
using Problem = pops::VariableInversionProblem<Dim, pops::StateVec<2>, pops::ProviderValues<1>,
                                               pops::StateVec<2>, RecoveryFailure>;

struct ClosedFormSource {
  static constexpr pops::PreparedProviderIdentity provider_identity() noexcept {
    return {"test.closed-form-inversion", 1};
  }
  void serialize_exact_parameters(pops::ExactContractBuilder& contract) const {
    contract.scalar(std::uint32_t{1});
  }
  pops::InversionResult<pops::StateVec<2>, RecoveryFailure> operator()(
      const pops::StateVec<2>& state, const pops::ProviderValues<1>& inputs,
      pops::InversionWorkspaceView workspace) const {
    if (workspace.as<pops::Real>() == nullptr)
      return decltype(operator()(state, inputs, workspace))::fail(RecoveryFailure::kNoConvergence);
    if (!(state[0] > 0.0))
      return decltype(operator()(state, inputs, workspace))::fail(RecoveryFailure::kInvalidState);
    return decltype(operator()(state, inputs, workspace))::success(
        pops::StateVec<2>{{state[0], state[1] / state[0] + inputs[0]}});
  }
};

struct IterativeSource {
  int iterations = 6;

  static constexpr pops::PreparedProviderIdentity provider_identity() noexcept {
    return {"test.iterative-inversion", 1};
  }
  void serialize_exact_parameters(pops::ExactContractBuilder& contract) const {
    contract.scalar(std::int32_t{iterations});
  }
  pops::InversionResult<pops::StateVec<2>, RecoveryFailure> operator()(
      const pops::StateVec<2>& state, const pops::ProviderValues<1>&,
      pops::InversionWorkspaceView workspace) const {
    auto* iterate = workspace.as<pops::Real>();
    if (iterate == nullptr || state[0] < 0.0)
      return decltype(operator()(state, {}, workspace))::fail(RecoveryFailure::kInvalidState);
    *iterate = state[0] > 1.0 ? state[0] : 1.0;
    for (int step = 0; step < iterations; ++step)
      *iterate = 0.5 * (*iterate + state[0] / *iterate);
    if (!std::isfinite(*iterate))
      return decltype(operator()(state, {}, workspace))::fail(RecoveryFailure::kNoConvergence);
    return decltype(operator()(state, {}, workspace))::success(
        pops::StateVec<2>{{*iterate, state[1]}});
  }
};

struct ConePredicate {
  pops::Real scale = 1.0;

  static constexpr pops::PreparedProviderIdentity provider_identity() noexcept {
    return {"test.realizable-cone", 1};
  }
  void serialize_exact_parameters(pops::ExactContractBuilder& contract) const {
    contract.scalar(scale);
  }
  bool operator()(const pops::StateVec<2>& candidate) const {
    return std::abs(candidate[1]) <= scale * candidate[0];
  }
};

struct SumPredicate {
  pops::Real upper = 4.0;

  static constexpr pops::PreparedProviderIdentity provider_identity() noexcept {
    return {"test.custom-sum-inequality", 1};
  }
  void serialize_exact_parameters(pops::ExactContractBuilder& contract) const {
    contract.scalar(upper);
  }
  bool operator()(const pops::StateVec<2>& candidate) const {
    return candidate[0] + candidate[1] < upper;
  }
};

struct ClampProjection {
  pops::Real lower_bound = 0.0;

  static constexpr pops::PreparedProviderIdentity provider_identity() noexcept {
    return {"test.explicit-projection", 1};
  }
  void serialize_exact_parameters(pops::ExactContractBuilder& contract) const {
    contract.scalar(lower_bound);
  }
  pops::ProjectionResult<pops::StateVec<2>> operator()(const pops::StateVec<2>& candidate,
                                                       const pops::ProviderValues<1>&) const {
    auto projected = candidate;
    const bool changed = projected[0] <= lower_bound;
    if (changed)
      projected[0] = lower_bound + 1.0e-6;
    return {projected, changed};
  }
};

template <int Dim>
void exercise_dimension() {
  ensure_kokkos();
  Problem<Dim> problem("conservative:pair", "provider-pack:one", "primitive:pair",
                       "recovery-failure:v1", {sizeof(pops::Real), alignof(pops::Real)});
  pops::PreparedVariableInversion closed(problem, ClosedFormSource{});
  const void* allocation = closed.workspace_allocation_identity();
  const pops::ProviderValues<1> inputs{{0.25}};

  auto first = closed.attempt(pops::StateVec<2>{{2.0, 4.0}}, inputs);
  ASSERT_TRUE(first.succeeded());
  auto consumed = first.consume();
  ASSERT_TRUE(consumed.succeeded());
  EXPECT_DOUBLE_EQ(consumed.candidate()[1], 2.25);
  EXPECT_THROW(first.consume(), std::logic_error);

  pops::StateVec<2> published{{7.0, 9.0}};
  auto invalid = closed.attempt(pops::StateVec<2>{{-1.0, 4.0}}, inputs);
  ASSERT_FALSE(invalid.succeeded());
  EXPECT_EQ(invalid.failure(), RecoveryFailure::kInvalidState);
  auto rejected = invalid.consume();
  EXPECT_FALSE(rejected.succeeded());
  EXPECT_DOUBLE_EQ(published[0], 7.0);
  EXPECT_DOUBLE_EQ(published[1], 9.0);

  auto retry = closed.attempt(pops::StateVec<2>{{4.0, 8.0}}, inputs);
  EXPECT_EQ(closed.workspace_allocation_identity(), allocation);
  EXPECT_TRUE(retry.consume().succeeded());

  pops::PreparedVariableInversion iterative(problem, IterativeSource{});
  auto iterated = iterative.attempt(pops::StateVec<2>{{9.0, 3.0}}, inputs).consume();
  ASSERT_TRUE(iterated.succeeded());
  EXPECT_NEAR(iterated.candidate()[0], 3.0, 1.0e-12);

  pops::AdmissibleSet admissible(
      pops::FiniteComponents<0, 2>{10}, pops::PositiveComponent<0>{0.0, 11},
      pops::RealizabilityConstraint<ConePredicate>{ConePredicate{2.0}, 12},
      pops::CustomInequality<SumPredicate>{SumPredicate{4.0}, 13});
  EXPECT_TRUE(admissible.evaluate(pops::StateVec<2>{{1.0, 1.5}}).accepted);
  const auto nonfinite =
      admissible.evaluate(pops::StateVec<2>{{std::numeric_limits<pops::Real>::infinity(), 0.0}});
  EXPECT_FALSE(nonfinite.accepted);
  EXPECT_EQ(nonfinite.kind, pops::AdmissibilityConstraintKind::kFinite);
  const auto nonpositive = admissible.evaluate(pops::StateVec<2>{{0.0, 0.0}});
  EXPECT_EQ(nonpositive.diagnostic_code, 11u);
  const auto unrealizable = admissible.evaluate(pops::StateVec<2>{{1.0, 3.0}});
  EXPECT_EQ(unrealizable.diagnostic_code, 12u);
  const auto custom = admissible.evaluate(pops::StateVec<2>{{2.0, 2.5}});
  EXPECT_EQ(custom.diagnostic_code, 13u);

  pops::ProjectionProvider<Dim, pops::StateVec<2>, pops::ProviderValues<1>, ClampProjection>
      projection("primitive:pair", "provider-pack:one", ClampProjection{0.0});
  auto explicit_projection = projection.project(pops::StateVec<2>{{-1.0, 3.0}}, inputs);
  EXPECT_TRUE(explicit_projection.changed());
  const auto projected = std::move(explicit_projection).consume();
  EXPECT_GT(projected[0], 0.0);

  EXPECT_FALSE(problem.exact_contract().empty());
  EXPECT_FALSE(closed.collective_contract().empty());
  EXPECT_FALSE(admissible.exact_contract().empty());
  EXPECT_FALSE(projection.collective_contract().empty());
}

template <int Dim>
void exercise_identity_model_recovery() {
  using Model = pops::nd::ScalarAdvection<Dim>;
  using Recovery = pops::PreparedModelVariableInversionRecovery<Model>;
  static_assert(std::same_as<typename Recovery::Candidate, typename Model::State>);

  pops::RealVector<Dim> velocity{};
  Recovery recovery(Model::prepare(velocity));
  const void* allocation = recovery.workspace_allocation_identity();

  const pops::Real finite[1] = {pops::Real(2.5)};
  const auto accepted = recovery.recover(finite);
  EXPECT_EQ(accepted.outcome.status, pops::RecoveryStatus::kRecovered);
  EXPECT_DOUBLE_EQ(accepted.outcome.value[0], finite[0]);
  EXPECT_FALSE(accepted.projection_attempted);
  EXPECT_EQ(recovery.workspace_allocation_identity(), allocation);

  const pops::Real nonfinite[1] = {std::numeric_limits<pops::Real>::infinity()};
  const auto rejected = recovery.recover(nonfinite);
  EXPECT_EQ(rejected.outcome.status, pops::RecoveryStatus::kRejected);
  EXPECT_EQ(rejected.outcome.cause, pops::RecoveryCause::kExplicitRejection);
  EXPECT_EQ(rejected.outcome.reason_code,
            static_cast<std::uint32_t>(pops::VariableRecoveryInversionFailure::kSourceRejected));
  EXPECT_EQ(recovery.workspace_allocation_identity(), allocation);
}

TEST(GenericInversionAdmissibility, ExactRankedClosedFormAndIterativeProviders) {
  exercise_dimension<1>();
  exercise_dimension<2>();
  exercise_dimension<3>();

  Problem<1> dim1("state", "inputs", "candidate", "failure", {8, 8});
  Problem<2> dim2("state", "inputs", "candidate", "failure", {8, 8});
  pops::PreparedVariableInversion first(dim1, ClosedFormSource{});
  pops::PreparedVariableInversion same(dim1, ClosedFormSource{});
  EXPECT_EQ(first.collective_contract(), same.collective_contract());
  EXPECT_NE(dim1.exact_contract(), dim2.exact_contract());
}

TEST(GenericInversionAdmissibility, OwnedParameterConstraintExecutesAndRetainsItsIncarnation) {
  ensure_kokkos();
  std::array<double, 65> values{};
  values[64] = 2;
  const auto capacity = pops::RuntimeParams::checked_storage_bytes(values.size());
  OwnedParameterPredicate predicate{pops::RuntimeParams::from_host(values, capacity)};
  const pops::AdmissibleSet accepted(pops::FiniteComponents<0, 1>(1),
                                     pops::CustomInequality<OwnedParameterPredicate>(predicate, 2));
  static_assert(pops::device_capture_v<decltype(accepted)>);
  const auto contract = accepted.exact_contract();
  values[64] = 0;
  predicate.parameters = pops::RuntimeParams::from_host(values, capacity);
  const pops::AdmissibleSet replaced(pops::FiniteComponents<0, 1>(1),
                                     pops::CustomInequality<OwnedParameterPredicate>(predicate, 2));
  EXPECT_NE(contract, replaced.exact_contract());
  EXPECT_EQ(contract, accepted.exact_contract());
  Kokkos::View<int*> result("owned-admissibility", 3);
  Kokkos::parallel_for(
      "owned-admissibility", 3, KOKKOS_LAMBDA(const int index) {
        const pops::StateVec<1> candidate{{pops::Real(index + 1)}};
        const auto decision = accepted.evaluate(candidate);
        result(index) = decision.accepted ? 0 : int(decision.diagnostic_code);
      });
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result);
  EXPECT_EQ(host(0), 0);
  EXPECT_EQ(host(1), 0);
  EXPECT_EQ(host(2), 2);
}

TEST(GenericInversionAdmissibility, IdentityModelUsesTypedStateCandidateInEveryDimension) {
  exercise_identity_model_recovery<1>();
  exercise_identity_model_recovery<2>();
  exercise_identity_model_recovery<3>();
}

// A recovery that can refuse: density first, velocity = momentum / density, admissible only for a
// positive density. A zero density gives an infinite velocity, which the finite constraint refuses.
struct PositiveDensityModel {
  using State = pops::StateVec<2>;
  using Prim = pops::StateVec<2>;
  static constexpr int dimension = pops::kNativeDimension;
  static constexpr int n_vars = 2;
  using pops_device_capture_members = pops::DeviceCaptureMembers<>;

  POPS_HD Prim to_primitive(const State& state) const {
    Prim primitive{};
    primitive[0] = state[0];
    primitive[1] = state[1] / state[0];
    return primitive;
  }
  POPS_HD State to_conservative(const Prim& primitive) const {
    State state{};
    state[0] = primitive[0];
    state[1] = primitive[0] * primitive[1];
    return state;
  }
  POPS_HD bool recovery_admissible(const Prim& primitive, int* failing_component) const {
    if (primitive[0] > pops::Real(0))
      return true;
    if (failing_component != nullptr)
      *failing_component = 0;
    return false;
  }
};

// Cell kinds by ordinal: recovered, non-finite conservative, negative density, zero density.
POPS_HD inline void census_cell_values(std::int64_t ordinal, pops::Real& density,
                                       pops::Real& momentum) {
  switch (ordinal % 5) {
    case 1:
      density = std::numeric_limits<pops::Real>::quiet_NaN();
      momentum = pops::Real(1);
      return;
    case 2:
      density = pops::Real(-1);
      momentum = pops::Real(1);
      return;
    case 3:
      density = pops::Real(0);
      momentum = pops::Real(1);
      return;
    default:
      density = pops::Real(2 + ordinal % 3);
      momentum = pops::Real(3);
  }
}

TEST(GenericInversionAdmissibility, DeviceCandidateCensusMatchesThePreparedHostRecovery) {
  ensure_kokkos();
  constexpr int D = pops::kNativeDimension;
  pops::Extent<D> cells{}, ghosts{}, ranks_extent{};
  for (int axis = 0; axis < D; ++axis) {
    cells[axis] = 4;
    ghosts[axis] = 1;
    ranks_extent[axis] = 1;
  }
  const auto box = pops::Box<D>::from_extents(cells);
  const auto layout = pops::mesh::BoxArray<D>::from_domain(box, cells);
  const pops::mesh::RankSpace<D> ranks(pops::Index<D>{}, ranks_extent);
  const auto distribution = pops::mesh::Distribution<D>::replicated(layout, ranks);
  pops::MultiFab<D> candidate(layout, distribution, pops::Index<D>{}, 2, ghosts);
  pops::MultiFab<D> active(layout, distribution, pops::Index<D>{}, 1, ghosts);
  // Ghost cells hold NaN: the census reads valid cells only.
  candidate.set_val(std::numeric_limits<pops::Real>::quiet_NaN());
  active.set_val(pops::Real(1));
  ASSERT_EQ(candidate.local_size(), 1u);
  const pops::Box<D> valid = candidate.fab(0).box();
  const pops::Index<D> lower = valid.lo;
  const pops::Extent<D> extent = valid.extent();
  const auto values = candidate.fab(0).view();
  const auto mask = active.fab(0).view();
  Kokkos::parallel_for(
      "census-cells", Kokkos::RangePolicy<Kokkos::IndexType<std::int64_t>>(0, valid.numPts()),
      KOKKOS_LAMBDA(const std::int64_t ordinal) {
        const auto index = pops::detail::cell_index_from_ordinal(lower, extent, ordinal);
        census_cell_values(ordinal, values(index, 0), values(index, 1));
        // The negative-density cells are inactive.
        mask(index) = ordinal % 5 == 2 ? pops::Real(0) : pops::Real(1);
      });
  Kokkos::fence();

  // The reference: the prepared host recovery on each cell, after the finite check.
  pops::PreparedModelVariableInversionRecovery<PositiveDensityModel> recovery(
      PositiveDensityModel{});
  pops::CandidateRecoveryCensus expected, expected_active;
  for (std::int64_t ordinal = 0; ordinal < valid.numPts(); ++ordinal) {
    pops::Real conservative[2]{};
    census_cell_values(ordinal, conservative[0], conservative[1]);
    const bool finite = std::isfinite(conservative[0]) && std::isfinite(conservative[1]);
    const bool recovered = finite && recovery.recover(conservative).outcome.publication_permitted();
    pops::CandidateRecoveryCensus cell;
    cell.nonfinite_conservative = finite ? 0 : 1;
    cell.unrecoverable = finite && !recovered ? 1 : 0;
    expected.nonfinite_conservative += cell.nonfinite_conservative;
    expected.unrecoverable += cell.unrecoverable;
    if (ordinal % 5 != 2) {
      expected_active.nonfinite_conservative += cell.nonfinite_conservative;
      expected_active.unrecoverable += cell.unrecoverable;
    }
  }
  const std::int64_t points = valid.numPts();
  const auto kind_count = [points](std::int64_t kind) {
    return points / 5 + (points % 5 > kind ? 1 : 0);
  };
  // Every kind is present and the host recovery refuses exactly the negative and zero densities.
  ASSERT_GT(kind_count(3), 0);
  EXPECT_EQ(expected.nonfinite_conservative, kind_count(1));
  EXPECT_EQ(expected.unrecoverable, kind_count(2) + kind_count(3));
  EXPECT_EQ(expected_active.unrecoverable, kind_count(3));

  const auto check = pops::make_model_candidate_recovery_check<D>(PositiveDensityModel{});
  const pops::CandidateRecoveryCensus all = check(candidate, nullptr);
  EXPECT_EQ(all.nonfinite_conservative, expected.nonfinite_conservative);
  EXPECT_EQ(all.unrecoverable, expected.unrecoverable);
  const pops::CandidateRecoveryCensus masked = check(candidate, &active);
  EXPECT_EQ(masked.nonfinite_conservative, expected_active.nonfinite_conservative);
  EXPECT_EQ(masked.unrecoverable, expected_active.unrecoverable);

  pops::MultiFab<D> narrow(layout, distribution, pops::Index<D>{}, 1, ghosts);
  EXPECT_THROW((void)check(narrow, nullptr), std::invalid_argument);
  EXPECT_THROW((void)check(candidate, &candidate), std::invalid_argument);
}

TEST(GenericInversionAdmissibility, DiagnosticAndScheduleCollisionsFailClosed) {
  EXPECT_THROW(
      (pops::AdmissibleSet(pops::FiniteComponents<0, 1>{7}, pops::PositiveComponent<0>{0.0, 7})),
      std::invalid_argument);

  const pops::EnforcementSchedule schedule(std::array<pops::EnforcementRule, 5>{
      pops::EnforcementRule{true, true}, pops::EnforcementRule{true, false},
      pops::EnforcementRule{true, true}, pops::EnforcementRule{true, false},
      pops::EnforcementRule{true, false}});
  EXPECT_TRUE(schedule.at(pops::EnforcementPhase::kInitialization).project_if_invalid);
  EXPECT_FALSE(schedule.at(pops::EnforcementPhase::kAcceptance).project_if_invalid);
  EXPECT_FALSE(schedule.exact_contract().empty());

  EXPECT_THROW((pops::EnforcementSchedule(std::array<pops::EnforcementRule, 5>{
                   pops::EnforcementRule{false, true}, {}, {}, {}, {}})),
               std::invalid_argument);
}

}  // namespace
