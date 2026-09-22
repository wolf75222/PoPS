#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <pops/mesh/boundary/fill_boundary.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/index/box.hpp>
#include <pops/mesh/index/extent.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution.hpp>
#include <pops/mesh/layout/rank_space.hpp>
#include <pops/mesh/storage/field_view.hpp>
#include <pops/numerics/elliptic/linear/generic_krylov.hpp>
#include <pops/numerics/elliptic/poisson/poisson_operator.hpp>

namespace pops {
namespace {

inline constexpr int kDim = 2;
using TestBox = Box<kDim>;
using TestLayout = mesh::BoxArray<kDim>;
using TestRankSpace = mesh::RankSpace<kDim>;
using TestDistribution = mesh::Distribution<kDim>;
using TestGeometry = pops::Geometry<kDim>;
using TestField = pops::MultiFab<kDim>;
using TestVectorDistribution = pops::PreparedVectorDistribution<kDim>;
using TestAffineOperatorProvider = pops::PreparedAffineOperatorProvider<kDim>;
using TestAffineOperatorCallbacks = pops::PreparedAffineOperatorSessionCallbacks<kDim>;
using TestAffineOperatorSession = pops::PreparedAffineOperatorSession<kDim>;
using TestAffineProblem = pops::PreparedAffineLinearProblem<kDim>;
using TestLinearPreconditioner = pops::PreparedLinearPreconditioner<kDim>;
using TestLinearPreconditionerProvider = pops::PreparedLinearPreconditionerProvider<kDim>;
using TestLinearPreconditionerCallbacks = pops::PreparedLinearPreconditionerSessionCallbacks<kDim>;
using TestLinearPreconditionerSession = pops::PreparedLinearPreconditionerSession<kDim>;
using TestKrylovMethod = pops::PreparedKrylovMethod<kDim>;
using TestKrylovMethodProvider = pops::PreparedKrylovMethodProvider<kDim>;
using TestKrylovSolveContext = pops::PreparedKrylovSolveContext<kDim>;
using TestKrylovInvocation = pops::PreparedKrylovInvocation<kDim>;
using TestKrylovWorkspace = pops::KrylovWorkspace<kDim>;
using TestKrylovFootprint = pops::KrylovFootprint<kDim>;
using TestKrylovControls = pops::KrylovControls<kDim>;
using TestKrylovProblemFacts = pops::KrylovMethodProblemFacts<kDim>;
using TestKrylovWorkspaceRequest = pops::KrylovWorkspaceRequest<kDim>;
using TestNullspacePolicy = pops::PreparedNullspacePolicy<kDim>;

Index<kDim> rank_coordinate(int rank) {
  return Index<kDim>{rank, 0};
}

TestRankSpace world_rank_space() {
  return TestRankSpace{Index<kDim>{}, Extent<kDim>{n_ranks(), 1}};
}

Extent<kDim> extent(int value) {
  return Extent<kDim>{value, value};
}

TestDistribution round_robin_distribution(const TestLayout& layout) {
  std::vector<Index<kDim>> owners;
  owners.reserve(layout.size());
  for (std::size_t global = 0; global < layout.size(); ++global)
    owners.push_back(rank_coordinate(static_cast<int>(global) % n_ranks()));
  return TestDistribution::partitioned(layout, world_rank_space(), std::move(owners));
}

TestField make_field(const TestLayout& layout, const TestDistribution& distribution, int components,
                     int ghosts) {
  return TestField(layout, distribution, rank_coordinate(my_rank()), components, extent(ghosts));
}

HaloScheduleBudget halo_budget(std::size_t boxes) {
  constexpr std::size_t images = 64;
  return HaloScheduleBudget{{boxes, boxes * (boxes - 1) / 2},
                            boxes * boxes * images,
                            boxes * boxes * images * static_cast<std::size_t>(2 * kDim),
                            images,
                            boxes,
                            1'000'000,
                            1'000'000,
                            1'000'000};
}

OperatorEvaluationSnapshot test_snapshot(const TestField& prototype) {
  return {{UINT64_C(11), UINT64_C(12), UINT64_C(13), UINT64_C(14)},
          1,
          0,
          0,
          1,
          std::bit_cast<std::uint64_t>(1.0),
          0,
          1,
          detail::layout_fingerprint(prototype),
          {UINT64_C(21), UINT64_C(22), UINT64_C(23), UINT64_C(24)}};
}

constexpr Real kHelmholtzAlpha = Real(1e-3);

struct FillRhsKernel {
  FieldView<Real, kDim> values{};
  Real sign = Real(1);

  POPS_HD void operator()(const Index<kDim>& index) const {
    values(index, 0) =
        sign * (Real(1) + Real(0.125) * Real(index[0]) + Real(0.25) * Real(index[1]));
  }
};

void fill_rhs(TestField& field, Real sign) {
  for (std::size_t local = 0; local < field.local_size(); ++local)
    for_each_cell(field.box(local), FillRhsKernel{field.fab(local).view(), sign});
}

struct AbsoluteDifferenceKernel {
  FieldView<const Real, kDim> left{};
  FieldView<const Real, kDim> right{};

  POPS_HD Real operator()(const Index<kDim>& index) const {
    const Real difference = left(index, 0) - right(index, 0);
    return difference < Real(0) ? -difference : difference;
  }
};

struct PositiveDiagonalKernel {
  FieldView<Real, kDim> out{};
  FieldView<const Real, kDim> in{};

  POPS_HD void operator()(const Index<kDim>& index) const {
    out(index, 0) = (Real(1) + Real(index[0])) * in(index, 0);
  }
};

struct NonNanFlagKernel {
  FieldView<const Real, kDim> values{};
  POPS_HD Real operator()(const Index<kDim>& index) const {
    return values(index, 0) == values(index, 0) ? Real(1) : Real(0);
  }
};

Real max_abs_diff(const TestField& left, const TestField& right) {
  Real local = Real(0);
  for (std::size_t patch = 0; patch < left.local_size(); ++patch)
    local = std::max(
        local, for_each_cell_reduce_max(
                   left.box(patch),
                   AbsoluteDifferenceKernel{left.fab(patch).view(), right.fab(patch).view()}));
  return static_cast<Real>(all_reduce_max(static_cast<double>(local)));
}

bool has_remote_face_neighbor(const TestLayout& boxes, const TestDistribution& distribution,
                              int rank) {
  const Index<kDim> local_rank = rank_coordinate(rank);
  for (std::size_t i = 0; i < boxes.size(); ++i) {
    if (distribution.owner(i) != local_rank)
      continue;
    for (std::size_t j = 0; j < boxes.size(); ++j) {
      if (distribution.owner(j) == local_rank)
        continue;
      const TestBox& left = boxes[i];
      const TestBox& right = boxes[j];
      const bool x_face = (left.hi[0] + 1 == right.lo[0] || right.hi[0] + 1 == left.lo[0]) &&
                          left.lo[1] <= right.hi[1] && right.lo[1] <= left.hi[1];
      const bool y_face = (left.hi[1] + 1 == right.lo[1] || right.hi[1] + 1 == left.lo[1]) &&
                          left.lo[0] <= right.hi[0] && right.lo[0] <= left.hi[0];
      if (x_face || y_face)
        return true;
    }
  }
  return false;
}

#ifdef POPS_HAS_MPI
class ScopedMpiCommunicator {
 public:
  explicit ScopedMpiCommunicator(MPI_Comm communicator) : communicator_(communicator) {}
  ScopedMpiCommunicator(const ScopedMpiCommunicator&) = delete;
  ScopedMpiCommunicator& operator=(const ScopedMpiCommunicator&) = delete;
  ~ScopedMpiCommunicator() {
    if (communicator_ != MPI_COMM_NULL && comm_active())
      (void)MPI_Comm_free(&communicator_);
  }

  [[nodiscard]] MPI_Comm get() const noexcept { return communicator_; }

 private:
  MPI_Comm communicator_ = MPI_COMM_NULL;
};
#endif

struct ConcurrentSessionProbe {
  std::atomic<int> next_session{0};
  std::atomic<bool> armed{false};
  std::mutex mutex;
  std::condition_variable ready;
  int arrivals = 0;
  bool timed_out = false;
  std::set<int> solve_session_ids;
  std::set<const void*> solve_resource_addresses;

  void rendezvous(int session_id, const void* resource_address) {
    std::unique_lock lock(mutex);
    solve_session_ids.insert(session_id);
    solve_resource_addresses.insert(resource_address);
    ++arrivals;
    if (arrivals == 2) {
      ready.notify_all();
      return;
    }
    if (!ready.wait_for(lock, std::chrono::seconds(5), [&] { return arrivals == 2; }))
      timed_out = true;
  }
};

struct ConcurrentSessionState {
  std::shared_ptr<ConcurrentSessionProbe> probe;
  int session_id = 0;
  std::atomic<bool> first_solve_apply{true};

  void apply(TestField& out, const TestField& in) {
    if (probe->armed.load(std::memory_order_acquire) &&
        first_solve_apply.exchange(false, std::memory_order_acq_rel))
      probe->rendezvous(session_id, this);
    detail::PreparedFieldAlgebra::copy(out, in);
  }

  [[nodiscard]] std::size_t allocation_count() const noexcept { return 0u; }
};

struct ConcurrentOperatorSessionState {
  std::shared_ptr<ConcurrentSessionProbe> probe;
  int session_id = 0;
  std::atomic<bool> first_solve_apply{true};
  TestGeometry geometry;
  const ExecutionLane* lane = nullptr;
  TestField scratch;
  HaloSchedule<kDim> halo_schedule;
  std::unique_ptr<HaloExchange<kDim>> halo_exchange;

  ConcurrentOperatorSessionState(std::shared_ptr<ConcurrentSessionProbe> session_probe, int id,
                                 TestGeometry session_geometry, const ExecutionLane& session_lane,
                                 const TestLayout& boxes, const TestDistribution& mapping)
      : probe(std::move(session_probe)),
        session_id(id),
        geometry(session_geometry),
        lane(&session_lane),
        scratch(make_field(boxes, mapping, 1, 1)),
        halo_schedule(prepare_halo_schedule(
            scratch, geometry.domain(),
            BoundaryTopology<kDim>::axis_periodic(std::array<bool, kDim>{true, true}),
            halo_budget(boxes.size()))) {
    if (halo_schedule.has_remote_jobs()) {
      HaloExchangeContext context{};
      context.context_generation = static_cast<std::uint64_t>(2 * session_id + 1);
      context.schedule_generation = static_cast<std::uint64_t>(2 * session_id + 2);
      halo_exchange = std::make_unique<HaloExchange<kDim>>(halo_schedule, session_lane, context);
    }
  }

  void apply(TestField& out, const TestField& in) {
    if (probe->armed.load(std::memory_order_acquire) &&
        first_solve_apply.exchange(false, std::memory_order_acq_rel))
      probe->rendezvous(session_id, &scratch);
    TestField& mutable_input = const_cast<TestField&>(in);
    if (halo_exchange)
      fill_boundary(mutable_input, *halo_exchange, *lane);
    else
      fill_boundary(mutable_input, halo_schedule);
    elliptic::mg::apply_poisson_operator_valid(in, geometry, scratch, Real(1) / kHelmholtzAlpha);
    detail::PreparedFieldAlgebra::lincomb(out, kHelmholtzAlpha, scratch, Real(0), in);
  }

  [[nodiscard]] std::size_t allocation_count() const noexcept { return 1u; }
};

class LongReasonKrylovProvider final : public TestKrylovMethodProvider {
 public:
  static constexpr std::size_t kReasonBytes = 16 * 1024 + 37;
  enum class Behavior { kNumericalFailure, kExplicitTerminalFailure, kUnverifiedSolved };
  explicit LongReasonKrylovProvider(Behavior behavior = Behavior::kNumericalFailure)
      : behavior_(behavior) {}

  std::string_view identity() const noexcept override {
    return "pops.test.krylov.long-report-reason";
  }
  std::uint64_t interface_version() const noexcept override { return 1; }
  std::string_view collective_contract() const noexcept override {
    return "pops.test.krylov.long-report-reason@1";
  }
  KrylovMethodValidation validate_controls(const KrylovMethodControls&,
                                           const PreparedProviderOptions&) const noexcept override {
    return KrylovMethodValidation::accept();
  }
  KrylovMethodValidation validate_problem(const TestKrylovProblemFacts& facts,
                                          const PreparedProviderOptions&) const noexcept override {
    return facts.has_preconditioner
               ? KrylovMethodValidation::reject(1, "long-reason probe is unpreconditioned")
               : KrylovMethodValidation::accept();
  }
  KrylovWorkspaceRequirements workspace_requirements(
      const TestKrylovWorkspaceRequest&, const PreparedProviderOptions&) const override {
    return {.field_count = 2, .initial_residual_field = 1};
  }
  SolveReport solve(TestKrylovSolveContext& context,
                    const PreparedProviderOptions&) const override {
    SolveReport report =
        context.report(context.initial_physical_residual(), 1,
                       behavior_ == Behavior::kUnverifiedSolved ? SolveStatus::kSolved
                                                                : SolveStatus::kIterationLimit);
    if (behavior_ == Behavior::kExplicitTerminalFailure)
      report.action = SolveAction::kFailRun;
    report.reason.assign(kReasonBytes, 'r');
    return report;
  }

 private:
  Behavior behavior_;
};

class SingleFieldKrylovProvider final : public TestKrylovMethodProvider {
 public:
  std::string_view identity() const noexcept override {
    return "pops.test.krylov.single-field-workspace";
  }
  std::uint64_t interface_version() const noexcept override { return 1; }
  std::string_view collective_contract() const noexcept override {
    return "pops.test.krylov.single-field-workspace@1";
  }
  KrylovMethodValidation validate_controls(const KrylovMethodControls&,
                                           const PreparedProviderOptions&) const noexcept override {
    return KrylovMethodValidation::accept();
  }
  KrylovMethodValidation validate_problem(const TestKrylovProblemFacts&,
                                          const PreparedProviderOptions&) const noexcept override {
    return KrylovMethodValidation::accept();
  }
  KrylovWorkspaceRequirements workspace_requirements(
      const TestKrylovWorkspaceRequest&, const PreparedProviderOptions&) const override {
    return {.field_count = 1, .initial_residual_field = 0};
  }
  SolveReport solve(TestKrylovSolveContext&, const PreparedProviderOptions&) const override {
    throw std::logic_error("invalid single-field provider must be rejected before solve");
  }
};

struct SessionLifecycleProbe {
  std::atomic<int> factories{0};
  std::atomic<int> states{0};
  std::atomic<int> prepares{0};
  std::atomic<int> applies{0};
};

struct SessionLifecycleState {
  explicit SessionLifecycleState(std::shared_ptr<SessionLifecycleProbe> lifecycle_probe)
      : probe(std::move(lifecycle_probe)) {
    probe->states.fetch_add(1, std::memory_order_relaxed);
  }

  void prepare() { probe->prepares.fetch_add(1, std::memory_order_relaxed); }
  void apply(TestField& out, const TestField& in) {
    probe->applies.fetch_add(1, std::memory_order_relaxed);
    detail::PreparedFieldAlgebra::copy(out, in);
  }
  [[nodiscard]] std::size_t allocation_count() const noexcept { return 0u; }

  std::shared_ptr<SessionLifecycleProbe> probe;
};

class BlockingPrepareGate {
 public:
  void arm() {
    std::lock_guard lock(mutex_);
    armed_ = true;
    entered_ = false;
    released_ = false;
    timed_out_ = false;
  }

  void block_if_armed() {
    std::unique_lock lock(mutex_);
    if (!armed_)
      return;
    armed_ = false;
    entered_ = true;
    changed_.notify_all();
    if (!changed_.wait_for(lock, std::chrono::seconds(10), [&] { return released_; }))
      timed_out_ = true;
  }

  [[nodiscard]] bool wait_until_entered() {
    std::unique_lock lock(mutex_);
    return changed_.wait_for(lock, std::chrono::seconds(10), [&] { return entered_; });
  }

  void release() {
    {
      std::lock_guard lock(mutex_);
      released_ = true;
    }
    changed_.notify_all();
  }

  [[nodiscard]] bool timed_out() const {
    std::lock_guard lock(mutex_);
    return timed_out_;
  }

 private:
  mutable std::mutex mutex_;
  std::condition_variable changed_;
  bool armed_ = false;
  bool entered_ = false;
  bool released_ = false;
  bool timed_out_ = false;
};

TestAffineOperatorProvider blocking_prepare_operator_provider(
    const std::shared_ptr<BlockingPrepareGate>& gate) {
  return TestAffineOperatorProvider::trusted_extension(
      {"pops.test.krylov.blocking-prepare-operator", 1}, {}, [gate](const ExecutionLane&) {
        return TestAffineOperatorCallbacks{[gate] { gate->block_if_armed(); },
                                           [](TestField& out, const TestField& in) {
                                             detail::PreparedFieldAlgebra::copy(out, in);
                                           },
                                           [] { return std::size_t{0}; }};
      });
}

struct PotentiallyThrowingSessionContract {
  void prepare() {}
  PreparedApplyResult apply(TestField&, const TestField&) { return PreparedApplyResult::success(); }
  [[nodiscard]] std::size_t allocation_count() const { return 0; }
};

struct NothrowSessionContract {
  void prepare() {}
  PreparedApplyResult apply(TestField&, const TestField&) noexcept {
    return PreparedApplyResult::success();
  }
  [[nodiscard]] std::size_t allocation_count() const { return 0; }
};

static_assert(!PreparedAffineOperatorSessionSource<kDim, PotentiallyThrowingSessionContract>);
static_assert(!PreparedLinearPreconditionerSessionSource<kDim, PotentiallyThrowingSessionContract>);
static_assert(PreparedAffineOperatorSessionSource<kDim, NothrowSessionContract>);
static_assert(PreparedLinearPreconditionerSessionSource<kDim, NothrowSessionContract>);

TestAffineOperatorProvider lifecycle_operator_provider(
    const std::shared_ptr<SessionLifecycleProbe>& probe) {
  return TestAffineOperatorProvider::trusted_extension(
      {"pops.test.krylov.lifecycle-operator", 1}, {}, [probe](const ExecutionLane&) {
        probe->factories.fetch_add(1, std::memory_order_relaxed);
        auto state = std::make_shared<SessionLifecycleState>(probe);
        return TestAffineOperatorCallbacks{
            [state] { state->prepare(); },
            [state](TestField& out, const TestField& in) { state->apply(out, in); },
            [state] { return state->allocation_count(); }};
      });
}

TestLinearPreconditionerProvider lifecycle_preconditioner_provider(
    const std::shared_ptr<SessionLifecycleProbe>& probe) {
  return TestLinearPreconditionerProvider::trusted_extension(
      {"pops.test.krylov.lifecycle-preconditioner", 1}, {}, [probe](const ExecutionLane&) {
        probe->factories.fetch_add(1, std::memory_order_relaxed);
        auto state = std::make_shared<SessionLifecycleState>(probe);
        return TestLinearPreconditionerCallbacks{
            [state] { state->prepare(); },
            [state](TestField& out, const TestField& in) { state->apply(out, in); },
            [state] { return state->allocation_count(); }};
      });
}

TEST(test_krylov_workspace_reentrancy,
     provider_workspace_with_only_one_core_field_is_rejected_before_materialization) {
  const TestKrylovMethod method(
      std::make_shared<SingleFieldKrylovProvider>(),
      PreparedProviderOptions{"pops.test.krylov.single-field-workspace.options@1", {}});
  EXPECT_THROW((void)method.workspace_requirements(TestKrylovWorkspaceRequest{
                   TestKrylovFootprint{1, extent(0), false}, TestVectorDistribution::Distributed,
                   detail::PreparedFieldAlgebra::kRobustDotPayloadWidth}),
               std::invalid_argument);
}

TEST(test_krylov_workspace_reentrancy,
     trusted_prepare_exceptions_propagate_but_hot_apply_exceptions_publish_failure_status) {
  const TestLayout boxes(std::vector<TestBox>{TestBox{Index<kDim>{0, 0}, Index<kDim>{1, 1}}});
  const TestDistribution mapping = round_robin_distribution(boxes);
  TestField input = make_field(boxes, mapping, 1, 0);
  TestField output = make_field(boxes, mapping, 1, 0);
  input.set_val(Real(1));
  output.set_val(Real(7));
  const ExecutionLane lane = ExecutionLane::world("pops.test.krylov.trusted-failure");

  TestAffineOperatorProvider provider = TestAffineOperatorProvider::trusted_extension(
      {"pops.test.krylov.throwing-trusted-operator", 1}, {}, [](const ExecutionLane&) {
        return TestAffineOperatorCallbacks{[] { throw std::runtime_error("prepare failure"); },
                                           [](TestField& out, const TestField& in) {
                                             detail::PreparedFieldAlgebra::copy(out, in);
                                           },
                                           [] { return std::size_t{0}; }};
      });
  TestAffineOperatorSession session = provider.make_session(lane);

  EXPECT_THROW(session.prepare(), std::runtime_error);

  output.set_val(Real(9));
  TestLinearPreconditionerProvider preconditioner_provider =
      TestLinearPreconditionerProvider::trusted_extension(
          {"pops.test.krylov.throwing-trusted-preconditioner", 1}, {}, [](const ExecutionLane&) {
            return TestLinearPreconditionerCallbacks{
                {},
                [](TestField&, const TestField&) { throw std::runtime_error("apply failure"); },
                [] { return std::size_t{0}; }};
          });
  TestLinearPreconditionerSession preconditioner_session =
      preconditioner_provider.make_session(lane);
  EXPECT_NO_THROW(preconditioner_session.prepare());
  const PreparedApplyResult unknown_failure = preconditioner_session.apply(output, input);
  EXPECT_FALSE(unknown_failure.succeeded());
  EXPECT_EQ(unknown_failure.kind, PreparedApplyFailureKind::kUnknownException);
  EXPECT_EQ(unknown_failure.action, PreparedApplyFailureAction::kFailRun);
  EXPECT_EQ(preconditioner_session.allocation_count(), 0u);
  for (std::size_t local = 0; local < output.local_size(); ++local)
    EXPECT_EQ(for_each_cell_reduce_max(output.box(local),
                                       NonNanFlagKernel{std::as_const(output).fab(local).view()}),
              Real(0));

  constexpr std::uint32_t reason_code = UINT32_C(0x52454a31);
  TestAffineOperatorProvider flux_provider = TestAffineOperatorProvider::trusted_extension(
      {"pops.test.krylov.flux-failure-operator", 1}, {}, [](const ExecutionLane&) {
        return TestAffineOperatorCallbacks{{},
                                           [](TestField&, const TestField&) {
                                             throw FluxEvaluationFailure(EvaluationStatus::kReject,
                                                                         reason_code,
                                                                         "test_flux_phase");
                                           },
                                           [] { return std::size_t{0}; }};
      });
  TestAffineOperatorSession flux_session = flux_provider.make_session(lane);
  flux_session.prepare();
  const PreparedApplyResult flux_failure = flux_session.apply(output, input);
  EXPECT_FALSE(flux_failure.succeeded());
  EXPECT_EQ(flux_failure.kind, PreparedApplyFailureKind::kFluxEvaluation);
  EXPECT_EQ(flux_failure.action, PreparedApplyFailureAction::kRejectAttempt);
  EXPECT_EQ(flux_failure.evaluation_status, EvaluationStatus::kReject);
  EXPECT_EQ(flux_failure.reason_code, reason_code);
  EXPECT_EQ(flux_failure.phase(), "test_flux_phase");
  EXPECT_FALSE(flux_failure.phase_truncated);
}

TEST(test_krylov_workspace_reentrancy,
     repeated_prepare_and_bind_refresh_sessions_without_rematerializing_their_source) {
  const TestBox domain{Index<kDim>{0, 0}, Index<kDim>{3, 3}};
  const TestLayout boxes = TestLayout::from_domain(domain, extent(2));
  const TestDistribution mapping = round_robin_distribution(boxes);
  TestField prototype = make_field(boxes, mapping, 1, 0);
  prototype.set_val(Real(0));
  OperatorEvaluationSnapshot snapshot = test_snapshot(prototype);
  const auto operator_probe = std::make_shared<SessionLifecycleProbe>();
  const auto preconditioner_probe = std::make_shared<SessionLifecycleProbe>();
  TestAffineOperatorProvider shared_operator = lifecycle_operator_provider(operator_probe);
  TestLinearPreconditionerProvider shared_preconditioner =
      lifecycle_preconditioner_provider(preconditioner_probe);
  const TestKrylovFootprint footprint{1, extent(0), true};

  TestAffineProblem first_problem(
      prototype, shared_operator, TestLinearPreconditioner(prototype, shared_preconditioner),
      LinearOperatorProperties::general(), footprint, TestNullspacePolicy::nonsingular(),
      [&snapshot] { return snapshot; });
  TestKrylovWorkspace workspace(prototype, bicgstab_krylov_method<kDim>(), footprint);

  first_problem.prepare(snapshot);
  workspace.bind(first_problem);
  EXPECT_EQ(operator_probe->factories.load(), 2);
  EXPECT_EQ(operator_probe->states.load(), 2);
  EXPECT_EQ(operator_probe->prepares.load(), 2);
  EXPECT_EQ(preconditioner_probe->factories.load(), 2);
  EXPECT_EQ(preconditioner_probe->states.load(), 2);
  EXPECT_EQ(preconditioner_probe->prepares.load(), 2);

  ++snapshot.revision;
  ++snapshot.macro_step;
  first_problem.prepare(snapshot);
  workspace.bind(first_problem);
  EXPECT_EQ(operator_probe->factories.load(), 2);
  EXPECT_EQ(operator_probe->states.load(), 2);
  EXPECT_EQ(operator_probe->prepares.load(), 4);
  EXPECT_EQ(preconditioner_probe->factories.load(), 2);
  EXPECT_EQ(preconditioner_probe->states.load(), 2);
  EXPECT_EQ(preconditioner_probe->prepares.load(), 4);

  // A second problem can share the exact immutable provider source. Its own problem-level session is
  // private, while the workspace can refresh its existing private session instead of keying on the
  // problem object's address.
  TestAffineProblem same_source_problem(
      prototype, shared_operator, TestLinearPreconditioner(prototype, shared_preconditioner),
      LinearOperatorProperties::general(), footprint, TestNullspacePolicy::nonsingular(),
      [&snapshot] { return snapshot; });
  same_source_problem.prepare(snapshot);
  workspace.bind(same_source_problem);
  EXPECT_EQ(operator_probe->factories.load(), 3);
  EXPECT_EQ(operator_probe->states.load(), 3);
  EXPECT_EQ(operator_probe->prepares.load(), 6);
  EXPECT_EQ(preconditioner_probe->factories.load(), 3);
  EXPECT_EQ(preconditioner_probe->states.load(), 3);
  EXPECT_EQ(preconditioner_probe->prepares.load(), 6);

  // The same semantic contract from a new concrete provider source must not inherit an opaque
  // session created by the old source.
  TestAffineProblem different_source_problem(
      prototype, lifecycle_operator_provider(operator_probe),
      TestLinearPreconditioner(prototype, lifecycle_preconditioner_provider(preconditioner_probe)),
      LinearOperatorProperties::general(), footprint, TestNullspacePolicy::nonsingular(),
      [&snapshot] { return snapshot; });
  different_source_problem.prepare(snapshot);
  workspace.bind(different_source_problem);
  EXPECT_EQ(operator_probe->factories.load(), 5);
  EXPECT_EQ(operator_probe->states.load(), 5);
  EXPECT_EQ(operator_probe->prepares.load(), 8);
  EXPECT_EQ(preconditioner_probe->factories.load(), 5);
  EXPECT_EQ(preconditioner_probe->states.load(), 5);
  EXPECT_EQ(preconditioner_probe->prepares.load(), 8);
}

TEST(test_krylov_workspace_reentrancy,
     workspace_rebind_reserves_mutation_during_blocking_operator_prepare) {
  comm_init();
  const TestLayout boxes(std::vector<TestBox>{TestBox{Index<kDim>{0, 0}, Index<kDim>{3, 3}}});
  const TestDistribution mapping = round_robin_distribution(boxes);
  TestField prototype = make_field(boxes, mapping, 1, 0);
  prototype.set_val(Real(0));
  OperatorEvaluationSnapshot snapshot = test_snapshot(prototype);
  const auto gate = std::make_shared<BlockingPrepareGate>();
  const TestKrylovFootprint footprint{1, extent(0), false};
  TestAffineProblem problem(prototype, blocking_prepare_operator_provider(gate),
                            TestLinearPreconditioner::identity(),
                            LinearOperatorProperties::symmetric_positive_definite(), footprint,
                            TestNullspacePolicy::nonsingular(), [&snapshot] { return snapshot; });
  const TestKrylovMethod method = cg_krylov_method<kDim>();
  TestKrylovWorkspace workspace(prototype, method, footprint);
  TestField iterate = make_field(boxes, mapping, 1, 0);
  TestField rhs = make_field(boxes, mapping, 1, 0);
  iterate.set_val(Real(0));
  rhs.set_val(Real(1));
  const TestKrylovControls controls{method, Real(1e-12), Real(0), 4};
  problem.prepare(snapshot);
  workspace.bind(problem);

  gate->arm();
  bool rebind_completed = false;
  std::exception_ptr rebind_failure;
  std::thread rebind([&] {
    try {
      workspace.bind(problem);
      rebind_completed = true;
    } catch (...) {
      rebind_failure = std::current_exception();
    }
  });

  const bool entered_locally = gate->wait_until_entered();
  const bool entered_on_every_rank = all_reduce_min(entered_locally ? 1L : 0L) != 0;
  bool rejected_as_logic_error = false;
  std::string rejection;
  if (entered_on_every_rank) {
    try {
      (void)detail::prepare_krylov_solve_in_place(problem, workspace, iterate, rhs, controls);
    } catch (const std::logic_error& error) {
      rejected_as_logic_error = true;
      rejection = error.what();
    } catch (const std::exception& error) {
      rejection = error.what();
    } catch (...) {
      rejection = "non-standard exception";
    }
  }

  gate->release();
  rebind.join();

  EXPECT_TRUE(entered_on_every_rank);
  EXPECT_FALSE(gate->timed_out());
  EXPECT_EQ(rebind_failure, nullptr);
  EXPECT_TRUE(rebind_completed);
  if (entered_on_every_rank) {
    EXPECT_EQ(rejection,
              "KrylovWorkspace is already reserved by another prepared bind or solve invocation");
    EXPECT_EQ(all_reduce_min(rejected_as_logic_error ? 1L : 0L), 1L);
    EXPECT_TRUE(all_ranks_agree_exact_ordered_byte_pairs(
        {{std::string_view("workspace-rebind-reservation"), std::string_view(rejection)}}));
  }
}

TEST(test_krylov_workspace_reentrancy,
     problem_prepare_reserves_mutation_during_blocking_resource_freeze) {
  comm_init();
  const TestLayout boxes(std::vector<TestBox>{TestBox{Index<kDim>{0, 0}, Index<kDim>{3, 3}}});
  const TestDistribution mapping = round_robin_distribution(boxes);
  TestField prototype = make_field(boxes, mapping, 1, 0);
  prototype.set_val(Real(0));
  OperatorEvaluationSnapshot snapshot = test_snapshot(prototype);
  const auto gate = std::make_shared<BlockingPrepareGate>();
  const TestKrylovFootprint footprint{1, extent(0), false};
  TestAffineProblem problem(
      prototype,
      TestAffineOperatorProvider::trusted_reentrant(
          [](TestField& out, const TestField& in) { detail::PreparedFieldAlgebra::copy(out, in); },
          [] { return std::size_t{0}; }),
      TestLinearPreconditioner::identity(), LinearOperatorProperties::symmetric_positive_definite(),
      footprint, TestNullspacePolicy::nonsingular(), [&snapshot] { return snapshot; },
      [gate] { gate->block_if_armed(); });
  const TestKrylovMethod method = cg_krylov_method<kDim>();
  TestKrylovWorkspace workspace(prototype, method, footprint);
  TestField iterate = make_field(boxes, mapping, 1, 0);
  TestField rhs = make_field(boxes, mapping, 1, 0);
  iterate.set_val(Real(0));
  rhs.set_val(Real(1));
  const TestKrylovControls controls{method, Real(1e-12), Real(0), 4};
  problem.prepare(snapshot);
  workspace.bind(problem);

  ++snapshot.revision;
  ++snapshot.macro_step;
  gate->arm();
  bool prepare_completed = false;
  std::exception_ptr prepare_failure;
  std::thread prepare([&] {
    try {
      problem.prepare(snapshot);
      prepare_completed = true;
    } catch (...) {
      prepare_failure = std::current_exception();
    }
  });

  const bool entered_locally = gate->wait_until_entered();
  const bool entered_on_every_rank = all_reduce_min(entered_locally ? 1L : 0L) != 0;
  bool rejected_as_logic_error = false;
  std::string rejection;
  if (entered_on_every_rank) {
    try {
      (void)detail::prepare_krylov_solve_in_place(problem, workspace, iterate, rhs, controls);
    } catch (const std::logic_error& error) {
      rejected_as_logic_error = true;
      rejection = error.what();
    } catch (const std::exception& error) {
      rejection = error.what();
    } catch (...) {
      rejection = "non-standard exception";
    }
  }

  gate->release();
  prepare.join();

  EXPECT_TRUE(entered_on_every_rank);
  EXPECT_FALSE(gate->timed_out());
  EXPECT_EQ(prepare_failure, nullptr);
  EXPECT_TRUE(prepare_completed);
  EXPECT_TRUE(problem.prepared());
  if (problem.prepared())
    EXPECT_EQ(problem.snapshot(), snapshot);
  if (entered_on_every_rank) {
    EXPECT_EQ(rejection,
              "prepared affine problem is being mutated or its operator requires exclusive "
              "access to its external execution context");
    EXPECT_EQ(all_reduce_min(rejected_as_logic_error ? 1L : 0L), 1L);
    EXPECT_TRUE(all_ranks_agree_exact_ordered_byte_pairs(
        {{std::string_view("problem-prepare-reservation"), std::string_view(rejection)}}));
  }
}

TEST(test_krylov_workspace_reentrancy,
     rank_local_problem_construction_failure_is_published_before_lane_unwind) {
#ifndef POPS_HAS_MPI
  GTEST_SKIP() << "rank-local constructor divergence requires MPI";
#else
  comm_init();
  if (n_ranks() < 2)
    GTEST_SKIP() << "rank-local constructor divergence requires multiple MPI ranks";

  const TestLayout boxes(std::vector<TestBox>{TestBox{Index<kDim>{0, 0}, Index<kDim>{1, 1}}});
  const TestDistribution mapping = round_robin_distribution(boxes);
  TestField prototype = make_field(boxes, mapping, 1, 0);
  prototype.set_val(Real(0));
  OperatorEvaluationSnapshot snapshot = test_snapshot(prototype);
  const LinearOperatorProperties properties = my_rank() == 0
                                                  ? LinearOperatorProperties{operator_property_bit(
                                                        LinearOperatorProperty::kPositiveDefinite)}
                                                  : LinearOperatorProperties::general();

  std::string rejection;
  bool invalid_argument = false;
  try {
    (void)TestAffineProblem(prototype,
                            TestAffineOperatorProvider::trusted_reentrant(
                                [](TestField& out, const TestField& in) {
                                  detail::PreparedFieldAlgebra::copy(out, in);
                                },
                                [] { return std::size_t{0}; }),
                            TestLinearPreconditioner::identity(), properties,
                            TestKrylovFootprint{1, extent(0), false},
                            TestNullspacePolicy::nonsingular(), [&snapshot] { return snapshot; });
  } catch (const std::invalid_argument& error) {
    invalid_argument = true;
    rejection = error.what();
  } catch (const std::exception& error) {
    rejection = error.what();
  }

  EXPECT_TRUE(invalid_argument);
  EXPECT_FALSE(rejection.empty());
  EXPECT_EQ(rejection,
            "PreparedAffineLinearProblem received invalid construction arguments on at least one "
            "communicator rank");
  EXPECT_EQ(all_reduce_min(invalid_argument ? 1L : 0L), 1L);
  EXPECT_EQ(all_reduce_min(rejection.empty() ? 0L : 1L), 1L);
  EXPECT_TRUE(all_ranks_agree_exact_ordered_byte_pairs(
      {{std::string_view("prepared-problem-constructor-failure"), std::string_view(rejection)}}));
#endif
}

// The injection belongs only to the shared candidate allocator. It neither replaces global new
// nor reaches a numerical/provider allocation after the constructor has entered its private lane.
struct CandidateAllocationState {
  bool fail_once = false;
  int allocations = 0;
};

template <class T>
struct CandidateAllocator {
  using value_type = T;
  CandidateAllocationState* state;
  explicit CandidateAllocator(CandidateAllocationState& value) noexcept : state(&value) {}
  template <class U>
  CandidateAllocator(const CandidateAllocator<U>& other) noexcept : state(other.state) {}
  T* allocate(std::size_t count) {
    ++state->allocations;
    if (std::exchange(state->fail_once, false))
      throw std::bad_alloc();
    return std::allocator<T>{}.allocate(count);
  }
  void deallocate(T* pointer, std::size_t count) noexcept {
    std::allocator<T>{}.deallocate(pointer, count);
  }
  template <class U>
  bool operator==(const CandidateAllocator<U>& other) const noexcept {
    return state == other.state;
  }
};

void verify_shared_construction_failure_and_retry(bool fail_problem) {
  comm_init();
  const auto embedding = ExecutionLane::duplicate_world_collectively("test.shared-construction");
#ifdef POPS_HAS_MPI
  const auto parent =
      ExecutionCommunicator::borrowed(embedding.identity(), embedding.native_handle());
#else
  const auto parent = ExecutionCommunicator::world();
#endif
  const TestLayout boxes(std::vector<TestBox>{TestBox{Index<kDim>{0, 0}, Index<kDim>{1, 1}}});
  const TestDistribution mapping = round_robin_distribution(boxes);
  TestField iterate = make_field(boxes, mapping, 1, 0);
  TestField rhs = make_field(boxes, mapping, 1, 0);
  rhs.set_val(Real(1));
  OperatorEvaluationSnapshot snapshot = test_snapshot(iterate);
  const TestKrylovFootprint footprint{1, extent(0), false};
  const auto method = cg_krylov_method<kDim>();

  for (const bool fail_allocation : {true, false}) {
    SCOPED_TRACE(fail_allocation ? "shared owner allocation" : "typed input preparation");
    iterate.set_val(Real(0));
    CandidateAllocationState problem_allocation{fail_problem && fail_allocation && my_rank() == 0};
    CandidateAllocationState workspace_allocation{!fail_problem && fail_allocation &&
                                                  my_rank() == 0};
    bool fail_problem_input = fail_problem && !fail_allocation && my_rank() == 0;
    bool fail_workspace_input = !fail_problem && !fail_allocation && my_rank() == 0;
    int problem_inputs = 0;
    int workspace_inputs = 0;
    auto make_problem = [&]() {
      return TestAffineProblem::make_shared_collectively(
          parent, "test.shared-problem",
          [&]() {
            ++problem_inputs;
            if (std::exchange(fail_problem_input, false))
              throw std::invalid_argument("injected problem input preparation");
            return TestAffineProblem::ConstructionInputs{
                std::cref(iterate),
                TestAffineOperatorProvider::trusted_reentrant(
                    [](TestField& out, const TestField& in) {
                      detail::PreparedFieldAlgebra::copy(out, in);
                    },
                    [] { return std::size_t{0}; }),
                TestLinearPreconditioner::identity(),
                LinearOperatorProperties::symmetric_positive_definite(),
                footprint,
                TestNullspacePolicy::nonsingular(),
                [&snapshot] { return snapshot; },
                {},
                TestVectorDistribution::Distributed};
          },
          CandidateAllocator<std::optional<TestAffineProblem>>{problem_allocation});
    };
    auto make_workspace = [&]() {
      return TestKrylovWorkspace::make_shared_collectively(
          parent, "test.shared-workspace",
          [&]() {
            ++workspace_inputs;
            if (std::exchange(fail_workspace_input, false))
              throw std::invalid_argument("injected workspace input preparation");
            return TestKrylovWorkspace::ConstructionInputs{"test.shared-workspace.materialization",
                                                           std::cref(iterate), method, footprint,
                                                           TestVectorDistribution::Distributed};
          },
          CandidateAllocator<std::optional<TestKrylovWorkspace>>{workspace_allocation});
    };
    std::string diagnostic;
    try {
      if (fail_problem)
        (void)make_problem();
      else
        (void)make_workspace();
    } catch (const std::exception& error) {
      diagnostic = error.what();
    }
    // Both failures must leave the supplied parent usable and publish the same refusal. In the
    // allocation branch the failing rank cannot even prepare its typed arguments, let alone T.
    EXPECT_EQ(all_reduce_min(diagnostic.empty() ? 0L : 1L, parent.communicator()), 1L);
    EXPECT_TRUE(all_ranks_agree_exact_ordered_byte_pairs(
        {{std::string_view("shared-construction-refusal"), std::string_view(diagnostic)}},
        parent.communicator()));
    EXPECT_EQ(fail_problem ? problem_inputs : workspace_inputs,
              fail_allocation && my_rank() == 0 ? 0 : 1);
    EXPECT_EQ(fail_problem ? problem_allocation.allocations : workspace_allocation.allocations, 1);
    EXPECT_EQ(max_abs_diff(iterate, rhs), Real(1));

    // The same allocator and input closures retry successfully. Copies of the aliasing owner keep
    // the in-place object and private communicator alive after the original handle is released.
    auto problem = make_problem();
    auto workspace = make_workspace();
    auto retained_problem = problem;
    auto retained_workspace = workspace;
    std::weak_ptr<TestAffineProblem> problem_lifetime = problem;
    std::weak_ptr<TestKrylovWorkspace> workspace_lifetime = workspace;
    problem.reset();
    workspace.reset();
    EXPECT_FALSE(problem_lifetime.expired());
    EXPECT_FALSE(workspace_lifetime.expired());
    retained_problem->prepare(snapshot);
    retained_workspace->bind(*retained_problem);
    const SolveReport report =
        detail::solve_prepared_affine_in_place(*retained_problem, *retained_workspace, iterate, rhs,
                                               TestKrylovControls{method, Real(1e-12), Real(0), 3});
    EXPECT_EQ(report.status, SolveStatus::kSolved);
    EXPECT_EQ(max_abs_diff(iterate, rhs), Real(0));
    retained_workspace.reset();
    retained_problem.reset();
    EXPECT_TRUE(workspace_lifetime.expired());
    EXPECT_TRUE(problem_lifetime.expired());
  }
}

TEST(test_krylov_workspace_reentrancy,
     shared_problem_allocation_and_input_failures_converge_before_constructor_and_retry) {
  verify_shared_construction_failure_and_retry(true);
}

TEST(test_krylov_workspace_reentrancy,
     shared_workspace_allocation_and_input_failures_converge_before_constructor_and_retry) {
  verify_shared_construction_failure_and_retry(false);
}

TEST(test_krylov_workspace_reentrancy,
     distinct_workspaces_run_fresh_operator_and_preconditioner_sessions_concurrently) {
  comm_init();
  const char* expected_ranks = std::getenv("POPS_TEST_EXPECT_RANKS");
  if (expected_ranks != nullptr)
    ASSERT_EQ(n_ranks(), std::atoi(expected_ranks));

  const TestBox domain{Index<kDim>{0, 0}, Index<kDim>{7, 7}};
  const TestLayout boxes = TestLayout::from_domain(domain, extent(2));
  const TestDistribution mapping = round_robin_distribution(boxes);
  const TestGeometry geometry = TestGeometry::from_bounds(
      domain, RealVector<kDim>{Real(0), Real(0)}, RealVector<kDim>{Real(1), Real(1)});
  TestField prototype = make_field(boxes, mapping, 1, 1);
  prototype.set_val(Real(0));
  EXPECT_GT(prototype.local_size(), 0);
  if (n_ranks() > 1)
    EXPECT_TRUE(has_remote_face_neighbor(boxes, mapping, my_rank()))
        << "the MPI variant must exchange a real inter-rank face halo";
  OperatorEvaluationSnapshot snapshot = test_snapshot(prototype);
  const auto operator_probe = std::make_shared<ConcurrentSessionProbe>();
  const auto preconditioner_probe = std::make_shared<ConcurrentSessionProbe>();

  TestAffineOperatorProvider operator_provider = TestAffineOperatorProvider::trusted_extension(
      {"pops.test.krylov.concurrent-operator", 1}, {},
      [operator_probe, boxes, mapping, geometry](const ExecutionLane& lane) {
        auto state = std::make_shared<ConcurrentOperatorSessionState>(
            operator_probe, operator_probe->next_session.fetch_add(1, std::memory_order_relaxed),
            geometry, lane, boxes, mapping);
        return TestAffineOperatorCallbacks{
            [] {}, [state](TestField& out, const TestField& in) { state->apply(out, in); },
            [state] { return state->allocation_count(); }};
      });

  TestLinearPreconditionerProvider provider = TestLinearPreconditionerProvider::trusted_extension(
      {"pops.test.krylov.concurrent-preconditioner", 1}, {},
      [preconditioner_probe](const ExecutionLane&) {
        auto state = std::make_shared<ConcurrentSessionState>();
        state->probe = preconditioner_probe;
        state->session_id =
            preconditioner_probe->next_session.fetch_add(1, std::memory_order_relaxed);
        return TestLinearPreconditionerCallbacks{
            [] {}, [state](TestField& out, const TestField& in) { state->apply(out, in); },
            [state] { return state->allocation_count(); }};
      });
  const TestKrylovFootprint footprint{1, extent(1), true};
  TestAffineProblem problem(prototype, std::move(operator_provider),
                            TestLinearPreconditioner(prototype, std::move(provider)),
                            LinearOperatorProperties::general(), footprint,
                            TestNullspacePolicy::nonsingular(), [&snapshot] { return snapshot; });
  const TestKrylovMethod method = bicgstab_krylov_method<kDim>();
  TestKrylovWorkspace oracle_left_workspace(prototype, method, footprint);
  TestKrylovWorkspace oracle_right_workspace(prototype, method, footprint);
  TestKrylovWorkspace left_workspace(prototype, method, footprint);
  TestKrylovWorkspace right_workspace(prototype, method, footprint);
  problem.prepare(snapshot);
  oracle_left_workspace.bind(problem);
  oracle_right_workspace.bind(problem);
  left_workspace.bind(problem);
  right_workspace.bind(problem);

  TestField oracle_left_iterate = make_field(boxes, mapping, 1, 1);
  TestField oracle_right_iterate = make_field(boxes, mapping, 1, 1);
  TestField oracle_left_rhs = make_field(boxes, mapping, 1, 1);
  TestField oracle_right_rhs = make_field(boxes, mapping, 1, 1);
  TestField left_iterate = make_field(boxes, mapping, 1, 1);
  TestField right_iterate = make_field(boxes, mapping, 1, 1);
  TestField left_rhs = make_field(boxes, mapping, 1, 1);
  TestField right_rhs = make_field(boxes, mapping, 1, 1);
  oracle_left_iterate.set_val(Real(0));
  oracle_right_iterate.set_val(Real(0));
  left_iterate.set_val(Real(0));
  right_iterate.set_val(Real(0));
  oracle_left_rhs.set_val(Real(0));
  oracle_right_rhs.set_val(Real(0));
  left_rhs.set_val(Real(0));
  right_rhs.set_val(Real(0));
  fill_rhs(oracle_left_rhs, Real(1));
  fill_rhs(oracle_right_rhs, Real(-2));
  fill_rhs(left_rhs, Real(1));
  fill_rhs(right_rhs, Real(-2));
  const TestKrylovControls controls{method, Real(1e-12), Real(0), 100};

  const SolveReport oracle_left_report = detail::solve_prepared_affine_in_place(
      problem, oracle_left_workspace, oracle_left_iterate, oracle_left_rhs, controls);
  const SolveReport oracle_right_report = detail::solve_prepared_affine_in_place(
      problem, oracle_right_workspace, oracle_right_iterate, oracle_right_rhs, controls);
  ASSERT_TRUE(oracle_left_report.solved()) << oracle_left_report.reason;
  ASSERT_TRUE(oracle_right_report.solved()) << oracle_right_report.reason;

  // Every rank materializes the independent invocation communicators in the same control order.
  // The worker entry order is then deliberately reversed on odd ranks. If both solves accidentally
  // share WORLD (or any one collective trace), this opposite interleaving deadlocks or mismatches.
  TestKrylovInvocation left_invocation = detail::prepare_krylov_solve_in_place(
      problem, left_workspace, left_iterate, left_rhs, controls);
  TestKrylovInvocation right_invocation = detail::prepare_krylov_solve_in_place(
      problem, right_workspace, right_iterate, right_rhs, controls);
  SolveReport left_report;
  SolveReport right_report;
  std::exception_ptr left_failure;
  std::exception_ptr right_failure;
  std::mutex launch_mutex;
  std::condition_variable launch_ready;
  bool left_entered = false;
  bool right_entered = false;
  operator_probe->armed.store(true, std::memory_order_release);
  preconditioner_probe->armed.store(true, std::memory_order_release);

  auto run_left = [&] {
    {
      std::lock_guard lock(launch_mutex);
      left_entered = true;
    }
    launch_ready.notify_all();
    try {
      left_report = left_invocation.execute();
    } catch (...) {
      left_failure = std::current_exception();
    }
  };
  auto run_right = [&] {
    {
      std::lock_guard lock(launch_mutex);
      right_entered = true;
    }
    launch_ready.notify_all();
    try {
      right_report = right_invocation.execute();
    } catch (...) {
      right_failure = std::current_exception();
    }
  };

  std::thread left;
  std::thread right;
  bool first_worker_entered = false;
  if ((my_rank() % 2) == 0) {
    left = std::thread(run_left);
    {
      std::unique_lock lock(launch_mutex);
      first_worker_entered =
          launch_ready.wait_for(lock, std::chrono::seconds(5), [&] { return left_entered; });
    }
    right = std::thread(run_right);
  } else {
    right = std::thread(run_right);
    {
      std::unique_lock lock(launch_mutex);
      first_worker_entered =
          launch_ready.wait_for(lock, std::chrono::seconds(5), [&] { return right_entered; });
    }
    left = std::thread(run_left);
  }
  left.join();
  right.join();

  EXPECT_TRUE(first_worker_entered);
  EXPECT_EQ(left_failure, nullptr);
  EXPECT_EQ(right_failure, nullptr);
  EXPECT_FALSE(operator_probe->timed_out);
  EXPECT_EQ(operator_probe->arrivals, 2);
  EXPECT_EQ(operator_probe->solve_session_ids.size(), 2u);
  EXPECT_EQ(operator_probe->solve_resource_addresses.size(), 2u);
  EXPECT_FALSE(preconditioner_probe->timed_out);
  EXPECT_EQ(preconditioner_probe->arrivals, 2);
  EXPECT_EQ(preconditioner_probe->solve_session_ids.size(), 2u);
  EXPECT_EQ(preconditioner_probe->solve_resource_addresses.size(), 2u);
  EXPECT_TRUE(left_report.solved()) << left_report.reason;
  EXPECT_TRUE(right_report.solved()) << right_report.reason;
  EXPECT_EQ(left_report.status, oracle_left_report.status);
  EXPECT_EQ(right_report.status, oracle_right_report.status);
  EXPECT_EQ(left_report.iters, oracle_left_report.iters);
  EXPECT_EQ(right_report.iters, oracle_right_report.iters);
  EXPECT_LT(max_abs_diff(left_iterate, oracle_left_iterate), Real(1e-12));
  EXPECT_LT(max_abs_diff(right_iterate, oracle_right_iterate), Real(1e-12));
}

TEST(test_krylov_workspace_reentrancy,
     identical_provider_reason_larger_than_the_old_fixed_capacity_is_published_exactly) {
  const TestLayout boxes(std::vector<TestBox>{TestBox{Index<kDim>{0, 0}, Index<kDim>{1, 1}}});
  const TestDistribution mapping = round_robin_distribution(boxes);
  TestField iterate = make_field(boxes, mapping, 1, 0);
  TestField rhs = make_field(boxes, mapping, 1, 0);
  iterate.set_val(Real(0));
  rhs.set_val(Real(1));
  OperatorEvaluationSnapshot snapshot = test_snapshot(iterate);
  const auto provider = std::make_shared<LongReasonKrylovProvider>();
  const TestKrylovMethod method(
      provider, PreparedProviderOptions{"pops.test.krylov.long-report-reason.options@1", {}});
  const TestKrylovFootprint footprint{1, extent(0), false};
  TestAffineProblem problem(
      iterate,
      TestAffineOperatorProvider::trusted_reentrant(
          [](TestField& out, const TestField& in) { detail::PreparedFieldAlgebra::copy(out, in); },
          [] { return std::size_t{0}; }),
      TestLinearPreconditioner::identity(), LinearOperatorProperties::general(), footprint,
      TestNullspacePolicy::nonsingular(), [&snapshot] { return snapshot; });
  TestKrylovWorkspace workspace(iterate, method, footprint);
  problem.prepare(snapshot);
  workspace.bind(problem);

  const SolveReport report = detail::solve_prepared_affine_in_place(
      problem, workspace, iterate, rhs, TestKrylovControls{method, Real(0), Real(0), 1});

  EXPECT_EQ(report.status, SolveStatus::kIterationLimit);
  EXPECT_EQ(report.reason.size(), LongReasonKrylovProvider::kReasonBytes);
  EXPECT_EQ(report.reason, std::string(LongReasonKrylovProvider::kReasonBytes, 'r'));
}

TEST(test_krylov_workspace_reentrancy,
     authored_numerical_actions_preserve_cg_failure_rollback_and_terminal_evaluation_guards) {
  const TestLayout boxes(std::vector<TestBox>{TestBox{Index<kDim>{0, 0}, Index<kDim>{3, 1}}});
  const TestDistribution mapping = round_robin_distribution(boxes);
  TestField iterate = make_field(boxes, mapping, 1, 0);
  TestField rhs = make_field(boxes, mapping, 1, 0);
  TestField accepted = make_field(boxes, mapping, 1, 0);
  iterate.set_val(Real(0));
  accepted.set_val(Real(0));
  rhs.set_val(Real(1));
  const OperatorEvaluationSnapshot snapshot = test_snapshot(iterate);
  const TestKrylovMethod method = cg_krylov_method<kDim>();
  const TestKrylovFootprint footprint{1, extent(0), false};
  bool fatal_operator_failure = false;
  TestAffineProblem problem(
      iterate,
      TestAffineOperatorProvider::trusted_reentrant(
          [&](TestField& out, const TestField& in) {
            if (fatal_operator_failure)
              throw std::runtime_error("terminal numerical-policy oracle");
            for (std::size_t local = 0; local < out.local_size(); ++local)
              for_each_cell(out.box(local),
                            PositiveDiagonalKernel{out.fab(local).view(), in.fab(local).view()});
          },
          [] { return std::size_t{0}; }),
      TestLinearPreconditioner::identity(), LinearOperatorProperties::symmetric_positive_definite(),
      footprint, TestNullspacePolicy::nonsingular(), [&snapshot] { return snapshot; });
  TestKrylovWorkspace workspace(iterate, method, footprint);
  problem.prepare(snapshot);
  workspace.bind(problem);
  // A four-eigenvalue diagonal problem cannot converge in one CG iteration. The same numerical
  // failure is retryable only if that exact status was selected, without publishing its candidate.
  TestKrylovControls invalid{method, Real(1e-12), Real(0), 1};
  invalid.failure_actions.iteration_limit = SolveAction::kNone;
  EXPECT_THROW((void)detail::validate_controls(invalid), std::invalid_argument);
  EXPECT_THROW((void)solve_prepared_affine_outcome(problem, workspace, iterate, rhs, invalid),
               std::invalid_argument);
  for (const KrylovFailureActions policy :
       {KrylovFailureActions{},
        KrylovFailureActions{SolveAction::kFailRun, SolveAction::kFailRun,
                             SolveAction::kRejectAttempt},
        KrylovFailureActions{SolveAction::kRejectAttempt, SolveAction::kRejectAttempt,
                             SolveAction::kFailRun}}) {
    auto outcome =
        solve_prepared_affine_outcome(problem, workspace, iterate, rhs,
                                      TestKrylovControls{method, Real(1e-12), Real(0), 1, policy});
    EXPECT_EQ(outcome.report().status, SolveStatus::kIterationLimit);
    EXPECT_EQ(outcome.report().action, policy.iteration_limit);
    EXPECT_GT(outcome.report().residual_norm, Real(0.1));
    EXPECT_EQ(max_abs_diff(iterate, accepted), Real(0));
    const auto consumption = policy.iteration_limit == SolveAction::kRejectAttempt
                                 ? SolveConsumption::kRejectAttempt
                                 : SolveConsumption::kFailRun;
    if (consumption == SolveConsumption::kFailRun)
      EXPECT_THROW((void)outcome.consume(SolveConsumption::kRejectAttempt), std::logic_error);
    EXPECT_EQ(outcome.consume(consumption).action, policy.iteration_limit);
    EXPECT_EQ(max_abs_diff(iterate, accepted), Real(0));
  }
  const KrylovFailureActions reject{SolveAction::kRejectAttempt, SolveAction::kRejectAttempt,
                                    SolveAction::kRejectAttempt};
  fatal_operator_failure = true;
  auto terminal =
      solve_prepared_affine_outcome(problem, workspace, iterate, rhs,
                                    TestKrylovControls{method, Real(1e-12), Real(0), 1, reject});
  EXPECT_EQ(terminal.report().status, SolveStatus::kInvalidEvaluation);
  EXPECT_EQ(terminal.report().action, SolveAction::kFailRun);
  EXPECT_THROW((void)terminal.consume(SolveConsumption::kRejectAttempt), std::logic_error);
  (void)terminal.consume(SolveConsumption::kFailRun);
  EXPECT_EQ(max_abs_diff(iterate, accepted), Real(0));
  fatal_operator_failure = false;
  auto solved =
      solve_prepared_affine_outcome(problem, workspace, iterate, rhs,
                                    TestKrylovControls{method, Real(1e-12), Real(0), 8, reject});
  EXPECT_TRUE(solved.report().solved_value_available());
  EXPECT_EQ(max_abs_diff(iterate, accepted), Real(0));
  (void)solved.consume(SolveConsumption::kAccept);
  EXPECT_GT(max_abs_diff(iterate, accepted), Real(0.9));
}

TEST(test_krylov_workspace_reentrancy,
     numerical_actions_do_not_downgrade_explicit_provider_failure_or_false_convergence) {
  const TestLayout boxes(std::vector<TestBox>{TestBox{Index<kDim>{0, 0}, Index<kDim>{1, 1}}});
  const TestDistribution mapping = round_robin_distribution(boxes);
  TestField iterate = make_field(boxes, mapping, 1, 0);
  TestField rhs = make_field(boxes, mapping, 1, 0);
  iterate.set_val(Real(0));
  rhs.set_val(Real(1));
  const OperatorEvaluationSnapshot snapshot = test_snapshot(iterate);
  const TestKrylovFootprint footprint{1, extent(0), false};
  TestAffineProblem problem(
      iterate,
      TestAffineOperatorProvider::trusted_reentrant(
          [](TestField& out, const TestField& in) { detail::PreparedFieldAlgebra::copy(out, in); },
          [] { return std::size_t{0}; }),
      TestLinearPreconditioner::identity(), LinearOperatorProperties::general(), footprint,
      TestNullspacePolicy::nonsingular(), [&snapshot] { return snapshot; });
  problem.prepare(snapshot);
  for (const auto behavior : {LongReasonKrylovProvider::Behavior::kExplicitTerminalFailure,
                              LongReasonKrylovProvider::Behavior::kUnverifiedSolved}) {
    const TestKrylovMethod method(
        std::make_shared<LongReasonKrylovProvider>(behavior),
        PreparedProviderOptions{"pops.test.krylov.long-report-reason.options@1", {}});
    TestKrylovWorkspace workspace(iterate, method, footprint);
    workspace.bind(problem);
    auto outcome = solve_prepared_affine_outcome(
        problem, workspace, iterate, rhs,
        TestKrylovControls{method,
                           Real(1e-12),
                           Real(0),
                           1,
                           {SolveAction::kRejectAttempt, SolveAction::kRejectAttempt,
                            SolveAction::kRejectAttempt}});
    EXPECT_EQ(outcome.report().status,
              behavior == LongReasonKrylovProvider::Behavior::kExplicitTerminalFailure
                  ? SolveStatus::kIterationLimit
                  : SolveStatus::kInvalidEvaluation);
    EXPECT_EQ(outcome.report().action, SolveAction::kFailRun);
    EXPECT_FALSE(outcome.report().solved_value_available());
    EXPECT_THROW((void)outcome.consume(SolveConsumption::kRejectAttempt), std::logic_error);
    (void)outcome.consume(SolveConsumption::kFailRun);
  }
}

TEST(test_krylov_workspace_reentrancy,
     exclusive_operator_rejects_overlapping_materialized_invocations_uniformly) {
  comm_init();
  const TestBox domain{Index<kDim>{0, 0}, Index<kDim>{3, 3}};
  const TestLayout boxes = TestLayout::from_domain(domain, extent(2));
  const TestDistribution mapping = round_robin_distribution(boxes);
  TestField prototype = make_field(boxes, mapping, 1, 0);
  prototype.set_val(Real(0));
  OperatorEvaluationSnapshot snapshot = test_snapshot(prototype);
  TestAffineOperatorProvider operator_provider = TestAffineOperatorProvider::trusted_extension(
      {"pops.test.krylov.exclusive-operator", 1}, {},
      [](const ExecutionLane&) {
        return TestAffineOperatorCallbacks{{},
                                           [](TestField& out, const TestField& in) {
                                             detail::PreparedFieldAlgebra::copy(out, in);
                                           },
                                           [] { return std::size_t{0}; }};
      },
      PreparedOperatorConcurrency::Exclusive);
  const TestKrylovFootprint footprint{1, extent(0), false};
  TestAffineProblem problem(prototype, std::move(operator_provider),
                            TestLinearPreconditioner::identity(),
                            LinearOperatorProperties::symmetric_positive_definite(), footprint,
                            TestNullspacePolicy::nonsingular(), [&snapshot] { return snapshot; });
  const TestKrylovMethod method = cg_krylov_method<kDim>();
  TestKrylovWorkspace first_workspace(prototype, method, footprint);
  TestKrylovWorkspace second_workspace(prototype, method, footprint);
  problem.prepare(snapshot);
  first_workspace.bind(problem);
  second_workspace.bind(problem);

  TestField first_iterate = make_field(boxes, mapping, 1, 0);
  TestField second_iterate = make_field(boxes, mapping, 1, 0);
  TestField first_rhs = make_field(boxes, mapping, 1, 0);
  TestField second_rhs = make_field(boxes, mapping, 1, 0);
  first_iterate.set_val(Real(0));
  second_iterate.set_val(Real(0));
  first_rhs.set_val(Real(2));
  second_rhs.set_val(Real(-3));
  const TestKrylovControls controls{method, Real(1e-12), Real(0), 4};

  {
    TestKrylovInvocation first = detail::prepare_krylov_solve_in_place(
        problem, first_workspace, first_iterate, first_rhs, controls);
    std::string rejection;
    try {
      (void)detail::prepare_krylov_solve_in_place(problem, second_workspace, second_iterate,
                                                  second_rhs, controls);
    } catch (const std::logic_error& error) {
      rejection = error.what();
    }
    EXPECT_EQ(rejection,
              "prepared affine problem is being mutated or its operator requires exclusive "
              "access to its external execution context");
    const SolveReport first_report = first.execute();
    EXPECT_TRUE(first_report.solved()) << first_report.reason;
  }

  TestKrylovInvocation second = detail::prepare_krylov_solve_in_place(
      problem, second_workspace, second_iterate, second_rhs, controls);
  const SolveReport second_report = second.execute();
  EXPECT_TRUE(second_report.solved()) << second_report.reason;
}

TEST(test_krylov_workspace_reentrancy,
     exclusive_operator_lease_is_shared_by_provider_copies_across_prepared_problems) {
  comm_init();
  const TestBox domain{Index<kDim>{0, 0}, Index<kDim>{3, 3}};
  const TestLayout boxes = TestLayout::from_domain(domain, extent(2));
  const TestDistribution mapping = round_robin_distribution(boxes);
  TestField prototype = make_field(boxes, mapping, 1, 0);
  prototype.set_val(Real(0));
  OperatorEvaluationSnapshot first_snapshot = test_snapshot(prototype);
  OperatorEvaluationSnapshot second_snapshot = test_snapshot(prototype);
  TestAffineOperatorProvider first_provider = TestAffineOperatorProvider::trusted_extension(
      {"pops.test.krylov.shared-exclusive-operator", 1}, {},
      [](const ExecutionLane&) {
        return TestAffineOperatorCallbacks{{},
                                           [](TestField& out, const TestField& in) {
                                             detail::PreparedFieldAlgebra::copy(out, in);
                                           },
                                           [] { return std::size_t{0}; }};
      },
      PreparedOperatorConcurrency::Exclusive);
  TestAffineOperatorProvider second_provider = first_provider;
  const TestKrylovFootprint footprint{1, extent(0), false};
  TestAffineProblem first_problem(
      prototype, std::move(first_provider), TestLinearPreconditioner::identity(),
      LinearOperatorProperties::symmetric_positive_definite(), footprint,
      TestNullspacePolicy::nonsingular(), [&first_snapshot] { return first_snapshot; });
  TestAffineProblem second_problem(
      prototype, std::move(second_provider), TestLinearPreconditioner::identity(),
      LinearOperatorProperties::symmetric_positive_definite(), footprint,
      TestNullspacePolicy::nonsingular(), [&second_snapshot] { return second_snapshot; });
  const TestKrylovMethod method = cg_krylov_method<kDim>();
  TestKrylovWorkspace first_workspace(prototype, method, footprint);
  TestKrylovWorkspace second_workspace(prototype, method, footprint);
  first_problem.prepare(first_snapshot);
  second_problem.prepare(second_snapshot);
  first_workspace.bind(first_problem);
  second_workspace.bind(second_problem);

  TestField first_iterate = make_field(boxes, mapping, 1, 0);
  TestField second_iterate = make_field(boxes, mapping, 1, 0);
  TestField first_rhs = make_field(boxes, mapping, 1, 0);
  TestField second_rhs = make_field(boxes, mapping, 1, 0);
  TestField second_operator_output = make_field(boxes, mapping, 1, 0);
  first_iterate.set_val(Real(0));
  second_iterate.set_val(Real(0));
  first_rhs.set_val(Real(2));
  second_rhs.set_val(Real(-3));
  const TestKrylovControls controls{method, Real(1e-12), Real(0), 4};

  {
    TestKrylovInvocation first = detail::prepare_krylov_solve_in_place(
        first_problem, first_workspace, first_iterate, first_rhs, controls);
    std::string prepare_rejection;
    try {
      second_problem.prepare(second_snapshot);
    } catch (const std::logic_error& error) {
      prepare_rejection = error.what();
    }
    EXPECT_EQ(prepare_rejection,
              "PreparedAffineLinearProblem cannot be prepared while another prepared operation "
              "is active");
    std::string bind_rejection;
    try {
      second_workspace.bind(second_problem);
    } catch (const std::logic_error& error) {
      bind_rejection = error.what();
    }
    EXPECT_EQ(bind_rejection,
              "KrylovWorkspace cannot be rebound while its prepared problem is mutating or in "
              "exclusive use");
    std::string direct_apply_rejection;
    try {
      second_problem.apply_linear(second_operator_output, second_iterate);
    } catch (const std::logic_error& error) {
      direct_apply_rejection = error.what();
    }
    EXPECT_EQ(
        direct_apply_rejection,
        "PreparedAffineLinearProblem::apply_linear cannot use its public operator session while "
        "another prepared operation is active");
    std::string rejection;
    try {
      (void)detail::prepare_krylov_solve_in_place(second_problem, second_workspace, second_iterate,
                                                  second_rhs, controls);
    } catch (const std::logic_error& error) {
      rejection = error.what();
    }
    EXPECT_EQ(rejection,
              "prepared affine problem is being mutated or its operator requires exclusive "
              "access to its external execution context");
    const SolveReport first_report = first.execute();
    EXPECT_TRUE(first_report.solved()) << first_report.reason;
  }

  TestKrylovInvocation second = detail::prepare_krylov_solve_in_place(
      second_problem, second_workspace, second_iterate, second_rhs, controls);
  const SolveReport second_report = second.execute();
  EXPECT_TRUE(second_report.solved()) << second_report.reason;
}

TEST(test_krylov_workspace_reentrancy,
     public_problem_sessions_reject_overlap_with_independent_workspace_invocation) {
  comm_init();
  const TestBox domain{Index<kDim>{0, 0}, Index<kDim>{3, 3}};
  const TestLayout boxes = TestLayout::from_domain(domain, extent(2));
  const TestDistribution mapping = round_robin_distribution(boxes);
  TestField prototype = make_field(boxes, mapping, 1, 0);
  prototype.set_val(Real(0));
  OperatorEvaluationSnapshot snapshot = test_snapshot(prototype);
  const auto operator_probe = std::make_shared<SessionLifecycleProbe>();
  const auto preconditioner_probe = std::make_shared<SessionLifecycleProbe>();
  const TestKrylovFootprint footprint{1, extent(0), true};
  TestAffineProblem problem(
      prototype, lifecycle_operator_provider(operator_probe),
      TestLinearPreconditioner(prototype, lifecycle_preconditioner_provider(preconditioner_probe)),
      LinearOperatorProperties::general(), footprint, TestNullspacePolicy::nonsingular(),
      [&snapshot] { return snapshot; });
  const TestKrylovMethod method = bicgstab_krylov_method<kDim>();
  TestKrylovWorkspace workspace(prototype, method, footprint);
  problem.prepare(snapshot);
  workspace.bind(problem);

  TestField iterate = make_field(boxes, mapping, 1, 0);
  TestField rhs = make_field(boxes, mapping, 1, 0);
  TestField output = make_field(boxes, mapping, 1, 0);
  iterate.set_val(Real(0));
  rhs.set_val(Real(2));
  output.set_val(Real(0));
  const TestKrylovControls controls{method, Real(1e-12), Real(0), 4};

  {
    TestKrylovInvocation invocation =
        detail::prepare_krylov_solve_in_place(problem, workspace, iterate, rhs, controls);
    std::string apply_rejection;
    try {
      problem.apply_linear(output, rhs);
    } catch (const std::logic_error& error) {
      apply_rejection = error.what();
    }
    EXPECT_EQ(
        apply_rejection,
        "PreparedAffineLinearProblem::apply_linear cannot use its public operator session while "
        "another prepared operation is active");

    std::string residual_rejection;
    try {
      problem.true_residual(output, rhs, iterate);
    } catch (const std::logic_error& error) {
      residual_rejection = error.what();
    }
    EXPECT_EQ(
        residual_rejection,
        "PreparedAffineLinearProblem::true_residual cannot use its public operator session while "
        "another prepared operation is active");

    std::string preconditioner_rejection;
    try {
      problem.apply_preconditioner(output, rhs);
    } catch (const std::logic_error& error) {
      preconditioner_rejection = error.what();
    }
    EXPECT_EQ(
        preconditioner_rejection,
        "PreparedAffineLinearProblem::apply_preconditioner cannot use its public session while "
        "another prepared operation is active");

    const SolveReport report = invocation.execute();
    EXPECT_TRUE(report.solved()) << report.reason;
  }

  EXPECT_NO_THROW(problem.apply_linear(output, rhs));
  EXPECT_NO_THROW(problem.true_residual(output, rhs, iterate));
  EXPECT_NO_THROW(problem.apply_preconditioner(output, rhs));
}

TEST(test_krylov_workspace_reentrancy,
     exclusive_source_lease_is_released_after_prepare_exception_across_problems) {
  comm_init();
  const TestBox domain{Index<kDim>{0, 0}, Index<kDim>{3, 3}};
  const TestLayout boxes = TestLayout::from_domain(domain, extent(2));
  const TestDistribution mapping = round_robin_distribution(boxes);
  TestField prototype = make_field(boxes, mapping, 1, 0);
  prototype.set_val(Real(0));
  OperatorEvaluationSnapshot first_snapshot = test_snapshot(prototype);
  OperatorEvaluationSnapshot second_snapshot = test_snapshot(prototype);
  TestAffineOperatorProvider first_provider = TestAffineOperatorProvider::trusted_extension(
      {"pops.test.krylov.exception-safe-exclusive-operator", 1}, {},
      [](const ExecutionLane&) {
        return TestAffineOperatorCallbacks{{},
                                           [](TestField& out, const TestField& in) {
                                             detail::PreparedFieldAlgebra::copy(out, in);
                                           },
                                           [] { return std::size_t{0}; }};
      },
      PreparedOperatorConcurrency::Exclusive);
  TestAffineOperatorProvider second_provider = first_provider;
  const TestKrylovFootprint footprint{1, extent(0), false};
  TestAffineProblem first_problem(
      prototype, std::move(first_provider), TestLinearPreconditioner::identity(),
      LinearOperatorProperties::symmetric_positive_definite(), footprint,
      TestNullspacePolicy::nonsingular(), [&first_snapshot] { return first_snapshot; },
      [] {
        if (my_rank() == 0)
          throw std::runtime_error("rank-local frozen-resource failure");
      });
  TestAffineProblem second_problem(
      prototype, std::move(second_provider), TestLinearPreconditioner::identity(),
      LinearOperatorProperties::symmetric_positive_definite(), footprint,
      TestNullspacePolicy::nonsingular(), [&second_snapshot] { return second_snapshot; });

  std::string prepare_rejection;
  if (n_ranks() == 1) {
    try {
      first_problem.prepare(first_snapshot);
    } catch (const std::runtime_error& error) {
      prepare_rejection = error.what();
    }
    EXPECT_EQ(prepare_rejection, "rank-local frozen-resource failure");
  } else {
    try {
      first_problem.prepare(first_snapshot);
    } catch (const std::logic_error& error) {
      prepare_rejection = error.what();
    }
    EXPECT_EQ(prepare_rejection,
              "prepared resource freeze failed on at least one communicator rank");
  }

  const TestKrylovMethod method = cg_krylov_method<kDim>();
  TestKrylovWorkspace workspace(prototype, method, footprint);
  EXPECT_NO_THROW(second_problem.prepare(second_snapshot));
  EXPECT_NO_THROW(workspace.bind(second_problem));
  TestField iterate = make_field(boxes, mapping, 1, 0);
  TestField rhs = make_field(boxes, mapping, 1, 0);
  iterate.set_val(Real(0));
  rhs.set_val(Real(2));
  const SolveReport report = detail::solve_prepared_affine_in_place(
      second_problem, workspace, iterate, rhs, TestKrylovControls{method, Real(1e-12), Real(0), 4});
  EXPECT_TRUE(report.solved()) << report.reason;
}

TEST(test_krylov_workspace_reentrancy,
     workspace_bind_rejects_affine_zero_response_drift_between_provider_sessions) {
  comm_init();
  const TestBox domain{Index<kDim>{0, 0}, Index<kDim>{3, 3}};
  const TestLayout boxes = TestLayout::from_domain(domain, extent(2));
  const TestDistribution mapping = round_robin_distribution(boxes);
  TestField prototype = make_field(boxes, mapping, 1, 0);
  prototype.set_val(Real(0));
  OperatorEvaluationSnapshot snapshot = test_snapshot(prototype);
  auto next_session = std::make_shared<std::atomic<int>>(0);
  TestAffineOperatorProvider operator_provider = TestAffineOperatorProvider::trusted_extension(
      {"pops.test.krylov.session-dependent-affine-constant", 1}, {},
      [next_session](const ExecutionLane&) {
        const Real offset =
            static_cast<Real>(next_session->fetch_add(1, std::memory_order_relaxed));
        return TestAffineOperatorCallbacks{{},
                                           [offset](TestField& out, const TestField& in) {
                                             out.set_val(offset);
                                             detail::PreparedFieldAlgebra::axpy(out, Real(1), in);
                                           },
                                           [] { return std::size_t{0}; }};
      });
  const TestKrylovFootprint footprint{1, extent(0), false};
  TestAffineProblem problem(prototype, std::move(operator_provider),
                            TestLinearPreconditioner::identity(),
                            LinearOperatorProperties::symmetric_positive_definite(), footprint,
                            TestNullspacePolicy::nonsingular(), [&snapshot] { return snapshot; });
  const TestKrylovMethod method = cg_krylov_method<kDim>();
  TestKrylovWorkspace workspace(prototype, method, footprint);
  problem.prepare(snapshot);

  std::string rejection;
  try {
    workspace.bind(problem);
  } catch (const std::runtime_error& error) {
    rejection = error.what();
  }
  EXPECT_EQ(next_session->load(std::memory_order_relaxed), 2);
  EXPECT_EQ(rejection,
            "KrylovWorkspace affine-operator session disagrees with the prepared problem's "
            "exact zero response");
}

TEST(test_krylov_workspace_reentrancy,
     workspace_bind_publishes_rank_local_zero_response_comparison_failure_uniformly) {
  comm_init();
  const TestBox domain{Index<kDim>{0, 0}, Index<kDim>{3, 3}};
  const TestLayout boxes = TestLayout::from_domain(domain, extent(2));
  const TestDistribution mapping = round_robin_distribution(boxes);
  TestField prototype = make_field(boxes, mapping, 1, 0);
  prototype.set_val(Real(0));
  OperatorEvaluationSnapshot snapshot = test_snapshot(prototype);
  auto next_session = std::make_shared<std::atomic<int>>(0);
  TestAffineOperatorProvider operator_provider = TestAffineOperatorProvider::trusted_extension(
      {"pops.test.krylov.rank-local-session-layout-corruption", 1}, {},
      [next_session](const ExecutionLane& lane) {
        const int session = next_session->fetch_add(1, std::memory_order_relaxed);
        const bool corrupt_output_layout = session == 1 && lane.rank() == 0;
        return TestAffineOperatorCallbacks{
            {},
            [corrupt_output_layout](TestField& out, const TestField& in) {
              if (!corrupt_output_layout) {
                detail::PreparedFieldAlgebra::copy(out, in);
                return;
              }
              const TestLayout incompatible_boxes(
                  std::vector<TestBox>{TestBox{Index<kDim>{0, 0}, Index<kDim>{0, 0}}});
              const TestDistribution incompatible_mapping =
                  round_robin_distribution(incompatible_boxes);
              out = make_field(incompatible_boxes, incompatible_mapping, 1, 0);
              out.set_val(Real(0));
            },
            [] { return std::size_t{0}; }};
      });
  const TestKrylovFootprint footprint{1, extent(0), false};
  TestAffineProblem problem(prototype, std::move(operator_provider),
                            TestLinearPreconditioner::identity(),
                            LinearOperatorProperties::symmetric_positive_definite(), footprint,
                            TestNullspacePolicy::nonsingular(), [&snapshot] { return snapshot; });
  const TestKrylovMethod method = cg_krylov_method<kDim>();
  TestKrylovWorkspace workspace(prototype, method, footprint);
  problem.prepare(snapshot);

  std::string rejection;
  try {
    workspace.bind(problem);
  } catch (const std::runtime_error& error) {
    rejection = error.what();
  }
  EXPECT_EQ(next_session->load(std::memory_order_relaxed), 2);
  EXPECT_EQ(rejection,
            "KrylovWorkspace affine-operator zero-response comparison failed on at least one "
            "communicator rank");
}

TEST(test_krylov_workspace_reentrancy,
     prepared_problem_and_workspace_execute_on_an_embedding_owned_congruent_communicator) {
#ifndef POPS_HAS_MPI
  GTEST_SKIP() << "custom communicator execution requires MPI";
#else
  comm_init();
  const char* expected_ranks = std::getenv("POPS_TEST_EXPECT_RANKS");
  if (expected_ranks != nullptr)
    ASSERT_EQ(n_ranks(), std::atoi(expected_ranks));
  if (n_ranks() < 2)
    GTEST_SKIP() << "a remote neighbor requires multiple MPI ranks";

  MPI_Comm embedding_owned = MPI_COMM_NULL;
  ASSERT_EQ(MPI_Comm_dup(MPI_COMM_WORLD, &embedding_owned), MPI_SUCCESS);
  ScopedMpiCommunicator owned_parent(embedding_owned);
  const ExecutionCommunicator parent = ExecutionCommunicator::borrowed(
      "pops.test.krylov.embedding-owned-world-congruent", embedding_owned);

  {
    const TestBox domain{Index<kDim>{0, 0}, Index<kDim>{7, 7}};
    const TestLayout boxes = TestLayout::from_domain(domain, extent(2));
    const TestDistribution mapping = round_robin_distribution(boxes);
    const TestGeometry geometry = TestGeometry::from_bounds(
        domain, RealVector<kDim>{Real(0), Real(0)}, RealVector<kDim>{Real(1), Real(1)});
    TestField prototype = make_field(boxes, mapping, 1, 1);
    prototype.set_val(Real(0));
    ASSERT_GT(prototype.local_size(), 0);
    ASSERT_TRUE(has_remote_face_neighbor(boxes, mapping, my_rank()));
    OperatorEvaluationSnapshot snapshot = test_snapshot(prototype);
    const auto probe = std::make_shared<ConcurrentSessionProbe>();

    TestAffineOperatorProvider operator_provider = TestAffineOperatorProvider::trusted_extension(
        {"pops.test.krylov.embedding-communicator-operator", 1}, {},
        [probe, boxes, mapping, geometry](const ExecutionLane& lane) {
          auto state = std::make_shared<ConcurrentOperatorSessionState>(
              probe, probe->next_session.fetch_add(1, std::memory_order_relaxed), geometry, lane,
              boxes, mapping);
          return TestAffineOperatorCallbacks{
              [] {}, [state](TestField& out, const TestField& in) { state->apply(out, in); },
              [state] { return state->allocation_count(); }};
        });

    const TestKrylovFootprint footprint{1, extent(1), false};
    TestAffineProblem problem(parent, "pops.test.krylov.embedding-communicator-problem", prototype,
                              std::move(operator_provider), TestLinearPreconditioner::identity(),
                              LinearOperatorProperties::symmetric_positive_definite(), footprint,
                              TestNullspacePolicy::nonsingular(), [&snapshot] { return snapshot; });
    const TestKrylovMethod method = cg_krylov_method<kDim>();
    TestKrylovWorkspace workspace(parent, "pops.test.krylov.embedding-communicator-workspace",
                                  prototype, method, footprint);
    problem.prepare(snapshot);
    workspace.bind(problem);

    TestField iterate = make_field(boxes, mapping, 1, 1);
    TestField rhs = make_field(boxes, mapping, 1, 1);
    iterate.set_val(Real(0));
    rhs.set_val(Real(0));
    fill_rhs(rhs, Real(1));
    const SolveReport report = detail::solve_prepared_affine_in_place(
        problem, workspace, iterate, rhs, TestKrylovControls{method, Real(1e-12), Real(0), 100});
    EXPECT_TRUE(report.solved()) << report.reason;
    EXPECT_GT(report.iters, 0);
  }
#endif
}

TEST(test_krylov_workspace_reentrancy, noncongruent_split_communicator_is_rejected_explicitly) {
#ifndef POPS_HAS_MPI
  GTEST_SKIP() << "custom communicator validation requires MPI";
#else
  comm_init();
  if (n_ranks() < 2)
    GTEST_SKIP() << "a noncongruent split requires multiple MPI ranks";

  MPI_Comm split = MPI_COMM_NULL;
  ASSERT_EQ(MPI_Comm_split(MPI_COMM_WORLD, my_rank() % 2, my_rank(), &split), MPI_SUCCESS);
  ScopedMpiCommunicator owned_split(split);
  try {
    (void)ExecutionCommunicator::borrowed("pops.test.krylov.noncongruent-split", split);
    FAIL() << "a communicator with a different rank space must not be accepted";
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string(error.what()).find("preserve the MPI_COMM_WORLD rank space"),
              std::string::npos);
  }
#endif
}

TEST(test_krylov_workspace_reentrancy,
     gmres_confirms_true_residual_before_classifying_the_iteration_cap) {
  comm_init();
  // A=I and b=(1,1), but left P=diag(1,1e-14) makes the first Arnoldi residual tiny
  // while the scientific residual remains approximately one. An estimate may request a true
  // check; it cannot make the iteration cap turn that unconverged value into Solved.
  const TestLayout boxes(std::vector<TestBox>{
      TestBox{Index<kDim>{0, 0}, Index<kDim>{0, 0}},
      TestBox{Index<kDim>{1, 0}, Index<kDim>{1, 0}}});
  const TestDistribution mapping = round_robin_distribution(boxes);
  TestField iterate = make_field(boxes, mapping, 1, 0);
  TestField rhs = make_field(boxes, mapping, 1, 0);
  iterate.set_val(Real(0));
  rhs.set_val(Real(1));
  const OperatorEvaluationSnapshot snapshot = test_snapshot(iterate);
  const TestKrylovMethod method = gmres_krylov_method<kDim>(2);
  const TestKrylovFootprint footprint{1, extent(0), true};
  TestLinearPreconditioner preconditioner(
      iterate, TestLinearPreconditionerProvider::trusted_extension(
          {"pops.test.krylov.cap-anisotropic-preconditioner", 1}, {},
          [](const ExecutionLane&) {
            return TestLinearPreconditionerCallbacks{
                [] {},
                [](TestField& out, const TestField& in) {
                  for (std::size_t local = 0; local < out.local_size(); ++local) {
                    const auto output = out.fab(local).view();
                    const auto values = in.fab(local).view();
                    for_each_cell(out.box(local), [=] POPS_HD(const Index<kDim>& index) {
                      output(index, 0) =
                          (index[0] == 0 ? Real(1) : Real(1e-14)) * values(index, 0);
                    });
                  }
                  Kokkos::fence();
                },
                [] { return std::size_t{0}; }};
          }));
  TestAffineProblem problem(
      iterate,
      TestAffineOperatorProvider::trusted_reentrant(
          [](TestField& out, const TestField& in) { detail::PreparedFieldAlgebra::copy(out, in); },
          [] { return std::size_t{0}; }),
      std::move(preconditioner), LinearOperatorProperties::general(), footprint,
      TestNullspacePolicy::nonsingular(), [&snapshot] { return snapshot; });
  TestKrylovWorkspace workspace(iterate, method, footprint);
  problem.prepare(snapshot);
  workspace.bind(problem);

  for (const auto norm : {KrylovPhysicalNorm::metric_l2, KrylovPhysicalNorm::component_linf}) {
    SCOPED_TRACE(static_cast<int>(norm));
    TestKrylovControls controls{method, Real(0), Real(1e-12), 1};
    controls.physical_norm = norm;
    iterate.set_val(Real(0));
    const auto capped =
        detail::solve_prepared_affine_in_place(problem, workspace, iterate, rhs, controls);
    EXPECT_EQ(capped.status, SolveStatus::kIterationLimit) << capped.reason;
    EXPECT_EQ(capped.action, SolveAction::kFailRun);
    EXPECT_EQ(capped.iters, 1);
    EXPECT_GT(capped.residual_norm, Real(0.9));

    // A second recurrence starts from the measured physical residual and reaches the solution
    // exactly at its second iteration. The cap must not downgrade that verified convergence.
    controls.max_iterations = 2;
    iterate.set_val(Real(0));
    const auto converged =
        detail::solve_prepared_affine_in_place(problem, workspace, iterate, rhs, controls);
    EXPECT_TRUE(converged.solved()) << converged.reason;
    EXPECT_EQ(converged.iters, 2);
    EXPECT_LE(converged.residual_norm, Real(1e-12));
    EXPECT_LT(max_abs_diff(iterate, rhs), Real(1e-12));
  }
}

TEST(test_krylov_workspace_reentrancy, gmres_infinity_stopping_keeps_euclidean_arnoldi) {
  comm_init();
  // This three-eigenvalue system needs a genuine Arnoldi recurrence. In infinity mode the
  // initial recurrence scale is 10, whereas its Euclidean beta is sqrt(200)/10, not one.
  // The same scientific vector is replicated, so neither norm may multiply by the MPI size.
  const TestLayout boxes(std::vector<TestBox>{
      TestBox{Index<kDim>{0, 0}, Index<kDim>{0, 0}}});
  const TestDistribution mapping = TestDistribution::replicated(boxes, world_rank_space());
  const auto vectors = TestVectorDistribution::replicated();
  TestField iterate = make_field(boxes, mapping, 3, 0);
  TestField rhs = make_field(boxes, mapping, 3, 0);
  iterate.set_val(Real(0));
  for (std::size_t local = 0; local < rhs.local_size(); ++local) {
    const auto values = rhs.fab(local).view();
    for_each_cell(rhs.box(local), [=] POPS_HD(const Index<kDim>& index) {
      values(index, 0) = Real(6);
      values(index, 1) = Real(10);
      values(index, 2) = Real(8);
    });
  }
  Kokkos::fence();
  const OperatorEvaluationSnapshot snapshot = test_snapshot(iterate);
  const TestKrylovMethod method = gmres_krylov_method<kDim>(3);
  const TestKrylovFootprint footprint{3, extent(0), false};
  int applications = 0;
  TestAffineProblem problem(
      iterate,
      TestAffineOperatorProvider::trusted_reentrant(
          [&applications](TestField& out, const TestField& in) {
            ++applications;
            for (std::size_t local = 0; local < out.local_size(); ++local) {
              const auto output = out.fab(local).view();
              const auto input = in.fab(local).view();
              for_each_cell(out.box(local), [=] POPS_HD(const Index<kDim>& index) {
                output(index, 0) = Real(4) * input(index, 0) + input(index, 1);
                output(index, 1) = input(index, 0) + Real(3) * input(index, 1) + input(index, 2);
                output(index, 2) = input(index, 1) + Real(2) * input(index, 2);
              });
            }
            Kokkos::fence();
          },
          [] { return std::size_t{0}; }),
      TestLinearPreconditioner::identity(), LinearOperatorProperties::symmetric_positive_definite(),
      footprint, TestNullspacePolicy::nonsingular(), [&snapshot] { return snapshot; }, {}, vectors);
  TestKrylovWorkspace workspace(iterate, method, footprint, vectors);
  problem.prepare(snapshot);
  workspace.bind(problem);
  const auto candidate_bits = [](const TestField& field) {
    std::vector<RealBits> result;
    for (std::size_t local = 0; local < field.local_size(); ++local) {
      auto host = field.fab(local).create_host_mirror();
      field.fab(local).copy_to_host(host);
      for (std::size_t index = 0; index < host.size(); ++index)
        result.push_back(std::bit_cast<RealBits>(host(index)));
    }
    return result;
  };
  for (const auto norm : {KrylovPhysicalNorm::metric_l2, KrylovPhysicalNorm::component_linf}) {
    SCOPED_TRACE(static_cast<int>(norm));
    iterate.set_val(Real(0));
    TestKrylovControls controls{method, Real(0), Real(1e-11), 3};
    controls.physical_norm = norm;
    applications = 0;
    const auto report =
        detail::solve_prepared_affine_in_place(problem, workspace, iterate, rhs, controls);
    const int original_applications = applications;
    const auto original_candidate = candidate_bits(iterate);
    EXPECT_TRUE(report.solved()) << report.reason;
    EXPECT_EQ(report.iters, 3);
    EXPECT_DOUBLE_EQ(report.reference_residual_norm,
                     norm == KrylovPhysicalNorm::component_linf ? Real(10) : std::sqrt(Real(200)));
    Real error = Real(0);
    Real residual = Real(0);
    for (std::size_t local = 0; local < iterate.local_size(); ++local) {
      const auto values = iterate.fab(local).view();
      error = std::max(error, for_each_cell_reduce_max(
          iterate.box(local), [=] POPS_HD(const Index<kDim>& index) {
            Real result = Real(0);
            for (int component = 0; component < 3; ++component) {
              const Real difference = values(index, component) - Real(component + 1);
              result = Kokkos::fmax(result, Kokkos::fabs(difference));
            }
            return result;
          }));
      residual = std::max(residual, for_each_cell_reduce_max(
          iterate.box(local), [=] POPS_HD(const Index<kDim>& index) {
            const Real r0 = Real(6) - Real(4) * values(index, 0) - values(index, 1);
            const Real r1 = Real(10) - values(index, 0) - Real(3) * values(index, 1) - values(index, 2);
            const Real r2 = Real(8) - values(index, 1) - Real(2) * values(index, 2);
            return Kokkos::fmax(Kokkos::fabs(r0), Kokkos::fmax(Kokkos::fabs(r1), Kokkos::fabs(r2)));
          }));
    }
    EXPECT_LT(all_reduce_max(static_cast<double>(error)), 1e-10);
    EXPECT_LE(all_reduce_max(static_cast<double>(residual)), 1e-11);
    EXPECT_LE(report.residual_norm, Real(1e-11));

    GmresDiagnosticTrace trace;
    controls.diagnostic_trace = &trace;
    iterate.set_val(Real(0));
    applications = 0;
    const auto observed =
        detail::solve_prepared_affine_in_place(problem, workspace, iterate, rhs, controls);
    EXPECT_EQ(candidate_bits(iterate), original_candidate);
    EXPECT_EQ(applications, original_applications);
    EXPECT_EQ(observed.status, report.status);
    EXPECT_EQ(observed.action, report.action);
    EXPECT_EQ(observed.iters, report.iters);
    EXPECT_EQ(observed.reason, report.reason);
    EXPECT_DOUBLE_EQ(observed.residual_norm, report.residual_norm);
    EXPECT_DOUBLE_EQ(observed.reference_residual_norm, report.reference_residual_norm);
    EXPECT_DOUBLE_EQ(observed.rel_residual, report.rel_residual);
    ASSERT_EQ(trace.size, 1u);
    EXPECT_FALSE(trace.overflow);
    EXPECT_EQ(trace.cycles[0].begin_iteration, 0);
    EXPECT_EQ(trace.cycles[0].end_iteration, 3);
    EXPECT_EQ(trace.cycles[0].dimension, 3);
    EXPECT_DOUBLE_EQ(trace.cycles[0].final_residual, report.residual_norm);
  }
  // The first exact GMRES candidate has residual (-265,8,333)/157. Its infinity norm is
  // 2.121... and its Euclidean norm is 2.711.... At tau=2.25 only the explicitly requested
  // infinity solve is converged. This exercises the recurrence and independent final verifier,
  // not merely the early initial-residual return.
  for (const auto norm : {KrylovPhysicalNorm::metric_l2, KrylovPhysicalNorm::component_linf}) {
    SCOPED_TRACE(static_cast<int>(norm));
    iterate.set_val(Real(0));
    TestKrylovControls controls{method, Real(0), Real(2.25), 1};
    controls.physical_norm = norm;
    const auto report =
        detail::solve_prepared_affine_in_place(problem, workspace, iterate, rhs, controls);
    EXPECT_EQ(report.iters, 1);
    if (norm == KrylovPhysicalNorm::component_linf) {
      EXPECT_TRUE(report.solved()) << report.reason;
      EXPECT_NEAR(report.residual_norm, Real(333) / Real(157), Real(1e-13));
    } else {
      EXPECT_EQ(report.status, SolveStatus::kIterationLimit) << report.reason;
      EXPECT_NEAR(report.residual_norm, std::sqrt(Real(181178)) / Real(157), Real(1e-13));
    }
  }
  // The same exact first-column residual has Linf 333/157 > 2, whereas its
  // Euclidean estimate meets the mapped threshold. The observer must preserve that refused
  // estimate and the following true-residual restart without altering numerical authority.
  GmresDiagnosticTrace trace;
  TestKrylovControls controls{method, Real(0), Real(2), 512};
  controls.physical_norm = KrylovPhysicalNorm::component_linf;
  controls.diagnostic_trace = &trace;
  iterate.set_val(Real(0));
  const auto short_cycle =
      detail::solve_prepared_affine_in_place(problem, workspace, iterate, rhs, controls);
  EXPECT_TRUE(short_cycle.solved()) << short_cycle.reason;
  ASSERT_GT(trace.size, 1u);
  EXPECT_EQ(trace.cycles[0].dimension, 1);
  EXPECT_NE(trace.cycles[0].end_flags & 1u, 0u);
  EXPECT_GT(trace.cycles[0].final_residual, controls.abs_tol);

  controls.max_iterations = 513;
  applications = 0;
  EXPECT_THROW((void)detail::solve_prepared_affine_in_place(
                   problem, workspace, iterate, rhs, controls), std::invalid_argument);
  EXPECT_EQ(applications, 0);
  controls.max_iterations = 3;
  controls.abs_tol = Real(1e-11);
  if (n_ranks() > 1) {
    controls.diagnostic_trace = my_rank() == 0 ? &trace : nullptr;
    EXPECT_THROW((void)detail::solve_prepared_affine_in_place(
                     problem, workspace, iterate, rhs, controls), std::logic_error);
    EXPECT_EQ(applications, 0);
  }
  controls.diagnostic_trace = &trace;
  iterate.set_val(Real(0));
  EXPECT_TRUE(detail::solve_prepared_affine_in_place(
                  problem, workspace, iterate, rhs, controls).solved());
  EXPECT_EQ(trace.size, 1u);  // Reset, not append to the previous invocation.
}

TEST(test_krylov_workspace_reentrancy, gmres_infinity_relative_reference_is_independent_of_warm_start) {
  const TestLayout boxes(std::vector<TestBox>{
      TestBox{Index<kDim>{0, 0}, Index<kDim>{0, 0}},
      TestBox{Index<kDim>{1, 0}, Index<kDim>{1, 0}},
      TestBox{Index<kDim>{2, 0}, Index<kDim>{2, 0}},
      TestBox{Index<kDim>{3, 0}, Index<kDim>{3, 0}}});
  const TestDistribution mapping = round_robin_distribution(boxes);
  TestField iterate = make_field(boxes, mapping, 1, 0);
  TestField rhs = make_field(boxes, mapping, 1, 0);
  iterate.set_val(Real(0));
  rhs.set_val(Real(0));
  const OperatorEvaluationSnapshot snapshot = test_snapshot(iterate);
  const TestKrylovMethod method = gmres_krylov_method<kDim>(2);
  const TestKrylovFootprint footprint{1, extent(0), false};
  TestAffineProblem problem(
      iterate,
      TestAffineOperatorProvider::trusted_reentrant(
          [](TestField& out, const TestField& in) { detail::PreparedFieldAlgebra::copy(out, in); },
          [] { return std::size_t{0}; }),
      TestLinearPreconditioner::identity(), LinearOperatorProperties::general(), footprint,
      TestNullspacePolicy::nonsingular(), [&snapshot] { return snapshot; });
  TestKrylovWorkspace workspace(iterate, method, footprint);
  problem.prepare(snapshot);
  workspace.bind(problem);
  const Real tolerance = std::ldexp(Real(1), -20);
  for (int mode = 0; mode < 3; ++mode) {
    SCOPED_TRACE(mode);
    for (std::size_t local = 0; local < iterate.local_size(); ++local) {
      const auto values = iterate.fab(local).view();
      const auto load = rhs.fab(local).view();
      for_each_cell(iterate.box(local), [=] POPS_HD(const Index<kDim>& index) {
        load(index, 0) = Real(1) + Real(0.25) * Real(index[0]);
        values(index, 0) = load(index, 0) - Real(0.75) * tolerance;
      });
    }
    Kokkos::fence();
    TestKrylovControls controls{method, Real(0), tolerance, 2};
    controls.physical_norm = mode == 0 ? KrylovPhysicalNorm::metric_l2
                                       : KrylovPhysicalNorm::component_linf;
    if (mode == 2) {
      controls.rel_tol = tolerance / Real(1.75);
      controls.abs_tol = Real(0);
    }
    const auto report =
        detail::solve_prepared_affine_in_place(problem, workspace, iterate, rhs, controls);
    EXPECT_TRUE(report.solved()) << report.reason;
    if (mode == 0) {
      EXPECT_EQ(report.iters, 1);
      EXPECT_DOUBLE_EQ(report.reference_residual_norm, std::sqrt(Real(7.875)));
    } else {
      EXPECT_EQ(report.iters, 0);
      EXPECT_DOUBLE_EQ(report.reference_residual_norm, Real(1.75));
      EXPECT_DOUBLE_EQ(report.residual_norm, Real(0.75) * tolerance);
      EXPECT_DOUBLE_EQ(report.rel_residual, Real(0.75) * tolerance / Real(1.75));
    }
  }
}

TEST(test_krylov_workspace_reentrancy, gmres_infinity_affine_reference_precedes_normalization) {
  const TestLayout boxes(std::vector<TestBox>{
      TestBox{Index<kDim>{0, 0}, Index<kDim>{0, 0}},
      TestBox{Index<kDim>{1, 0}, Index<kDim>{1, 0}}});
  const TestDistribution mapping = round_robin_distribution(boxes);
  for (const int exponent : {-900, 900}) {
    SCOPED_TRACE(exponent);
    // Four ulps separate b from A(0); the warm-start residual is one ulp. Squaring these
    // physical values underflows/overflows, and ||b|| would give the wrong relative reference.
    const Real offset = std::ldexp(Real(1), exponent);
    const Real load = std::ldexp(Real(1), exponent - 50);
    TestField iterate = make_field(boxes, mapping, 1, 0);
    TestField rhs = make_field(boxes, mapping, 1, 0);
    iterate.set_val(Real(0.75) * load);
    rhs.set_val(offset + load);
    const OperatorEvaluationSnapshot snapshot = test_snapshot(iterate);
    const TestKrylovMethod method = gmres_krylov_method<kDim>(2);
    const TestKrylovFootprint footprint{1, extent(0), false};
    TestAffineProblem problem(
        iterate,
        TestAffineOperatorProvider::trusted_reentrant(
            [offset](TestField& out, const TestField& in) {
              for (std::size_t local = 0; local < out.local_size(); ++local) {
                const auto output = out.fab(local).view();
                const auto input = in.fab(local).view();
                for_each_cell(out.box(local), [=] POPS_HD(const Index<kDim>& index) {
                  output(index, 0) = input(index, 0) + offset;
                });
              }
              Kokkos::fence();
            },
            [] { return std::size_t{0}; }),
        TestLinearPreconditioner::identity(), LinearOperatorProperties::general(), footprint,
        TestNullspacePolicy::nonsingular(), [&snapshot] { return snapshot; });
    TestKrylovWorkspace workspace(iterate, method, footprint);
    problem.prepare(snapshot);
    workspace.bind(problem);
    TestKrylovControls controls{method, Real(0.3), Real(0), 2};
    controls.physical_norm = KrylovPhysicalNorm::component_linf;
    const auto report =
        detail::solve_prepared_affine_in_place(problem, workspace, iterate, rhs, controls);
    EXPECT_TRUE(report.solved()) << report.reason;
    EXPECT_EQ(report.iters, 0);
    EXPECT_DOUBLE_EQ(report.reference_residual_norm, load);
    EXPECT_DOUBLE_EQ(report.residual_norm, Real(0.25) * load);
    EXPECT_DOUBLE_EQ(report.rel_residual, Real(0.25));
  }
}

TEST(test_krylov_workspace_reentrancy, physical_stopping_norm_is_authenticated_before_callbacks) {
  const TestLayout boxes(std::vector<TestBox>{
      TestBox{Index<kDim>{0, 0}, Index<kDim>{0, 0}},
      TestBox{Index<kDim>{1, 0}, Index<kDim>{1, 0}}});
  const TestDistribution mapping = round_robin_distribution(boxes);
  TestField iterate = make_field(boxes, mapping, 1, 0);
  TestField rhs = make_field(boxes, mapping, 1, 0);
  iterate.set_val(Real(0));
  rhs.set_val(Real(1));
  const OperatorEvaluationSnapshot snapshot = test_snapshot(iterate);
  const TestKrylovFootprint footprint{1, extent(0), false};
  std::atomic<int> applications{0};
  TestAffineProblem problem(
      iterate,
      TestAffineOperatorProvider::trusted_reentrant(
          [&applications](TestField& out, const TestField& in) {
            ++applications;
            detail::PreparedFieldAlgebra::copy(out, in);
          },
          [] { return std::size_t{0}; }),
      TestLinearPreconditioner::identity(), LinearOperatorProperties::symmetric_positive_definite(),
      footprint, TestNullspacePolicy::nonsingular(), [&snapshot] { return snapshot; });
  problem.prepare(snapshot);
  const TestKrylovMethod gmres = gmres_krylov_method<kDim>(2);
  TestKrylovWorkspace workspace(iterate, gmres, footprint);
  workspace.bind(problem);
  const int prepared_applications = applications.load();
  TestKrylovControls unknown{gmres, Real(1e-12), Real(0), 2};
  unknown.physical_norm = static_cast<KrylovPhysicalNorm>(19);
  EXPECT_THROW((void)detail::solve_prepared_affine_in_place(problem, workspace, iterate, rhs, unknown),
               std::invalid_argument);
  EXPECT_EQ(applications.load(), prepared_applications);
  if (n_ranks() > 1) {
    TestKrylovControls asymmetric{gmres, Real(1e-12), Real(0), 2};
    asymmetric.physical_norm = my_rank() == 0 ? KrylovPhysicalNorm::component_linf
                                             : KrylovPhysicalNorm::metric_l2;
    EXPECT_THROW((void)detail::solve_prepared_affine_in_place(problem, workspace, iterate, rhs,
                                                            asymmetric), std::logic_error);
    EXPECT_THROW(workspace.require_bound(problem, asymmetric), std::logic_error);
    EXPECT_EQ(applications.load(), prepared_applications);
    // A rank-local unknown value must also reach the collective rejection before a callback.
    if (my_rank() != 0)
      unknown.physical_norm = KrylovPhysicalNorm::component_linf;
    EXPECT_THROW((void)detail::solve_prepared_affine_in_place(problem, workspace, iterate, rhs,
                                                            unknown), std::logic_error);
    EXPECT_EQ(applications.load(), prepared_applications);
  }
  for (const auto method : {cg_krylov_method<kDim>(), bicgstab_krylov_method<kDim>(),
                            richardson_krylov_method<kDim>(Real(1))}) {
    SCOPED_TRACE(std::string(method.identity()));
    TestKrylovWorkspace other_workspace(iterate, method, footprint);
    other_workspace.bind(problem);
    const int before = applications.load();
    TestKrylovControls unsupported{method, Real(1e-12), Real(0), 2};
    unsupported.physical_norm = KrylovPhysicalNorm::component_linf;
    EXPECT_THROW((void)detail::solve_prepared_affine_in_place(problem, other_workspace, iterate, rhs,
                                                            unsupported), std::invalid_argument);
    EXPECT_EQ(applications.load(), before);
  }
  TestKrylovProblemFacts facts{problem.properties(), footprint, problem.vector_distribution(),
                              problem.metric().robust_payload_width(), true, false};
  EXPECT_TRUE(gmres.validate_problem(facts).accepted());
  facts.physical_norm = KrylovPhysicalNorm::component_linf;
  EXPECT_FALSE(gmres.validate_problem(facts).accepted());
}

TEST(test_krylov_workspace_reentrancy, gmres_infinity_refuses_nonfinite_physical_input) {
  // The second owner holds the only NaN in MPI2. The infinity reduction must not hide it.
  const TestLayout boxes(std::vector<TestBox>{
      TestBox{Index<kDim>{0, 0}, Index<kDim>{0, 0}},
      TestBox{Index<kDim>{1, 0}, Index<kDim>{1, 0}}});
  const TestDistribution mapping = round_robin_distribution(boxes);
  TestField iterate = make_field(boxes, mapping, 1, 0);
  TestField rhs = make_field(boxes, mapping, 1, 0);
  iterate.set_val(Real(0));
  rhs.set_val(Real(1));
  const OperatorEvaluationSnapshot snapshot = test_snapshot(iterate);
  const TestKrylovMethod method = gmres_krylov_method<kDim>(2);
  const TestKrylovFootprint footprint{1, extent(0), false};
  TestAffineProblem problem(
      iterate,
      TestAffineOperatorProvider::trusted_reentrant(
          [](TestField& out, const TestField& in) { detail::PreparedFieldAlgebra::copy(out, in); },
          [] { return std::size_t{0}; }),
      TestLinearPreconditioner::identity(), LinearOperatorProperties::general(), footprint,
      TestNullspacePolicy::nonsingular(), [&snapshot] { return snapshot; });
  TestKrylovWorkspace workspace(iterate, method, footprint);
  problem.prepare(snapshot);
  workspace.bind(problem);
  const Real invalid = std::numeric_limits<Real>::quiet_NaN();
  for (std::size_t local = 0; local < iterate.local_size(); ++local) {
    const auto values = iterate.fab(local).view();
    for_each_cell(iterate.box(local), [=] POPS_HD(const Index<kDim>& index) {
      values(index, 0) = index[0] == 1 ? invalid : Real(0);
    });
  }
  Kokkos::fence();
  TestKrylovControls controls{method, Real(1e-12), Real(0), 2};
  controls.physical_norm = KrylovPhysicalNorm::component_linf;
  const auto report =
      detail::solve_prepared_affine_in_place(problem, workspace, iterate, rhs, controls);
  EXPECT_EQ(report.status, SolveStatus::kInvalidEvaluation);
  EXPECT_EQ(report.action, SolveAction::kFailRun);
  EXPECT_EQ(report.iters, 0);
}

TEST(test_krylov_workspace_reentrancy, gmres_infinity_refuses_prepared_nullspace_before_callbacks) {
  // The complete two-cell matrix [[1,-1],[-1,1]] has precisely the constant nullspace.
  // One rank owns the box; extra MPI ranks still participate in the same refusal boundary.
  const TestLayout boxes(std::vector<TestBox>{
      TestBox{Index<kDim>{0, 0}, Index<kDim>{1, 0}}});
  const TestDistribution mapping = round_robin_distribution(boxes);
  TestField iterate = make_field(boxes, mapping, 1, 0);
  TestField rhs = make_field(boxes, mapping, 1, 0);
  iterate.set_val(Real(0));
  rhs.set_val(Real(0));
  const OperatorEvaluationSnapshot snapshot = test_snapshot(iterate);
  const TestKrylovMethod method = gmres_krylov_method<kDim>(2);
  const TestKrylovFootprint footprint{1, extent(0), false};
  std::atomic<int> applications{0};
  auto nullspace = constant_mean_zero_nullspace<kDim>(
      "test://krylov/infinity-unsupported-nullspace@1", "two-cell constant nullspace");
  TestAffineProblem problem(
      iterate,
      TestAffineOperatorProvider::trusted_reentrant(
          [&applications](TestField& out, const TestField& in) {
            ++applications;
            for (std::size_t local = 0; local < out.local_size(); ++local) {
              const auto output = out.fab(local).view();
              const auto input = in.fab(local).view();
              for_each_cell(out.box(local), [=] POPS_HD(const Index<kDim>& index) {
                const Index<kDim> other{1 - index[0], 0};
                output(index, 0) = input(index, 0) - input(other, 0);
              });
            }
            Kokkos::fence();
          },
          [] { return std::size_t{0}; }),
      TestLinearPreconditioner::identity(),
      LinearOperatorProperties::symmetric_positive_definite_on_nullspace_complement(), footprint,
      TestNullspacePolicy::preserving(std::move(nullspace)), [&snapshot] { return snapshot; });
  TestKrylovWorkspace workspace(iterate, method, footprint);
  problem.prepare(snapshot);
  workspace.bind(problem);
  TestKrylovControls controls{method, Real(1e-12), Real(0), 2};
  controls.physical_norm = KrylovPhysicalNorm::component_linf;
  const int before = applications.load();
  EXPECT_THROW((void)detail::solve_prepared_affine_in_place(problem, workspace, iterate, rhs, controls),
               std::invalid_argument);
  EXPECT_EQ(applications.load(), before);
  controls.physical_norm = KrylovPhysicalNorm::metric_l2;
  const auto report =
      detail::solve_prepared_affine_in_place(problem, workspace, iterate, rhs, controls);
  EXPECT_TRUE(report.solved()) << report.reason;
  EXPECT_EQ(report.iters, 0);
}

TEST(test_krylov_workspace_reentrancy,
     gmres_single_column_recovers_masked_representable_stagnation) {
  comm_init();
  const TestLayout boxes(std::vector<TestBox>{TestBox{Index<kDim>{0, 0}, Index<kDim>{0, 0}}});
  const TestDistribution mapping = TestDistribution::replicated(boxes, world_rank_space());
  const auto vectors = TestVectorDistribution::replicated();
  const Real base = Real(1.5);
  const Real ulp = std::nextafter(base, std::numeric_limits<Real>::infinity()) - base;
  const Real tolerance = Real(0.75) * ulp;
  const Real high_scale = std::ldexp(Real(1), 2 * std::numeric_limits<Real>::max_exponent / 3);
  const Real low_scale = Real(1) / high_scale;
  // All entries are binary fractions. For the first matrix, the exact inverse proposes
  // (+15/32,-15/32) ulp: both ordinary additions stagnate. Only equation 0 exceeds tau.
  // Moving x_0 by one ulp passes; moving both components would fail the true residual.
  // The second matrix is a counterexample to assuming that the mask promises descent:
  // the proposed x_0 step creates a residual of two ulps in equation 1 and must be rejected.
  for (const bool coupled_refusal : {false, true}) {
    for (const int restart : {1, 3}) {
      for (const Real preconditioner_scale : {Real(1), high_scale, low_scale}) {
        SCOPED_TRACE(coupled_refusal);
        SCOPED_TRACE(restart);
        SCOPED_TRACE(preconditioner_scale);
        TestField iterate = make_field(boxes, mapping, 2, 0);
        TestField rhs = make_field(boxes, mapping, 2, 0);
        const Real initial_first = coupled_refusal ? std::nextafter(base, Real(2)) : base;
        const Real row_factor = coupled_refusal ? Real(1) : Real(0.25);
        const Real inverse_factor = coupled_refusal ? Real(0.5) : Real(2);
        const Real first_rhs = (coupled_refusal ? Real(1.875) : Real(0.9375)) * ulp;
        const Real second_rhs = coupled_refusal ? Real(3) : Real(0.75);
        const auto reset = [&] {
          for (std::size_t local = 0; local < iterate.local_size(); ++local) {
            const auto x = iterate.fab(local).view();
            const auto right = rhs.fab(local).view();
            for_each_cell(iterate.box(local), [=] POPS_HD(const Index<kDim>& index) {
              x(index, 0) = initial_first;
              x(index, 1) = base;
              right(index, 0) = first_rhs;
              right(index, 1) = second_rhs;
            });
          }
          Kokkos::fence();
        };
        reset();
        const OperatorEvaluationSnapshot snapshot = test_snapshot(iterate);
        const TestKrylovMethod method = gmres_krylov_method<kDim>(restart);
        const TestKrylovFootprint footprint{2, extent(0), true};
        auto preconditioner = TestLinearPreconditionerProvider::trusted_extension(
            {"pops.test.krylov.masked-stagnation-inverse", 1}, {}, [=](const ExecutionLane&) {
              return TestLinearPreconditionerCallbacks{
                  [] {},
                  [=](TestField& out, const TestField& in) {
                    for (std::size_t local = 0; local < out.local_size(); ++local) {
                      const auto output = out.fab(local).view();
                      const auto input = in.fab(local).view();
                      for_each_cell(out.box(local), [=] POPS_HD(const Index<kDim>& index) {
                        output(index, 0) =
                            preconditioner_scale *
                            (Real(0.5) * input(index, 0) + inverse_factor * input(index, 1));
                        output(index, 1) =
                            preconditioner_scale *
                            (-Real(0.5) * input(index, 0) + inverse_factor * input(index, 1));
                      });
                    }
                    Kokkos::fence();
                  },
                  [] { return std::size_t{0}; }};
            });
        TestAffineProblem problem(
            iterate,
            TestAffineOperatorProvider::trusted_reentrant(
                [=](TestField& out, const TestField& in) {
                  for (std::size_t local = 0; local < out.local_size(); ++local) {
                    const auto output = out.fab(local).view();
                    const auto input = in.fab(local).view();
                    for_each_cell(out.box(local), [=] POPS_HD(const Index<kDim>& index) {
                      output(index, 0) = input(index, 0) - input(index, 1);
                      output(index, 1) = row_factor * (input(index, 0) + input(index, 1));
                    });
                  }
                  Kokkos::fence();
                },
                [] { return std::size_t{0}; }),
            TestLinearPreconditioner(iterate, std::move(preconditioner), vectors),
            LinearOperatorProperties::general(), footprint, TestNullspacePolicy::nonsingular(),
            [&snapshot] { return snapshot; }, {}, vectors);
        TestKrylovWorkspace workspace(iterate, method, footprint, vectors);
        problem.prepare(snapshot);
        workspace.bind(problem);
        const auto allocated = workspace.allocation_count();
        for (const auto norm :
             {KrylovPhysicalNorm::metric_l2, KrylovPhysicalNorm::component_linf}) {
          for (const int maximum : {1, 2, 1}) {
            reset();
            TestKrylovControls controls{method, Real(0), tolerance, maximum};
            controls.physical_norm = norm;
            GmresDiagnosticTrace trace;
            controls.diagnostic_trace = &trace;
            const auto report =
                detail::solve_prepared_affine_in_place(problem, workspace, iterate, rhs, controls);
            const bool recover = norm == KrylovPhysicalNorm::component_linf && maximum == 2;
            EXPECT_EQ(report.solved(), recover && !coupled_refusal) << report.reason;
            EXPECT_EQ(report.iters, maximum);
            EXPECT_EQ(workspace.allocation_count(), allocated);
            ASSERT_EQ(trace.size, static_cast<std::size_t>(maximum));
            for (std::size_t cycle = 0; cycle < trace.size; ++cycle)
              EXPECT_EQ(trace.cycles[cycle].dimension, 1);
            Real residual = 0;
            for (std::size_t local = 0; local < iterate.local_size(); ++local) {
              auto host = iterate.fab(local).create_host_mirror();
              iterate.fab(local).copy_to_host(host);
              const Real expected_first =
                  recover ? std::nextafter(initial_first, Real(2)) : initial_first;
              EXPECT_EQ(host(0), expected_first);
              EXPECT_EQ(host(1), base);
              const Real r0 = first_rhs - (host(0) - host(1));
              const Real r1 = second_rhs - row_factor * (host(0) + host(1));
              residual = std::max(residual, std::max(std::abs(r0), std::abs(r1)));
            }
            residual = all_reduce_max(residual);
            if (recover && !coupled_refusal)
              EXPECT_LE(residual, tolerance);
            else
              EXPECT_GT(residual, tolerance);
            if (norm == KrylovPhysicalNorm::component_linf)
              EXPECT_EQ(report.residual_norm, residual);
          }
        }
      }
    }
  }
}

TEST(test_krylov_workspace_reentrancy,
     gmres_damps_a_rejected_single_column_rounding_cycle) {
  comm_init();
  const TestLayout boxes(
      std::vector<TestBox>{TestBox{Index<kDim>{0, 0}, Index<kDim>{0, 0}}});
  const TestDistribution mapping =
      TestDistribution::replicated(boxes, world_rank_space());
  const auto vectors = TestVectorDistribution::replicated();
  const Real base = Real(1.5);
  const Real ulp = std::nextafter(base, Real(2)) - base;
  const Real second_base = Real(0.375);
  const Real second_ulp = std::nextafter(second_base, Real(1)) - second_base;
  const Real first_rhs = -Real(31) / Real(16) * ulp;
  const Real second_rhs = second_base - Real(3) * second_ulp;
  const Real tolerance = ulp / Real(4);
  const Real high_scale =
      std::ldexp(Real(1), 2 * std::numeric_limits<Real>::max_exponent / 3);
  const Real low_scale = Real(1) / high_scale;
  // A=[[1,-1],[1/8,1/8]], with its exact inverse. Ordinary corrections
  // alternate between offsets (-4,-3) and (-3,-2) ulp from (1.5,1.5), both
  // above tau. After true non-descent, half a correction plus the existing
  // lost-update recovery reaches (-4,-2): its actual residual is (1/16,0) ulp.
  // No estimate may accept early.
  for (const int restart : {1, 3}) {
    for (const Real preconditioner_scale : {Real(1), high_scale, low_scale}) {
      SCOPED_TRACE(restart);
      SCOPED_TRACE(preconditioner_scale);
      TestField iterate = make_field(boxes, mapping, 2, 0);
      TestField rhs = make_field(boxes, mapping, 2, 0);
      const auto reset = [&] {
        for (std::size_t local = 0; local < iterate.local_size(); ++local) {
          const auto x = iterate.fab(local).view();
          const auto right = rhs.fab(local).view();
          for_each_cell(iterate.box(local),
                        [=] POPS_HD(const Index<kDim> &index) {
                          x(index, 0) = base - ulp;
                          x(index, 1) = base;
                          right(index, 0) = first_rhs;
                          right(index, 1) = second_rhs;
                        });
        }
        Kokkos::fence();
      };
      reset();
      const OperatorEvaluationSnapshot snapshot = test_snapshot(iterate);
      const TestKrylovMethod method = gmres_krylov_method<kDim>(restart);
      const TestKrylovFootprint footprint{2, extent(0), true};
      auto preconditioner = TestLinearPreconditionerProvider::trusted_extension(
          {"pops.test.krylov.rounding-cycle-inverse", 1}, {},
          [=](const ExecutionLane &) {
            return TestLinearPreconditionerCallbacks{
                [] {},
                [=](TestField &out, const TestField &in) {
                  for (std::size_t local = 0; local < out.local_size();
                       ++local) {
                    const auto output = out.fab(local).view();
                    const auto input = in.fab(local).view();
                    for_each_cell(
                        out.box(local), [=] POPS_HD(const Index<kDim> &index) {
                          output(index, 0) = preconditioner_scale *
                                             (Real(0.5) * input(index, 0) +
                                              Real(4) * input(index, 1));
                          output(index, 1) = preconditioner_scale *
                                             (-Real(0.5) * input(index, 0) +
                                              Real(4) * input(index, 1));
                        });
                  }
                  Kokkos::fence();
                },
                [] { return std::size_t{0}; }};
          });
      TestAffineProblem problem(
          iterate,
          TestAffineOperatorProvider::trusted_reentrant(
              [=](TestField &out, const TestField &in) {
                for (std::size_t local = 0; local < out.local_size(); ++local) {
                  const auto output = out.fab(local).view();
                  const auto input = in.fab(local).view();
                  for_each_cell(
                      out.box(local), [=] POPS_HD(const Index<kDim> &index) {
                        output(index, 0) = input(index, 0) - input(index, 1);
                        output(index, 1) =
                            Real(0.125) * (input(index, 0) + input(index, 1));
                      });
                }
                Kokkos::fence();
              },
              [] { return std::size_t{0}; }),
          TestLinearPreconditioner(iterate, std::move(preconditioner), vectors),
          LinearOperatorProperties::general(), footprint,
          TestNullspacePolicy::nonsingular(), [&snapshot] { return snapshot; },
          {}, vectors);
      TestKrylovWorkspace workspace(iterate, method, footprint, vectors);
      problem.prepare(snapshot);
      workspace.bind(problem);
      const auto allocated = workspace.allocation_count();
      for (const auto norm : {KrylovPhysicalNorm::metric_l2,
                              KrylovPhysicalNorm::component_linf}) {
        for (const int maximum : {2, 3, 2}) {
          SCOPED_TRACE(maximum);
          reset();
          TestKrylovControls controls{method, Real(0), tolerance, maximum};
          controls.physical_norm = norm;
          GmresDiagnosticTrace trace;
          controls.diagnostic_trace = &trace;
          const auto report = detail::solve_prepared_affine_in_place(
              problem, workspace, iterate, rhs, controls);
          const bool solved =
              norm == KrylovPhysicalNorm::component_linf && maximum == 3;
          EXPECT_EQ(report.solved(), solved) << report.reason;
          EXPECT_EQ(report.iters, maximum);
          EXPECT_EQ(workspace.allocation_count(), allocated);
          ASSERT_EQ(trace.size, static_cast<std::size_t>(maximum));
          for (std::size_t cycle = 0; cycle < trace.size; ++cycle)
            EXPECT_EQ(trace.cycles[cycle].dimension, 1);
          Real residual = 0;
          for (std::size_t local = 0; local < iterate.local_size(); ++local) {
            auto host = iterate.fab(local).create_host_mirror();
            iterate.fab(local).copy_to_host(host);
            EXPECT_EQ(host(0), base - Real(maximum == 2 ? 3 : 4) * ulp);
            EXPECT_EQ(host(1),
                      base - Real(maximum == 2 || solved ? 2 : 3) * ulp);
            const Real r0 = first_rhs - (host(0) - host(1));
            const Real r1 = second_rhs - Real(0.125) * (host(0) + host(1));
            residual = std::max(residual, std::max(std::abs(r0), std::abs(r1)));
          }
          residual = all_reduce_max(residual);
          if (solved)
            EXPECT_LE(residual, tolerance);
          else
            EXPECT_GT(residual, tolerance);
          if (norm == KrylovPhysicalNorm::component_linf)
            EXPECT_EQ(report.residual_norm, residual);
        }
      }
    }
  }
}

TEST(test_krylov_workspace_reentrancy,
     scaled_stagnation_update_preserves_masks_and_exponent_guards) {
  comm_init();
  const TestLayout boxes(std::vector<TestBox>{TestBox{Index<kDim>{0, 0}, Index<kDim>{0, 0}}});
  const TestDistribution mapping = TestDistribution::replicated(boxes, world_rank_space());
  TestField destination = make_field(boxes, mapping, 1, 0);
  TestField reference = make_field(boxes, mapping, 1, 0);
  TestField source = make_field(boxes, mapping, 1, 0);
  TestField mask = make_field(boxes, mapping, 1, 0);
  const Real base = Real(1.5);
  const Real ulp = std::nextafter(base, Real(2)) - base;
  using detail::ScaledScalar;
  const auto run = [&](Real value, const ScaledScalar& coefficient, Real input, bool enabled,
                       bool expect_adjacent, bool negative, bool restrict_to_mask = false) {
    destination.set_val(value);
    reference.set_val(value);
    source.set_val(input);
    mask.set_val(enabled ? Real(1) : Real(0));
    detail::ScaledFieldAlgebra::axpy(reference, coefficient, source);
    const bool promoted = detail::ScaledFieldAlgebra::axpy_adjacent_if_stagnant(
        destination, coefficient, source, mask, restrict_to_mask);
    EXPECT_EQ(promoted, destination.local_size() != 0 && expect_adjacent);
    for (std::size_t local = 0; local < destination.local_size(); ++local) {
      auto actual = destination.fab(local).create_host_mirror();
      destination.fab(local).copy_to_host(actual);
      auto ordinary = reference.fab(local).create_host_mirror();
      reference.fab(local).copy_to_host(ordinary);
      const Real expected =
          restrict_to_mask && !enabled ? value
          : expect_adjacent
              ? std::nextafter(value, negative ? -std::numeric_limits<Real>::infinity()
                                               : std::numeric_limits<Real>::infinity())
              : ordinary(0);
      if (std::isnan(expected))
        EXPECT_TRUE(std::isnan(actual(0)));
      else
        EXPECT_EQ(std::bit_cast<RealBits>(actual(0)), std::bit_cast<RealBits>(expected));
    }
  };
  run(base, ScaledScalar::from(Real(0.25) * ulp), Real(1), false, false, false);
  run(base, ScaledScalar::from(Real(0.25) * ulp), Real(1), true, true, false);
  run(base, ScaledScalar::from(-Real(0.25) * ulp), Real(1), true, true, true);
  run(base, ScaledScalar::from(ulp), Real(1), false, false, false, true);
  run(base, ScaledScalar::from(ulp), Real(1), true, false, false, true);
  run(base, ScaledScalar::from(Real(0.25) * ulp), Real(1), true, true, false, true);
  run(base, ScaledScalar::zero(), Real(1), true, false, false);
  run(base, ScaledScalar::from(Real(1)), Real(0), true, false, false);
  run(std::numeric_limits<Real>::max(), ScaledScalar::from(Real(1)), Real(1), true, false, false);
  run(base, ScaledScalar::from(Real(1)), std::numeric_limits<Real>::quiet_NaN(), true, false,
      false);
  const Real high = std::ldexp(Real(1), 2 * std::numeric_limits<Real>::max_exponent / 3);
  const Real low = Real(1) / high;
  const auto huge = ScaledScalar::product(ScaledScalar::from(high), ScaledScalar::from(high));
  const auto tiny = ScaledScalar::product(ScaledScalar::from(low), ScaledScalar::from(low));
  Real materialized = Real(0);
  EXPECT_FALSE(huge.try_materialize(materialized));
  EXPECT_FALSE(tiny.try_materialize(materialized));
  ASSERT_TRUE(huge.try_apply(low, materialized));
  EXPECT_EQ(materialized, high);
  ASSERT_TRUE(tiny.try_apply(high, materialized));
  EXPECT_EQ(materialized, low);
  run(Real(0), huge, low, true, false, false);
  run(base, tiny, high, true, true, false);
  run(base, ScaledScalar::negated(tiny), high, true, true, true);
  run(-high, huge, low, false, false, false);
}

TEST(test_krylov_workspace_reentrancy, gmres_coordinates_coupled_representable_updates) {
  comm_init();
  std::vector<TestBox> block_boxes;
  for (int block = 0; block < 4; ++block)
    block_boxes.emplace_back(Index<kDim>{block, 0}, Index<kDim>{block, 0});
  const TestLayout boxes(std::move(block_boxes));
  const TestDistribution mapping = round_robin_distribution(boxes);
  const Real base = Real(1.5);
  const Real ulp = std::nextafter(base, Real(2)) - base;
  const Real tolerance = Real(0.75) * ulp;
  const Real high = std::ldexp(Real(1), 2 * std::numeric_limits<Real>::max_exponent / 3);
  const Real low = Real(1) / high;
  // Independent copies of T=tridiag(-1,2,-1) act on x-1.5. The exact
  // correction (5/16,-3/8,5/16) ulp is initially lost. Promoting every
  // unconverged component produces a coupled rounding cycle. The native
  // solution (0,-1,0) ulp has residual (0,5/8,0) ulp, below the same tau.
  // One active block checks collective activation with empty contributors;
  // four copies check exact global maxima shared by distinct MPI owners.
  for (const int active_blocks : {1, 4}) {
    for (const int restart : {1, 4}) {
      for (const Real scale : {Real(1), high, low}) {
        SCOPED_TRACE(active_blocks);
        SCOPED_TRACE(restart);
        SCOPED_TRACE(scale);
        TestField iterate = make_field(boxes, mapping, 3, 0);
        TestField rhs = make_field(boxes, mapping, 3, 0);
        const auto reset = [&] {
          for (std::size_t local = 0; local < iterate.local_size(); ++local) {
            const auto x = iterate.fab(local).view();
            const auto right = rhs.fab(local).view();
            for_each_cell(iterate.box(local), [=] POPS_HD(const Index<kDim>& index) {
              const Real amplitude = index[0] < active_blocks ? ulp : Real(0);
              x(index, 0) = base;
              x(index, 1) = base;
              x(index, 2) = base;
              right(index, 0) = amplitude;
              right(index, 1) = -Real(11) / Real(8) * amplitude;
              right(index, 2) = amplitude;
            });
          }
          Kokkos::fence();
        };
        reset();
        const auto snapshot = test_snapshot(iterate);
        const auto method = gmres_krylov_method<kDim>(restart);
        const TestKrylovFootprint footprint{3, extent(0), true};
        bool fail_after_coordinated_update = false;
        int applications_after_second_cycle = 0;
        GmresDiagnosticTrace* active_trace = nullptr;
        auto preconditioner = TestLinearPreconditionerProvider::trusted_extension(
            {"pops.test.krylov.coupled-rounding-inverse", 1}, {}, [=](const ExecutionLane&) {
              return TestLinearPreconditionerCallbacks{
                  [] {},
                  [=](TestField& out, const TestField& in) {
                    for (std::size_t local = 0; local < out.local_size(); ++local) {
                      const auto output = out.fab(local).view();
                      const auto input = in.fab(local).view();
                      for_each_cell(out.box(local), [=] POPS_HD(const Index<kDim>& index) {
                        const Real a = input(index, 0), b = input(index, 1), c = input(index, 2);
                        output(index, 0) = scale * (Real(3) * a + Real(2) * b + c) / Real(4);
                        output(index, 1) = scale * (a + Real(2) * b + c) / Real(2);
                        output(index, 2) = scale * (a + Real(2) * b + Real(3) * c) / Real(4);
                      });
                    }
                    Kokkos::fence();
                  },
                  [] { return std::size_t{0}; }};
            });
        TestAffineProblem problem(
            iterate,
            TestAffineOperatorProvider::trusted_reentrant(
                [=, &fail_after_coordinated_update, &applications_after_second_cycle,
                 &active_trace](TestField& out, const TestField& in) {
                  for (std::size_t local = 0; local < out.local_size(); ++local) {
                    const auto output = out.fab(local).view();
                    const auto input = in.fab(local).view();
                    for_each_cell(out.box(local), [=] POPS_HD(const Index<kDim>& index) {
                      const Real a = input(index, 0) - base;
                      const Real b = input(index, 1) - base;
                      const Real c = input(index, 2) - base;
                      output(index, 0) = Real(2) * a - b;
                      output(index, 1) = -a + Real(2) * b - c;
                      output(index, 2) = -b + Real(2) * c;
                    });
                  }
                  Kokkos::fence();
                  if (fail_after_coordinated_update && active_trace && active_trace->size == 2) {
                    ++applications_after_second_cycle;
                    // One linear Arnoldi application precedes the true-residual callback.
                    // The last MPI owner has no active mask when only block zero is forced.
                    if (applications_after_second_cycle == 2 && my_rank() == n_ranks() - 1)
                      throw std::runtime_error("failure after a coordinated update");
                  }
                },
                [] { return std::size_t{0}; }),
            TestLinearPreconditioner(iterate, std::move(preconditioner)),
            LinearOperatorProperties::general(), footprint, TestNullspacePolicy::nonsingular(),
            [&snapshot] { return snapshot; });
        TestKrylovWorkspace workspace(iterate, method, footprint);
        problem.prepare(snapshot);
        workspace.bind(problem);
        const auto allocations = workspace.allocation_count();
        for (const auto norm :
             {KrylovPhysicalNorm::component_linf, KrylovPhysicalNorm::metric_l2}) {
          for (const int maximum : {4, 5, 4}) {
            SCOPED_TRACE(maximum);
            reset();
            TestKrylovControls controls{method, Real(0), tolerance, maximum};
            controls.physical_norm = norm;
            GmresDiagnosticTrace trace;
            controls.diagnostic_trace = &trace;
            const auto report =
                detail::solve_prepared_affine_in_place(problem, workspace, iterate, rhs, controls);
            const bool solved = norm == KrylovPhysicalNorm::component_linf && maximum == 5;
            EXPECT_EQ(report.solved(), solved) << report.reason;
            EXPECT_EQ(report.iters, maximum);
            EXPECT_EQ(workspace.allocation_count(), allocations);
            ASSERT_EQ(trace.size, static_cast<std::size_t>(maximum));
            for (std::size_t cycle = 0; cycle < trace.size; ++cycle)
              EXPECT_EQ(trace.cycles[cycle].dimension, 1);
            Real residual = Real(0);
            for (std::size_t local = 0; local < iterate.local_size(); ++local) {
              auto values = iterate.fab(local).create_host_mirror();
              iterate.fab(local).copy_to_host(values);
              const bool active = iterate.box(local).lo[0] < active_blocks;
              if (solved) {
                EXPECT_EQ(values(0), base);
                EXPECT_EQ(values(1), active ? base - ulp : base);
                EXPECT_EQ(values(2), base);
              }
              const Real a = values(0) - base, b = values(1) - base, c = values(2) - base;
              const Real amplitude = active ? ulp : Real(0);
              residual = std::max(residual, std::abs(amplitude - (Real(2) * a - b)));
              residual = std::max(
                  residual, std::abs(-Real(11) / Real(8) * amplitude - (-a + Real(2) * b - c)));
              residual = std::max(residual, std::abs(amplitude - (-b + Real(2) * c)));
            }
            residual = all_reduce_max(residual);
            if (solved)
              EXPECT_LE(residual, tolerance);
            else
              EXPECT_GT(residual, tolerance);
            if (norm == KrylovPhysicalNorm::component_linf)
              EXPECT_EQ(report.residual_norm, residual);
          }
        }
        if (active_blocks == 1 && restart == 1 && scale == Real(1)) {
          reset();
          GmresDiagnosticTrace fault_trace;
          active_trace = &fault_trace;
          fail_after_coordinated_update = true;
          TestKrylovControls controls{method, Real(0), tolerance, 6};
          controls.physical_norm = KrylovPhysicalNorm::component_linf;
          controls.diagnostic_trace = &fault_trace;
          const auto rejected =
              detail::solve_prepared_affine_in_place(problem, workspace, iterate, rhs, controls);
          EXPECT_EQ(rejected.status, SolveStatus::kInvalidEvaluation);
          EXPECT_EQ(rejected.action, SolveAction::kFailRun);
          EXPECT_EQ(fault_trace.size, std::size_t{3});
          EXPECT_EQ(workspace.allocation_count(), allocations);
          fail_after_coordinated_update = false;
          active_trace = nullptr;
          reset();
          controls.max_iterations = 5;
          controls.diagnostic_trace = nullptr;
          const auto retry =
              detail::solve_prepared_affine_in_place(problem, workspace, iterate, rhs, controls);
          EXPECT_TRUE(retry.solved()) << retry.reason;
          EXPECT_EQ(retry.iters, 5);
          EXPECT_LE(retry.residual_norm, tolerance);
          EXPECT_EQ(workspace.allocation_count(), allocations);
        }
      }
    }
  }
}

TEST(test_krylov_workspace_reentrancy, gmres_rejects_unrepresentable_permuted_equations) {
  comm_init();
  const TestLayout boxes(std::vector<TestBox>{TestBox{Index<kDim>{0, 0}, Index<kDim>{0, 0}}});
  const auto mapping = TestDistribution::replicated(boxes, world_rank_space());
  const auto vectors = TestVectorDistribution::replicated();
  const Real base = Real(1.5);
  const Real ulp = std::nextafter(base, Real(2)) - base;
  const Real tolerance = ulp / Real(4);
  TestField iterate = make_field(boxes, mapping, 2, 0);
  TestField rhs = make_field(boxes, mapping, 2, 0);
  const auto reset = [&] {
    for (std::size_t local = 0; local < iterate.local_size(); ++local) {
      const auto x = iterate.fab(local).view();
      const auto right = rhs.fab(local).view();
      for_each_cell(iterate.box(local), [=] POPS_HD(const Index<kDim>& index) {
        x(index, 0) = base;
        x(index, 1) = base;
        right(index, 0) = Real(3) / Real(8) * ulp;
        right(index, 1) = ulp / Real(8);
      });
    }
    Kokkos::fence();
  };
  reset();
  const auto snapshot = test_snapshot(iterate);
  const auto method = gmres_krylov_method<kDim>(3);
  const TestKrylovFootprint footprint{2, extent(0), true};
  // A permutes the unknowns: an equation maximum does not identify its own
  // variable. No representable x_1 can make |3/8 ulp-(x_1-1.5)| <= 1/4 ulp.
  // A recovery proposal must therefore remain a refusal under every cap.
  auto preconditioner = TestLinearPreconditionerProvider::trusted_extension(
      {"pops.test.krylov.permuted-rounding-inverse", 1}, {}, [=](const ExecutionLane&) {
        return TestLinearPreconditionerCallbacks{
            [] {},
            [](TestField& out, const TestField& in) {
              for (std::size_t local = 0; local < out.local_size(); ++local) {
                const auto output = out.fab(local).view();
                const auto input = in.fab(local).view();
                for_each_cell(out.box(local), [=] POPS_HD(const Index<kDim>& index) {
                  output(index, 0) = input(index, 1);
                  output(index, 1) = input(index, 0);
                });
              }
              Kokkos::fence();
            },
            [] { return std::size_t{0}; }};
      });
  TestAffineProblem problem(
      iterate,
      TestAffineOperatorProvider::trusted_reentrant(
          [=](TestField& out, const TestField& in) {
            for (std::size_t local = 0; local < out.local_size(); ++local) {
              const auto output = out.fab(local).view();
              const auto input = in.fab(local).view();
              for_each_cell(out.box(local), [=] POPS_HD(const Index<kDim>& index) {
                output(index, 0) = input(index, 1) - base;
                output(index, 1) = input(index, 0) - base;
              });
            }
            Kokkos::fence();
          },
          [] { return std::size_t{0}; }),
      TestLinearPreconditioner(iterate, std::move(preconditioner), vectors),
      LinearOperatorProperties::general(), footprint, TestNullspacePolicy::nonsingular(),
      [&snapshot] { return snapshot; }, {}, vectors);
  TestKrylovWorkspace workspace(iterate, method, footprint, vectors);
  problem.prepare(snapshot);
  workspace.bind(problem);
  const auto allocations = workspace.allocation_count();
  for (const int maximum : {2, 4, 2}) {
    SCOPED_TRACE(maximum);
    reset();
    TestKrylovControls controls{method, Real(0), tolerance, maximum};
    controls.physical_norm = KrylovPhysicalNorm::component_linf;
    const auto report =
        detail::solve_prepared_affine_in_place(problem, workspace, iterate, rhs, controls);
    EXPECT_FALSE(report.solved());
    EXPECT_EQ(report.status, SolveStatus::kIterationLimit);
    EXPECT_EQ(report.iters, maximum);
    EXPECT_EQ(workspace.allocation_count(), allocations);
    Real residual = Real(0);
    for (std::size_t local = 0; local < iterate.local_size(); ++local) {
      auto values = iterate.fab(local).create_host_mirror();
      iterate.fab(local).copy_to_host(values);
      residual = std::max(residual, std::abs(Real(3) / Real(8) * ulp - (values(1) - base)));
      residual = std::max(residual, std::abs(ulp / Real(8) - (values(0) - base)));
    }
    residual = all_reduce_max(residual);
    EXPECT_EQ(report.residual_norm, residual);
    EXPECT_GE(residual, Real(3) / Real(8) * ulp);
    EXPECT_GT(residual, tolerance);
  }
}
}  // namespace
}  // namespace pops
