#include <gtest/gtest.h>

#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/numerics/elliptic/linear/generic_krylov.hpp>
#include <pops/numerics/elliptic/poisson/poisson_operator.hpp>

namespace pops {
namespace {

OperatorEvaluationSnapshot test_snapshot(const MultiFab& prototype) {
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

void fill_rhs(MultiFab& field, Real sign) {
  field.sync_host();
  for (int li = 0; li < field.local_size(); ++li) {
    Array4 values = field.fab(li).array();
    const Box2D valid = field.box(li);
    for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
      for (int i = valid.lo[0]; i <= valid.hi[0]; ++i)
        values(i, j) = sign * (Real(1) + Real(0.125) * Real(i) + Real(0.25) * Real(j));
  }
}

Real max_abs_diff(const MultiFab& left, const MultiFab& right) {
  Real local = Real(0);
  left.sync_host();
  right.sync_host();
  for (int li = 0; li < left.local_size(); ++li) {
    const ConstArray4 left_values = left.fab(li).const_array();
    const ConstArray4 right_values = right.fab(li).const_array();
    const Box2D valid = left.box(li);
    for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
      for (int i = valid.lo[0]; i <= valid.hi[0]; ++i)
        local = std::fmax(local, std::fabs(left_values(i, j) - right_values(i, j)));
  }
  return static_cast<Real>(all_reduce_max(static_cast<double>(local)));
}

bool has_remote_face_neighbor(const BoxArray& boxes, const DistributionMapping& mapping, int rank) {
  for (int i = 0; i < boxes.size(); ++i) {
    if (mapping[i] != rank)
      continue;
    for (int j = 0; j < boxes.size(); ++j) {
      if (mapping[j] == rank)
        continue;
      const Box2D& left = boxes[i];
      const Box2D& right = boxes[j];
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

  void apply(MultiFab& out, const MultiFab& in) {
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
  Geometry geometry;
  BCRec boundary;
  const ExecutionLane* lane = nullptr;
  MultiFab scratch;

  ConcurrentOperatorSessionState(std::shared_ptr<ConcurrentSessionProbe> session_probe, int id,
                                 Geometry session_geometry, BCRec session_boundary,
                                 const ExecutionLane& session_lane, const BoxArray& boxes,
                                 const DistributionMapping& mapping)
      : probe(std::move(session_probe)),
        session_id(id),
        geometry(session_geometry),
        boundary(session_boundary),
        lane(&session_lane),
        scratch(boxes, mapping, 1, 1) {}

  void apply(MultiFab& out, const MultiFab& in) {
    if (probe->armed.load(std::memory_order_acquire) &&
        first_solve_apply.exchange(false, std::memory_order_acq_rel))
      probe->rendezvous(session_id, &scratch);
    MultiFab& mutable_input = const_cast<MultiFab&>(in);
    fill_ghosts(mutable_input, geometry.domain, boundary, *lane);
    apply_laplacian(in, geometry, scratch);
    detail::PreparedFieldAlgebra::lincomb(out, Real(1), in, -kHelmholtzAlpha, scratch);
  }

  [[nodiscard]] std::size_t allocation_count() const noexcept { return 1u; }
};

class LongReasonKrylovProvider final : public PreparedKrylovMethodProvider {
 public:
  static constexpr std::size_t kReasonBytes = 16 * 1024 + 37;

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
  KrylovMethodValidation validate_problem(const KrylovMethodProblemFacts& facts,
                                          const PreparedProviderOptions&) const noexcept override {
    return facts.has_preconditioner
               ? KrylovMethodValidation::reject(1, "long-reason probe is unpreconditioned")
               : KrylovMethodValidation::accept();
  }
  KrylovWorkspaceRequirements workspace_requirements(
      const KrylovWorkspaceRequest&, const PreparedProviderOptions&) const override {
    return {.field_count = 2, .initial_residual_field = 1};
  }
  SolveReport solve(PreparedKrylovSolveContext& context,
                    const PreparedProviderOptions&) const override {
    SolveReport report =
        context.report(context.initial_physical_residual(), 1, SolveStatus::kIterationLimit);
    report.reason.assign(kReasonBytes, 'r');
    return report;
  }
};

class SingleFieldKrylovProvider final : public PreparedKrylovMethodProvider {
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
  KrylovMethodValidation validate_problem(const KrylovMethodProblemFacts&,
                                          const PreparedProviderOptions&) const noexcept override {
    return KrylovMethodValidation::accept();
  }
  KrylovWorkspaceRequirements workspace_requirements(
      const KrylovWorkspaceRequest&, const PreparedProviderOptions&) const override {
    return {.field_count = 1, .initial_residual_field = 0};
  }
  SolveReport solve(PreparedKrylovSolveContext&, const PreparedProviderOptions&) const override {
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
  void apply(MultiFab& out, const MultiFab& in) {
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

PreparedAffineOperatorProvider blocking_prepare_operator_provider(
    const std::shared_ptr<BlockingPrepareGate>& gate) {
  return PreparedAffineOperatorProvider::trusted_extension(
      {"pops.test.krylov.blocking-prepare-operator", 1}, {}, [gate](const ExecutionLane&) {
        return PreparedAffineOperatorSessionCallbacks{
            [gate] { gate->block_if_armed(); },
            [](MultiFab& out, const MultiFab& in) { detail::PreparedFieldAlgebra::copy(out, in); },
            [] { return std::size_t{0}; }};
      });
}

struct PotentiallyThrowingSessionContract {
  void prepare() {}
  PreparedApplyStatus apply(MultiFab&, const MultiFab&) { return PreparedApplyStatus::Success; }
  [[nodiscard]] std::size_t allocation_count() const { return 0; }
};

struct NothrowSessionContract {
  void prepare() {}
  PreparedApplyStatus apply(MultiFab&, const MultiFab&) noexcept {
    return PreparedApplyStatus::Success;
  }
  [[nodiscard]] std::size_t allocation_count() const { return 0; }
};

static_assert(!PreparedAffineOperatorSessionSource<PotentiallyThrowingSessionContract>);
static_assert(!PreparedLinearPreconditionerSessionSource<PotentiallyThrowingSessionContract>);
static_assert(PreparedAffineOperatorSessionSource<NothrowSessionContract>);
static_assert(PreparedLinearPreconditionerSessionSource<NothrowSessionContract>);

PreparedAffineOperatorProvider lifecycle_operator_provider(
    const std::shared_ptr<SessionLifecycleProbe>& probe) {
  return PreparedAffineOperatorProvider::trusted_extension(
      {"pops.test.krylov.lifecycle-operator", 1}, {}, [probe](const ExecutionLane&) {
        probe->factories.fetch_add(1, std::memory_order_relaxed);
        auto state = std::make_shared<SessionLifecycleState>(probe);
        return PreparedAffineOperatorSessionCallbacks{
            [state] { state->prepare(); },
            [state](MultiFab& out, const MultiFab& in) { state->apply(out, in); },
            [state] { return state->allocation_count(); }};
      });
}

PreparedLinearPreconditionerProvider lifecycle_preconditioner_provider(
    const std::shared_ptr<SessionLifecycleProbe>& probe) {
  return PreparedLinearPreconditionerProvider::trusted_extension(
      {"pops.test.krylov.lifecycle-preconditioner", 1}, {}, [probe](const ExecutionLane&) {
        probe->factories.fetch_add(1, std::memory_order_relaxed);
        auto state = std::make_shared<SessionLifecycleState>(probe);
        return PreparedLinearPreconditionerSessionCallbacks{
            [state] { state->prepare(); },
            [state](MultiFab& out, const MultiFab& in) { state->apply(out, in); },
            [state] { return state->allocation_count(); }};
      });
}

TEST(test_krylov_workspace_reentrancy,
     provider_workspace_with_only_one_core_field_is_rejected_before_materialization) {
  const PreparedKrylovMethod method(
      std::make_shared<SingleFieldKrylovProvider>(),
      PreparedProviderOptions{"pops.test.krylov.single-field-workspace.options@1", {}});
  EXPECT_THROW((void)method.workspace_requirements(KrylovWorkspaceRequest{
                   KrylovFootprint{1, 0, false}, PreparedVectorDistribution::Distributed,
                   detail::PreparedFieldAlgebra::kRobustDotPayloadWidth}),
               std::invalid_argument);
}

TEST(test_krylov_workspace_reentrancy,
     trusted_prepare_exceptions_propagate_but_hot_apply_exceptions_publish_failure_status) {
  const BoxArray boxes(std::vector<Box2D>{Box2D{{0, 0}, {1, 1}}});
  const DistributionMapping mapping(boxes.size(), n_ranks());
  MultiFab input(boxes, mapping, 1, 0);
  MultiFab output(boxes, mapping, 1, 0);
  input.set_val(Real(1));
  output.set_val(Real(7));
  const ExecutionLane lane = ExecutionLane::world("pops.test.krylov.trusted-failure");

  PreparedAffineOperatorProvider provider = PreparedAffineOperatorProvider::trusted_extension(
      {"pops.test.krylov.throwing-trusted-operator", 1}, {}, [](const ExecutionLane&) {
        return PreparedAffineOperatorSessionCallbacks{
            [] { throw std::runtime_error("prepare failure"); },
            [](MultiFab& out, const MultiFab& in) { detail::PreparedFieldAlgebra::copy(out, in); },
            [] { return std::size_t{0}; }};
      });
  PreparedAffineOperatorSession session = provider.make_session(lane);

  EXPECT_THROW(session.prepare(), std::runtime_error);

  output.set_val(Real(9));
  PreparedLinearPreconditionerProvider preconditioner_provider =
      PreparedLinearPreconditionerProvider::trusted_extension(
          {"pops.test.krylov.throwing-trusted-preconditioner", 1}, {}, [](const ExecutionLane&) {
            return PreparedLinearPreconditionerSessionCallbacks{
                {},
                [](MultiFab&, const MultiFab&) { throw std::runtime_error("apply failure"); },
                [] { return std::size_t{0}; }};
          });
  PreparedLinearPreconditionerSession preconditioner_session =
      preconditioner_provider.make_session(lane);
  EXPECT_NO_THROW(preconditioner_session.prepare());
  EXPECT_EQ(preconditioner_session.apply(output, input), PreparedApplyStatus::Failure);
  EXPECT_EQ(preconditioner_session.allocation_count(), 0u);
  output.sync_host();
  for (int local = 0; local < output.local_size(); ++local) {
    const ConstArray4 values = output.fab(local).const_array();
    const Box2D valid = output.box(local);
    for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
      for (int i = valid.lo[0]; i <= valid.hi[0]; ++i)
        EXPECT_TRUE(std::isnan(values(i, j)));
  }
}

TEST(test_krylov_workspace_reentrancy,
     repeated_prepare_and_bind_refresh_sessions_without_rematerializing_their_source) {
  const Box2D domain{{0, 0}, {3, 3}};
  const BoxArray boxes = BoxArray::from_domain(domain, 2);
  const DistributionMapping mapping(boxes.size(), n_ranks());
  MultiFab prototype(boxes, mapping, 1, 0);
  prototype.set_val(Real(0));
  OperatorEvaluationSnapshot snapshot = test_snapshot(prototype);
  const auto operator_probe = std::make_shared<SessionLifecycleProbe>();
  const auto preconditioner_probe = std::make_shared<SessionLifecycleProbe>();
  PreparedAffineOperatorProvider shared_operator = lifecycle_operator_provider(operator_probe);
  PreparedLinearPreconditionerProvider shared_preconditioner =
      lifecycle_preconditioner_provider(preconditioner_probe);
  const KrylovFootprint footprint{1, 0, true};

  PreparedAffineLinearProblem first_problem(
      prototype, shared_operator, PreparedLinearPreconditioner(prototype, shared_preconditioner),
      LinearOperatorProperties::general(), footprint, PreparedNullspacePolicy::nonsingular(),
      [&snapshot] { return snapshot; });
  KrylovWorkspace workspace(prototype, bicgstab_krylov_method(), footprint);

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
  PreparedAffineLinearProblem same_source_problem(
      prototype, shared_operator, PreparedLinearPreconditioner(prototype, shared_preconditioner),
      LinearOperatorProperties::general(), footprint, PreparedNullspacePolicy::nonsingular(),
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
  PreparedAffineLinearProblem different_source_problem(
      prototype, lifecycle_operator_provider(operator_probe),
      PreparedLinearPreconditioner(prototype,
                                   lifecycle_preconditioner_provider(preconditioner_probe)),
      LinearOperatorProperties::general(), footprint, PreparedNullspacePolicy::nonsingular(),
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
  const BoxArray boxes(std::vector<Box2D>{Box2D{{0, 0}, {3, 3}}});
  const DistributionMapping mapping(boxes.size(), n_ranks());
  MultiFab prototype(boxes, mapping, 1, 0);
  prototype.set_val(Real(0));
  OperatorEvaluationSnapshot snapshot = test_snapshot(prototype);
  const auto gate = std::make_shared<BlockingPrepareGate>();
  const KrylovFootprint footprint{1, 0, false};
  PreparedAffineLinearProblem problem(
      prototype, blocking_prepare_operator_provider(gate), PreparedLinearPreconditioner::identity(),
      LinearOperatorProperties::symmetric_positive_definite(), footprint,
      PreparedNullspacePolicy::nonsingular(), [&snapshot] { return snapshot; });
  const PreparedKrylovMethod method = cg_krylov_method();
  KrylovWorkspace workspace(prototype, method, footprint);
  MultiFab iterate(boxes, mapping, 1, 0);
  MultiFab rhs(boxes, mapping, 1, 0);
  iterate.set_val(Real(0));
  rhs.set_val(Real(1));
  const KrylovControls controls{method, Real(1e-12), Real(0), 4};
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
      (void)prepare_krylov_solve(problem, workspace, iterate, rhs, controls);
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
  const BoxArray boxes(std::vector<Box2D>{Box2D{{0, 0}, {3, 3}}});
  const DistributionMapping mapping(boxes.size(), n_ranks());
  MultiFab prototype(boxes, mapping, 1, 0);
  prototype.set_val(Real(0));
  OperatorEvaluationSnapshot snapshot = test_snapshot(prototype);
  const auto gate = std::make_shared<BlockingPrepareGate>();
  const KrylovFootprint footprint{1, 0, false};
  PreparedAffineLinearProblem problem(
      prototype,
      PreparedAffineOperatorProvider::trusted_reentrant(
          [](MultiFab& out, const MultiFab& in) { detail::PreparedFieldAlgebra::copy(out, in); },
          [] { return std::size_t{0}; }),
      PreparedLinearPreconditioner::identity(),
      LinearOperatorProperties::symmetric_positive_definite(), footprint,
      PreparedNullspacePolicy::nonsingular(), [&snapshot] { return snapshot; },
      [gate] { gate->block_if_armed(); });
  const PreparedKrylovMethod method = cg_krylov_method();
  KrylovWorkspace workspace(prototype, method, footprint);
  MultiFab iterate(boxes, mapping, 1, 0);
  MultiFab rhs(boxes, mapping, 1, 0);
  iterate.set_val(Real(0));
  rhs.set_val(Real(1));
  const KrylovControls controls{method, Real(1e-12), Real(0), 4};
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
      (void)prepare_krylov_solve(problem, workspace, iterate, rhs, controls);
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

  const BoxArray boxes(std::vector<Box2D>{Box2D{{0, 0}, {1, 1}}});
  const DistributionMapping mapping(boxes.size(), n_ranks());
  MultiFab prototype(boxes, mapping, 1, 0);
  prototype.set_val(Real(0));
  OperatorEvaluationSnapshot snapshot = test_snapshot(prototype);
  const LinearOperatorProperties properties = my_rank() == 0
                                                  ? LinearOperatorProperties{operator_property_bit(
                                                        LinearOperatorProperty::kPositiveDefinite)}
                                                  : LinearOperatorProperties::general();

  std::string rejection;
  bool invalid_argument = false;
  try {
    (void)PreparedAffineLinearProblem(
        prototype,
        PreparedAffineOperatorProvider::trusted_reentrant(
            [](MultiFab& out, const MultiFab& in) { detail::PreparedFieldAlgebra::copy(out, in); },
            [] { return std::size_t{0}; }),
        PreparedLinearPreconditioner::identity(), properties, KrylovFootprint{1, 0, false},
        PreparedNullspacePolicy::nonsingular(), [&snapshot] { return snapshot; });
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

TEST(test_krylov_workspace_reentrancy,
     distinct_workspaces_run_fresh_operator_and_preconditioner_sessions_concurrently) {
  comm_init();
  const char* expected_ranks = std::getenv("POPS_TEST_EXPECT_RANKS");
  if (expected_ranks != nullptr)
    ASSERT_EQ(n_ranks(), std::atoi(expected_ranks));

  const Box2D domain{{0, 0}, {7, 7}};
  const BoxArray boxes = BoxArray::from_domain(domain, 2);
  const DistributionMapping mapping(boxes.size(), n_ranks());
  const Geometry geometry{domain, Real(0), Real(1), Real(0), Real(1)};
  const BCRec periodic{};
  MultiFab prototype(boxes, mapping, 1, 1);
  prototype.set_val(Real(0));
  EXPECT_GT(prototype.local_size(), 0);
  if (n_ranks() > 1)
    EXPECT_TRUE(has_remote_face_neighbor(boxes, mapping, my_rank()))
        << "the MPI variant must exchange a real inter-rank face halo";
  OperatorEvaluationSnapshot snapshot = test_snapshot(prototype);
  const auto operator_probe = std::make_shared<ConcurrentSessionProbe>();
  const auto preconditioner_probe = std::make_shared<ConcurrentSessionProbe>();

  PreparedAffineOperatorProvider operator_provider =
      PreparedAffineOperatorProvider::trusted_extension(
          {"pops.test.krylov.concurrent-operator", 1}, {},
          [operator_probe, boxes, mapping, geometry, periodic](const ExecutionLane& lane) {
            auto state = std::make_shared<ConcurrentOperatorSessionState>(
                operator_probe,
                operator_probe->next_session.fetch_add(1, std::memory_order_relaxed), geometry,
                periodic, lane, boxes, mapping);
            return PreparedAffineOperatorSessionCallbacks{
                [] {}, [state](MultiFab& out, const MultiFab& in) { state->apply(out, in); },
                [state] { return state->allocation_count(); }};
          });

  PreparedLinearPreconditionerProvider provider =
      PreparedLinearPreconditionerProvider::trusted_extension(
          {"pops.test.krylov.concurrent-preconditioner", 1}, {},
          [preconditioner_probe](const ExecutionLane&) {
            auto state = std::make_shared<ConcurrentSessionState>();
            state->probe = preconditioner_probe;
            state->session_id =
                preconditioner_probe->next_session.fetch_add(1, std::memory_order_relaxed);
            return PreparedLinearPreconditionerSessionCallbacks{
                [] {}, [state](MultiFab& out, const MultiFab& in) { state->apply(out, in); },
                [state] { return state->allocation_count(); }};
          });
  const KrylovFootprint footprint{1, 1, true};
  PreparedAffineLinearProblem problem(prototype, std::move(operator_provider),
                                      PreparedLinearPreconditioner(prototype, std::move(provider)),
                                      LinearOperatorProperties::general(), footprint,
                                      PreparedNullspacePolicy::nonsingular(),
                                      [&snapshot] { return snapshot; });
  const PreparedKrylovMethod method = bicgstab_krylov_method();
  KrylovWorkspace oracle_left_workspace(prototype, method, footprint);
  KrylovWorkspace oracle_right_workspace(prototype, method, footprint);
  KrylovWorkspace left_workspace(prototype, method, footprint);
  KrylovWorkspace right_workspace(prototype, method, footprint);
  problem.prepare(snapshot);
  oracle_left_workspace.bind(problem);
  oracle_right_workspace.bind(problem);
  left_workspace.bind(problem);
  right_workspace.bind(problem);

  MultiFab oracle_left_iterate(boxes, mapping, 1, 1);
  MultiFab oracle_right_iterate(boxes, mapping, 1, 1);
  MultiFab oracle_left_rhs(boxes, mapping, 1, 1);
  MultiFab oracle_right_rhs(boxes, mapping, 1, 1);
  MultiFab left_iterate(boxes, mapping, 1, 1);
  MultiFab right_iterate(boxes, mapping, 1, 1);
  MultiFab left_rhs(boxes, mapping, 1, 1);
  MultiFab right_rhs(boxes, mapping, 1, 1);
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
  const KrylovControls controls{method, Real(1e-12), Real(0), 100};

  const SolveReport oracle_left_report = solve_prepared_affine(
      problem, oracle_left_workspace, oracle_left_iterate, oracle_left_rhs, controls);
  const SolveReport oracle_right_report = solve_prepared_affine(
      problem, oracle_right_workspace, oracle_right_iterate, oracle_right_rhs, controls);
  ASSERT_TRUE(oracle_left_report.solved()) << oracle_left_report.reason;
  ASSERT_TRUE(oracle_right_report.solved()) << oracle_right_report.reason;

  // Every rank materializes the independent invocation communicators in the same control order.
  // The worker entry order is then deliberately reversed on odd ranks. If both solves accidentally
  // share WORLD (or any one collective trace), this opposite interleaving deadlocks or mismatches.
  PreparedKrylovInvocation left_invocation =
      prepare_krylov_solve(problem, left_workspace, left_iterate, left_rhs, controls);
  PreparedKrylovInvocation right_invocation =
      prepare_krylov_solve(problem, right_workspace, right_iterate, right_rhs, controls);
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
  const BoxArray boxes(std::vector<Box2D>{Box2D{{0, 0}, {1, 1}}});
  const DistributionMapping mapping(boxes.size(), n_ranks());
  MultiFab iterate(boxes, mapping, 1, 0);
  MultiFab rhs(boxes, mapping, 1, 0);
  iterate.set_val(Real(0));
  rhs.set_val(Real(1));
  OperatorEvaluationSnapshot snapshot = test_snapshot(iterate);
  const auto provider = std::make_shared<LongReasonKrylovProvider>();
  const PreparedKrylovMethod method(
      provider, PreparedProviderOptions{"pops.test.krylov.long-report-reason.options@1", {}});
  const KrylovFootprint footprint{1, 0, false};
  PreparedAffineLinearProblem problem(
      iterate,
      PreparedAffineOperatorProvider::trusted_reentrant(
          [](MultiFab& out, const MultiFab& in) { detail::PreparedFieldAlgebra::copy(out, in); },
          [] { return std::size_t{0}; }),
      PreparedLinearPreconditioner::identity(), LinearOperatorProperties::general(), footprint,
      PreparedNullspacePolicy::nonsingular(), [&snapshot] { return snapshot; });
  KrylovWorkspace workspace(iterate, method, footprint);
  problem.prepare(snapshot);
  workspace.bind(problem);

  const SolveReport report = solve_prepared_affine(problem, workspace, iterate, rhs,
                                                   KrylovControls{method, Real(0), Real(0), 1});

  EXPECT_EQ(report.status, SolveStatus::kIterationLimit);
  EXPECT_EQ(report.reason.size(), LongReasonKrylovProvider::kReasonBytes);
  EXPECT_EQ(report.reason, std::string(LongReasonKrylovProvider::kReasonBytes, 'r'));
}

TEST(test_krylov_workspace_reentrancy,
     exclusive_operator_rejects_overlapping_materialized_invocations_uniformly) {
  comm_init();
  const Box2D domain{{0, 0}, {3, 3}};
  const BoxArray boxes = BoxArray::from_domain(domain, 2);
  const DistributionMapping mapping(boxes.size(), n_ranks());
  MultiFab prototype(boxes, mapping, 1, 0);
  prototype.set_val(Real(0));
  OperatorEvaluationSnapshot snapshot = test_snapshot(prototype);
  PreparedAffineOperatorProvider operator_provider =
      PreparedAffineOperatorProvider::trusted_extension(
          {"pops.test.krylov.exclusive-operator", 1}, {},
          [](const ExecutionLane&) {
            return PreparedAffineOperatorSessionCallbacks{{},
                                                          [](MultiFab& out, const MultiFab& in) {
                                                            detail::PreparedFieldAlgebra::copy(out,
                                                                                               in);
                                                          },
                                                          [] { return std::size_t{0}; }};
          },
          PreparedOperatorConcurrency::Exclusive);
  const KrylovFootprint footprint{1, 0, false};
  PreparedAffineLinearProblem problem(
      prototype, std::move(operator_provider), PreparedLinearPreconditioner::identity(),
      LinearOperatorProperties::symmetric_positive_definite(), footprint,
      PreparedNullspacePolicy::nonsingular(), [&snapshot] { return snapshot; });
  const PreparedKrylovMethod method = cg_krylov_method();
  KrylovWorkspace first_workspace(prototype, method, footprint);
  KrylovWorkspace second_workspace(prototype, method, footprint);
  problem.prepare(snapshot);
  first_workspace.bind(problem);
  second_workspace.bind(problem);

  MultiFab first_iterate(boxes, mapping, 1, 0);
  MultiFab second_iterate(boxes, mapping, 1, 0);
  MultiFab first_rhs(boxes, mapping, 1, 0);
  MultiFab second_rhs(boxes, mapping, 1, 0);
  first_iterate.set_val(Real(0));
  second_iterate.set_val(Real(0));
  first_rhs.set_val(Real(2));
  second_rhs.set_val(Real(-3));
  const KrylovControls controls{method, Real(1e-12), Real(0), 4};

  {
    PreparedKrylovInvocation first =
        prepare_krylov_solve(problem, first_workspace, first_iterate, first_rhs, controls);
    std::string rejection;
    try {
      (void)prepare_krylov_solve(problem, second_workspace, second_iterate, second_rhs, controls);
    } catch (const std::logic_error& error) {
      rejection = error.what();
    }
    EXPECT_EQ(rejection,
              "prepared affine problem is being mutated or its operator requires exclusive "
              "access to its external execution context");
    const SolveReport first_report = first.execute();
    EXPECT_TRUE(first_report.solved()) << first_report.reason;
  }

  PreparedKrylovInvocation second =
      prepare_krylov_solve(problem, second_workspace, second_iterate, second_rhs, controls);
  const SolveReport second_report = second.execute();
  EXPECT_TRUE(second_report.solved()) << second_report.reason;
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
    const Box2D domain{{0, 0}, {7, 7}};
    const BoxArray boxes = BoxArray::from_domain(domain, 2);
    const DistributionMapping mapping(boxes.size(), n_ranks());
    const Geometry geometry{domain, Real(0), Real(1), Real(0), Real(1)};
    const BCRec periodic{};
    MultiFab prototype(boxes, mapping, 1, 1);
    prototype.set_val(Real(0));
    ASSERT_GT(prototype.local_size(), 0);
    ASSERT_TRUE(has_remote_face_neighbor(boxes, mapping, my_rank()));
    OperatorEvaluationSnapshot snapshot = test_snapshot(prototype);
    const auto probe = std::make_shared<ConcurrentSessionProbe>();

    PreparedAffineOperatorProvider operator_provider =
        PreparedAffineOperatorProvider::trusted_extension(
            {"pops.test.krylov.embedding-communicator-operator", 1}, {},
            [probe, boxes, mapping, geometry, periodic](const ExecutionLane& lane) {
              auto state = std::make_shared<ConcurrentOperatorSessionState>(
                  probe, probe->next_session.fetch_add(1, std::memory_order_relaxed), geometry,
                  periodic, lane, boxes, mapping);
              return PreparedAffineOperatorSessionCallbacks{
                  [] {}, [state](MultiFab& out, const MultiFab& in) { state->apply(out, in); },
                  [state] { return state->allocation_count(); }};
            });

    const KrylovFootprint footprint{1, 1, false};
    PreparedAffineLinearProblem problem(
        parent, "pops.test.krylov.embedding-communicator-problem", prototype,
        std::move(operator_provider), PreparedLinearPreconditioner::identity(),
        LinearOperatorProperties::symmetric_positive_definite(), footprint,
        PreparedNullspacePolicy::nonsingular(), [&snapshot] { return snapshot; });
    const PreparedKrylovMethod method = cg_krylov_method();
    KrylovWorkspace workspace(parent, "pops.test.krylov.embedding-communicator-workspace",
                              prototype, method, footprint);
    problem.prepare(snapshot);
    workspace.bind(problem);

    MultiFab iterate(boxes, mapping, 1, 1);
    MultiFab rhs(boxes, mapping, 1, 1);
    iterate.set_val(Real(0));
    rhs.set_val(Real(0));
    fill_rhs(rhs, Real(1));
    const SolveReport report = solve_prepared_affine(
        problem, workspace, iterate, rhs, KrylovControls{method, Real(1e-12), Real(0), 100});
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

}  // namespace
}  // namespace pops
