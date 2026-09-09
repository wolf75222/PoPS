#include <gtest/gtest.h>
#include <pops/runtime/program/prepared_amr_spatial_residual.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/runtime/program/amr_program_context.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/physics/bricks/bricks.hpp>
#include <pops/physics/fluids/euler.hpp>
#include "amr_tagging_test_authority.hpp"
#include "explicit_amr_program.hpp"

namespace {
class CommEnvironment final : public ::testing::Environment {
 public:
  void SetUp() override { pops::comm_init(); }
  void TearDown() override { pops::comm_finalize(); }
};
[[maybe_unused]] const auto* environment = ::testing::AddGlobalTestEnvironment(new CommEnvironment);
using namespace pops;

MultiFab<2> field(bool fine, bool replicated) {
  const Box<2> domain =
      fine ? Box<2>{Index<2>{8, 8}, Index<2>{23, 23}} : Box<2>{Index<2>{0, 0}, Index<2>{15, 15}};
  const auto boxes = mesh::BoxArray<2>::from_domain(domain, Extent<2>{8, 8});
  const mesh::RankSpace<2> ranks{Index<2>{}, Extent<2>{n_ranks(), 1}};
  std::vector<Index<2>> owners;
  for (std::size_t patch = 0; patch < boxes.size(); ++patch)
    owners.push_back(Index<2>{static_cast<int>(patch % n_ranks()), 0});
  const auto distribution = replicated ? mesh::Distribution<2>::replicated(boxes, ranks)
                                       : mesh::Distribution<2>::partitioned(boxes, ranks, owners);
  return MultiFab<2>(boxes, distribution, Index<2>{my_rank(), 0}, 1, Extent<2>{});
}
}  // namespace

TEST(AmrSpatialNorm, MixedOwnershipCountsOnlyThePhysicalActiveCover) {
  using namespace pops;
  const auto lane = ExecutionLane::world("test.amr-spatial.active-cover");
  for (const auto ownership :
       {std::array<bool, 2>{false, false}, {true, false}, {false, true}, {true, true}}) {
    auto coarse = field(false, ownership[0]), fine = field(true, ownership[1]);
    auto coarse_mask = field(false, ownership[0]), fine_mask = field(true, ownership[1]);
    coarse.set_val(0);
    fine.set_val(0);
    fine_mask.set_val(1);
    for (std::size_t local = 0; local < coarse_mask.local_size(); ++local) {
      const auto view = coarse_mask.fab(local).view();
      for_each_cell(coarse_mask.box(local), [=] POPS_HD(const Index<2>& cell) {
        view(cell, 0) =
            cell[0] >= 4 && cell[0] <= 11 && cell[1] >= 4 && cell[1] <= 11 ? Real(0) : Real(1);
      });
    }
    const std::array<const MultiFab<2>*, 2> layouts{&coarse, &fine},
        masks{&coarse_mask, &fine_mask};
    const std::array<Real, 2> measures{Real(1) / 256, Real(1) / 1024};
    FieldNewtonOptions options;
    options.tolerance = Real(2);
    AmrFieldNewtonKrylovWorkspace<2> workspace(layouts, masks, measures, options);
    const std::array<MultiFab<2>*, 2> destinations{&coarse, &fine};
    auto residual = [](const auto& q, auto& output, int) {
      for (std::size_t level = 0; level < q.size(); ++level) {
        output[level].set_val(level == 0 ? Real(1) : Real(3));
        saxpy(output[level], Real(-1), q[level]);
      }
    };
    auto derivative = [](const auto&, const auto& direction, auto& output, int) {
      for (std::size_t level = 0; level < output.size(); ++level)
        lincomb(output[level], Real(1), direction[level], Real(0), direction[level]);
    };
    const auto report = workspace.solve(destinations, residual, derivative, [](auto&) {}, lane);
    ASSERT_TRUE(report.solved_value_available()) << report.reason;
    // 3/4 uncovered coarse cells with residual 1, 1/4 fine volume with residual 3.
    EXPECT_NEAR(report.residual_norm, std::sqrt(Real(3)), Real(1e-14));
    EXPECT_EQ(report.iters, 0);
  }
}

TEST(AmrSpatialNorm, PreparedNewtonProjectsCompositeOwnershipAcrossGmresRestarts) {
  using namespace pops;
  using runtime::program::PreparedAmrSpatialResidual;
  const auto lane = ExecutionLane::world("test.amr-spatial.prepared-composite-cover");
  for (const auto ownership :
       {std::array<bool, 2>{false, false}, {true, false}, {false, true}, {true, true}}) {
    auto coarse = field(false, ownership[0]), fine = field(true, ownership[1]);
    auto coarse_eb = field(false, ownership[0]), fine_eb = field(true, ownership[1]);
    auto coarse_cover = field(false, ownership[0]), fine_cover = field(true, ownership[1]);
    coarse.set_val(0);
    fine.set_val(0);
    coarse_eb.set_val(1);
    fine_eb.set_val(1);
    fine_cover.set_val(1);
    for (std::size_t local = 0; local < coarse_cover.local_size(); ++local) {
      const auto cover = coarse_cover.fab(local).view();
      const auto eb = coarse_eb.fab(local).view();
      for_each_cell(coarse_cover.box(local), [=] POPS_HD(const Index<2>& cell) {
        cover(cell, 0) =
            cell[0] >= 4 && cell[0] <= 11 && cell[1] >= 4 && cell[1] <= 11 ? Real(0) : Real(1);
        eb(cell, 0) = cell[0] == 0 && cell[1] == 0 ? Real(0) : Real(1);
      });
    }
    const std::array<const MultiFab<2>*, 2> layouts{&coarse, &fine}, embedded{&coarse_eb, &fine_eb},
        coverage{&coarse_cover, &fine_cover};
    const std::array<Real, 2> measures{Real(1) / 256, Real(1) / 1024};
    FieldNewtonOptions options;
    options.tolerance = Real(1e-12);
    options.linear_tolerance = Real(1e-8);
    // Three active diagonal values cannot be solved by one two-vector Krylov cycle.
    options.restart = 2;
    PreparedAmrSpatialResidual<2> workspace(layouts, embedded, measures, options, Real(1e-5),
                                            coverage);
    workspace.stage(0, coarse, &coarse);
    workspace.stage(1, fine, &fine);
    auto evaluate = [](const auto& q, const auto&, auto& result, int) {
      for (std::size_t level = 0; level < result.size(); ++level)
        for (std::size_t local = 0; local < result[level].local_size(); ++local) {
          const auto values = q[level].fab(local).view();
          const auto output = result[level].fab(local).view();
          const auto box = result[level].box(local);
          // Each coarse patch has an active exterior corner. Avoid the one EB-inactive cell.
          const Index<2> active_reference{box.lo[0] == 0 ? 1 : box.hi[0],
                                          box.lo[1] == 0 ? 0 : box.hi[1]};
          for_each_cell(box, [=] POPS_HD(const Index<2>& cell) {
            const bool excluded =
                level == 0 && ((cell[0] >= 4 && cell[0] <= 11 && cell[1] >= 4 && cell[1] <= 11) ||
                               (cell[0] == 0 && cell[1] == 0));
            const Real diagonal = Real(1) + Real(0.25) * Real(cell[0] % 3);
            // Excluded equations depend on an active DOF: their true finite-difference JVP
            // is nonzero. Both Arnoldi work_ and restarted-GMRES image_ must project it out,
            // or the excluded entries contaminate a basis and evolve the frozen candidate.
            output(cell, 0) = excluded ? Real(5) + Real(2) * values(active_reference, 0)
                                       : diagonal * values(cell, 0) - Real(1);
          });
        }
    };
    const auto report = workspace.solve(evaluate, lane);
    ASSERT_TRUE(report.solved_value_available()) << report.reason;
    EXPECT_NEAR(report.reference_residual_norm, std::sqrt(Real(255) / 256), Real(1e-14));
    // Without a restart, each Newton iteration can invoke at most restart JVPs. Exceeding
    // that bound proves a restart image was evaluated, independently of convergence roundoff.
    EXPECT_GT(workspace.derivative_evaluations(), options.restart * report.iters);
    for (std::size_t level = 0; level < workspace.levels(); ++level) {
      const auto& solved = workspace.candidate(level);
      sync_host();
      for (std::size_t local = 0; local < solved.local_size(); ++local) {
        const auto values = solved.fab(local).view();
        for (std::int64_t ordinal = 0; ordinal < solved.box(local).numPts(); ++ordinal) {
          const auto box = solved.box(local);
          const Index<2> cell{box.lo[0] + static_cast<int>(ordinal % box.length(0)),
                              box.lo[1] + static_cast<int>(ordinal / box.length(0))};
          const bool excluded =
              level == 0 && ((cell[0] >= 4 && cell[0] <= 11 && cell[1] >= 4 && cell[1] <= 11) ||
                             (cell[0] == 0 && cell[1] == 0));
          const Real diagonal = Real(1) + Real(0.25) * Real(cell[0] % 3);
          EXPECT_NEAR(values(cell, 0), excluded ? Real(0) : Real(1) / diagonal, Real(1e-12));
        }
      }
    }
  }
}

// This facade fixture matches the Dim2 public implicit qualification matrix.
#if POPS_NATIVE_DIM == 2
namespace pops::runtime::program {
// White-box access is limited to adversarial evidence setup; the tested operation
// remains the real collective materialization entrypoint used by generated Programs.
struct AmrSpatialReconciliationTestAccess {
  using Context = AmrProgramContext<2>;
  using Ledger = Context::multiblock_flux_ledger_type;
  static void verify_scratch_ownership(Context& ctx) {
    auto checkpoint = ctx.capture_accepted_context_snapshot_();
    const auto& accepted = ctx.state(0);
    const int owner = ctx.scratch_prototype_owner_(accepted);
    auto& trial = ctx.scratch_state(900, 1, accepted);
    auto& solved = ctx.scratch_state(900, 0, accepted);
    auto& mapped = ctx.scratch_state(901, 0, accepted);
    auto& status = ctx.scalar_scratch(901, 0, accepted, 1, 0);
    EXPECT_NE(&trial, &solved);
    EXPECT_NE(&trial, &mapped);
    trial.set_val(Real(7));
    solved.set_val(Real(11));
    mapped.set_val(Real(3));
    auto& rhs = ctx.rhs_scratch(902, 0, mapped);
    EXPECT_EQ(ctx.scratch_prototype_owner_(trial), owner);
    EXPECT_EQ(ctx.scratch_prototype_owner_(mapped), owner);
    EXPECT_EQ(ctx.scratch_prototype_owner_(rhs), owner);
    EXPECT_EQ(&ctx.rhs_scratch(902, 0, mapped), &rhs);
    EXPECT_EQ(norm_inf(trial), trial.local_size() ? Real(7) : Real(0));
    EXPECT_EQ(norm_inf(solved), solved.local_size() ? Real(11) : Real(0));
    EXPECT_EQ(norm_inf(mapped), mapped.local_size() ? Real(3) : Real(0));
    auto detached = ctx.scratch_state_like(mapped);
    EXPECT_THROW(ctx.rhs_scratch(903, 0, detached), std::invalid_argument);
    // Identical shape and a real block owner cannot authorize another level's object.
    const Context::ScratchKey foreign_key{Context::ScratchKind::State, ctx.active_level_ + 1, owner,
                                          904, 0};
    auto& foreign = ctx.scratches_.emplace(foreign_key, std::move(detached)).first->second;
    EXPECT_THROW(ctx.rhs_scratch(903, 0, foreign), std::invalid_argument);
    ctx.scratches_.erase(foreign_key);
    // Exercise the actual accepted-context rollback protocol. It retains the epoch
    // while retiring scratch objects, so callbacks must reacquire rather than borrow
    // pointers captured before the rejected attempt.
    auto restored = checkpoint->prepare_restore();
    restored->publish_restore();
    EXPECT_THROW(ctx.rhs_scratch(903, 0, trial), std::invalid_argument);
    auto& next_trial = ctx.scratch_state(900, 1, ctx.state(0));
    auto& next_mapped = ctx.scratch_state(901, 0, ctx.state(0));
    auto& next_status = ctx.scalar_scratch(901, 0, ctx.state(0), 1, 0);
    EXPECT_NE(&next_trial, &trial);
    EXPECT_NE(&next_mapped, &mapped);
    EXPECT_NE(&next_status, &status);
    EXPECT_EQ(ctx.scratch_prototype_owner_(next_trial), owner);
    EXPECT_EQ(ctx.scratch_prototype_owner_(next_mapped), owner);
    EXPECT_EQ(ctx.scratch_prototype_owner_(next_status), owner);
    EXPECT_NO_THROW(ctx.rhs_scratch(902, 0, next_mapped));
  }
  static void prepare(Context& ctx, MultiFab<2>& coarse, MultiFab<2>& middle, MultiFab<2>& fine,
                      Ledger& incoming, Ledger& outgoing) {
    ctx.active_level_ = 1;
    ctx.active_subcycling_attempt_ = 7;
    ctx.active_subcycling_window_ = {{1, 0, {0, 1}, 0}, {1, 0, {1, 1}, 1}};
    ctx.active_attempt_states_ = {&middle};
    ctx.active_incoming_flux_ = {&incoming};
    ctx.active_outgoing_flux_ = {&outgoing};
    ctx.active_block_identities_ = {"tracer"};
    ctx.spatial_consumed_flux_.clear();
    auto first = incoming, second = outgoing;
    first.commit();
    second.commit();
    ctx.spatial_consumed_flux_.emplace(
        std::make_pair(0, 0),
        Context::SpatialConsumedFlux{7, ctx.spatial_weighted_fragments(first), &coarse, &middle});
    ctx.spatial_consumed_flux_.emplace(
        std::make_pair(0, 1),
        Context::SpatialConsumedFlux{7, ctx.spatial_weighted_fragments(second), &middle, &fine});
  }
  static void damage(Context& ctx, int fault, MultiFab<2>& unrelated) {
    if (my_rank() != 0)
      return;
    if (fault == 1)
      ctx.spatial_consumed_flux_.erase({0, 0});
    if (fault == 2)
      ctx.spatial_consumed_flux_.clear();
    if (fault == 3)
      ++ctx.spatial_consumed_flux_.at({0, 0}).attempt;
    if (fault == 4)
      ctx.spatial_consumed_flux_.at({0, 0}).child_candidate = &unrelated;
    if (fault == 5)
      ctx.spatial_consumed_flux_.at({0, 1}).weighted_fragments.back() ^= 1;
    if (fault == 6)
      ctx.active_attempt_states_[0] = &unrelated;
  }
  static void materialize(Context& ctx, MultiFab<2>& candidate) {
    ctx.materialize_active_flux_expression_(0, candidate);
  }
  static std::size_t proofs(const Context& ctx) { return ctx.spatial_consumed_flux_.size(); }
  static bool consume(Context& ctx, std::size_t parent, MultiFab<2>& coarse, MultiFab<2>& fine,
                      const Ledger& flux) {
    const ::pops::amr::ClockWindow window{{static_cast<int>(parent), 0, {0, 1}, 0},
                                          {static_cast<int>(parent), 0, {1, 1}, 1}};
    Context::multiblock_reflux_context_type context{
        0, "tracer", parent, 7, window, coarse, fine, flux, ::pops::amr::RefinementRatio<2>{2, 2},
        {}};
    return ctx.consume_spatial_reconciliation(context);
  }
};
}  // namespace pops::runtime::program

namespace {
using SpatialAccess = pops::runtime::program::AmrSpatialReconciliationTestAccess;
SpatialAccess::Ledger predictor_ledger(int parent, double weight = 1) {
  SpatialAccess::Ledger result({8, 8, 1});
  result.begin(7);
  for (auto role :
       {pops::amr::reflux::FaceLedgerRole::Coarse, pops::amr::reflux::FaceLedgerRole::Fine}) {
    pops::amr::reflux::FaceFluxFragmentKey<2> key;
    key.owner = "tracer";
    key.state = "tracer/state";
    key.stage = "predictor/dt-power/1/weight/1/1";
    key.levels = {parent, parent + 1};
    key.role = role;
    key.axis = 0;
    key.attempt = 7;
    key.clock = {parent + (role == pops::amr::reflux::FaceLedgerRole::Fine), 0, {0, 1}, 0};
    pops::amr::reflux::FaceFluxFragmentMeasure measure;
    measure.stage_weight = {static_cast<std::int64_t>(weight), 1};
    measure.substep_begin = {0, 1};
    measure.substep_end = {1, 1};
    measure.substep_duration = 1;
    measure.face_measure = 0.125;
    result.accumulate(key, measure, {pops::Real(3)});
  }
  return result;
}
}  // namespace

TEST(AmrSpatialMaterialization, PendingProofSurvivesPublicationAndRejectsRankLocalDamage) {
  using namespace pops;
  AmrSystemConfig<2> config;
  config.shape = Extent<2>{16, 16};
  config.periodicity = {true, true};
  config.level_count = 2;
  config.regrid_every = 0;
  AmrSystem<2> system(config);
  test::install_amr_runtime_authority(system, "tests.spatial-proof/runtime@1");
  system.set_temporal_relations({2}, {1}, {"integral_only"});
  system.install_block_state_route("tracer", "tests.spatial-proof/tracer/state@1");
  system.install_hyperbolic_boundary("tracer", "tests.spatial-proof/boundary@1", 1,
                                     {"periodic", "periodic", "periodic", "periodic"},
                                     std::vector<double>(16, 0), {"xlo", "xhi", "ylo", "yhi"},
                                     {"density", "momentum:0", "momentum:1", "energy"},
                                     "tests.spatial-proof/tracer/state@1");
  using Model = CompositeModel<EulerND<2>, NoSource, NoElliptic>;
  add_compiled_model<2>(system, "tracer",
                        Model{{}, {}, EulerND<2>::prepare(Real(1.4)), NoSource{}, NoElliptic{}},
                        "minmod", "rusanov", "conservative", "explicit", 1.4, 1, 1, {}, {}, 0.0,
                        static_cast<double>(kWenoEpsilon), false, "tests.spatial-proof/flux@1");
  std::vector<double> initial(4 * 256, 0);
  std::fill(initial.begin(), initial.begin() + 256, 1);
  std::fill(initial.begin() + 3 * 256, initial.end(), 2.5);
  system.set_conservative_state("tracer", initial);
  system.install_program_step([](double) {});
  system.set_program_block_map({0});
  (void)system.mass("tracer");
  SpatialAccess::Context context(system.engine(), &system);
  SpatialAccess::verify_scratch_ownership(context);
  auto coarse = field(false, false), middle = field(false, false), fine = field(true, false);
  auto unrelated = field(false, false);
  coarse.set_val(1);
  middle.set_val(2);
  fine.set_val(3);
  unrelated.set_val(4);
  for (int fault = 0; fault <= 8; ++fault) {
    auto incoming = predictor_ledger(0), outgoing = predictor_ledger(1);
    auto attempt = [&] {
      SpatialAccess::prepare(context, coarse, middle, fine, incoming, outgoing);
      SpatialAccess::damage(context, fault, unrelated);
      if (my_rank() == 0 && fault == 7)
        outgoing = predictor_ledger(1, 2);
      if (my_rank() == 0 && fault == 8)
        outgoing.rollback();
      SpatialAccess::materialize(context, middle);
      EXPECT_EQ(SpatialAccess::proofs(context), 2u);
      EXPECT_EQ(incoming.pending_size(), 2u);
      EXPECT_EQ(incoming.published_size(), 0u);
      incoming.commit();
      outgoing.commit();
      EXPECT_TRUE(SpatialAccess::consume(context, 0, coarse, middle, incoming));
      EXPECT_TRUE(SpatialAccess::consume(context, 1, middle, fine, outgoing));
      EXPECT_EQ(SpatialAccess::proofs(context), 0u);
    };
    if (fault == 0) {
      EXPECT_NO_THROW(context.with_spatial_reconciliation_attempt(attempt));
    } else {
      EXPECT_THROW(context.with_spatial_reconciliation_attempt(attempt), std::exception);
      EXPECT_EQ(SpatialAccess::proofs(context), 0u);
      incoming.rollback();
      if (!(my_rank() == 0 && fault == 8))
        outgoing.rollback();
      EXPECT_EQ(incoming.published_size() + outgoing.published_size(), 0u);
      EXPECT_EQ(incoming.pending_size() + outgoing.pending_size(), 0u);
    }
    EXPECT_DOUBLE_EQ(reduce_min_local(coarse), 1);
    EXPECT_DOUBLE_EQ(reduce_min_local(middle), 2);
    EXPECT_DOUBLE_EQ(reduce_min_local(fine), 3);
  }
}

#endif  // POPS_NATIVE_DIM == 2
