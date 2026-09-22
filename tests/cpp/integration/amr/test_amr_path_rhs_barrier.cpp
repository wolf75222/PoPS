#include <gtest/gtest.h>

#include "explicit_amr_program.hpp"
#include "../../support/generated_fan_li15.hpp"
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#if POPS_NATIVE_DIM == 2
namespace {
using Real = pops::Real;
using Raw = pops::StateVec<15>;
using Field = pops::MultiFab<2>;
using System = pops::AmrSystem<2>;
using Context = pops::runtime::program::AmrProgramContext<2>;
constexpr const char* kProgram = "tests.amr-path-rhs/program@1";
constexpr const char* kFamily = "tests.amr-path-rhs/forward-euler@1";
constexpr int kRate = 101;
constexpr double kDt = 1e-4;

// Independent scalar-normal recurrence, in the authored q-outer raw ordering.
// This fixture never calls the production Hermite reconstruction to initialize its state.
Raw gaussian(Real density, Real ux, Real uy, Real variance = Real(1)) {
  Real mx[5]{1, ux}, my[5]{1, uy};
  for (int degree = 2; degree < 5; ++degree) {
    mx[degree] = ux * mx[degree - 1] + Real(degree - 1) * variance * mx[degree - 2];
    my[degree] = uy * my[degree - 1] + Real(degree - 1) * variance * my[degree - 2];
  }
  Raw state{};
  int slot = 0;
  for (int q = 0; q <= 4; ++q)
    for (int p = 0; p <= 4 - q; ++p)
      state[slot++] = density * mx[p] * my[q];
  return state;
}

struct Model : test_fan_li15::Kernel {
  using State = Raw;
  using Primitive = Raw;
  struct Schema {
    using Conservative = Raw;
    using Primitive = Raw;
    static constexpr int nvars = 15;
  };
  static constexpr int dimension = 2, n_vars = 15, n_providers = 0;
  static constexpr bool path_conservative = true;
  static constexpr std::string_view path_operator_identity() {
    return "tests.amr-path-rhs/FanLi15-q-outer/gx=(1,0)/gy=(0,1)@1";
  }
  POPS_HD static constexpr std::array<bool, 4> path_zero_measure_faces() { return {}; }
  template <int Axis, class Providers>
  POPS_HD std::array<Real, 2> path_covector(const Providers&) const {
    static_assert(Axis == 0 || Axis == 1);
    return Axis == 0 ? std::array<Real, 2>{1, 0} : std::array<Real, 2>{0, 1};
  }
  static constexpr pops::PreparedProviderIdentity provider_identity() {
    return {"tests.amr-path-rhs/raw15-model", 1};
  }
  void serialize_exact_parameters(pops::ExactContractBuilder& contract) const {
    contract.text(path_operator_identity());
  }
  static pops::VariableSet conservative_vars() {
    std::vector<std::string> names;
    std::vector<pops::VariableRole> roles(15, pops::VariableRole::Scalar);
    for (int component = 0; component < 15; ++component)
      names.push_back("raw" + std::to_string(component));
    roles[0] = pops::VariableRole::Density;
    return {pops::VariableKind::Conservative, std::move(names), 15, std::move(roles)};
  }
  static pops::VariableSet primitive_vars() {
    auto result = conservative_vars();
    result.kind = pops::VariableKind::Primitive;
    return result;
  }
  POPS_HD bool path_admissible(const State& state) const {
    Real raw[15];
    for (int component = 0; component < 15; ++component)
      raw[component] = state[component];
    return test_fan_li15::Kernel::admissibility(raw) ==
           pops::PathStatus::Success;
  }
  POPS_HD pops::nd::StateConversionStatus admissibility(const State& state) const {
    return path_admissible(state) ? pops::nd::StateConversionStatus::Success
                                 : pops::nd::StateConversionStatus::NonPositivePressure;
  }
  POPS_HD pops::nd::StateConversion<State> recover(const State& state) const {
    return {state, admissibility(state)};
  }
  POPS_HD pops::nd::StateConversion<State> make_conservative(const State& state) const {
    return recover(state);
  }
  template <int Axis>
  POPS_HD State flux(const State& state) const {
    Real raw[15];
    for (int component = 0; component < 15; ++component)
      raw[component] = state[component];
    const auto result = test_fan_li15::flux(
        raw, Axis == 0 ? Real(1) : Real(0), Axis == 1 ? Real(1) : Real(0));
    State flux{};
    for (int component = 0; component < 15; ++component)
      flux[component] = result.flux.values[component];
    return flux;
  }
  template <int Axis>
  POPS_HD Real max_wave_speed(const State& state) const {
    Real raw[15];
    for (int component = 0; component < 15; ++component)
      raw[component] = state[component];
    const auto result = test_fan_li15::path_integral(
        raw, raw, Axis == 0 ? Real(1) : Real(0), Axis == 1 ? Real(1) : Real(0));
    return result.succeeded() ? result.speed_bound : std::numeric_limits<Real>::quiet_NaN();
  }
  POPS_HD State source(const State&, const pops::ProviderValues<0>&) const { return {}; }
  POPS_HD Real elliptic_rhs(const State&) const { return Real(0); }
};

struct OrdinaryModel : Model {
  static constexpr bool path_conservative = false;
  static constexpr pops::PreparedProviderIdentity provider_identity() {
    return {"tests.amr-path-rhs/ordinary-raw15-control", 1};
  }
};

Raw accepted_state(int level) {
  return level == 0 ? gaussian(1, 0, 0) : gaussian(Real(1.125), Real(0.125), 0);
}
Raw trace_state(int level, bool hot = false) {
  Raw result = gaussian(level == 0 ? Real(1) : Real(1.25),
                        level == 0 ? Real(0.25) : Real(-0.5),
                        level == 0 ? Real(-0.125) : Real(0.25),
                        hot ? Real(1e8) : Real(1));
  // A finite third raw moment changes the nonconservative fourth-order residual while leaving
  // the strict rho/Theta domain unchanged. This makes an equal/opposite NCP ledger observable.
  result[3] += level == 0 ? Real(0.125) : Real(-0.25);
  return result;
}

POPS_HD Real trace_scale(int level, int x, int y, bool nonconstant) {
  if (!nonconstant)
    return Real(1);
  return level == 0 ? Real(1) + Real(x) / Real(16)
                    : Real(1) + (y % 2 == 0 ? Real(1) : Real(-1)) / Real(32);
}

Raw trace_at(int level, int x, int y, bool nonconstant) {
  Raw result = trace_state(level);
  const Real scale = trace_scale(level, x, y, nonconstant);
  for (int component = 0; component < 15; ++component)
    result[component] *= scale;
  return result;
}

void fill_uniform(Field& field, const Raw& state) {
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    const auto values = field.fab(local).view();
    pops::for_each_cell(field.box(local), [=] POPS_HD(const pops::Index<2>& cell) {
      for (int component = 0; component < 15; ++component)
        values(cell, component) = state[component];
    });
  }
  pops::device_fence();
}

void fill_trace(Field& field, int level, bool hot, bool nonconstant) {
  const Raw state = trace_state(level, hot);
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    const auto values = field.fab(local).view();
    pops::for_each_cell(field.box(local), [=] POPS_HD(const pops::Index<2>& cell) {
      const Real scale = trace_scale(level, cell[0], cell[1], nonconstant);
      for (int component = 0; component < 15; ++component)
        values(cell, component) = scale * state[component];
    });
  }
  pops::device_fence();
}

template <class Expected>
void expect_cells(const Field& field, Expected expected, bool active_coarse_only = false) {
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    const auto& fab = field.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const auto grown = fab.grown_box();
    const auto cells = field.box(local);
    for (int j = cells.lo[1]; j <= cells.hi[1]; ++j)
      for (int i = cells.lo[0]; i <= cells.hi[0]; ++i) {
        if (active_coarse_only && i < 2)
          continue;
        const Raw reference = expected(i, j);
        for (int component = 0; component < 15; ++component) {
          const auto offset = static_cast<std::size_t>(component * grown.numPts() +
              i - grown.lo[0] + (j - grown.lo[1]) * grown.length(0));
          EXPECT_NEAR(host(offset), reference[component],
                      Real(2e-11) * (Real(1) + std::abs(reference[component])))
              << "cell=" << i << ',' << j << " component=" << component;
        }
      }
  }
}

pops::PathInterfaceResult<15> tuple(const Raw& left, const Raw& right, int axis = 0) {
  Real l[15], r[15];
  for (int component = 0; component < 15; ++component) {
    l[component] = left[component];
    r[component] = right[component];
  }
  auto result = test_fan_li15::interface(
      l, r, axis == 0 ? Real(1) : Real(0), axis == 1 ? Real(1) : Real(0));
  if (!result.succeeded())
    throw std::runtime_error("independent path test tuple failed its strict domain");
  return result;
}

// The refined strip spans the periodic y extent. Coarse recipient faces are the explicit mean
// of two independently evaluated fine subfaces, always using the actual parent cell mean.
Raw expected_rhs(int level, int x, int y = 0, bool nonconstant = false) {
  const Raw center = trace_at(level, x, y, nonconstant);
  const Raw left = level == 0 ? trace_at(0, (x + 7) % 8, y, nonconstant)
      : x == 0 ? trace_at(0, 7, y / 2, nonconstant)
               : trace_at(1, x - 1, y, nonconstant);
  const Raw right = level == 0 ? trace_at(0, (x + 1) % 8, y, nonconstant)
      : x == 3 ? trace_at(0, 2, y / 2, nonconstant)
               : trace_at(1, x + 1, y, nonconstant);
  auto lower0 = tuple(left, center), lower1 = lower0;
  auto upper0 = tuple(center, right), upper1 = upper0;
  if (level == 0 && x == 2) {
    lower0 = tuple(trace_at(1, 3, 2 * y, nonconstant), center);
    lower1 = tuple(trace_at(1, 3, 2 * y + 1, nonconstant), center);
  }
  if (level == 0 && x == 7) {
    upper0 = tuple(center, trace_at(1, 0, 2 * y, nonconstant));
    upper1 = tuple(center, trace_at(1, 0, 2 * y + 1, nonconstant));
  }
  const Real inverse_spacing = level == 0 ? Real(8) : Real(16);
  Raw result{};
  for (int component = 0; component < 15; ++component) {
    result[component] = inverse_spacing * Real(0.5) * (
        lower0.conservative_flux.values[component] + lower0.right_ncp.values[component] +
        lower1.conservative_flux.values[component] + lower1.right_ncp.values[component] -
        upper0.conservative_flux.values[component] + upper0.left_ncp.values[component] -
        upper1.conservative_flux.values[component] + upper1.left_ncp.values[component]);
  }
  if (level == 1) {
    const auto lower = tuple(trace_at(1, x, (y + 15) % 16, nonconstant), center, 1);
    const auto upper = tuple(center, trace_at(1, x, (y + 1) % 16, nonconstant), 1);
    for (int component = 0; component < 15; ++component)
      result[component] += inverse_spacing * (
          lower.conservative_flux.values[component] + lower.right_ncp.values[component] -
          upper.conservative_flux.values[component] + upper.left_ncp.values[component]);
  }
  return result;
}

enum class Failure {
  None, HotTrace, InvalidTrace, ForeignOutput, ForeignSource, LegacyResidual, AfterPublication
};
struct Evidence {
  Failure failure = Failure::None;
  bool nonconstant = false;
  int gathered = 0, published = 0, continued = 0, completed = 0;
};
struct Fixture {
  std::unique_ptr<System> system;
  std::shared_ptr<Context> context;
  std::shared_ptr<Evidence> evidence;
};

enum class TransferPolicy { Omitted, Injection, LimitedLinear, Polynomial5, WideInjection };

void add_fine_level(Fixture& fixture);

template <class BlockModel = Model>
Fixture prepare_fixture(TransferPolicy prolongation = TransferPolicy::Injection,
                        TransferPolicy coarse_fine = TransferPolicy::Injection,
                        bool refine = true) {
  pops::comm_init();
  pops::AmrSystemConfig<2> config;
  config.explicit_bootstrap = true;
  config.regrid_every = 0;
  config.distribute_coarse = true;
  config.shape = pops::Extent<2>{8, 8};
  config.periodicity = {true, true};
  config.coarse_max_grid = pops::Extent<2>{2, 8};
  auto system = std::make_unique<System>(config);
  pops::test::install_amr_runtime_authority(*system, "tests.amr-path-rhs/runtime");
  system->set_temporal_relations({1}, {1}, {"integral_only"});
  system->install_block_state_route("raw15", "tests.amr-path-rhs/state");
  const auto install_transfer = [&](const std::string& operation, TransferPolicy policy) {
    if (policy == TransferPolicy::Omitted)
      return;
    const bool injection = policy == TransferPolicy::Injection ||
                           policy == TransferPolicy::WideInjection;
    const int order = injection ? 1 : policy == TransferPolicy::Polynomial5 ? 5 : 2;
    const std::string kernel = injection ? "conservative_injection"
        : operation == "prolongation" ? "conservative_linear"
        : policy == TransferPolicy::Polynomial5 ? "conservative_polynomial5_coarse_fine"
                                                : "conservative_coarse_fine";
    pops::Extent<2> ghosts{};
    for (int axis = 0; axis < 2; ++axis)
      ghosts[axis] = operation == "prolongation" && injection ? 0
          : policy == TransferPolicy::WideInjection ? 2 : policy == TransferPolicy::Polynomial5 ? 3 : 1;
    system->register_bootstrap_transfer_route(
        "tests.amr-path-rhs/" + operation, {"tests.amr-path-rhs/state"},
        "tests.amr-path-rhs/" + kernel + "@1", "cell", "cell", "conservative", "dense",
        operation, kernel, order, ghosts, config.transition_ratios.front());
  };
  install_transfer("prolongation", prolongation);
  install_transfer("coarse_fine_fill", coarse_fine);
  pops::add_compiled_model<2>(*system, "raw15", BlockModel{}, "none", "rusanov",
      "conservative", "explicit", 1.4, 1, 1, {}, {}, 0.0,
      static_cast<double>(pops::kWenoEpsilon), false, "tests.amr-path-rhs/physical-flux");
  std::vector<double> initial(15 * 64);
  const auto root_state = accepted_state(0);
  for (int component = 0; component < 15; ++component)
    for (int cell = 0; cell < 64; ++cell)
      initial[static_cast<std::size_t>(component * 64 + cell)] = root_state[component];
  system->set_conservative_state("raw15", initial);
  auto context = pops::runtime::program::make_program_execution_provider(system.get());
  system->set_conservative_state("raw15", initial);
  auto evidence = std::make_shared<Evidence>();
  System* facade = system.get();
  context->configure_primary_clock("tests.amr-path-rhs/clock");
  context->install([context, evidence, facade](double dt) {
    context->advance_mapping_hierarchy(dt, [context, evidence, facade](double level_dt) {
      ++evidence->gathered;
      context->set_stage_time(0, 1);
      const int level = context->level();
      auto& state = context->state(0);
      auto& input = context->scratch_state(100, 0, state);
      fill_trace(input, level, evidence->failure == Failure::HotTrace, evidence->nonconstant);
      if (evidence->failure == Failure::InvalidTrace && level == 1 && input.local_size() != 0)
        input.fab(0).set_val(Real(-1));
      auto& rhs = context->rhs_scratch(kRate, 0, state);
      if (evidence->failure == Failure::LegacyResidual)
        context->rhs_into(0, input, rhs, kRate);
      const auto source = evidence->failure == Failure::ForeignSource && level == 1 ? 99 : 100;
      const auto trace = context->capture_rhs_input_trace(0, input, kRate, source);
      context->stage_path_rhs(0, input,
          evidence->failure == Failure::ForeignOutput ? state : rhs,
          kRate, kFamily, &trace, context->path_rhs_courant());
      // The producer may reuse its live scratch after the immutable SSA trace is captured.
      // Both the current level and the parent fill must read the retained stage, not this value.
      input.set_val(Real(-123));
      context->suspend_hierarchy_barrier(Context::HierarchyBarrierKind::spatial_rhs, kRate,
          kProgram, "raw15/input100/path101", 0, 1,
          [context, evidence, facade] {
            EXPECT_EQ(evidence->gathered, facade->n_levels());
            EXPECT_EQ(evidence->continued, 0);
            EXPECT_EQ(facade->time(), 0.0);
            for (int level = 0; level < facade->n_levels(); ++level)
              expect_cells(facade->engine()->hierarchy().state(level),
                           [level](int, int) { return accepted_state(level); });
            context->publish_staged_path_rhs(kRate);
            ++evidence->published;
            for (int level = 0; level < facade->n_levels(); ++level) {
              const auto& evaluation = facade->prepared_amr_level_evaluation(level);
              ASSERT_TRUE(evaluation.path_faces);
              EXPECT_EQ(evaluation.path_faces->operator_identity, Model::path_operator_identity());
              expect_cells(evaluation.residual,
                           [level, evidence](int x, int y) {
                             return expected_rhs(level, x, y, evidence->nonconstant);
                           }, level == 0);
            }
            if (evidence->failure == Failure::AfterPublication &&
                context->prepared_execution_lane().rank() == 0)
              throw std::runtime_error("injected path barrier failure after tuple publication");
          },
          [context, evidence, &state, &rhs, level_dt] {
            ++evidence->continued;
            EXPECT_EQ(evidence->published, 1);
            context->axpy(state, Real(level_dt), rhs);
          });
    }, context, [evidence] { ++evidence->completed; });
  }, context);
  system->set_program_block_map({0});
  context->install_flux_temporal_families({{0, kRate, 1, kFamily}});
  using Budget = System::PreparedAmrProgramFluxExpressionBlockBudget;
  system->install_prepared_amr_program_flux_expression_budget(kProgram, {Budget{1, 1}}, 0, 0);

  Fixture fixture{std::move(system), std::move(context), std::move(evidence)};
  if (refine)
    add_fine_level(fixture);
  return fixture;
}

void add_fine_level(Fixture& fixture) {
  auto& system = fixture.system;
  auto& context = fixture.context;
  const auto& parent = system->engine()->hierarchy().layout(0);
  const pops::mesh::BoxArray<2> boxes(std::vector<pops::Box<2>>{
      pops::Box<2>{pops::Index<2>{0, 0}, pops::Index<2>{1, 7}}});
  pops::amr::tagging::ClusterOptions<2> options;
  options.min_efficiency = 0.7;
  options.min_box_size.fill(1);
  options.max_box_size.fill(16);
  options.budget = {16, 256, 8192, 64, 1U << 20};
  pops::amr::tagging::ClusterResultIdentity<2> identity{
      "tests.amr-path-rhs/cluster", parent.exact_identity(), options, {}, boxes.boxes()};
  const std::array<int, 2> ratio{2, 2};
  auto prepared = context->prepare_regrid(0, pops::amr::RefinementRatio<2>(ratio),
      {boxes, std::move(identity)}, {.clustered_parent_layout = {16, 120},
       .fine_layout = {16, 120},
       .load_balance = {16, 16, std::numeric_limits<std::int64_t>::max()}});
  const auto& accepted = system->engine()->hierarchy().state(0);
  Field child(prepared.fine_layout()->patches(), prepared.fine_layout()->distribution(),
              accepted.local_rank(), accepted.ncomp(), accepted.ghosts());
  fill_uniform(child, accepted_state(1));
  context->publish_regrid(std::move(prepared), std::move(child));
}

void expect_success(const Fixture& fixture) {
  EXPECT_EQ(fixture.evidence->gathered, 2);
  EXPECT_EQ(fixture.evidence->published, 1);
  EXPECT_EQ(fixture.evidence->continued, 2);
  EXPECT_EQ(fixture.evidence->completed, 1);
  EXPECT_DOUBLE_EQ(fixture.system->time(), kDt);
  for (int level = 0; level < 2; ++level)
    expect_cells(fixture.system->engine()->hierarchy().state(level), [level, &fixture](int x, int y) {
      Raw expected = accepted_state(level);
      const Raw rhs = expected_rhs(level, x, y, fixture.evidence->nonconstant);
      for (int component = 0; component < 15; ++component)
        expected[component] += Real(kDt) * rhs[component];
      return expected;
    }, level == 0);
  EXPECT_THROW(fixture.context->path_rhs_courant(), std::logic_error);
}
}  // namespace

TEST(AmrPathRhsBarrier, CanonicalSubfacesAndIndependentSidesPrecedeUpdatesAtPeriodicSeam) {
  const auto nonconservative = tuple(trace_state(0), trace_state(1));
  Real magnitude = 0;
  for (int component = 0; component < 15; ++component)
    magnitude += std::abs(nonconservative.left_ncp.values[component]);
  ASSERT_GT(magnitude, Real(1e-8));
  auto fixture = prepare_fixture();
  ASSERT_EQ(fixture.system->n_levels(), 2);
  EXPECT_DOUBLE_EQ(fixture.system->step_cfl(0.3, 1e-12, kDt, 0), kDt);
  expect_success(fixture);
}

TEST(AmrPathRhsBarrier, NonconstantParentMeansAndUnequalSubfacesUseExactInjection) {
  const Raw parent = trace_at(0, 2, 0, true);
  const Raw fine = trace_at(1, 3, 0, true);
  Raw reconstructed = parent;
  const Raw slope = trace_state(0);
  // The first fine ghost center lies one quarter coarse-cell width below its parent center.
  // This monotone three-cell profile has nonzero limited-linear slopes in every nonzero row.
  for (int component = 0; component < 15; ++component)
    reconstructed[component] -= slope[component] / Real(64);
  const auto injected = tuple(fine, parent);
  const auto linear = tuple(fine, reconstructed);
  const auto other_subface = tuple(trace_at(1, 3, 1, true), parent);
  ASSERT_GT(std::abs(injected.conservative_flux.values[0] - linear.conservative_flux.values[0]),
            Real(1e-5));
  ASSERT_GT(std::abs(injected.conservative_flux.values[0] -
                     other_subface.conservative_flux.values[0]), Real(1e-5));
  auto fixture = prepare_fixture();
  fixture.evidence->nonconstant = true;
  EXPECT_DOUBLE_EQ(fixture.system->step_cfl(0.3, 1e-12, kDt, 0), kDt);
  expect_success(fixture);
}

TEST(AmrPathRhsBarrier, ReconstructedOrMissingTransferRoutesRefuseBeforeHierarchyPublication) {
  using Policy = TransferPolicy;
  const std::array policies{
      std::pair{Policy::Omitted, Policy::Omitted},
      std::pair{Policy::Omitted, Policy::Injection},
      std::pair{Policy::Injection, Policy::Omitted},
      std::pair{Policy::LimitedLinear, Policy::Injection},
      std::pair{Policy::Injection, Policy::LimitedLinear},
      std::pair{Policy::Injection, Policy::Polynomial5},
      std::pair{Policy::Injection, Policy::WideInjection}};
  for (const auto& [prolongation, coarse_fine] : policies) {
    SCOPED_TRACE(std::to_string(static_cast<int>(prolongation)) + "/" +
                 std::to_string(static_cast<int>(coarse_fine)));
    auto fixture = prepare_fixture(prolongation, coarse_fine, false);
    ASSERT_EQ(fixture.system->n_levels(), 1);
    const auto accepted = fixture.system->program_accepted_state();
    const auto topology = fixture.system->engine()->topology_epoch();
    const auto materialization = fixture.system->engine()->materialization_generation();
    try {
      add_fine_level(fixture);
      FAIL() << "path hierarchy accepted a missing or reconstructed transfer route";
    } catch (const std::invalid_argument& error) {
      EXPECT_NE(std::string(error.what()).find("AMR path state"), std::string::npos);
    }
    EXPECT_EQ(fixture.system->n_levels(), 1);
    EXPECT_EQ(fixture.system->engine()->topology_epoch(), topology);
    EXPECT_EQ(fixture.system->engine()->materialization_generation(), materialization);
    EXPECT_EQ(fixture.system->program_accepted_state(), accepted);
    EXPECT_DOUBLE_EQ(fixture.system->time(), 0.0);
    EXPECT_EQ(fixture.system->macro_step(), 0);
    expect_cells(fixture.system->engine()->hierarchy().state(0),
                 [](int, int) { return accepted_state(0); });
  }
}

TEST(AmrPathRhsBarrier, OrdinaryBlocksRetainDefaultAndLimitedLinearTransferAdmission) {
  using Policy = TransferPolicy;
  for (const auto policy : {Policy::Omitted, Policy::LimitedLinear, Policy::Injection}) {
    auto fixture = prepare_fixture<OrdinaryModel>(policy, policy);
    EXPECT_EQ(fixture.system->n_levels(), 2);
    EXPECT_DOUBLE_EQ(fixture.system->time(), 0.0);
    for (int level = 0; level < 2; ++level)
      expect_cells(fixture.system->engine()->hierarchy().state(level),
                   [level](int, int) { return accepted_state(level); });
  }
}

TEST(AmrPathRhsBarrier, ActualStageCflDomainAndIdentityFailuresRollbackThenPermitRetry) {
  auto fixture = prepare_fixture();
  const auto accepted = fixture.system->program_accepted_state();
  for (Failure failure : {Failure::HotTrace, Failure::InvalidTrace, Failure::ForeignOutput,
                          Failure::ForeignSource, Failure::LegacyResidual,
                          Failure::AfterPublication}) {
    *fixture.evidence = Evidence{};
    fixture.evidence->failure = failure;
    EXPECT_THROW(fixture.system->step_cfl(0.3, 1e-12, kDt, 0), std::exception);
    EXPECT_EQ(fixture.evidence->continued, 0);
    EXPECT_EQ(fixture.evidence->completed, 0);
    EXPECT_DOUBLE_EQ(fixture.system->time(), 0.0);
    EXPECT_EQ(fixture.system->program_accepted_state(), accepted);
    EXPECT_THROW(fixture.context->path_rhs_courant(), std::logic_error);
    for (int level = 0; level < 2; ++level)
      expect_cells(fixture.system->engine()->hierarchy().state(level),
                   [level](int, int) { return accepted_state(level); });
  }
  *fixture.evidence = Evidence{};
  fixture.system->step_cfl(0.3, 1e-12, kDt, 0);
  expect_success(fixture);
}

TEST(AmrPathRhsBarrier, FixedDtCannotInventAuthoredCourantAndFailedAttemptCanRetry) {
  auto fixture = prepare_fixture();
  const auto accepted = fixture.system->program_accepted_state();
  EXPECT_THROW(fixture.system->step(kDt), std::exception);
  EXPECT_EQ(fixture.evidence->published, 0);
  EXPECT_EQ(fixture.evidence->continued, 0);
  EXPECT_EQ(fixture.system->program_accepted_state(), accepted);
  EXPECT_THROW(fixture.context->path_rhs_courant(), std::logic_error);
  *fixture.evidence = Evidence{};
  fixture.system->step_cfl(0.3, 1e-12, kDt, 0);
  expect_success(fixture);
}
#else
TEST(AmrPathRhsBarrier, RequiresTwoDimensionalNativeMomentCarrier) {
  GTEST_SKIP() << "The authenticated Fan-Li15 path has a two-dimensional moment carrier";
}
#endif
