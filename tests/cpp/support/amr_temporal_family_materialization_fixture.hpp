#pragma once

#include <pops/amr/reflux/metric_reflux.hpp>
#include <cmath>

namespace {
enum class TemporalFluxFault {
  None, MissingBasisFamily, ForeignBasisFamily, MissingCoarseFamily, ForeignCoarseFamily,
  MissingDeclaredProducer, ForeignRhs, ForeignProvider, ForeignBasis, ForeignStage,
  WeightMeasureMismatch, ExtraCoarseStage, IncompleteWindow, IncompleteFineFaces,
  BadClockDuration, WeightsDoNotClose
};

// White-box metadata setup invokes the real collective materialization and metric-reflux
// entrypoints. Numeric payloads are a constant-density independent quadrature oracle.
struct TemporalFluxFixture {
  using Rational = pops::amr::Rational;
  using Basis = SpatialAccess::Basis;
  using Provider = SpatialAccess::Provider;
  using Expression = SpatialAccess::Expression;
  using Role = pops::amr::reflux::FaceLedgerRole;
  using Key = pops::amr::reflux::FaceFluxFragmentKey<2>;
  static constexpr auto family = "tests.spatial-family/declared-occurrence@1";
  static constexpr auto state_identity = "tests.spatial-family/tracer/state@1";
  std::unique_ptr<pops::AmrSystem<2>> system;
  std::unique_ptr<SpatialAccess::Context> context;
  pops::MultiFab<2> candidate;
  SpatialAccess::Ledger ledger;
  TemporalFluxFault fault;
  bool declared, migrated;

  TemporalFluxFixture(TemporalFluxFault damage = TemporalFluxFault::None,
                      bool installed = true, bool interpolate = true)
      : candidate(field(true, false)), ledger({64, 64, 1}), fault(damage),
        declared(installed), migrated(interpolate) {
    using namespace pops;
    AmrSystemConfig<2> config;
    config.shape = Extent<2>{16, 16};
    config.periodicity = {true, true};
    config.level_count = 2;
    config.regrid_every = 0;
    system = std::make_unique<AmrSystem<2>>(config);
    test::install_amr_runtime_authority(*system, "tests.spatial-family/runtime@1");
    system->set_temporal_relations({2}, {1}, {"integral_only"});
    system->install_block_state_route("tracer", state_identity);
    system->install_hyperbolic_boundary("tracer", "tests.spatial-family/boundary@1", 1,
        {"periodic", "periodic", "periodic", "periodic"}, std::vector<double>(16, 0),
        {"xlo", "xhi", "ylo", "yhi"}, {"density", "momentum:0", "momentum:1", "energy"},
        state_identity);
    using Model = CompositeModel<EulerND<2>, NoSource, NoElliptic>;
    add_compiled_model<2>(*system, "tracer",
        Model{{}, {}, EulerND<2>::prepare(Real(1.4)), NoSource{}, NoElliptic{}},
        "minmod", "rusanov", "conservative", "explicit", 1.4, 1, 1, {}, {}, 0.0,
        static_cast<double>(kWenoEpsilon), false, "tests.spatial-family/flux@1");
    std::vector<double> initial(4 * 256, 0);
    std::fill(initial.begin(), initial.begin() + 256, 1);
    std::fill(initial.begin() + 3 * 256, initial.end(), 2.5);
    system->set_conservative_state("tracer", initial);
    system->install_program_step([](double) {});
    system->set_program_block_map({0});
    (void)system->mass("tracer");
    context = std::make_unique<SpatialAccess::Context>(system->engine(), system.get());
    if (declared) {
      if (fault == TemporalFluxFault::MissingDeclaredProducer)
        context->install_flux_temporal_families({
            {0, 2, static_cast<int>(Provider::PreparedDefaultFlux), family},
            {0, 19, static_cast<int>(Provider::PreparedDefaultFlux), "tests.unrelated-family"}});
      else
        context->install_flux_temporal_families(
            {{0, 2, static_cast<int>(Provider::PreparedDefaultFlux), family}});
    }
    candidate.set_val(Real(17));
    ledger.begin(7);
    for (int term = 0; term < 2; ++term) {
      const auto weight = coarse_weight(term);
      Key key;
      key.owner = "tracer";
      key.state = state_identity;
      key.levels = {0, 1};
      key.axis = 0;
      key.face = key.coarse_face = Index<2>{4, 4};
      key.clock = {0, 0, {0, 1}, 0};
      key.attempt = 7;
      key.role = Role::Coarse;
      key.temporal_family = declared ? family : "";
      key.stage = literal_stage(term == 0 ? 1 : 3, weight);
      if (my_rank() == 0 && fault == TemporalFluxFault::MissingCoarseFamily)
        key.temporal_family.clear();
      if (my_rank() == 0 && fault == TemporalFluxFault::ForeignCoarseFamily)
        key.temporal_family = "tests.foreign-family";
      const auto measure_weight = my_rank() == 0 && term == 0 &&
          fault == TemporalFluxFault::WeightMeasureMismatch ? Rational{5, 4} : weight;
      ledger.accumulate(key, {measure_weight, {0, 1}, {1, 1}, 1, 1}, {Real(5)});
      if (my_rank() == 0 && term == 0 && fault == TemporalFluxFault::ExtraCoarseStage) {
        key.stage = literal_stage(9, {0, 1});
        ledger.accumulate(key, {{0, 1}, {0, 1}, {1, 1}, 1, 1}, {Real(5)});
      }
    }
  }

  static Rational coarse_weight(int term) { return term == 0 ? Rational{3, 2} : Rational{-1, 2}; }
  static std::string literal_stage(std::uint64_t id, Rational weight) {
    // Fixed external contract for this fixture's single provider/RHS/stage, not the production renderer.
    return "pops.program-flux-expression.v1/provider/1/rhs/2/point-stage/2/basis/" +
        std::to_string(id) + "/dt-power/1/weight/" + std::to_string(weight.numerator) + "/" +
        std::to_string(weight.denominator) + "/stage/0/1";
  }
  auto query() const {
    pops::amr::reflux::CoarseFaceRefluxKey<2> result;
    result.owner = "tracer";
    result.state = state_identity;
    result.levels = {0, 1};
    result.axis = 0;
    result.coarse_face = pops::Index<2>{4, 4};
    result.attempt = 7;
    result.macro_step = 0;
    return result;
  }
  static constexpr auto budget() { return pops::amr::reflux::MetricRefluxBudget{4, 64, 16}; }

  void materialize(int substep) {
    using namespace pops;
    const bool damage = my_rank() == 0;
    const double duration = damage && substep == 1 &&
        fault == TemporalFluxFault::BadClockDuration ? 0.75 : 0.5;
    context->prepare_spatial_collectively([&] {
    const auto faces = amr::reflux::fine_faces_for_coarse_face(
        query(), amr::RefinementRatio<2>{2, 2}, amr::reflux::FaceRefinementMapping<2>{}, budget());
    Expression expression;
    for (int term = 0; term < 2; ++term) {
      Basis basis;
      basis.identity = term == 0 ? 1 : 3;
      basis.runtime_block = 0;
      basis.level = 1;
      basis.provider = Provider::PreparedDefaultFlux;
      basis.rhs_identity = 2;
      basis.temporal_family = declared ? family : "";
      basis.point = {"clock.macro", 0, 1, substep, 2, {0, 1}, duration, substep * 0.5,
                     "tests.spatial-family/program@1", "tests.spatial-family/rhs@1", "program-rhs"};
      basis.window = {{1, 0, {substep, 2}, substep * 0.5},
                      {1, 0, {substep + 1, 2}, substep * 0.5 + duration}};
      for (std::size_t index = 0; index < faces.size(); ++index) {
        if (damage && fault == TemporalFluxFault::IncompleteFineFaces && index == 1)
          continue;
        basis.faces.push_back({Role::Fine, 0, faces[index], query().coarse_face, 0.5, {Real(5)}});
      }
      if (damage && term == 0) {
        if (fault == TemporalFluxFault::MissingBasisFamily) basis.temporal_family.clear();
        if (fault == TemporalFluxFault::ForeignBasisFamily) basis.temporal_family = "tests.foreign-family";
        if (fault == TemporalFluxFault::ForeignRhs) basis.point.stage = basis.rhs_identity = 19;
        if (fault == TemporalFluxFault::ForeignProvider) basis.provider = Provider::PreparedResidual;
        if (fault == TemporalFluxFault::ForeignBasis) basis.identity = 9;
        if (fault == TemporalFluxFault::ForeignStage) {
          basis.point.stage_fraction = {1, 2};
          basis.point.physical_time += duration * 0.5;
        }
      }
      Rational weight = migrated && substep == 0
          ? (term == 0 ? Rational{5, 4} : Rational{-1, 4}) : coarse_weight(term);
      if (damage && term == 1 && substep == 0 && fault == TemporalFluxFault::WeightsDoNotClose)
        weight = {-1, 8};
      expression.emplace(basis.identity, SpatialAccess::Term{
          std::make_shared<const Basis>(std::move(basis)), {{1, weight}}});
    }
    SpatialAccess::prepare_temporal_expression(*context, candidate, ledger, substep, duration,
                                               std::move(expression));
    if (damage && substep == 0 && fault == TemporalFluxFault::MissingDeclaredProducer)
      SpatialAccess::erase_declared_producer(*context);
    });
    SpatialAccess::materialize(*context, candidate);
  }

  void verify_unchanged_candidate() {
    context->prepare_spatial_collectively([&] {
      for (std::size_t local = 0; local < candidate.local_size(); ++local) {
        const auto host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, candidate.fab(local).storage());
        for (std::size_t element = 0; element < host.extent(0); ++element) {
          EXPECT_TRUE(std::isfinite(static_cast<double>(host(element))));
          EXPECT_EQ(host(element), pops::Real(17));
        }
      }
    });
  }

  auto metric() {
    using Result = pops::amr::reflux::MetricFaceReflux<std::vector<pops::Real>>;
    std::optional<Result> result;
    context->prepare_spatial_collectively([&] {
      auto published = ledger;
      if (published.in_transaction()) published.commit();
      result.emplace(pops::amr::reflux::metric_reflux(
          published, query(), pops::amr::RefinementRatio<2>{2, 2},
          pops::amr::reflux::FaceRefinementMapping<2>{}, budget(),
          [](auto& output, double scale, const auto& input) {
            if (output.empty()) output.resize(input.size(), pops::Real(0));
            if (output.size() != input.size()) throw std::logic_error("fixture payload shape changed");
            for (std::size_t component = 0; component < input.size(); ++component)
              output[component] += static_cast<pops::Real>(scale) * input[component];
          }));
    });
    return std::move(*result);
  }

  void receive_positive() {
    materialize(0);
    materialize(1);
    EXPECT_EQ(ledger.pending_size(), 10u);
    ledger.commit();
    EXPECT_EQ(ledger.published_size(), 10u);
    const auto result = metric();
    ASSERT_EQ(result.coarse_integrated.size(), 1u);
    ASSERT_EQ(result.fine_integrated.size(), 1u);
    ASSERT_EQ(result.mismatch.size(), 1u);
    EXPECT_DOUBLE_EQ(result.coarse_integrated[0], 5);
    EXPECT_DOUBLE_EQ(result.fine_integrated[0], 5);
    EXPECT_DOUBLE_EQ(result.mismatch[0], 0);
    EXPECT_DOUBLE_EQ(result.coarse_weighted_measure, 1);
    EXPECT_DOUBLE_EQ(result.fine_weighted_measure, 1);
    for (const auto& entry : ledger.published_entries(0)) {
      const auto term = entry.key.stage.find("/basis/1/") != std::string::npos ? 0 : 1;
      const auto expected_weight = entry.key.role == Role::Fine &&
          entry.measure.substep_begin == Rational{0, 1} && migrated
          ? (term == 0 ? Rational{5, 4} : Rational{-1, 4}) : coarse_weight(term);
      EXPECT_EQ(entry.measure.stage_weight, expected_weight);
      EXPECT_EQ(entry.key.stage, literal_stage(term == 0 ? 1 : 3, expected_weight));
      EXPECT_EQ(entry.key.temporal_family, declared ? family : "");
    }
    verify_unchanged_candidate();
  }

  void receive_refusal(bool coverage = false) {
    std::string reason;
    try {
      materialize(0);
      if (coverage) {
        if (fault != TemporalFluxFault::IncompleteWindow) materialize(1);
        (void)metric();
      }
    } catch (const std::exception& error) { reason = error.what(); }
    const std::string prefix = pops::n_ranks() > 1
        ? "composite spatial preparation failed collectively; rank 0: " : "";
    std::string cause = "AMR Program coarse/fine flux operator identities differ before face-flux publication";
    if (fault == TemporalFluxFault::MissingBasisFamily ||
        fault == TemporalFluxFault::ForeignBasisFamily ||
        fault == TemporalFluxFault::MissingDeclaredProducer ||
        fault == TemporalFluxFault::ForeignRhs || fault == TemporalFluxFault::ForeignProvider)
      cause = "AMR Program flux basis lacks its declared producer family";
    if (fault == TemporalFluxFault::IncompleteWindow)
      cause = "ND metric reflux substeps do not cover the complete exact clock window";
    if (fault == TemporalFluxFault::IncompleteFineFaces)
      cause = "ND metric reflux has an incomplete fine tangential face product in one clock-stage slice";
    if (fault == TemporalFluxFault::BadClockDuration)
      cause = "ND metric reflux substeps disagree on their physical clock rate";
    if (fault == TemporalFluxFault::WeightsDoNotClose)
      cause = "ND metric reflux stage weights do not close one accepted substep";
    EXPECT_TRUE(reason.starts_with(prefix + cause)) << reason;
    if (coverage || cause == "AMR Program flux basis lacks its declared producer family")
      EXPECT_EQ(reason, prefix + cause);
    EXPECT_TRUE(ledger.in_transaction());
    EXPECT_EQ(ledger.published_size(), 0u);
    ledger.rollback();
    EXPECT_EQ(ledger.pending_size(), 0u);
    EXPECT_EQ(ledger.published_size(), 0u);
    verify_unchanged_candidate();
  }
};
}  // namespace
