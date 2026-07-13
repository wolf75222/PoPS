#include <pops/runtime/config/component_interfaces.hpp>

#include <gtest/gtest.h>

#include <string>
#include <utility>
#include <vector>

namespace {

struct Context {};

struct FluxComponent {
  std::vector<std::string> requirements() const { return {"state", "normal"}; }
  double stability() const { return 1.0; }
  pops::component::EvaluationOutcome<double> evaluate(Context&) const {
    return pops::component::EvaluationOutcome<double>::ok(2.0);
  }
};

struct BoundaryComponent {
  std::vector<std::string> providers() const { return {"state", "logical_time"}; }
  int stencil() const { return 1; }
};

struct TaggerComponent {
  std::vector<std::string> requirements() const { return {"indicator"}; }
  std::string lower(Context&) const { return "tagger-plan"; }
};

struct ClusteringComponent {
  std::string lower(Context&) const { return "cluster-plan"; }
  std::vector<std::string> effects() const { return {"topology"}; }
};

struct TransferComponent {
  int stencil() const { return 2; }
  std::string restart() const { return "stateless"; }
};

struct RefluxComponent {
  std::vector<std::string> effects() const { return {"conservative-correction"}; }
  std::string report() const { return "reflux-report"; }
};

struct SolverComponent {
  pops::component::EvaluationOutcome<int> evaluate(Context&) const {
    return pops::component::EvaluationOutcome<int>::reject("non-converged");
  }
  std::string restart() const { return "warm-start"; }
  std::string report() const { return "solve-report"; }
};

struct WriterComponent {
  std::vector<std::string> effects() const { return {"io"}; }
  std::string format(const double& value) const { return std::to_string(value); }
  std::string report() const { return "writer-report"; }
};

static_assert(pops::component::Requirement<FluxComponent>);
static_assert(pops::component::Stability<FluxComponent>);
static_assert(pops::component::FallibleEvaluation<FluxComponent, Context&>);
static_assert(pops::component::Provider<BoundaryComponent>);
static_assert(pops::component::Stencil<BoundaryComponent>);
static_assert(pops::component::Requirement<TaggerComponent>);
static_assert(pops::component::Lowering<TaggerComponent, Context>);
static_assert(pops::component::Lowering<ClusteringComponent, Context>);
static_assert(pops::component::Effects<ClusteringComponent>);
static_assert(pops::component::Stencil<TransferComponent>);
static_assert(pops::component::Restart<TransferComponent>);
static_assert(pops::component::Effects<RefluxComponent>);
static_assert(pops::component::Report<RefluxComponent>);
static_assert(pops::component::FallibleEvaluation<SolverComponent, Context&>);
static_assert(pops::component::Restart<SolverComponent>);
static_assert(pops::component::Format<WriterComponent, double>);
static_assert(pops::component::Report<WriterComponent>);

pops::component::RegistrationRecord record(std::string id, std::string semantic) {
  return {
      std::move(id),
      "test.external",
      {},
      {"external", "pops://external.test/package", std::move(semantic), "manifest-digest"},
  };
}

TEST(ComponentInterfaces, FallibleOutcomeKeepsTransactionActionExplicit) {
  Context context;
  const auto flux = FluxComponent{}.evaluate(context);
  EXPECT_EQ(flux.status, pops::component::EvaluationStatus::kOk);
  ASSERT_TRUE(flux.value.has_value());
  EXPECT_EQ(*flux.value, 2.0);

  const auto solve = SolverComponent{}.evaluate(context);
  EXPECT_EQ(solve.status, pops::component::EvaluationStatus::kReject);
  EXPECT_EQ(solve.reason, "non-converged");
  EXPECT_THROW(pops::component::EvaluationOutcome<int>::retry(""), std::invalid_argument);
}

TEST(ComponentInterfaces, RegistryIsCollisionSafeIdempotentAndExplicitlyFrozen) {
  pops::component::Registry registry;
  const auto& first = registry.register_component(record("pops://external.test/flux@1.0.0", "s1"));
  EXPECT_EQ(first.component_type, "test.external");
  EXPECT_EQ(registry.revision(), 1u);

  const auto& repeated =
      registry.register_component(record("pops://external.test/flux@1.0.0", "s1"));
  EXPECT_EQ(&first, &repeated);
  EXPECT_EQ(registry.revision(), 1u);

  EXPECT_THROW(
      registry.register_component(record("pops://external.test/flux@1.0.0", "different")),
      std::invalid_argument);
  EXPECT_EQ(registry.revision(), 1u);

  registry.freeze();
  EXPECT_TRUE(registry.frozen());
  EXPECT_THROW(
      registry.register_component(record("pops://external.test/writer@1.0.0", "s2")),
      std::logic_error);
}

}  // namespace
