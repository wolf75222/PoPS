#include <pops/runtime/program/field_provider_read_authority.hpp>
#include <gtest/gtest.h>

namespace {
using namespace pops::runtime::system;
using pops::runtime::program::field_provider_read_authority;
constexpr int Dim = 2;
AuxiliaryOutput<Dim> output(const char* component) {
  AuxiliaryStorageShape<Dim> shape;
  shape.spatial_rank = Dim;
  shape.value_components = 1;
  AuxiliaryComponentContract contract;
  contract.representation = "cell-average";
  contract.centering = "cell";
  contract.layout = "shared";
  contract.value_kind = "scalar";
  return {{"owner", "auxiliary", "inputs", component}, contract, shape};
}
AuxiliaryDependency<Dim> dependency(const AuxiliaryOutput<Dim>& value) {
  return {value.key, value.contract, value.shape};
}
AuxiliaryEvaluationPoint point(int step) {
  AuxiliaryEvaluationPoint result;
  result.clock = "macro";
  result.accepted_step = step;
  result.event = AuxiliaryEvaluationEvent::before_residual;
  return result;
}
void publish(ExactAuxiliaryRegistry<Dim>& registry, const char* name, int step) {
  auto candidate = registry.begin_external_publication(point(step), {name});
  candidate.stage_external(name);
  candidate.accept();
}
TEST(FieldProviderReadAuthority, ChangedUsedPublicationDiffersButOtherProviderIsIrrelevant) {
  ExactAuxiliaryRegistry<Dim> registry;
  const auto first = output("first"), second = output("second");
  const AuxiliaryEvaluationPolicy policy{AuxiliaryEvaluationEvent::before_residual,
                                         AuxiliaryFreshness::evaluation};
  registry.add({"first", AuxiliaryProviderKind::input, policy, {first}, {}});
  registry.add({"second", AuxiliaryProviderKind::input, policy, {second}, {}});
  registry.add_consumer_plan({"consumer", {{dependency(first), 0}, {dependency(second), 1}}});
  registry.seal();
  publish(registry, "first", 0);
  publish(registry, "second", 0);
  const auto original = field_provider_read_authority(registry, "consumer", 0, 1ULL << 20);
  publish(registry, "second", 1);
  EXPECT_EQ(original, field_provider_read_authority(registry, "consumer", 0, 1ULL << 20));
  publish(registry, "first", 1);
  EXPECT_NE(original, field_provider_read_authority(registry, "consumer", 0, 1ULL << 20));
  EXPECT_THROW(field_provider_read_authority(registry, "consumer", 0, 8), std::length_error);
  EXPECT_THROW(field_provider_read_authority(registry, "consumer", 2, 1ULL << 20),
               std::invalid_argument);
}
TEST(FieldProviderReadAuthority, PointProjectionMustBeCertifiedByCompiledProducer) {
  using namespace pops::runtime::program::provider_read_detail;
  using R = AuxiliaryPointReads;
  auto first = point(2);
  first.stage = 7;
  first.nonlinear_iteration = 8;
  pops::runtime::multiblock::BoundaryEvaluationPoint native;
  native.clock = "macro";
  native.tick = 2;
  native.stage = 7;
  native.stage_fraction = {1, 2};
  native.dt = .25;
  native.physical_time = .5;
  first.qualify_physical_evaluation(native);
  auto second = first;
  second.stage = 19;
  second.nonlinear_iteration = 20;
  // Requalify the immutable physical payload against the new consumer stage.
  native.stage = 19;
  second.qualify_physical_evaluation(native);
  auto encode = [](const auto& value, R reads) {
    std::string out;
    Writer writer{&out};
    point(writer, value, reads);
    return out;
  };
  const R analytic{R::clock | R::physical_time};
  EXPECT_EQ(encode(first, analytic), encode(second, analytic));
  EXPECT_NE(encode(first, R{}), encode(second, R{}));
  second.physical_evaluation->physical_time = .75;
  EXPECT_NE(encode(first, analytic), encode(second, analytic));
  EXPECT_THROW((PreparedAuxiliaryProvider<Dim>{
                   "external",
                   AuxiliaryProviderKind::input,
                   {AuxiliaryEvaluationEvent::before_residual, AuxiliaryFreshness::evaluation},
                   {output("first")},
                   {},
                   std::nullopt,
                   R{0}}),
               std::invalid_argument);
}
TEST(FieldProviderReadAuthority, DerivedReadClosesItsActualDependencyPublication) {
  using Provider = PreparedAuxiliaryProvider<Dim>;
  const auto input = output("input"), derived = output("derived");
  ExactAuxiliaryRegistry<Dim> registry;
  const AuxiliaryEvaluationPolicy policy{AuxiliaryEvaluationEvent::before_residual,
                                         AuxiliaryFreshness::evaluation};
  registry.add({"input",
                AuxiliaryProviderKind::input,
                {AuxiliaryEvaluationEvent::before_residual, AuxiliaryFreshness::once},
                {input},
                {}});
  registry.add({"derived",
                AuxiliaryProviderKind::derived,
                policy,
                {derived},
                {dependency(input)},
                Provider::launcher_type::trusted_extension(
                    {"test.closed-dependency", 1}, "test.closed-dependency",
                    [](const AuxiliaryKernelLaunchContext<Dim>&) {}),
                AuxiliaryPointReads{0}});
  registry.add_consumer_plan({"consumer", {{dependency(derived), 0}}});
  registry.seal();
  publish(registry, "input", 0);
  {
    auto candidate = registry.begin_publication(point(0));
    candidate.launch_ready_native();
    candidate.accept();
  }
  const auto original = field_provider_read_authority(registry, "consumer", 0, 1ULL << 20);
  {
    auto candidate = registry.begin_publication(point(1));
    candidate.launch_ready_native();
    candidate.accept();
  }
  EXPECT_EQ(original, field_provider_read_authority(registry, "consumer", 0, 1ULL << 20));
  publish(registry, "input", 2);
  {
    auto candidate = registry.begin_publication(point(2));
    candidate.launch_ready_native();
    candidate.accept();
  }
  EXPECT_NE(original, field_provider_read_authority(registry, "consumer", 0, 1ULL << 20));
}
}  // namespace
