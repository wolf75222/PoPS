// ADC-588: typed FieldContext + AuxLayout. These are HOST-ONLY descriptors WRAPPING the fixed
// aux-component truth of core/state/state.hpp -- they add no numerics. This suite pins:
//  - handle -> real aux component lookup (and that the base contract lands on components 0..2);
//  - a mistyped output fails loud with a message naming the known outputs;
//  - a FieldContext solved for (problem, block, stage) refuses to be read as another triple;
//  - duplicate handles / duplicate components / out-of-range components are rejected.
#include <gtest/gtest.h>

#include <string>

#include <pops/core/state/state.hpp>  // kAuxBaseComps / kAuxNamedBase / kAuxMaxComps
#include <pops/runtime/context/aux_layout.hpp>
#include <pops/runtime/context/field_context.hpp>

namespace {

TEST(AuxLayout, DefaultPoissonLayoutMirrorsBaseContract) {
  const pops::AuxLayout layout = pops::default_poisson_layout();
  // The default layout is the historical base contract: phi=0, grad_x=1, grad_y=2.
  EXPECT_EQ(layout.component_of("phi"), 0);
  EXPECT_EQ(layout.component_of("grad_x"), 1);
  EXPECT_EQ(layout.component_of("grad_y"), 2);
  EXPECT_EQ(layout.base_width(), pops::kAuxBaseComps);
  EXPECT_EQ(layout.width(), pops::kAuxBaseComps) << "no named channel beyond the base contract";
  // find() is the non-throwing report path.
  const pops::AuxChannel* phi = layout.find("phi");
  ASSERT_NE(phi, nullptr);
  EXPECT_EQ(phi->role, pops::FieldChannelRole::kPotential);
  EXPECT_EQ(layout.find("E_x"), nullptr) << "unknown handle -> nullptr, no throw";
}

TEST(AuxLayout, NamedChannelLandsAtCanonicalOrModelComponent) {
  pops::AuxLayout layout = pops::default_poisson_layout();
  // A canonical extra (B_z at component 3) and a model-named field at kAuxNamedBase.
  layout.add_channel("B_z", 3, pops::FieldChannelRole::kNamed);
  layout.add_channel("psi", pops::kAuxNamedBase, pops::FieldChannelRole::kNamed);
  EXPECT_EQ(layout.component_of("B_z"), 3);
  EXPECT_EQ(layout.component_of("psi"), pops::kAuxNamedBase);
  // width() is one past the highest bound component.
  EXPECT_EQ(layout.width(), pops::kAuxNamedBase + 1);
}

TEST(AuxLayout, UnknownOutputThrowsNamingKnownHandles) {
  const pops::AuxLayout layout = pops::default_poisson_layout();
  try {
    layout.component_of("E", "phi");
    FAIL() << "expected component_of to throw on unknown handle";
  } catch (const std::out_of_range& e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("'E'"), std::string::npos) << "names the missing handle";
    EXPECT_NE(msg.find("phi"), std::string::npos) << "lists known outputs / problem id";
  }
}

TEST(AuxLayout, RejectsDuplicateAndOutOfRangeChannels) {
  pops::AuxLayout layout = pops::default_poisson_layout();
  EXPECT_THROW(layout.add_channel("phi", 6, pops::FieldChannelRole::kNamed), std::invalid_argument)
      << "duplicate handle";
  EXPECT_THROW(layout.add_channel("alias", 0, pops::FieldChannelRole::kNamed),
               std::invalid_argument)
      << "component 0 already bound to phi";
  EXPECT_THROW(layout.add_channel("oob", pops::kAuxMaxComps, pops::FieldChannelRole::kNamed),
               std::out_of_range)
      << "component past kAuxMaxComps";
  EXPECT_THROW(layout.add_channel("neg", -1, pops::FieldChannelRole::kNamed), std::out_of_range);
}

TEST(FieldContext, MatchesRejectsWrongProviderOwnerOrStage) {
  const pops::AuxLayout layout = pops::default_poisson_layout();
  pops::FieldContext ctx;
  ctx.provider_identity = "case/field/provider-pack/2";
  ctx.owner_identity = "case/block/material";
  ctx.stage_id = 3;
  ctx.layout = &layout;
  EXPECT_TRUE(ctx.matches("case/field/provider-pack/2", "case/block/material", 3));
  EXPECT_FALSE(ctx.matches("case/field/provider-pack/2", "case/block/material", 4));
  EXPECT_FALSE(ctx.matches("case/field/provider-pack/2", "case/block/other", 3));
  EXPECT_FALSE(ctx.matches("case/field/provider-pack/5", "case/block/material", 3));
  EXPECT_FALSE(ctx.matches("", "case/block/material", 3)) << "there is no default wildcard";
}

TEST(FieldContext, ResolvesOutputThroughLayoutAndFailsWithoutOne) {
  const pops::AuxLayout layout = pops::default_poisson_layout();
  pops::FieldContext ctx;
  ctx.provider_identity = "case/field/provider-pack/0";
  ctx.owner_identity = "case/block/material";
  ctx.layout = &layout;
  EXPECT_EQ(ctx.component_of("grad_y"), 2);
  EXPECT_THROW(ctx.component_of("E"), std::out_of_range) << "unknown output still fails loud";

  pops::FieldContext bare;  // no layout bound
  EXPECT_THROW(bare.component_of("phi"), std::logic_error);
}

}  // namespace
