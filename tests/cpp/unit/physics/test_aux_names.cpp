#include <gtest/gtest.h>

#include <pops/core/state/aux_names.hpp>
#include <pops/core/state/state.hpp>

#include <cstddef>
#include <string_view>
#include <type_traits>

using namespace pops;

static_assert(aux_canonical_index<1>("phi") == 0);
static_assert(aux_canonical_index<1>("grad_x") == 1);
static_assert(aux_canonical_index<1>("grad_y") == -1);
static_assert(aux_canonical_index<1>("B_z") == 2);
static_assert(aux_canonical_index<1>("T_e") == 3);

static_assert(aux_canonical_index<2>("grad_y") == 2);
static_assert(aux_canonical_index<2>("grad_z") == -1);
static_assert(aux_canonical_index<2>("B_z") == 3);
static_assert(aux_canonical_index<2>("T_e") == 4);

static_assert(aux_canonical_index<3>("grad_z") == 3);
static_assert(aux_canonical_index<3>("B_z") == 4);
static_assert(aux_canonical_index<3>("T_e") == 5);
static_assert(aux_canonical_index<3>("kappa") == -1);

static_assert(kAuxBaseComps == kAuxBaseCompsFor<kNativeDimension>);
static_assert(kAuxNamedBase == kAuxNamedBaseFor<kNativeDimension>);
static_assert(kAuxMaxComps == kAuxMaxCompsFor<kNativeDimension>);
static_assert(std::is_same_v<Aux, AuxState<kNativeDimension>>);

namespace {

template <int Dim>
void check_ranked_names() {
  using layout = AuxComponentLayout<Dim>;
  EXPECT_EQ(kAuxCanonicalNamesFor<Dim>.size(), static_cast<std::size_t>(Dim + 3));
  EXPECT_EQ(aux_canonical_index<Dim>("phi"), layout::phi);
  EXPECT_EQ(aux_canonical_index<Dim>("B_z"), layout::b_z);
  EXPECT_EQ(aux_canonical_index<Dim>("T_e"), layout::t_e);
  EXPECT_EQ(aux_canonical_name<Dim>(layout::phi), "phi");
  EXPECT_EQ(aux_canonical_name<Dim>(layout::b_z), "B_z");
  EXPECT_EQ(aux_canonical_name<Dim>(layout::t_e), "T_e");
  EXPECT_EQ(aux_canonical_name<Dim>(layout::named_begin), std::string_view{});
}

}  // namespace

TEST(AuxNames, CanonicalIndicesFollowRankWithoutPhantomAxes) {
  check_ranked_names<1>();
  check_ranked_names<2>();
  check_ranked_names<3>();
  EXPECT_EQ(aux_canonical_index<1>("grad_y"), -1);
  EXPECT_EQ(aux_canonical_index<2>("grad_z"), -1);
}

TEST(AuxNames, NativeAliasesResolveTheBuildSpecialization) {
  EXPECT_EQ(aux_canonical_index("B_z"), kAuxBzComponent);
  EXPECT_EQ(aux_canonical_index("T_e"), kAuxTeComponent);
  EXPECT_EQ(aux_canonical_name(kAuxNamedBase), std::string_view{});
}
