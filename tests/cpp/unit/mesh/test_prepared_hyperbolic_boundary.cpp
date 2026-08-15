#include <gtest/gtest.h>

#include <pops/mesh/boundary/prepared_hyperbolic_boundary.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

using namespace pops;

namespace {

template <int Dim>
Extent<Dim> uniform_extent(int value) {
  Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim, class Function>
void for_each_host_index(const Box<Dim>& box, Function&& function) {
  for (std::int64_t linear = 0; linear < box.numPts(); ++linear) {
    std::int64_t remaining = linear;
    Index<Dim> index{};
    for (int axis = 0; axis < Dim; ++axis) {
      index[axis] = box.lo[axis] + static_cast<int>(remaining % box.length(axis));
      remaining /= box.length(axis);
    }
    function(index);
  }
}

template <int Dim>
std::size_t host_offset(const Box<Dim>& storage, const Index<Dim>& index, int component) {
  std::int64_t linear = 0;
  std::int64_t stride = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    linear += static_cast<std::int64_t>(index[axis] - storage.lo[axis]) * stride;
    stride *= storage.length(axis);
  }
  return static_cast<std::size_t>(component * storage.numPts() + linear);
}

template <int Dim>
MultiFab<Dim> one_patch_field(const Box<Dim>& domain, int ncomp, Extent<Dim> ghosts) {
  const mesh::BoxArray<Dim> layout(std::vector<Box<Dim>>{domain});
  const mesh::RankSpace<Dim> ranks{Index<Dim>{}, uniform_extent<Dim>(1)};
  const auto distribution = mesh::Distribution<Dim>::replicated(layout, ranks);
  return MultiFab<Dim>(layout, distribution, Index<Dim>{}, ncomp, ghosts);
}

template <int Dim, class Function>
void fill_valid(MultiFab<Dim>& state, Real sentinel, Function&& value) {
  state.set_val(sentinel);
  Fab<Dim>& fab = state.fab(0);
  auto host = fab.create_host_mirror();
  fab.copy_to_host(host);
  for_each_host_index(fab.box(), [&](const Index<Dim>& index) {
    for (int component = 0; component < state.ncomp(); ++component)
      host(host_offset(fab.grown_box(), index, component)) = value(index, component);
  });
  fab.copy_from_host(host);
}

template <int Dim>
Real value_at(const MultiFab<Dim>& state, const Index<Dim>& index, int component) {
  const Fab<Dim>& fab = state.fab(0);
  auto host = fab.create_host_mirror();
  fab.copy_to_host(host);
  return host(host_offset(fab.grown_box(), index, component));
}

template <int Dim>
std::vector<std::string> identities() {
  std::vector<std::string> result;
  result.reserve(static_cast<std::size_t>(2 * Dim));
  for (int face = 0; face < 2 * Dim; ++face)
    result.push_back("face:" + std::to_string(face));
  return result;
}

template <int Dim>
void check_periodic_axes() {
  const std::vector<std::string> laws(static_cast<std::size_t>(2 * Dim), "periodic");
  const auto boundary = prepare_hyperbolic_boundary<Dim>(
      laws, std::vector<double>(static_cast<std::size_t>(2 * Dim), 0.0), identities<Dim>(),
      {"Scalar"});
  const auto periodic = boundary.periodic_axes();
  for (int axis = 0; axis < Dim; ++axis)
    EXPECT_TRUE(periodic[static_cast<std::size_t>(axis)]);
}

template <int Dim>
struct QualifiedCharacteristicModel {
  using State = std::array<Real, 1>;
  static constexpr int dimension = Dim;
  static constexpr int n_vars = 1;
  static constexpr int characteristic_no_inflow_contract_version = 1;
  static constexpr int characteristic_no_inflow_dimension = Dim;
  static constexpr int characteristic_no_inflow_components = n_vars;
  static constexpr bool characteristic_no_inflow_conservative = true;

  int expected_axis = 0;
  int expected_side = -1;
  bool reject = false;
  bool nonfinite_output = false;

  POPS_HD bool characteristic_no_inflow(const State& interior, const State& reference, int axis,
                                        int outward_sign, State& ghost) const {
    if (axis != expected_axis || outward_sign != expected_side || reject)
      return false;
    ghost[0] = nonfinite_output ? std::numeric_limits<Real>::quiet_NaN()
                                : interior[0] + reference[0] + Real(100 * axis + 10 * outward_sign);
    return true;
  }
};

template <int Dim>
struct UnqualifiedCharacteristicModel {
  using State = std::array<Real, 1>;
  static constexpr int dimension = Dim;
  static constexpr int n_vars = 1;
};

template <int Dim>
struct IncompatibleCharacteristicModel {
  using State = std::array<Real, 2>;
  static constexpr int dimension = Dim;
  static constexpr int n_vars = 2;
  static constexpr int characteristic_no_inflow_contract_version = 1;
  static constexpr int characteristic_no_inflow_dimension = Dim;
  static constexpr int characteristic_no_inflow_components = n_vars;
  static constexpr bool characteristic_no_inflow_conservative = true;

  POPS_HD bool characteristic_no_inflow(const State&, const State&, int, int, State&) const {
    return true;
  }
};

template <int Dim>
void check_characteristic_axis_side_permutations() {
  for (int normal_axis = 0; normal_axis < Dim; ++normal_axis) {
    for (const int side : {-1, 1}) {
      std::vector<std::string> laws(static_cast<std::size_t>(2 * Dim), "foextrap");
      const int face = 2 * normal_axis + (side > 0 ? 1 : 0);
      laws[static_cast<std::size_t>(face)] = "characteristic_no_inflow";
      std::vector<double> values(static_cast<std::size_t>(2 * Dim), 0.0);
      values[static_cast<std::size_t>(face)] = Real(5 + face);
      const auto boundary =
          prepare_hyperbolic_boundary<Dim>(laws, values, identities<Dim>(), {"scalar"});
      const Box<Dim> domain = Box<Dim>::from_extents(uniform_extent<Dim>(3));
      auto state = one_patch_field(domain, 1, uniform_extent<Dim>(1));
      fill_valid(state, Real(-31), [](const Index<Dim>& index, int) {
        Real value = Real(1);
        for (int axis = 0; axis < Dim; ++axis)
          value += Real((axis + 1) * (index[axis] + 1));
        return value;
      });

      QualifiedCharacteristicModel<Dim> model{normal_axis, side, false, false};
      boundary.fill_physical_model_qualified(state, domain, model);
      Index<Dim> interior{};
      Index<Dim> ghost{};
      interior[normal_axis] = side < 0 ? domain.lo[normal_axis] : domain.hi[normal_axis];
      ghost[normal_axis] = interior[normal_axis] + side;
      const Real expected =
          value_at(state, interior, 0) + Real(5 + face + 100 * normal_axis + 10 * side);
      EXPECT_EQ(value_at(state, ghost, 0), expected)
          << "Dim=" << Dim << " axis=" << normal_axis << " side=" << side;
    }
  }
}

}  // namespace

TEST(test_prepared_hyperbolic_boundary, periodic_topology_is_compile_time_ranked) {
  check_periodic_axes<1>();
  check_periodic_axes<2>();
  check_periodic_axes<3>();
}

TEST(test_prepared_hyperbolic_boundary, one_dimensional_fixed_and_extrapolated_halos_are_exact) {
  const Box<1> domain{Index<1>{0}, Index<1>{2}};
  auto state = one_patch_field(domain, 1, Extent<1>{2});
  fill_valid(state, Real(-99), [](const Index<1>& index, int) { return Real(index[0] + 1); });
  const auto boundary = prepare_hyperbolic_boundary<1>({"dirichlet", "foextrap"}, {5.0, 0.0},
                                                       identities<1>(), {"Scalar"});
  boundary.fill_physical(state, domain);

  EXPECT_EQ(value_at(state, Index<1>{-1}, 0), Real(9));
  EXPECT_EQ(value_at(state, Index<1>{-2}, 0), Real(8));
  EXPECT_EQ(value_at(state, Index<1>{3}, 0), Real(3));
  EXPECT_EQ(value_at(state, Index<1>{4}, 0), Real(3));
}

TEST(test_prepared_hyperbolic_boundary, three_dimensional_slip_uses_axis_static_component_parity) {
  const Box<3> domain = Box<3>::from_extents(Extent<3>{2, 2, 2});
  auto state = one_patch_field(domain, 5, Extent<3>{1, 1, 1});
  fill_valid(state, Real(-99), [](const Index<3>&, int component) { return Real(component + 1); });
  const auto boundary = prepare_hyperbolic_boundary<3>(
      std::vector<std::string>(6, "slip_wall"), std::vector<double>(30, 0.0), identities<3>(),
      {"density", "momentum:0", "momentum:1", "momentum:2", "energy"});
  boundary.fill_physical(state, domain);

  EXPECT_EQ(value_at(state, Index<3>{-1, 0, 0}, 0), Real(1));
  EXPECT_EQ(value_at(state, Index<3>{-1, 0, 0}, 1), Real(-2));
  EXPECT_EQ(value_at(state, Index<3>{-1, 0, 0}, 2), Real(3));
  EXPECT_EQ(value_at(state, Index<3>{0, 2, 0}, 1), Real(2));
  EXPECT_EQ(value_at(state, Index<3>{0, 2, 0}, 2), Real(-3));
  EXPECT_EQ(value_at(state, Index<3>{0, 0, -1}, 3), Real(-4));
  EXPECT_EQ(value_at(state, Index<3>{-1, -1, 0}, 0), Real(-99));
}

TEST(test_prepared_hyperbolic_boundary, no_flux_is_enforced_on_the_post_riemann_face_field) {
  const Box<2> domain = Box<2>::from_extents(Extent<2>{2, 2});
  const auto boundary =
      prepare_hyperbolic_boundary<2>({"no_flux", "foextrap", "foextrap", "foextrap"},
                                     {0.0, 0.0, 0.0, 0.0}, identities<2>(), {"Scalar"});
  nd::FaceField<2> fluxes(domain, 1);
  fluxes.set_val(Real(4));
  boundary.apply_physical_flux_conditions(fluxes, domain);

  const Fab<2>& x_faces = fluxes.field<0>();
  auto x_host = x_faces.create_host_mirror();
  x_faces.copy_to_host(x_host);
  EXPECT_EQ(x_host(host_offset(x_faces.grown_box(), Index<2>{0, 0}, 0)), Real(0));
  EXPECT_EQ(x_host(host_offset(x_faces.grown_box(), Index<2>{1, 0}, 0)), Real(4));
  EXPECT_EQ(x_host(host_offset(x_faces.grown_box(), Index<2>{2, 0}, 0)), Real(4));

  const Fab<2>& y_faces = fluxes.field<1>();
  auto y_host = y_faces.create_host_mirror();
  y_faces.copy_to_host(y_host);
  EXPECT_EQ(y_host(host_offset(y_faces.grown_box(), Index<2>{0, 0}, 0)), Real(4));
}

TEST(test_prepared_hyperbolic_boundary,
     physical_fill_preflight_token_commits_one_complete_builtin_transaction) {
  const Box<1> domain{Index<1>{0}, Index<1>{2}};
  auto state = one_patch_field(domain, 1, Extent<1>{1});
  fill_valid(state, Real(-19), [](const Index<1>& index, int) { return Real(index[0] + 1); });
  const auto boundary = prepare_hyperbolic_boundary<1>({"dirichlet", "foextrap"}, {5.0, 0.0},
                                                       identities<1>(), {"Scalar"});
  auto preflight = boundary.preflight_physical(state, domain);
  EXPECT_EQ(value_at(state, Index<1>{-1}, 0), Real(-19));
  boundary.fill_physical_preflighted(state, std::move(preflight));
  EXPECT_EQ(value_at(state, Index<1>{-1}, 0), Real(9));
  EXPECT_EQ(value_at(state, Index<1>{3}, 0), Real(3));
}

TEST(test_prepared_hyperbolic_boundary,
     analytic_boundary_without_nd_coordinate_provider_is_refused) {
  EXPECT_THROW((void)prepare_hyperbolic_boundary<2>(
                   std::vector<std::string>(4, "foextrap"), std::vector<double>(4, 0.0),
                   identities<2>(), {"Scalar"}, false, {}, {}, {{"literal"}}, {{1.0}}, {""}),
               std::invalid_argument);
}

TEST(test_prepared_hyperbolic_boundary,
     characteristic_boundary_requires_model_qualification_before_physical_mutation) {
  const auto characteristic = prepare_hyperbolic_boundary<2>(
      {"characteristic_no_inflow", "foextrap", "foextrap", "foextrap"}, {1.0, 0.0, 0.0, 0.0},
      identities<2>(), {"Scalar"});
  const Box<2> domain = Box<2>::from_extents(Extent<2>{2, 2});
  auto state = one_patch_field(domain, 1, Extent<2>{1, 1});
  fill_valid(state, Real(-17), [](const Index<2>&, int) { return Real(3); });
  EXPECT_THROW(characteristic.preflight_physical(state, domain), std::logic_error);
  EXPECT_EQ(value_at(state, Index<2>{-1, 0}, 0), Real(-17));
  EXPECT_EQ(value_at(state, Index<2>{0, 0}, 0), Real(3));
}

TEST(test_prepared_hyperbolic_boundary, characteristic_metadata_preserves_the_compile_time_rank) {
  const auto characteristic = prepare_hyperbolic_boundary<3>(
      {"characteristic_no_inflow", "foextrap", "foextrap", "foextrap", "foextrap", "foextrap"},
      {1.0, 0.0, 0.0, 0.0, 0.0, 0.0}, identities<3>(), {"Scalar"});
  EXPECT_TRUE(characteristic.has_characteristic_no_inflow());
  EXPECT_EQ(characteristic.face(0, -1).law, HyperbolicBoundaryLaw::CharacteristicNoInflow);
  EXPECT_EQ(characteristic.face(2, 1).law, HyperbolicBoundaryLaw::Extrapolate);
}

TEST(test_prepared_hyperbolic_boundary,
     generated_characteristic_provider_executes_on_permuted_axes_in_all_ranks) {
  check_characteristic_axis_side_permutations<1>();
  check_characteristic_axis_side_permutations<2>();
  check_characteristic_axis_side_permutations<3>();
}

TEST(test_prepared_hyperbolic_boundary,
     characteristic_provider_absence_rejection_and_nonfinite_output_fail_before_publication) {
  const auto boundary = prepare_hyperbolic_boundary<2>(
      {"foextrap", "foextrap", "characteristic_no_inflow", "foextrap"}, {0.0, 0.0, 4.0, 0.0},
      identities<2>(), {"scalar"});
  const Box<2> domain = Box<2>::from_extents(Extent<2>{3, 3});
  auto state = one_patch_field(domain, 1, Extent<2>{1, 1});
  fill_valid(state, Real(-29), [](const Index<2>&, int) { return Real(2); });

  EXPECT_THROW(
      boundary.fill_physical_model_qualified(state, domain, UnqualifiedCharacteristicModel<2>{}),
      std::logic_error);
  EXPECT_EQ(value_at(state, Index<2>{0, -1}, 0), Real(-29));
  EXPECT_EQ(value_at(state, Index<2>{0, 0}, 0), Real(2));

  EXPECT_THROW(
      boundary.fill_physical_model_qualified(state, domain, IncompatibleCharacteristicModel<2>{}),
      std::invalid_argument);
  EXPECT_EQ(value_at(state, Index<2>{0, -1}, 0), Real(-29));
  EXPECT_EQ(value_at(state, Index<2>{-1, 0}, 0), Real(-29));
  EXPECT_EQ(value_at(state, Index<2>{0, 0}, 0), Real(2));

  EXPECT_THROW(boundary.fill_physical_model_qualified(
                   state, domain, QualifiedCharacteristicModel<2>{1, -1, false, true}),
               std::runtime_error);
  EXPECT_EQ(value_at(state, Index<2>{0, -1}, 0), Real(-29));
  EXPECT_EQ(value_at(state, Index<2>{-1, 0}, 0), Real(-29));
  EXPECT_EQ(value_at(state, Index<2>{0, 0}, 0), Real(2));

  EXPECT_THROW(boundary.fill_physical_model_qualified(
                   state, domain, QualifiedCharacteristicModel<2>{1, -1, true, false}),
               std::runtime_error);
  EXPECT_EQ(value_at(state, Index<2>{0, -1}, 0), Real(-29));
  EXPECT_EQ(value_at(state, Index<2>{-1, 0}, 0), Real(-29));
  EXPECT_EQ(value_at(state, Index<2>{0, 0}, 0), Real(2));
}

TEST(test_prepared_hyperbolic_boundary,
     characteristic_rank_local_rejection_rolls_back_collectively_and_retry_publishes) {
  constexpr int Dim = 1;
  auto lane = ExecutionLane::duplicate_world_collectively("test/characteristic-boundary-mpi");
  if (lane.size() != 2)
    GTEST_SKIP() << "requires exactly two MPI ranks";

  const Box<Dim> domain = Box<Dim>::from_extents(Extent<Dim>{3});
  const mesh::BoxArray<Dim> layout(std::vector<Box<Dim>>{domain});
  const mesh::RankSpace<Dim> ranks{Index<Dim>{}, Extent<Dim>{lane.size()}};
  const auto distribution = mesh::Distribution<Dim>::replicated(layout, ranks);
  MultiFab<Dim> state(layout, distribution, Index<Dim>{lane.rank()}, 1, Extent<Dim>{1});
  fill_valid(state, Real(-29), [](const Index<Dim>&, int) { return Real(3); });
  const auto boundary = prepare_hyperbolic_boundary<Dim>({"characteristic_no_inflow", "foextrap"},
                                                         {4.0, 0.0}, identities<Dim>(), {"scalar"});

  EXPECT_THROW(boundary.fill_physical_model_qualified(
                   state, domain, QualifiedCharacteristicModel<Dim>{0, -1, false, false}),
               std::logic_error);
  EXPECT_EQ(value_at(state, Index<Dim>{-1}, 0), Real(-29));

  EXPECT_THROW(
      boundary.fill_physical_model_qualified(
          state, domain, QualifiedCharacteristicModel<Dim>{0, -1, lane.rank() == 0, false}, lane),
      std::runtime_error);
  EXPECT_EQ(value_at(state, Index<Dim>{-1}, 0), Real(-29));
  EXPECT_EQ(value_at(state, Index<Dim>{3}, 0), Real(-29));

  boundary.fill_physical_model_qualified(
      state, domain, QualifiedCharacteristicModel<Dim>{0, -1, false, false}, lane);
  EXPECT_EQ(value_at(state, Index<Dim>{-1}, 0), Real(-3));
  EXPECT_EQ(value_at(state, Index<Dim>{3}, 0), Real(3));
}
