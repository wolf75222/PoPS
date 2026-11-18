#include <gtest/gtest.h>

#include <pops/mesh/boundary/prepared_hyperbolic_boundary.hpp>
#include <pops/mesh/geometry/geometry.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <cstring>
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

template <int Dim>
MultiFab<Dim> one_patch_field_for_lane(const Box<Dim>& domain, int ncomp, Extent<Dim> ghosts,
                                       const ExecutionLane& lane) {
  const mesh::BoxArray<Dim> layout(std::vector<Box<Dim>>{domain});
  Extent<Dim> rank_extent = uniform_extent<Dim>(1);
  rank_extent[0] = lane.size();
  Index<Dim> local_rank{};
  local_rank[0] = lane.rank();
  const mesh::RankSpace<Dim> ranks{Index<Dim>{}, rank_extent};
  const auto distribution = mesh::Distribution<Dim>::replicated(layout, ranks);
  return MultiFab<Dim>(layout, distribution, local_rank, ncomp, ghosts);
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
  auto lane = ExecutionLane::duplicate_world_collectively("test/characteristic-boundary-nd" +
                                                          std::to_string(Dim));
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
      auto state = one_patch_field_for_lane(domain, 1, uniform_extent<Dim>(1), lane);
      auto candidate = one_patch_field_for_lane(domain, 1, uniform_extent<Dim>(1), lane);
      fill_valid(state, Real(-31), [](const Index<Dim>& index, int) {
        Real value = Real(1);
        for (int axis = 0; axis < Dim; ++axis)
          value += Real((axis + 1) * (index[axis] + 1));
        return value;
      });

      QualifiedCharacteristicModel<Dim> model{normal_axis, side, false, false};
      boundary.fill_physical_model_qualified(state, domain, model, lane, candidate);
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

TEST(test_prepared_hyperbolic_boundary,
     two_dimensional_slip_preserves_out_of_plane_axial_reflection_parity) {
  using hyperbolic_boundary_detail::transform_from_semantic;
  const auto transform = transform_from_semantic<2>(VariableSemantic::axial(2));
  EXPECT_EQ(transform.parity, HyperbolicComponentParity::AxialVector);
  EXPECT_EQ(transform.axis, 2);
  EXPECT_EQ(transform.reflection_sign(0), Real(-1));
  EXPECT_EQ(transform.reflection_sign(1), Real(-1));
  EXPECT_THROW(transform_from_semantic<2>(VariableSemantic::axial(3)), std::invalid_argument);
  EXPECT_THROW(transform_from_semantic<2>(VariableSemantic::momentum(2)), std::invalid_argument);
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

TEST(test_prepared_hyperbolic_boundary, analytic_tables_install_and_fill_through_ranked_geometry) {
  const Box<1> domain{Index<1>{0}, Index<1>{1}};
  auto state = one_patch_field(domain, 1, Extent<1>{1});
  fill_valid(state, Real(-99), [](const Index<1>& index, int) { return Real(index[0] + 1); });
  const auto boundary =
      prepare_hyperbolic_boundary<1>({"dirichlet", "foextrap"}, {0.0, 0.0}, identities<1>(),
                                     {"Scalar"}, false, {}, {}, {{"x"}, {}}, {{0.0}, {}}, {"", ""});
  EXPECT_TRUE(boundary.has_analytic_state());
  EXPECT_THROW(boundary.fill_physical(state, domain), std::logic_error);
  EXPECT_EQ(value_at(state, Index<1>{-1}, 0), Real(-99));

  const auto geometry = Geometry<1>::from_bounds(domain, RealVector<1>{0.0}, RealVector<1>{2.0});
  boundary.fill_physical(state, geometry);
  EXPECT_EQ(value_at(state, Index<1>{-1}, 0), Real(-2));
  EXPECT_EQ(value_at(state, Index<1>{2}, 0), Real(2));
}

TEST(test_prepared_hyperbolic_boundary, analytic_z_coordinate_fills_three_dimensional_halos) {
  const Box<3> domain = Box<3>::from_extents(Extent<3>{2, 2, 2});
  auto state = one_patch_field(domain, 1, Extent<3>{1, 1, 1});
  fill_valid(state, Real(-99), [](const Index<3>&, int) { return Real(4); });
  std::vector<std::vector<std::string>> opcodes(6);
  std::vector<std::vector<double>> literals(6);
  opcodes[4] = {"z"};
  literals[4] = {0.0};
  const auto boundary = prepare_hyperbolic_boundary<3>(
      {"foextrap", "foextrap", "foextrap", "foextrap", "dirichlet", "foextrap"},
      std::vector<double>(6, 0.0), identities<3>(), {"Scalar"}, false, {}, {}, opcodes, literals,
      std::vector<std::string>(6, ""));
  const auto geometry =
      Geometry<3>::from_bounds(domain, RealVector<3>{0.0, 0.0, 0.0}, RealVector<3>{2.0, 2.0, 2.0});
  boundary.fill_physical(state, geometry);
  EXPECT_EQ(value_at(state, Index<3>{0, 0, -1}, 0), Real(-5));
  EXPECT_EQ(value_at(state, Index<3>{0, 0, 2}, 0), Real(4));
}

TEST(test_prepared_hyperbolic_boundary, analytic_domain_only_recovers_prepared_session_geometry) {
  const Box<1> domain{Index<1>{0}, Index<1>{1}};
  auto state = one_patch_field(domain, 1, Extent<1>{1});
  fill_valid(state, Real(-99), [](const Index<1>& index, int) { return Real(index[0] + 1); });
  const auto boundary =
      prepare_hyperbolic_boundary<1>({"dirichlet", "foextrap"}, {0.0, 0.0}, identities<1>(),
                                     {"Scalar"}, false, {}, {}, {{"x"}, {}}, {{0.0}, {}}, {"", ""});
  const auto geometry = Geometry<1>::from_bounds(domain, RealVector<1>{0.0}, RealVector<1>{2.0});
  const auto bound = boundary.with_prepared_geometry(geometry);
  bound.fill_physical(state, domain);
  EXPECT_EQ(value_at(state, Index<1>{-1}, 0), Real(-2));
}

TEST(test_prepared_hyperbolic_boundary, analytic_missing_origin_spacing_refuses_without_zeros) {
  const Box<1> domain{Index<1>{0}, Index<1>{1}};
  auto state = one_patch_field(domain, 1, Extent<1>{1});
  const auto boundary =
      prepare_hyperbolic_boundary<1>({"dirichlet", "foextrap"}, {0.0, 0.0}, identities<1>(),
                                     {"Scalar"}, false, {}, {}, {{"x"}, {}}, {{0.0}, {}}, {"", ""});
  const RealVector<1> missing{std::numeric_limits<Real>::quiet_NaN()};
  const RealVector<1> spacing{Real(1)};
  EXPECT_THROW(boundary.fill_physical(state, domain, missing, spacing), std::logic_error);
  EXPECT_THROW(
      boundary.fill_physical(state, domain, RealVector<1>{Real(0)}, RealVector<1>{Real(0)}),
      std::logic_error);
}

TEST(test_prepared_hyperbolic_boundary, analytic_z_is_refused_below_rank_three) {
  EXPECT_THROW((void)prepare_hyperbolic_boundary<2>(
                   {"dirichlet", "foextrap", "foextrap", "foextrap"}, std::vector<double>(4, 0.0),
                   identities<2>(), {"Scalar"}, false, {}, {}, {{"z"}, {}, {}, {}},
                   {{0.0}, {}, {}, {}}, std::vector<std::string>(4, "")),
               std::invalid_argument);
}

TEST(test_prepared_hyperbolic_boundary, analytic_ghost_depth_cannot_exceed_normal_extent) {
  const Box<1> domain{Index<1>{0}, Index<1>{0}};
  auto state = one_patch_field(domain, 1, Extent<1>{2});
  const auto boundary =
      prepare_hyperbolic_boundary<1>({"dirichlet", "foextrap"}, {0.0, 0.0}, identities<1>(),
                                     {"Scalar"}, false, {}, {}, {{"x"}, {}}, {{0.0}, {}}, {"", ""});
  const auto geometry = Geometry<1>::from_bounds(domain, RealVector<1>{0.0}, RealVector<1>{1.0});
  EXPECT_THROW(boundary.fill_physical(state, geometry), std::invalid_argument);
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
  auto lane = ExecutionLane::duplicate_world_collectively("test/characteristic-boundary-rejection");
  auto state = one_patch_field_for_lane(domain, 1, Extent<2>{1, 1}, lane);
  auto candidate = one_patch_field_for_lane(domain, 1, Extent<2>{1, 1}, lane);
  fill_valid(state, Real(-29), [](const Index<2>&, int) { return Real(2); });

  EXPECT_THROW(boundary.fill_physical_model_qualified(
                   state, domain, UnqualifiedCharacteristicModel<2>{}, lane, candidate),
               std::logic_error);
  EXPECT_EQ(value_at(state, Index<2>{0, -1}, 0), Real(-29));
  EXPECT_EQ(value_at(state, Index<2>{0, 0}, 0), Real(2));

  EXPECT_THROW(boundary.fill_physical_model_qualified(
                   state, domain, IncompatibleCharacteristicModel<2>{}, lane, candidate),
               std::invalid_argument);
  EXPECT_EQ(value_at(state, Index<2>{0, -1}, 0), Real(-29));
  EXPECT_EQ(value_at(state, Index<2>{-1, 0}, 0), Real(-29));
  EXPECT_EQ(value_at(state, Index<2>{0, 0}, 0), Real(2));

  EXPECT_THROW(
      boundary.fill_physical_model_qualified(
          state, domain, QualifiedCharacteristicModel<2>{1, -1, false, true}, lane, candidate),
      std::runtime_error);
  EXPECT_EQ(value_at(state, Index<2>{0, -1}, 0), Real(-29));
  EXPECT_EQ(value_at(state, Index<2>{-1, 0}, 0), Real(-29));
  EXPECT_EQ(value_at(state, Index<2>{0, 0}, 0), Real(2));

  EXPECT_THROW(
      boundary.fill_physical_model_qualified(
          state, domain, QualifiedCharacteristicModel<2>{1, -1, true, false}, lane, candidate),
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
  MultiFab<Dim> candidate(layout, distribution, Index<Dim>{lane.rank()}, 1, Extent<Dim>{1});
  fill_valid(state, Real(-29), [](const Index<Dim>&, int) { return Real(3); });
  const auto boundary = prepare_hyperbolic_boundary<Dim>({"characteristic_no_inflow", "foextrap"},
                                                         {4.0, 0.0}, identities<Dim>(), {"scalar"});

  EXPECT_THROW(boundary.fill_physical_model_qualified(
                   state, domain, QualifiedCharacteristicModel<Dim>{0, -1, false, false}),
               std::logic_error);
  EXPECT_EQ(value_at(state, Index<Dim>{-1}, 0), Real(-29));

  EXPECT_THROW(boundary.fill_physical_model_qualified(
                   state, domain, QualifiedCharacteristicModel<Dim>{0, -1, lane.rank() == 0, false},
                   lane, candidate),
               std::runtime_error);
  EXPECT_EQ(value_at(state, Index<Dim>{-1}, 0), Real(-29));
  EXPECT_EQ(value_at(state, Index<Dim>{3}, 0), Real(-29));

  boundary.fill_physical_model_qualified(
      state, domain, QualifiedCharacteristicModel<Dim>{0, -1, false, false}, lane, candidate);
  EXPECT_EQ(value_at(state, Index<Dim>{-1}, 0), Real(-3));
  EXPECT_EQ(value_at(state, Index<Dim>{3}, 0), Real(3));
}

TEST(test_prepared_hyperbolic_boundary, omitted_interface_faces_require_exact_external_owners) {
  const auto boundary =
      prepare_hyperbolic_boundary<2>({"external", "foextrap", "external", "external"},
                                     {0.0, 0.0, 0.0, 0.0}, identities<2>(), {"Scalar"});
  EXPECT_THROW(boundary.with_omitted_interface_faces({-1}), std::invalid_argument);
  EXPECT_THROW(boundary.with_omitted_interface_faces({4}), std::invalid_argument);
  EXPECT_THROW(boundary.with_omitted_interface_faces({0, 0}), std::invalid_argument);
  EXPECT_THROW(boundary.with_omitted_interface_faces({1}), std::invalid_argument);
  const auto shared = boundary.with_omitted_interface_faces({0, 3});
  EXPECT_EQ(shared.omitted_interface_faces(), (std::array<bool, 4>{true, false, false, true}));
  EXPECT_EQ(boundary.omitted_interface_faces(), (std::array<bool, 4>{}));
  EXPECT_EQ(
      shared
          .with_prepared_geometry(Geometry<2>::from_bounds(
              Box<2>::from_extents(Extent<2>{4, 4}), RealVector<2>{0, 0}, RealVector<2>{1, 1}))
          .omitted_interface_faces(),
      shared.omitted_interface_faces());
}

TEST(test_prepared_hyperbolic_boundary, fixed_state_conversion_retains_interface_omission) {
  const auto boundary = prepare_hyperbolic_boundary<1>(
                            {"external", "dirichlet"}, {0.0, 3.0}, identities<1>(), {"Scalar"},
                            false, {"conservative", "primitive"}, {"", "convert@1"})
                            .with_omitted_interface_faces({0});
  const auto converted = boundary.with_converted_fixed_states(
      [](const double* input, double* output) { output[0] = 2 * input[0]; });
  EXPECT_EQ(converted.face(0, 1).fixed_state.front(), Real(6));
  EXPECT_EQ(converted.omitted_interface_faces(), boundary.omitted_interface_faces());
}

TEST(test_prepared_hyperbolic_boundary, prescribed_trace_faces_require_unique_fixed_state_faces) {
  const auto boundary =
      prepare_hyperbolic_boundary<2>({"dirichlet", "foextrap", "dirichlet", "external"},
                                     {1.0, 0.0, 2.0, 0.0}, identities<2>(), {"Scalar"});
  EXPECT_THROW(boundary.with_prescribed_trace_faces({-1}), std::invalid_argument);
  EXPECT_THROW(boundary.with_prescribed_trace_faces({4}), std::invalid_argument);
  EXPECT_THROW(boundary.with_prescribed_trace_faces({0, 0}), std::invalid_argument);
  EXPECT_THROW(boundary.with_prescribed_trace_faces({1}), std::invalid_argument);
  EXPECT_THROW(boundary.with_prescribed_trace_faces({3}), std::invalid_argument);
  const auto prescribed = boundary.with_prescribed_trace_faces({2, 0});
  EXPECT_EQ(prescribed.prescribed_trace_faces(), (std::array<bool, 4>{true, false, true, false}));
  EXPECT_EQ(boundary.prescribed_trace_faces(), (std::array<bool, 4>{}));
  EXPECT_EQ(prescribed.face(0, -1).law, boundary.face(0, -1).law);
  EXPECT_EQ(
      prescribed
          .with_prepared_geometry(Geometry<2>::from_bounds(
              Box<2>::from_extents(Extent<2>{4, 4}), RealVector<2>{0, 0}, RealVector<2>{1, 1}))
          .prescribed_trace_faces(),
      prescribed.prescribed_trace_faces());
}

TEST(test_prepared_hyperbolic_boundary, fixed_state_conversion_retains_prescribed_trace_faces) {
  const auto boundary = prepare_hyperbolic_boundary<1>(
                            {"foextrap", "dirichlet"}, {0.0, 3.0}, identities<1>(), {"Scalar"},
                            false, {"conservative", "primitive"}, {"", "convert@1"})
                            .with_prescribed_trace_faces({1});
  const auto converted = boundary.with_converted_fixed_states(
      [](const double* input, double* output) { output[0] = 2 * input[0]; });
  EXPECT_EQ(converted.face(0, 1).fixed_state.front(), Real(6));
  EXPECT_EQ(converted.prescribed_trace_faces(), boundary.prescribed_trace_faces());
}

TEST(test_prepared_hyperbolic_boundary, boundary_flux_cannot_replace_a_reserved_interface_face) {
  const auto boundary = prepare_hyperbolic_boundary<1>({"external", "external"}, {0.0, 0.0},
                                                       identities<1>(), {"Scalar"})
                            .with_omitted_interface_faces({0});
  EXPECT_THROW(boundary.require_unreserved_boundary_flux_face(0, -1), std::invalid_argument);
  EXPECT_NO_THROW(boundary.require_unreserved_boundary_flux_face(0, 1));
  EXPECT_THROW(boundary.require_unreserved_boundary_flux_face(1, 1), std::out_of_range);
  EXPECT_THROW(boundary.require_unreserved_boundary_flux_face(0, 0), std::out_of_range);
}

// --- G04-a: the generated ghost policy of a package (SourceGhost faces) ---------------------

namespace {

/// A hand-written generated ghost policy: 2 g - u_k on the lower face of axis 0 (the native
/// FixedState law) and u_1 on its upper face (the native Extrapolate law).
template <int Dim>
struct MirrorGhostPolicy {
  static constexpr bool present = true;
  static constexpr const char* source_identity = "test:mirror";
  static constexpr int width = 1;
  static constexpr int depth = 3;
  static constexpr int reads = 3;
  static constexpr int face_count = 2 * Dim;
  static constexpr bool face_generated(int ordinal) { return ordinal == 0 || ordinal == 1; }
  static constexpr const char* face_identity(int ordinal) {
    return ordinal == 0 ? "ghost:low" : ordinal == 1 ? "ghost:high" : "";
  }

  Real g = Real(0);

  template <int Ordinal, class Inner>
  POPS_HD void value(int k, const Inner& inner, Real* ghost) const {
    if constexpr (Ordinal == 0)
      ghost[0] = Real(2) * g - inner(k, 0);
    else
      ghost[0] = inner(1, 0);
  }
};

/// Two components and a body that no native law has (R4 of the review): every face of axis 0.
template <int Dim>
struct PairGhostPolicy {
  static constexpr bool present = true;
  static constexpr const char* source_identity = "test:pair";
  static constexpr int width = 2;
  static constexpr int depth = 2;
  static constexpr int reads = 2;
  static constexpr int face_count = 2 * Dim;
  static constexpr bool face_generated(int ordinal) { return ordinal == 0 || ordinal == 1; }
  static constexpr const char* face_identity(int ordinal) {
    return ordinal == 0 ? "pair:low" : ordinal == 1 ? "pair:high" : "";
  }

  template <int Ordinal, class Inner>
  POPS_HD void value(int k, const Inner& inner, Real* ghost) const {
    ghost[0] = inner(1, 1) * inner(2, 0) + Real(k) + Real(10 * Ordinal);
    ghost[1] = Real(2) * inner(1, 0) - inner(2, 0);
  }
};

/// The mirror policy on its upper face only, reading one inner cell.
struct HighOnlyMirrorGhostPolicy : MirrorGhostPolicy<1> {
  static constexpr int reads = 1;
  static constexpr bool face_generated(int ordinal) { return ordinal == 1; }
};

/// The mirror policy generated for the depths 1 and 2 only.
struct ShallowMirrorGhostPolicy : MirrorGhostPolicy<1> {
  static constexpr int depth = 2;
  static constexpr int reads = 2;
};

template <int Dim>
std::vector<std::string> mirror_types(const char* low, const char* high, bool periodic_tangents) {
  std::vector<std::string> types(static_cast<std::size_t>(2 * Dim),
                                 periodic_tangents ? "periodic" : "foextrap");
  types[0] = low;
  types[1] = high;
  return types;
}

template <int Dim>
std::vector<std::string> mirror_sources(bool generated) {
  std::vector<std::string> sources(static_cast<std::size_t>(2 * Dim));
  if (generated) {
    sources[0] = "ghost:low";
    sources[1] = "ghost:high";
  }
  return sources;
}

/// Write f(index) into every stored cell of every local patch that lies inside the domain (the
/// valid cells and the halo cells that a halo exchange fills), the sentinel elsewhere.
template <int Dim, class Function>
void fill_inside_domain(MultiFab<Dim>& state, const Box<Dim>& domain, Real sentinel,
                        Function&& value) {
  state.set_val(sentinel);
  for (std::size_t local = 0; local < state.local_size(); ++local) {
    Fab<Dim>& fab = state.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    for_each_host_index(fab.grown_box(), [&](const Index<Dim>& index) {
      if (!domain.contains(index))
        return;
      for (int component = 0; component < state.ncomp(); ++component)
        host(host_offset(fab.grown_box(), index, component)) = value(index, component);
    });
    fab.copy_from_host(host);
  }
}

template <int Dim>
std::vector<Real> stored_values(const MultiFab<Dim>& state) {
  std::vector<Real> result;
  for (std::size_t local = 0; local < state.local_size(); ++local) {
    const Fab<Dim>& fab = state.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    for_each_host_index(fab.grown_box(), [&](const Index<Dim>& index) {
      for (int component = 0; component < state.ncomp(); ++component)
        result.push_back(host(host_offset(fab.grown_box(), index, component)));
    });
  }
  return result;
}

bool same_bits(Real left, Real right) {
  return std::memcmp(&left, &right, sizeof(Real)) == 0;
}

template <int Dim>
Real interior_value(const Index<Dim>& index, int component) {
  Real value = Real(0.3) + Real(component);
  for (int axis = 0; axis < Dim; ++axis)
    value += Real(0.17) * Real(axis + 1) * Real(index[axis]) + Real(0.01) * Real(axis);
  return value;
}

/// The generated 2 g - u_k and u_1 against the native FixedState and Extrapolate laws, bit for
/// bit, at depths 1 to 3, with periodic or physical tangent axes.
template <int Dim>
void check_generated_mirror_matches_native(bool periodic_tangents, Real g) {
  Extent<Dim> extents = uniform_extent<Dim>(3);
  extents[0] = 4;
  const Box<Dim> domain = Box<Dim>::from_extents(extents);
  const auto ghosts = uniform_extent<Dim>(3);
  std::vector<double> values(static_cast<std::size_t>(2 * Dim), 0.0);
  values[0] = g;
  const auto native = prepare_hyperbolic_boundary<Dim>(
      mirror_types<Dim>("dirichlet", "foextrap", periodic_tangents), values, identities<Dim>(),
      {"Scalar"});
  const auto generated = prepare_hyperbolic_boundary<Dim>(
      mirror_types<Dim>("source_ghost", "source_ghost", periodic_tangents),
      std::vector<double>(static_cast<std::size_t>(2 * Dim), 0.0), identities<Dim>(), {"Scalar"},
      false, {}, {}, {}, {}, {}, mirror_sources<Dim>(true));
  auto native_state = one_patch_field(domain, 1, ghosts);
  auto generated_state = one_patch_field(domain, 1, ghosts);
  fill_inside_domain(native_state, domain, Real(-7.25), interior_value<Dim>);
  fill_inside_domain(generated_state, domain, Real(-7.25), interior_value<Dim>);
  QualifiedCharacteristicModel<Dim> model{};
  native.fill_physical(native_state, domain);
  generated.fill_physical_model_qualified(generated_state, domain, model,
                                          MirrorGhostPolicy<Dim>{g});
  const auto expected = stored_values(native_state);
  const auto actual = stored_values(generated_state);
  ASSERT_EQ(expected.size(), actual.size());
  for (std::size_t index = 0; index < expected.size(); ++index)
    EXPECT_TRUE(same_bits(expected[index], actual[index]))
        << "Dim=" << Dim << " periodic=" << periodic_tangents << " index=" << index
        << " native=" << expected[index] << " generated=" << actual[index];
}

}  // namespace

TEST(test_prepared_hyperbolic_boundary,
     generated_mirror_ghosts_equal_the_native_fixed_and_extrapolated_laws_bit_for_bit) {
  for (const Real g : {Real(0.2), Real(0.05), Real(-1.75), Real(0)}) {
    check_generated_mirror_matches_native<1>(false, g);
    check_generated_mirror_matches_native<2>(false, g);
    check_generated_mirror_matches_native<2>(true, g);
    check_generated_mirror_matches_native<3>(false, g);
    check_generated_mirror_matches_native<3>(true, g);
  }
}

// C1 of the review of the note (review/g04/sonde_signe_de_zero.cpp): two inputs where the sign of
// zero differs. The native laws add their offset, which starts at +0; the generated ghost is the
// exact formula of the policy. Each route's value is pinned there; everywhere else the two are
// equal bit for bit.
TEST(test_prepared_hyperbolic_boundary,
     generated_mirror_ghosts_keep_the_sign_of_zero_of_their_formula) {
  const Box<1> domain{Index<1>{0}, Index<1>{2}};
  const auto fill_pair = [&](Real g, Real inner) {
    const auto native = prepare_hyperbolic_boundary<1>({"dirichlet", "foextrap"}, {g, 0.0},
                                                       identities<1>(), {"Scalar"});
    const auto generated = prepare_hyperbolic_boundary<1>(
        {"source_ghost", "source_ghost"}, {0.0, 0.0}, identities<1>(), {"Scalar"}, false, {}, {},
        {}, {}, {}, mirror_sources<1>(true));
    auto native_state = one_patch_field(domain, 1, Extent<1>{1});
    auto generated_state = one_patch_field(domain, 1, Extent<1>{1});
    fill_valid(native_state, Real(-99), [&](const Index<1>&, int) { return inner; });
    fill_valid(generated_state, Real(-99), [&](const Index<1>&, int) { return inner; });
    native.fill_physical(native_state, domain);
    QualifiedCharacteristicModel<1> model{};
    generated.fill_physical_model_qualified(generated_state, domain, model,
                                            MirrorGhostPolicy<1>{g});
    return std::array<Real, 4>{
        value_at(native_state, Index<1>{-1}, 0), value_at(generated_state, Index<1>{-1}, 0),
        value_at(native_state, Index<1>{3}, 0), value_at(generated_state, Index<1>{3}, 0)};
  };
  const Real zero = Real(0);
  const Real negative_zero = -zero;

  // FixedState, g = -0 and u_k = +0: the native law gives +0, the generated 2 g - u_k gives -0.
  const auto fixed = fill_pair(negative_zero, zero);
  EXPECT_EQ(fixed[0], Real(0));
  EXPECT_FALSE(std::signbit(fixed[0]));
  EXPECT_EQ(fixed[1], Real(0));
  EXPECT_TRUE(std::signbit(fixed[1]));
  // Extrapolate, u_1 = -0: the native law gives +0, the generated u_1 keeps -0.
  const auto extrapolated = fill_pair(Real(0.5), negative_zero);
  EXPECT_FALSE(std::signbit(extrapolated[2]));
  EXPECT_TRUE(std::signbit(extrapolated[3]));
  // Everywhere else in the probe grid of the review, the two routes are equal bit for bit.
  for (const Real g : {negative_zero, zero, Real(1), Real(0.3)})
    for (const Real inner : {negative_zero, zero, Real(0.5), Real(0.15)}) {
      const auto values = fill_pair(g, inner);
      const bool fixed_exception =
          std::signbit(g) && g == Real(0) && inner == Real(0) && !std::signbit(inner);
      const bool extrapolated_exception = inner == Real(0) && std::signbit(inner);
      if (!fixed_exception)
        EXPECT_TRUE(same_bits(values[0], values[1])) << "g=" << g << " inner=" << inner;
      if (!extrapolated_exception)
        EXPECT_TRUE(same_bits(values[2], values[3])) << "g=" << g << " inner=" << inner;
    }
}

TEST(test_prepared_hyperbolic_boundary, generated_ghosts_index_every_component_and_depth) {
  const Box<2> domain = Box<2>::from_extents(Extent<2>{3, 2});
  const auto boundary = prepare_hyperbolic_boundary<2>(
      {"source_ghost", "source_ghost", "foextrap", "foextrap"}, std::vector<double>(8, 0.0),
      identities<2>(), {"Scalar", "Scalar"}, false, {}, {}, {}, {}, {},
      {"pair:low", "pair:high", "", ""});
  auto state = one_patch_field(domain, 2, Extent<2>{2, 2});
  fill_valid(state, Real(-99), [](const Index<2>& index, int component) {
    return Real(1 + index[0] + 10 * index[1] + 100 * component);
  });
  QualifiedCharacteristicModel<2> model{};
  boundary.fill_physical_model_qualified(state, domain, model, PairGhostPolicy<2>{});
  for (const int row : {0, 1})
    for (const int k : {1, 2}) {
      const auto u = [&](int i, int component) { return Real(1 + i + 10 * row + 100 * component); };
      // Lower face: u_j at x = j - 1; upper face: u_j at x = 3 - j.
      const Index<2> low{-k, row};
      EXPECT_EQ(value_at(state, low, 0), u(0, 1) * u(1, 0) + Real(k));
      EXPECT_EQ(value_at(state, low, 1), Real(2) * u(0, 0) - u(1, 0));
      const Index<2> high{2 + k, row};
      EXPECT_EQ(value_at(state, high, 0), u(2, 1) * u(1, 0) + Real(k) + Real(10));
      EXPECT_EQ(value_at(state, high, 1), Real(2) * u(2, 0) - u(1, 0));
    }
  // Corners stay unfilled, as with the native laws.
  EXPECT_EQ(value_at(state, Index<2>{-1, -1}, 0), Real(-99));
}

// R5 of the review: two patches along the normal, the boundary patch narrower than the depth. The
// generated pass reads the halo of its patch as the native law does.
TEST(test_prepared_hyperbolic_boundary, generated_ghosts_read_the_patch_halo_as_the_native_law) {
  const Box<1> domain{Index<1>{0}, Index<1>{4}};
  const mesh::BoxArray<1> layout(
      std::vector<Box<1>>{Box<1>{Index<1>{0}, Index<1>{1}}, Box<1>{Index<1>{2}, Index<1>{4}}});
  const mesh::RankSpace<1> ranks{Index<1>{}, uniform_extent<1>(1)};
  const auto distribution = mesh::Distribution<1>::replicated(layout, ranks);
  MultiFab<1> native_state(layout, distribution, Index<1>{}, 1, Extent<1>{3});
  MultiFab<1> generated_state(layout, distribution, Index<1>{}, 1, Extent<1>{3});
  ASSERT_EQ(native_state.local_size(), 2u);
  fill_inside_domain(native_state, domain, Real(-5), interior_value<1>);
  fill_inside_domain(generated_state, domain, Real(-5), interior_value<1>);
  const auto native = prepare_hyperbolic_boundary<1>({"dirichlet", "foextrap"}, {0.2, 0.0},
                                                     identities<1>(), {"Scalar"});
  const auto generated = prepare_hyperbolic_boundary<1>(
      {"source_ghost", "source_ghost"}, {0.0, 0.0}, identities<1>(), {"Scalar"}, false, {}, {}, {},
      {}, {}, mirror_sources<1>(true));
  native.fill_physical(native_state, domain);
  QualifiedCharacteristicModel<1> model{};
  generated.fill_physical_model_qualified(generated_state, domain, model,
                                          MirrorGhostPolicy<1>{Real(0.2)});
  const auto expected = stored_values(native_state);
  const auto actual = stored_values(generated_state);
  ASSERT_EQ(expected.size(), actual.size());
  for (std::size_t index = 0; index < expected.size(); ++index)
    EXPECT_TRUE(same_bits(expected[index], actual[index])) << "index=" << index;
}

TEST(test_prepared_hyperbolic_boundary, source_ghost_faces_refuse_every_fill_without_their_policy) {
  const Box<1> domain{Index<1>{0}, Index<1>{3}};
  const auto boundary = prepare_hyperbolic_boundary<1>({"source_ghost", "source_ghost"}, {0.0, 0.0},
                                                       identities<1>(), {"Scalar"}, false, {}, {},
                                                       {}, {}, {}, mirror_sources<1>(true));
  EXPECT_TRUE(boundary.has_source_ghost());
  auto state = one_patch_field(domain, 1, Extent<1>{1});
  fill_valid(state, Real(-99), [](const Index<1>& index, int) { return Real(index[0]); });
  QualifiedCharacteristicModel<1> model{};
  try {
    boundary.fill_physical(state, domain);
    FAIL() << "a source ghost face was filled without its policy";
  } catch (const std::logic_error& error) {
    EXPECT_NE(std::string(error.what()).find("generated ghost policy of its package"),
              std::string::npos);
  }
  EXPECT_THROW(boundary.fill_physical_model_qualified(state, domain, model), std::logic_error);
  EXPECT_THROW((void)boundary.preflight_physical(state, domain), std::logic_error);
  auto lane = ExecutionLane::duplicate_world_collectively("test/source-ghost-refusal");
  auto candidate = one_patch_field_for_lane(domain, 1, Extent<1>{1}, lane);
  auto laned = one_patch_field_for_lane(domain, 1, Extent<1>{1}, lane);
  EXPECT_THROW(boundary.fill_physical_model_qualified(laned, domain, model, lane, candidate),
               std::logic_error);
  // With its policy, on the lane, the same boundary fills.
  boundary.fill_physical_model_qualified(laned, domain, model, lane, candidate,
                                         MirrorGhostPolicy<1>{Real(1)});
  EXPECT_EQ(value_at(state, Index<1>{-1}, 0), Real(-99));
}

TEST(test_prepared_hyperbolic_boundary, source_ghost_policies_are_authenticated_face_by_face) {
  const Box<1> domain{Index<1>{0}, Index<1>{3}};
  QualifiedCharacteristicModel<1> model{};
  auto state = one_patch_field(domain, 1, Extent<1>{1});
  fill_valid(state, Real(-99), [](const Index<1>& index, int) { return Real(index[0]); });
  // A face whose source identity differs from the policy of the package.
  const auto other = prepare_hyperbolic_boundary<1>({"source_ghost", "source_ghost"}, {0.0, 0.0},
                                                    identities<1>(), {"Scalar"}, false, {}, {}, {},
                                                    {}, {}, {"ghost:low", "ghost:other"});
  EXPECT_THROW(other.fill_physical_model_qualified(state, domain, model, MirrorGhostPolicy<1>{}),
               std::logic_error);
  // A face of the policy that the installed boundary gives to a native law.
  const auto native_face =
      prepare_hyperbolic_boundary<1>({"source_ghost", "foextrap"}, {0.0, 0.0}, identities<1>(),
                                     {"Scalar"}, false, {}, {}, {}, {}, {}, {"ghost:low", ""});
  EXPECT_THROW(
      native_face.fill_physical_model_qualified(state, domain, model, MirrorGhostPolicy<1>{}),
      std::logic_error);
  // A policy of another width.
  const auto scalar = prepare_hyperbolic_boundary<1>({"source_ghost", "source_ghost"}, {0.0, 0.0},
                                                     identities<1>(), {"Scalar"}, false, {}, {}, {},
                                                     {}, {}, {"pair:low", "pair:high"});
  EXPECT_THROW(scalar.fill_physical_model_qualified(state, domain, model, PairGhostPolicy<1>{}),
               std::invalid_argument);
  // Source identities exist exactly on source ghost faces.
  EXPECT_THROW(prepare_hyperbolic_boundary<1>({"source_ghost", "foextrap"}, {0.0, 0.0},
                                              identities<1>(), {"Scalar"}),
               std::invalid_argument);
  EXPECT_THROW(
      prepare_hyperbolic_boundary<1>({"foextrap", "foextrap"}, {0.0, 0.0}, identities<1>(),
                                     {"Scalar"}, false, {}, {}, {}, {}, {}, {"ghost:low", ""}),
      std::invalid_argument);
  EXPECT_THROW(
      prepare_hyperbolic_boundary<1>({"source_ghost", "foextrap"}, {1.0, 0.0}, identities<1>(),
                                     {"Scalar"}, false, {}, {}, {}, {}, {}, {"ghost:low", ""}),
      std::invalid_argument);
}

// C3 of the review: a domain narrower than the inner cells that a face reads, and a native halo
// that reaches a source ghost face, are refused by name.
TEST(test_prepared_hyperbolic_boundary, source_ghost_faces_refuse_a_domain_narrower_than_reads) {
  QualifiedCharacteristicModel<1> model{};
  const Box<1> narrow{Index<1>{0}, Index<1>{1}};
  const auto boundary = prepare_hyperbolic_boundary<1>({"source_ghost", "source_ghost"}, {0.0, 0.0},
                                                       identities<1>(), {"Scalar"}, false, {}, {},
                                                       {}, {}, {}, mirror_sources<1>(true));
  auto state = one_patch_field(narrow, 1, Extent<1>{1});
  fill_valid(state, Real(-99), [](const Index<1>& index, int) { return Real(index[0]); });
  try {
    boundary.fill_physical_model_qualified(state, narrow, model, MirrorGhostPolicy<1>{});
    FAIL() << "a source ghost face read past the narrow domain";
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string(error.what()).find("reads 3 inner cells"), std::string::npos);
  }
  // A state that stores ghost cells deeper than the generated depths.
  const Box<1> domain{Index<1>{0}, Index<1>{4}};
  auto deep = one_patch_field(domain, 1, Extent<1>{3});
  fill_valid(deep, Real(-99), [](const Index<1>& index, int) { return Real(index[0]); });
  try {
    boundary.fill_physical_model_qualified(deep, domain, model, ShallowMirrorGhostPolicy{});
    FAIL() << "a source ghost face left stored ghost cells deeper than its policy";
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string(error.what()).find("deeper than the depths 1..2"), std::string::npos);
  }
  // The native halo of a fixed-state face that reaches the source ghost face across one cell.
  const auto mixed =
      prepare_hyperbolic_boundary<1>({"dirichlet", "source_ghost"}, {1.0, 0.0}, identities<1>(),
                                     {"Scalar"}, false, {}, {}, {}, {}, {}, {"", "ghost:high"});
  const Box<1> one{Index<1>{0}, Index<1>{0}};
  auto single = one_patch_field(one, 1, Extent<1>{2});
  fill_valid(single, Real(-99), [](const Index<1>&, int) { return Real(3); });
  try {
    mixed.fill_physical_model_qualified(single, one, model, HighOnlyMirrorGhostPolicy{});
    FAIL() << "a native halo crossed a source ghost face";
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string(error.what()).find("reaches a source_ghost face"), std::string::npos);
  }
}
