/// @file
/// @brief Exact-ranked production-operator proofs for isotropic Fickian diffusion.

#include <gtest/gtest.h>

#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/numerics/spatial/operators/cartesian_operator.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace pops;

namespace {

template <int Dim>
class DiffusiveScalar : public nd::ScalarAdvection<Dim> {
 public:
  explicit DiffusiveScalar(Real diffusivity) : diffusivity_(diffusivity) {}

  [[nodiscard]] static constexpr PreparedProviderIdentity provider_identity() noexcept {
    return {"test.nd.diffusive-scalar", 1};
  }

  void serialize_exact_parameters(ExactContractBuilder& contract) const {
    contract.scalar(diffusivity_);
  }

  POPS_HD Real diffusivity() const { return diffusivity_; }

 private:
  Real diffusivity_ = Real(0);
};

static_assert(DiffusiveModel<DiffusiveScalar<1>>);
static_assert(nd::ConservationLaw<1, DiffusiveScalar<1>>);
static_assert(nd::ConservationLaw<2, DiffusiveScalar<2>>);
static_assert(nd::ConservationLaw<3, DiffusiveScalar<3>>);

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
std::size_t host_offset(const Box<Dim>& storage, const Index<Dim>& index, int component = 0) {
  std::int64_t linear = 0;
  std::int64_t stride = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    linear += static_cast<std::int64_t>(index[axis] - storage.lo[axis]) * stride;
    stride *= storage.length(axis);
  }
  return static_cast<std::size_t>(component * storage.numPts() + linear);
}

template <int Dim>
Extent<Dim> uniform_extent(int value) {
  Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim>
Index<Dim> periodic_image(Index<Dim> index, const Box<Dim>& domain) {
  for (int axis = 0; axis < Dim; ++axis) {
    const int length = static_cast<int>(domain.length(axis));
    while (index[axis] < domain.lo[axis])
      index[axis] += length;
    while (index[axis] > domain.hi[axis])
      index[axis] -= length;
  }
  return index;
}

template <int Dim, class Function>
void fill_fab(Fab<Dim>& field, Function&& value) {
  auto host = field.create_host_mirror();
  for_each_host_index(field.grown_box(), [&](const Index<Dim>& index) {
    host(host_offset(field.grown_box(), index)) = value(index);
  });
  field.copy_from_host(host);
}

template <int Dim>
Real fab_value(const Fab<Dim>& field, const Index<Dim>& index) {
  auto host = field.create_host_mirror();
  field.copy_to_host(host);
  return host(host_offset(field.grown_box(), index));
}

template <int Axis, int Dim>
void expect_all_face_values(const nd::FaceField<Dim>& faces, Real expected) {
  const auto& field = faces.template field<Axis>();
  auto host = field.create_host_mirror();
  field.copy_to_host(host);
  for (std::size_t index = 0; index < host.size(); ++index)
    EXPECT_EQ(host(index), expected);
  if constexpr (Axis + 1 < Dim)
    expect_all_face_values<Axis + 1>(faces, expected);
}

template <int Dim>
Geometry<Dim> unit_geometry(const Box<Dim>& domain) {
  RealVector<Dim> lower{};
  RealVector<Dim> upper{};
  for (int axis = 0; axis < Dim; ++axis)
    upper[axis] = Real(1);
  return Geometry<Dim>::from_bounds(domain, lower, upper);
}

template <int Dim>
void check_periodic_manufactured_mode() {
  constexpr int cells = 12;
  constexpr Real diffusivity = Real(0.07);
  constexpr Real two_pi = Real(6.283185307179586476925286766559);
  const Box<Dim> domain = Box<Dim>::from_extents(uniform_extent<Dim>(cells));
  const auto geometry = unit_geometry(domain);
  const auto op = nd::prepare_cartesian_operator<Dim>(geometry, DiffusiveScalar<Dim>{diffusivity});
  Fab<Dim> state(domain, 1, uniform_extent<Dim>(1));
  Fab<Dim> residual(domain, 1);
  nd::FaceField<Dim> faces(domain, 1);

  const auto mode = [&](const Index<Dim>& input) {
    const Index<Dim> index = periodic_image(input, domain);
    Real value = Real(1);
    for (int axis = 0; axis < Dim; ++axis)
      value += (Real(0.1) + Real(0.02) * axis) *
               std::cos(two_pi * (Real(index[axis] - domain.lo[axis]) + Real(0.5)) /
                        static_cast<Real>(cells));
    return value;
  };
  fill_fab(state, mode);
  op.materialize_face_fluxes(state, faces);
  op.assemble_residual_from_face_fluxes(faces, residual);

  Real integral = Real(0);
  for_each_host_index(domain, [&](const Index<Dim>& cell) {
    Real expected = Real(0);
    for (int axis = 0; axis < Dim; ++axis) {
      Index<Dim> lower = cell;
      Index<Dim> upper = cell;
      --lower[axis];
      ++upper[axis];
      expected +=
          diffusivity * Real(cells * cells) * (mode(lower) - Real(2) * mode(cell) + mode(upper));
    }
    const Real actual = fab_value(residual, cell);
    EXPECT_NEAR(actual, expected, Real(3e-12));
    integral += actual * op.metric().cell_measure(cell);
  });
  EXPECT_NEAR(integral, Real(0), Real(3e-13));

  Index<Dim> face = domain.lo;
  ++face[0];
  Index<Dim> left = face;
  --left[0];
  const auto& axis_faces = faces.template field<0>();
  const Real integrated = fab_value(axis_faces, face);
  const Real face_measure =
      op.metric().template oriented_face_area_vector<0, MetricFaceSide::Upper>(left)[0];
  const Real expected_face = -diffusivity * (mode(face) - mode(left)) * Real(cells) * face_measure;
  EXPECT_NEAR(integrated, expected_face, Real(3e-14));
}

template <int Dim>
void check_halo_contract() {
  const Box<Dim> domain = Box<Dim>::from_extents(uniform_extent<Dim>(5));
  const auto geometry = unit_geometry(domain);
  const auto op = nd::prepare_cartesian_operator<Dim>(geometry, DiffusiveScalar<Dim>{Real(0.2)});
  Fab<Dim> no_halo(domain, 1);
  nd::FaceField<Dim> rejected_faces(domain, 1);
  EXPECT_THROW(op.materialize_face_fluxes(no_halo, rejected_faces), std::invalid_argument);

  Fab<Dim> state(domain, 1, uniform_extent<Dim>(1));
  fill_fab(state, [](const Index<Dim>& index) {
    Real value = Real(0.5);
    for (int axis = 0; axis < Dim; ++axis)
      value += Real(0.25 + 0.05 * axis) * Real(index[axis]);
    return value;
  });
  nd::FaceField<Dim> faces(domain, 1);
  op.materialize_face_fluxes(state, faces);
  const auto& axis_faces = faces.template field<0>();
  Index<Dim> lower_face = domain.lo;
  Index<Dim> upper_face = domain.lo;
  upper_face[0] = domain.hi[0] + 1;
  EXPECT_NEAR(fab_value(axis_faces, lower_face), fab_value(axis_faces, upper_face), Real(2e-14));
  EXPECT_LT(fab_value(axis_faces, lower_face), Real(0));
}

template <int Dim>
void check_candidate_rollback() {
  const Box<Dim> domain = Box<Dim>::from_extents(uniform_extent<Dim>(4));
  const auto geometry = unit_geometry(domain);
  EXPECT_THROW((void)nd::prepare_cartesian_operator<Dim>(
                   geometry, DiffusiveScalar<Dim>{std::numeric_limits<Real>::quiet_NaN()}),
               std::invalid_argument);
  EXPECT_THROW(
      (void)nd::prepare_cartesian_operator<Dim>(geometry, DiffusiveScalar<Dim>{Real(-0.1)}),
      std::invalid_argument);

  const auto op = nd::prepare_cartesian_operator<Dim>(geometry, DiffusiveScalar<Dim>{Real(0.1)});
  Fab<Dim> state(domain, 1, uniform_extent<Dim>(1));
  fill_fab(state, [](const Index<Dim>&) { return Real(1); });
  auto state_host = state.create_host_mirror();
  state.copy_to_host(state_host);
  Index<Dim> bad_ghost = domain.lo;
  --bad_ghost[0];
  state_host(host_offset(state.grown_box(), bad_ghost)) = std::numeric_limits<Real>::quiet_NaN();
  state.copy_from_host(state_host);

  nd::FaceField<Dim> output(domain, 1);
  output.set_val(Real(17));
  EXPECT_THROW(op.materialize_face_fluxes(state, output), std::runtime_error);
  expect_all_face_values<0>(output, Real(17));
}

TEST(Diffusion, PeriodicManufacturedResidualAndIntegratedFacesAreExactRanked) {
  check_periodic_manufactured_mode<1>();
  check_periodic_manufactured_mode<2>();
  check_periodic_manufactured_mode<3>();
}

TEST(Diffusion, PhysicalAndPeriodicCallersMustSupplyTheExactHalo) {
  check_halo_contract<1>();
  check_halo_contract<2>();
  check_halo_contract<3>();
}

TEST(Diffusion, InvalidCandidatesAndConstitutiveDataNeverPublish) {
  check_candidate_rollback<1>();
  check_candidate_rollback<2>();
  check_candidate_rollback<3>();
}

}  // namespace
