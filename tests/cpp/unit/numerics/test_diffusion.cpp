/// @file
/// @brief Exact-ranked SSPRK2 integration of an isotropic periodic diffusion eigenmode.

#include <gtest/gtest.h>

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/time/integrators/time_steppers.hpp>

#include <Kokkos_MathematicalFunctions.hpp>

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

constexpr int kDim = pops::kNativeDimension;
constexpr pops::Real kPi = pops::Real(3.141592653589793238462643383279502884L);
using Field = pops::MultiFab<kDim>;

template <int Dim>
pops::Extent<Dim> filled_extent(std::int64_t value) {
  pops::Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim>
struct InitializeMode {
  pops::FieldView<pops::Real, Dim> state{};
  pops::Box<Dim> domain{};
  pops::Real amplitude = pops::Real(0);

  POPS_HD void operator()(const pops::Index<Dim>& cell) const {
    const pops::Real x = (static_cast<pops::Real>(cell[0] - domain.lo[0]) + pops::Real(0.5)) /
                         static_cast<pops::Real>(domain.length(0));
    state(cell) = pops::Real(1) + amplitude * Kokkos::cos(pops::Real(2) * kPi * x);
  }
};

template <int Dim>
struct PeriodicDiffusionResidual {
  pops::FieldView<const pops::Real, Dim> state{};
  pops::FieldView<pops::Real, Dim> residual{};
  pops::Box<Dim> domain{};
  pops::Real diffusivity = pops::Real(0);

  POPS_HD void operator()(const pops::Index<Dim>& cell) const {
    pops::Real laplacian = pops::Real(0);
    for (int axis = 0; axis < Dim; ++axis) {
      pops::Index<Dim> lower = cell;
      pops::Index<Dim> upper = cell;
      lower[axis] = cell[axis] == domain.lo[axis] ? domain.hi[axis] : cell[axis] - 1;
      upper[axis] = cell[axis] == domain.hi[axis] ? domain.lo[axis] : cell[axis] + 1;
      const pops::Real inverse_spacing = static_cast<pops::Real>(domain.length(axis));
      laplacian += (state(lower) - pops::Real(2) * state(cell) + state(upper)) * inverse_spacing *
                   inverse_spacing;
    }
    residual(cell) = diffusivity * laplacian;
  }
};

template <int Dim>
struct ModeProjection {
  pops::FieldView<const pops::Real, Dim> state{};
  pops::Box<Dim> domain{};

  POPS_HD pops::Real operator()(const pops::Index<Dim>& cell) const {
    const pops::Real x = (static_cast<pops::Real>(cell[0] - domain.lo[0]) + pops::Real(0.5)) /
                         static_cast<pops::Real>(domain.length(0));
    return (state(cell) - pops::Real(1)) * Kokkos::cos(pops::Real(2) * kPi * x);
  }
};

struct Fixture {
  explicit Fixture(int cells)
      : domain(pops::Box<kDim>::from_extents(filled_extent<kDim>(cells))),
        layout(std::vector<pops::Box<kDim>>{domain}),
        ranks(pops::Index<kDim>{}, filled_extent<kDim>(1)),
        distribution(pops::mesh::Distribution<kDim>::replicated(layout, ranks)),
        state(layout, distribution, pops::Index<kDim>{}, 1, filled_extent<kDim>(0)) {}

  pops::Real amplitude() const {
    pops::Real projection = pops::Real(0);
    for (std::size_t local = 0; local < state.local_size(); ++local)
      projection += pops::for_each_cell_reduce_sum(
          state.box(local), ModeProjection<kDim>{state.fab(local).view(), domain});
    return pops::Real(2) * projection / static_cast<pops::Real>(domain.numPts());
  }

  void initialize(pops::Real amplitude) {
    for (std::size_t local = 0; local < state.local_size(); ++local)
      pops::for_each_cell(state.box(local),
                          InitializeMode<kDim>{state.fab(local).view(), domain, amplitude});
  }

  auto residual(pops::Real diffusivity) {
    return [this, diffusivity](Field& candidate, Field& rate) {
      for (std::size_t local = 0; local < candidate.local_size(); ++local)
        pops::for_each_cell(
            candidate.box(local),
            PeriodicDiffusionResidual<kDim>{static_cast<const Field&>(candidate).fab(local).view(),
                                            rate.fab(local).view(), domain, diffusivity});
    };
  }

  pops::Box<kDim> domain;
  pops::mesh::BoxArray<kDim> layout;
  pops::mesh::RankSpace<kDim> ranks;
  pops::mesh::Distribution<kDim> distribution;
  Field state;
};

static_assert(pops::TimeStepperFor<pops::ForwardEuler<kDim>, kDim>);
static_assert(pops::TimeStepperFor<pops::SSPRK2Step<kDim>, kDim>);
static_assert(pops::TimeStepperFor<pops::SSPRK3Step<kDim>, kDim>);

TEST(Diffusion, ZeroCoefficientIsExactlyStaticInNativeRank) {
  Fixture fixture(16);
  fixture.initialize(pops::Real(1.0e-3));
  const pops::Real initial = fixture.amplitude();
  auto residual = fixture.residual(pops::Real(0));
  for (int step = 0; step < 20; ++step)
    pops::SSPRK2Step<kDim>{}.take_step(residual, fixture.state, pops::Real(2.0e-3));
  EXPECT_EQ(fixture.amplitude(), initial);
}

TEST(Diffusion, SSPRK2MatchesExactDiscreteModeAmplification) {
  constexpr int cells = 16;
  constexpr int steps = 100;
  constexpr pops::Real dt = pops::Real(2.0e-3);
  constexpr pops::Real diffusivity = pops::Real(0.05);
  Fixture fixture(cells);
  fixture.initialize(pops::Real(1.0e-3));
  const pops::Real initial = fixture.amplitude();
  auto residual = fixture.residual(diffusivity);
  pops::SSPRK2Step<kDim>::Scratch scratch(fixture.state);
  for (int step = 0; step < steps; ++step)
    pops::SSPRK2Step<kDim>{}.take_step(residual, fixture.state, dt, scratch);

  const pops::Real theta = pops::Real(2) * kPi / static_cast<pops::Real>(cells);
  const pops::Real eigenvalue = diffusivity * pops::Real(2) * (pops::Real(1) - Kokkos::cos(theta)) *
                                static_cast<pops::Real>(cells * cells);
  const pops::Real one_step =
      pops::Real(1) - eigenvalue * dt + pops::Real(0.5) * eigenvalue * eigenvalue * dt * dt;
  const pops::Real expected = initial * std::pow(one_step, steps);
  EXPECT_NEAR(fixture.amplitude(), expected, pops::Real(2.0e-12));
  EXPECT_LT(fixture.amplitude(), pops::Real(0.8) * initial);
}

}  // namespace
