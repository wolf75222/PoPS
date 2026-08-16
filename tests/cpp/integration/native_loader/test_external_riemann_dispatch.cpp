// End-to-end external C++ Riemann brick: build a real .so, dlopen it, dispatch its flux (ADC-463).
//
// test_external_brick.cpp covers the host identity registry in isolation. THIS test closes the
// deferred half of ADC-463 (Spec 3 section 21-22, criterion 20): an external brick shipped in a
// standalone .so is dlopen'd and its flux DISPATCHED into the finite-volume machinery in the SAME
// exact-ranked type system as a native flux, never through a per-cell string lookup. It mirrors
// test_amr_native_loader.cpp: the brick source is written and compiled to a .so at run time (so no
// committed binary), then loaded.
//
// VALIDATIONS:
//   1. ExternalBrickHandle dlopens the .so, reads its manifest, and exposes the brick's id +
//      requirements (the manifest is visible in the same registry native bricks would use).
//   2. The resolved residual entry point runs the brick's flux. The brick wraps pops::RusanovFlux, so
//      its residual is compared BIT-FOR-BIT against the native rusanov path (make_block "rusanov"):
//      the external brick is dispatched into identical numerics -> dmax == 0.
//   3. An unknown id is rejected with a clear error.

#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include "native_dso_compiler.hpp"
#include <pops/runtime/program/external_riemann_brick.hpp>

#include <pops/physics/bricks/bricks.hpp>  // CompositeModel / Euler / ...

#include "test_harness.hpp"  // pops::test::Checker

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <string>
#include <vector>

using pops::runtime::program::ExternalBrickHandle;

namespace {

constexpr double kGamma = 1.4;
constexpr const char* kModelIdentity =
    "0000000000000000000000000000000000000000000000000000000000000000";

// The C++ an advanced user ships in my_riemann.cpp: a NumericalFlux policy (here a thin wrapper over
// the native RusanovFlux so the test can prove BIT-IDENTICAL dispatch) + the two macros that register
// the brick and emit its static-dispatch ABI. The Model is fixed in the .so (Euler), exactly as a
// real external brick would pin its target model.
std::string brick_source() {
  // clang-format off
  return R"CPP(
#include <pops/runtime/program/external_riemann_brick.hpp>
#include <pops/physics/bricks/bricks.hpp>

// The user's flux: the same narrow PhysicalFlux + two typed traces + FaceContext contract as
// pops::RusanovFlux. Here it forwards to RusanovFlux so the test can assert bit-identical dispatch;
// a real brick would compute its own interface flux. POPS_HD: device-callable, no virtuals.
struct UserRusanov {
  template <pops::PhysicalFlux Physical>
  POPS_HD pops::FluxEvaluation<typename Physical::State> operator()(
      const Physical& physical, const typename Physical::Trace& left,
      const typename Physical::Trace& right, const pops::FaceContext& face) const {
    return pops::RusanovFlux{}(physical, left, right, face);
  }
};

namespace user_brick {
struct Model : pops::nd::IdealGasEuler<pops::kNativeDimension> {
  POPS_HD pops::Real elliptic_rhs(const State&) const { return pops::Real(0); }
};
}

POPS_DEFINE_EMPTY_EXTERNAL_RIEMANN_PROVIDER_ROUTES(user_brick::Model);
POPS_DEFINE_EXTERNAL_RIEMANN_BRICK(
    "my_riemann", UserRusanov, user_brick::Model,
    "0000000000000000000000000000000000000000000000000000000000000000",
    "physical_flux,provider_pack,stability_bound");
POPS_DEFINE_BRICK_MANIFEST();
)CPP";
  // clang-format on
}

std::string legacy_brick_source() {
  return R"CPP(
#include <pops/runtime/program/external_brick.hpp>
    POPS_REGISTER_BRICK("legacy_riemann", "riemann", "physical_flux");
    POPS_DEFINE_BRICK_MANIFEST();
    extern "C" void pops_brick_residual() {}
  )CPP";
}

using RefModel = pops::nd::IdealGasEuler<pops::kNativeDimension>;
using RefProviderValues = pops::ProviderValues<pops::provider_count<RefModel>()>;

static_assert(RefProviderValues::size == pops::provider_count<RefModel>());

// A smooth periodic Euler state in component-major layout c*cells + flattened-cell. The vector
// has the exact artifact rank: rho, one momentum per axis, then total energy.
template <int Dim>
std::vector<double> euler_state(const std::array<int, Dim>& shape) {
  std::size_t cells = 1;
  for (int axis = 0; axis < Dim; ++axis)
    cells *= static_cast<std::size_t>(shape[static_cast<std::size_t>(axis)]);
  std::vector<double> state(static_cast<std::size_t>(Dim + 2) * cells);
  const double pi = std::acos(-1.0);
  for (std::size_t cell = 0; cell < cells; ++cell) {
    std::size_t quotient = cell;
    double radius_squared = 0.0;
    double wave = 0.0;
    double velocity_squared = 0.0;
    std::array<double, Dim> velocity{};
    for (int axis = 0; axis < Dim; ++axis) {
      const int extent = shape[static_cast<std::size_t>(axis)];
      const int coordinate = static_cast<int>(quotient % static_cast<std::size_t>(extent));
      quotient /= static_cast<std::size_t>(extent);
      const double x = (static_cast<double>(coordinate) + 0.5) / extent - 0.5;
      radius_squared += x * x;
      wave += 0.04 * static_cast<double>(axis + 1) * std::sin((axis + 2) * pi * x);
      velocity[static_cast<std::size_t>(axis)] = 0.1 / static_cast<double>(axis + 1);
      velocity_squared +=
          velocity[static_cast<std::size_t>(axis)] * velocity[static_cast<std::size_t>(axis)];
    }
    const double rho = 1.0 + 0.3 * std::exp(-radius_squared / 0.02) + wave;
    state[cell] = rho;
    for (int axis = 0; axis < Dim; ++axis)
      state[static_cast<std::size_t>(axis + 1) * cells + cell] =
          rho * velocity[static_cast<std::size_t>(axis)];
    state[static_cast<std::size_t>(Dim + 1) * cells + cell] =
        1.0 / (kGamma - 1.0) + 0.5 * rho * velocity_squared;
  }
  return state;
}

template <int Dim>
struct ExactRankedAdvection {
  using Schema = pops::nd::ScalarStateSchema<Dim>;
  using State = typename Schema::Conservative;
  using Primitive = typename Schema::Primitive;
  static constexpr int dimension = Dim;
  static constexpr int n_vars = 1;

  POPS_HD pops::nd::StateConversion<Primitive> recover(const State& state) const {
    return {state, Kokkos::isfinite(state[0]) ? pops::nd::StateConversionStatus::Success
                                              : pops::nd::StateConversionStatus::NonFiniteState};
  }
  POPS_HD pops::nd::StateConversion<State> make_conservative(const Primitive& primitive) const {
    return {primitive, Kokkos::isfinite(primitive[0])
                           ? pops::nd::StateConversionStatus::Success
                           : pops::nd::StateConversionStatus::NonFiniteState};
  }
  POPS_HD pops::nd::StateConversionStatus admissibility(const State& state) const {
    return recover(state).status;
  }
  template <int Axis>
  POPS_HD State flux(const State& state) const {
    return State{static_cast<pops::Real>(Axis + 1) * state[0]};
  }
  template <int Axis>
  POPS_HD pops::Real max_wave_speed(const State&) const {
    return static_cast<pops::Real>(Axis + 1);
  }
};

struct ForwardRusanov {
  template <pops::PhysicalFlux Physical>
  POPS_HD pops::FluxEvaluation<typename Physical::State> operator()(
      const Physical& physical, const typename Physical::Trace& left,
      const typename Physical::Trace& right, const pops::FaceContext& face) const {
    return pops::RusanovFlux{}(physical, left, right, face);
  }
};

template <int Dim>
void check_exact_ranked_flat_residual() {
  std::array<int, Dim> shape{};
  std::array<double, Dim> spacing{};
  std::array<int, Dim> periodic{};
  std::size_t cells = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    shape[static_cast<std::size_t>(axis)] = 10 + axis;
    spacing[static_cast<std::size_t>(axis)] = 1.0 / shape[static_cast<std::size_t>(axis)];
    periodic[static_cast<std::size_t>(axis)] = 1;
    cells *= static_cast<std::size_t>(shape[static_cast<std::size_t>(axis)]);
  }
  std::vector<double> state(cells);
  for (std::size_t cell = 0; cell < cells; ++cell)
    state[cell] = 1.0 + 0.2 * std::sin(0.17 * static_cast<double>(cell));
  std::vector<double> external(cells, 0.0), native(cells, 0.0);
  pops::runtime::program::detail::external_residual<Dim, ExactRankedAdvection<Dim>, ForwardRusanov>(
      state.data(), external.data(), nullptr, shape.data(), spacing.data(), periodic.data(),
      "minmod", false, 0.0);
  pops::runtime::program::detail::external_residual<Dim, ExactRankedAdvection<Dim>,
                                                    pops::RusanovFlux>(
      state.data(), native.data(), nullptr, shape.data(), spacing.data(), periodic.data(), "minmod",
      false, 0.0);
  double magnitude = 0.0;
  for (std::size_t cell = 0; cell < cells; ++cell) {
    EXPECT_EQ(external[cell], native[cell]);
    EXPECT_TRUE(std::isfinite(external[cell]));
    magnitude = std::max(magnitude, std::fabs(external[cell]));
  }
  EXPECT_GT(magnitude, 1e-8);
}

}  // namespace

static int pops_run_test_external_riemann_dispatch() {
  const std::string tmp = std::string(POPS_TEST_TMPDIR) + "/external_riemann_" +
                          std::to_string(static_cast<long>(std::clock()));
  const std::string src = tmp + ".cpp", so = tmp + ".so";
  {
    std::ofstream f(src);
    f << brick_source();
  }
  const auto package = pops::test::native_dso::compile_shared(src, so);
  if (!package.ok) {
    pops::test::native_dso::report_compile_failure("test_external_riemann_dispatch", package);
    return 1;
  }

  pops::test::Checker chk;
  const std::string digest = pops::dynlib::AuthenticatedNativeFile(so).content_sha256();

  // (1) dlopen + manifest visibility + requirements surface.
  ExternalBrickHandle handle(so, "my_riemann", RefModel::n_vars, pops::provider_count<RefModel>(),
                             kModelIdentity, digest, true);
  handle.require_system_v7();
  chk(handle.id() == "my_riemann", "handle_id");
  chk(handle.dimension() == pops::kNativeDimension, "native_dimension_authenticated");
  chk(handle.nvars() == RefModel::n_vars, "state_shape_authenticated");
  chk(handle.provider_count() == pops::provider_count<RefModel>(), "provider_shape_authenticated");
  chk(handle.requirements() == "physical_flux,provider_pack,stability_bound",
      "requirements_surface");
  chk(handle.residual() != nullptr, "residual_resolved");
  // The dlopen registered the manifest in this image's process catalog too.
  const auto* entry = pops::runtime::program::BrickRegistry::instance().lookup("my_riemann");
  chk(entry != nullptr && entry->category == "riemann", "manifest_visible_in_registry");
  bool identity_threw = false;
  try {
    ExternalBrickHandle wrong_model(
        so, "my_riemann", RefModel::n_vars, pops::provider_count<RefModel>(),
        "1111111111111111111111111111111111111111111111111111111111111111", digest);
  } catch (const std::runtime_error& e) {
    identity_threw = true;
    chk(std::string(e.what()).find("different compiled model identity") != std::string::npos,
        "model_identity_error_is_actionable");
  }
  chk(identity_threw, "same_shape_different_model_rejected");
  // (2) BIT-IDENTICAL dispatch: external brick residual == native rusanov residual.
  constexpr int Dim = pops::kNativeDimension;
  const int n = 48;
  std::array<int, Dim> shape{};
  std::array<double, Dim> spacing{};
  std::size_t cells = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    shape[static_cast<std::size_t>(axis)] = n;
    spacing[static_cast<std::size_t>(axis)] = 1.0 / n;
    cells *= static_cast<std::size_t>(n);
  }
  const std::vector<double> U = euler_state<Dim>(shape);
  std::vector<double> Rext(static_cast<std::size_t>(RefModel::n_vars) * cells, 0.0);
  std::vector<double> Rnat(Rext.size(), 0.0);

  std::vector<double> residual_x_only, residual_y_only;
  for (int mask = 0; mask < (1 << Dim); ++mask) {
    std::array<int, Dim> periodic{};
    for (int axis = 0; axis < Dim; ++axis)
      periodic[static_cast<std::size_t>(axis)] = (mask >> axis) & 1;
    std::fill(Rext.begin(), Rext.end(), 0.0);
    std::fill(Rnat.begin(), Rnat.end(), 0.0);
    handle.residual()(U.data(), Rext.data(), /*provider_values=*/nullptr, shape.data(),
                      spacing.data(), periodic.data(), "minmod", /*recon_prim=*/0,
                      /*pos_floor=*/0.0, static_cast<double>(pops::kWenoEpsilon));
    pops::runtime::program::detail::external_residual<Dim, RefModel, pops::RusanovFlux>(
        U.data(), Rnat.data(), /*provider_values=*/nullptr, shape.data(), spacing.data(),
        periodic.data(), "minmod", /*recon_prim=*/false, /*pos_floor=*/0.0);
    double dmax = 0.0, nrm = 0.0;
    for (std::size_t k = 0; k < Rext.size(); ++k) {
      dmax = std::max(dmax, std::fabs(Rext[k] - Rnat[k]));
      nrm = std::max(nrm, std::fabs(Rnat[k]));
    }
    chk(nrm > 1e-8, "native_residual_nontrivial");
    chk(dmax == 0.0, "external_dispatch_bit_identical_to_native_rusanov");
    if constexpr (Dim >= 2) {
      bool x_only = periodic[0] != 0;
      bool y_only = periodic[1] != 0;
      for (int axis = 2; axis < Dim; ++axis) {
        x_only = x_only && periodic[static_cast<std::size_t>(axis)] == 0;
        y_only = y_only && periodic[static_cast<std::size_t>(axis)] == 0;
      }
      if (x_only && periodic[1] == 0)
        residual_x_only = Rext;
      if (y_only && periodic[0] == 0)
        residual_y_only = Rext;
    }
  }
  if constexpr (Dim >= 2) {
    double mixed_axis_difference = 0.0;
    for (std::size_t k = 0; k < residual_x_only.size(); ++k)
      mixed_axis_difference =
          std::max(mixed_axis_difference, std::fabs(residual_x_only[k] - residual_y_only[k]));
    chk(mixed_axis_difference > 1e-8, "x_only_and_y_only_are_not_flattened");
  }

  // (3) Unknown id -> clear error.
  bool threw = false;
  try {
    ExternalBrickHandle bad(so, "no_such_brick", RefModel::n_vars, pops::provider_count<RefModel>(),
                            kModelIdentity, digest);
  } catch (const std::runtime_error& e) {
    threw = true;
    const std::string msg = e.what();
    chk(msg.find("no_such_brick") != std::string::npos, "unknown_id_names_id");
  }
  chk(threw, "unknown_id_rejected");

  // (4) A v1/unversioned DSO is rejected before its old residual function can be called.
  const std::string legacy_src = tmp + "_legacy.cpp", legacy_so = tmp + "_legacy.so";
  {
    std::ofstream f(legacy_src);
    f << legacy_brick_source();
  }
  const auto legacy_package = pops::test::native_dso::compile_shared(legacy_src, legacy_so);
  if (!legacy_package.ok) {
    pops::test::native_dso::report_compile_failure("test_external_riemann_dispatch_legacy",
                                                   legacy_package);
    return 1;
  }
  threw = false;
  try {
    const std::string legacy_digest =
        pops::dynlib::AuthenticatedNativeFile(legacy_so).content_sha256();
    ExternalBrickHandle legacy(legacy_so, "legacy_riemann", RefModel::n_vars,
                               pops::provider_count<RefModel>(), kModelIdentity, legacy_digest);
  } catch (const std::runtime_error& e) {
    threw = true;
    const std::string msg = e.what();
    chk(msg.find("legacy") != std::string::npos, "legacy_abi_error_is_actionable");
  }
  chk(threw, "legacy_unversioned_abi_rejected");

  return chk.failed();
}

TEST(test_external_riemann_dispatch, Runs) {
  EXPECT_EQ(pops::test::RunTestBody(&pops_run_test_external_riemann_dispatch,
                                    "test_external_riemann_dispatch"),
            0);
}

TEST(test_external_riemann_dispatch, ExactRankedResidualRunsInOneTwoAndThreeDimensions) {
  check_exact_ranked_flat_residual<1>();
  check_exact_ranked_flat_residual<2>();
  check_exact_ranked_flat_residual<3>();
}
