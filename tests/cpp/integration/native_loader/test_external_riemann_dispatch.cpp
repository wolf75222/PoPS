// End-to-end external C++ Riemann brick: build a real .so, dlopen it, dispatch its flux (ADC-463).
//
// test_external_brick.cpp covers the host identity registry in isolation. THIS test closes the
// deferred half of ADC-463 (Spec 3 section 21-22, criterion 20): an external brick shipped in a
// standalone .so is dlopen'd and its flux DISPATCHED into the finite-volume machinery in the SAME
// type system as a native flux -- statically (build_block<Limiter, UserFlux> inside the .so), never a
// per-cell string lookup. It mirrors test_amr_native_loader.cpp: the brick source is written and
// compiled to a .so at run time (so no committed binary), then loaded.
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
using Model = pops::CompositeModel<pops::Euler, pops::NoSource, pops::BackgroundDensity>;
}

POPS_DEFINE_EXTERNAL_RIEMANN_BRICK(
    "my_riemann", UserRusanov, user_brick::Model,
    "test.euler-rusanov.v1",
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

// A smooth periodic Euler state (rho, mx, my, E) in component-major layout c*n*n + j*n + i.
std::vector<double> euler_state(int n) {
  const std::size_t nn = static_cast<std::size_t>(n) * n;
  std::vector<double> U(4 * nn);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const double x = (i + 0.5) / n - 0.5, y = (j + 0.5) / n - 0.5;
      const double pi = std::acos(-1.0);
      const double rho = 1.0 + 0.3 * std::exp(-(x * x + y * y) / 0.02) +
                         0.08 * std::sin(2.0 * pi * x) + 0.05 * std::cos(4.0 * pi * y);
      const double u = 0.1, v = -0.05, p = 1.0;
      const std::size_t k = static_cast<std::size_t>(j) * n + i;
      U[0 * nn + k] = rho;
      U[1 * nn + k] = rho * u;
      U[2 * nn + k] = rho * v;
      U[3 * nn + k] = p / (kGamma - 1.0) + 0.5 * rho * (u * u + v * v);
    }
  return U;
}

using RefModel = pops::CompositeModel<pops::Euler, pops::NoSource, pops::BackgroundDensity>;

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

  // (1) dlopen + manifest visibility + requirements surface.
  ExternalBrickHandle handle(so, "my_riemann", {}, RefModel::n_vars,
                             pops::aux_comps<RefModel>(), "test.euler-rusanov.v1");
  chk(handle.id() == "my_riemann", "handle_id");
  chk(handle.requirements() == "physical_flux,provider_pack,stability_bound",
      "requirements_surface");
  chk(handle.residual() != nullptr, "residual_resolved");
  // The dlopen registered the manifest in this image's process catalog too.
  const auto* entry = pops::runtime::program::BrickRegistry::instance().lookup("my_riemann");
  chk(entry != nullptr && entry->category == "riemann", "manifest_visible_in_registry");
  bool identity_threw = false;
  try {
    ExternalBrickHandle wrong_model(so, "my_riemann", {}, RefModel::n_vars,
                                    pops::aux_comps<RefModel>(), "different-model-hash");
  } catch (const std::runtime_error& e) {
    identity_threw = true;
    chk(std::string(e.what()).find("different compiled model identity") != std::string::npos,
        "model_identity_error_is_actionable");
  }
  chk(identity_threw, "same_shape_different_model_rejected");

  // (2) BIT-IDENTICAL dispatch: external brick residual == native rusanov residual.
  const int n = 48;
  const double dx = 1.0 / n, dy = 1.0 / n;
  const std::vector<double> U = euler_state(n);
  const std::size_t nn = static_cast<std::size_t>(n) * n;
  std::vector<double> Rext(4 * nn, 0.0), Rnat(4 * nn, 0.0);

  const std::array<pops::Periodicity, 4> topologies{{
      {false, false}, {true, false}, {false, true}, {true, true}}};
  std::vector<double> residual_x_only, residual_y_only;
  for (const auto periodicity : topologies) {
    std::fill(Rext.begin(), Rext.end(), 0.0);
    std::fill(Rnat.begin(), Rnat.end(), 0.0);
    // External brick: v2 carries x/y independently into the exact same static native leaf.
    handle.residual()(U.data(), Rext.data(), /*aux=*/nullptr, n, dx, dy,
                      periodicity.x ? 1 : 0, periodicity.y ? 1 : 0, "minmod",
                      /*recon_prim=*/0, /*pos_floor=*/0.0);
    pops::runtime::program::detail::external_residual<RefModel, pops::RusanovFlux>(
        U.data(), Rnat.data(), /*aux=*/nullptr, n, dx, dy, periodicity, "minmod",
        /*recon_prim=*/false, /*pos_floor=*/0.0);
    double dmax = 0.0, nrm = 0.0;
    for (std::size_t k = 0; k < Rext.size(); ++k) {
      dmax = std::max(dmax, std::fabs(Rext[k] - Rnat[k]));
      nrm = std::max(nrm, std::fabs(Rnat[k]));
    }
    chk(nrm > 1e-8, "native_residual_nontrivial");
    chk(dmax == 0.0, "external_dispatch_bit_identical_to_native_rusanov");
    if (periodicity.x && !periodicity.y)
      residual_x_only = Rext;
    if (!periodicity.x && periodicity.y)
      residual_y_only = Rext;
  }
  double mixed_axis_difference = 0.0;
  for (std::size_t k = 0; k < residual_x_only.size(); ++k)
    mixed_axis_difference =
        std::max(mixed_axis_difference, std::fabs(residual_x_only[k] - residual_y_only[k]));
  chk(mixed_axis_difference > 1e-8, "x_only_and_y_only_are_not_flattened");

  // (3) Unknown id -> clear error.
  bool threw = false;
  try {
    ExternalBrickHandle bad(so, "no_such_brick");
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
    ExternalBrickHandle legacy(legacy_so, "legacy_riemann");
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
