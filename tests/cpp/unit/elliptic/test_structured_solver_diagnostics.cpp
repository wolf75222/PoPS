#include <gtest/gtest.h>

#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/numerics/elliptic/mg/composite_fac_poisson.hpp>
#include <pops/numerics/elliptic/mg/geometric_mg.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace {
void set_env_var(const char* key, const char* value) {
#if defined(_WIN32)
  _putenv_s(key, value);
#else
  setenv(key, value, 1);
#endif
}
}  // namespace

TEST(test_structured_solver_diagnostics, Runs) {
  using namespace pops;
  static constexpr double kPi = 3.14159265358979323846;

  set_env_var("POPS_TRACE_SOLVE_FIELDS", "1");

  {
    const int n = 16;
    const Box2D dom = Box2D::from_extents(n, n);
    const Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
    const BoxArray ba = BoxArray::from_domain(dom, n);
    BCRec bc;
    bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;
    GeometricMG mg(geom, ba, bc);
    Array4 rhs = mg.rhs().fab(0).array();
    for_each_cell(dom, [rhs, geom](int i, int j) {
      const double x = geom.x_cell(i);
      const double y = geom.y_cell(j);
      rhs(i, j, 0) = -2.0 * kPi * kPi * std::sin(kPi * x) * std::sin(kPi * y);
    });
    mg.phi().set_val(0.0);
    mg.solve(Real(1e-6), 2);

    const RuntimeDiagnosticsReport& report = mg.diagnostics_report();
    ASSERT_TRUE(report.source == "pops.numerics.elliptic.geometric_mg") << "MG report source";
    ASSERT_TRUE(report.count("elliptic.mg.trace") > 0) << "MG trace events recorded structurally";
    ASSERT_TRUE(report.events.front().severity == "trace") << "MG trace severity";
  }

  {
    const int n = 16;
    const int ratio = 2;
    const Box2D dom = Box2D::from_extents(n, n);
    const Geometry geom_c{dom, 0.0, 1.0, 0.0, 1.0};
    const BoxArray ba_c = BoxArray::from_domain(dom, n);
    BCRec bc;
    bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;
    const int ic0 = n / 4;
    const int ic1 = 3 * n / 4 - 1;
    const Box2D fine_box{{ratio * ic0, ratio * ic0},
                         {ratio * ic1 + ratio - 1, ratio * ic1 + ratio - 1}};

    CompositeFacPoisson fac(geom_c, ba_c, bc, fine_box, ratio);
    fac.set_verbose(true);
    const Real residual = fac.solve(/*max_iters=*/1, /*fine_sweeps=*/2,
                                    /*rel_tol=*/Real(1e-8), /*abs_tol=*/Real(0));
    const RuntimeDiagnosticsReport& report = fac.diagnostics_report();
    ASSERT_TRUE(std::isfinite(static_cast<double>(residual))) << "FAC residual finite";
    ASSERT_TRUE(report.source == "pops.numerics.elliptic.composite_fac_poisson")
        << "FAC report source";
    ASSERT_TRUE(report.count("elliptic.fac.residual") > 0)
        << "FAC residual events recorded structurally";
    ASSERT_TRUE(report.events.front().component == "CompositeFacPoisson") << "FAC event component";
  }

  std::printf("OK test_structured_solver_diagnostics\n");
}
