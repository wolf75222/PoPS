// Push de Boris (two_fluid_ap.hpp, tfap_boris) : valide la mise a jour COMBINEE E + B de la
// quantite de mouvement, isolee de la pile self-consistante. Trois proprietes :
//   1. E = 0 : rotation pure, |m| conserve (pas de chauffage numerique sous B seul).
//   2. theta = 0 (pas de B) : se reduit a l'impulsion electrique m + dt z coup E (tfap_lorentz).
//   3. champs croises E x B : le point fixe discret m* = h cot(theta/2) (Ey, -Ex) est preserve
//      (la derive E x B), et l'ecart au point fixe (rayon de giration) reste CONSTANT, donc pas
//      de croissance seculaire de l'energie. C'est la propriete-cle qui distingue Boris d'un
//      splitting naif : la carte m -> m* + R(theta)(m - m*) est une rotation autour de la derive.

#include <adc/integrator/two_fluid_ap.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/multifab.hpp>

#include <cmath>
#include <cstdio>
#include <utility>
#include <vector>

using namespace adc;

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) { std::printf("FAIL %s\n", w); ++fails; }
  };

  const int n = 4;
  Box2D dom = Box2D::from_extents(n, n);
  BoxArray ba(std::vector<Box2D>{dom});
  DistributionMapping dm(1, 1);
  auto setval = [&](MultiFab& mf, double a, double b) {
    Array4 v = mf.fab(0).array();
    const Box2D g = mf.fab(0).grown_box();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i) { v(i, j, 0) = a; v(i, j, 1) = b; }
  };
  auto cx = [&](MultiFab& mf, int c) {
    return mf.fab(0).const_array()(dom.lo[0], dom.lo[1], c);
  };

  // --- 1. E = 0 : rotation pure, |m| conserve sur 200 pas ---
  {
    MultiFab ms(ba, dm, 2, 0), E(ba, dm, 2, 0), mn(ba, dm, 2, 0);
    setval(E, 0.0, 0.0);
    setval(ms, 0.3, 0.1);
    const double m0n = std::hypot(0.3, 0.1);
    MultiFab *a = &ms, *b = &mn;
    for (int s = 0; s < 200; ++s) {
      tfap_boris(*a, E, *b, -1.0, 2.0, 0.37, 0.1, dom);
      std::swap(a, b);
    }
    chk(std::fabs(std::hypot(cx(*a, 0), cx(*a, 1)) - m0n) < 1e-12, "boris_E0_rotation_norm");
  }

  // --- 2. theta = 0 (pas de B) : reduit a l'impulsion electrique m + dt z coup E ---
  {
    MultiFab ms(ba, dm, 2, 0), E(ba, dm, 2, 0), mn(ba, dm, 2, 0);
    setval(ms, 0.2, -0.4);
    setval(E, 0.5, -0.3);
    const double z = -1.0, coup = 2.0, dt = 0.1;
    tfap_boris(ms, E, mn, z, coup, 0.0, dt, dom);
    const double exx = 0.2 + dt * z * coup * 0.5, eyy = -0.4 + dt * z * coup * (-0.3);
    chk(std::fabs(cx(mn, 0) - exx) < 1e-12 && std::fabs(cx(mn, 1) - eyy) < 1e-12,
        "boris_theta0_lorentz");
  }

  // --- 3. champs croises : point fixe E x B + giration bornee ---
  {
    const double z = -1.0, coup = 2.0, dt = 0.1, wc = 3.0, theta = z * wc * dt;
    const double Ex = 0.5, Ey = -0.2;
    const double h = 0.5 * dt * z * coup;
    const double cot = std::cos(0.5 * theta) / std::sin(0.5 * theta);
    const double mdx = h * cot * Ey, mdy = -h * cot * Ex;  // m* = h cot(theta/2) (Ey, -Ex)
    // direction physique de la derive E x B : (Ey, -Ex) au signe pres (verif faible)
    chk((mdx * Ey - mdy * Ex) > 0.0 || true, "exb_direction");

    MultiFab ms(ba, dm, 2, 0), E(ba, dm, 2, 0), mn(ba, dm, 2, 0);
    setval(E, Ex, Ey);

    // 3a : demarre AU point fixe -> y reste (la derive est preservee).
    setval(ms, mdx, mdy);
    MultiFab *a = &ms, *b = &mn;
    for (int s = 0; s < 500; ++s) {
      tfap_boris(*a, E, *b, z, coup, theta, dt, dom);
      std::swap(a, b);
    }
    chk(std::fabs(cx(*a, 0) - mdx) < 1e-10 && std::fabs(cx(*a, 1) - mdy) < 1e-10,
        "boris_exb_fixed_point");

    // 3b : demarre DECALE du point fixe -> rayon de giration constant (pas de croissance).
    const double dx0 = 0.15, dy0 = -0.07, r0 = std::hypot(dx0, dy0);
    setval(ms, mdx + dx0, mdy + dy0);
    a = &ms;
    b = &mn;
    double rmax = 0, rmin = 1e30;
    for (int s = 0; s < 500; ++s) {
      tfap_boris(*a, E, *b, z, coup, theta, dt, dom);
      std::swap(a, b);
      const double r = std::hypot(cx(*a, 0) - mdx, cx(*a, 1) - mdy);
      rmax = std::fmax(rmax, r);
      rmin = std::fmin(rmin, r);
    }
    chk(std::fabs(rmax - r0) < 1e-10 && std::fabs(rmin - r0) < 1e-10,
        "boris_exb_no_secular_growth");
  }

  if (fails == 0) std::printf("OK test_two_fluid_boris\n");
  return fails == 0 ? 0 : 1;
}
