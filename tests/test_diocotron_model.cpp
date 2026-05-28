// Premier test du socle : prouver que la couche physique tient debout.
//
// 1. static_assert : Diocotron satisfait le concept PhysicalModel. C'est le
//    vrai test d'architecture : si le modele ne respecte pas le contrat, ca
//    ne compile pas.
// 2. controles numeriques : la derive E x B et le second membre elliptique
//    sont calcules correctement, sans aucun maillage (la physique est pure
//    arithmetique, independante du backend parallele).

#include <adc/core/physical_model.hpp>
#include <adc/model/diocotron.hpp>

#include <cmath>
#include <cstdio>

using namespace adc;

static_assert(PhysicalModel<Diocotron>,
              "Diocotron doit satisfaire le concept PhysicalModel");

int main() {
  int fails = 0;
  auto check = [&](Real got, Real expected, const char* what) {
    if (std::fabs(got - expected) > 1e-12) {
      std::printf("FAIL %s : got %g, expected %g\n", what, got, expected);
      ++fails;
    }
  };

  Diocotron m;
  m.B0 = 2.0;
  m.alpha = 1.0;
  m.n_i0 = 1.0;

  Diocotron::State u{};
  u[0] = 3.0;

  Aux a{};
  a.grad_x = 0.5;
  a.grad_y = -0.25;

  // v_E_x = -grad_y / B0 = 0.25 / 2 = 0.125
  // v_E_y =  grad_x / B0 = 0.5  / 2 = 0.25
  check(m.flux(u, a, 0)[0], 3.0 * 0.125, "flux_x");
  check(m.flux(u, a, 1)[0], 3.0 * 0.25, "flux_y");
  check(m.max_wave_speed(u, a, 0), 0.125, "wavespeed_x");
  check(m.max_wave_speed(u, a, 1), 0.25, "wavespeed_y");

  // source nulle (transport pur, le couplage passe par le flux)
  check(m.source(u, a)[0], 0.0, "source");

  // second membre Poisson : alpha (n_e - n_i0) = 1 * (3 - 1) = 2
  check(m.elliptic_rhs(u), 2.0, "elliptic_rhs");

  if (fails == 0) std::printf("OK test_diocotron_model\n");
  return fails == 0 ? 0 : 1;
}
