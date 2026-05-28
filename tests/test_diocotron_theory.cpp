// Validation de l'eigensolver radial du diocotron contre la theorie lineaire
// publiee (Davidson-Felice ; Hoffart et al. arXiv:2510.11808 fig 5.4) :
// pour (a, b, Rw) = (6, 8, 16), les taux de croissance theoriques sont
//   gamma_3 ~ 0.772, gamma_4 ~ 0.911, gamma_5 ~ 0.683.
// On verifie que l'eigensolver les reproduit a mieux que 2 %.

#include <adc/analysis/diocotron_growth.hpp>

#include <cmath>
#include <cstdio>

using namespace adc;

int main() {
  int fails = 0;
  auto chk = [&](double got, double ref, double tol, const char* w) {
    const double rel = std::fabs(got - ref) / ref;
    if (rel > tol) {
      std::printf("FAIL %s : got %.4f, ref %.4f (rel %.3f)\n", w, got, ref, rel);
      ++fails;
    }
  };

  const double a = 6, b = 8, Rw = 16;
  const double g3 = diocotron_growth_rate(3, a, b, Rw, 1.0, 0.04, 1600);
  const double g4 = diocotron_growth_rate(4, a, b, Rw, 1.0, 0.04, 1600);
  const double g5 = diocotron_growth_rate(5, a, b, Rw, 1.0, 0.04, 1600);
  std::printf("eigensolver : g3=%.4f g4=%.4f g5=%.4f (ref 0.772/0.911/0.683)\n",
              g3, g4, g5);

  chk(g3, 0.772, 0.02, "gamma_3");
  chk(g4, 0.911, 0.02, "gamma_4");
  chk(g5, 0.683, 0.02, "gamma_5");
  // le spectre culmine au mode 4 (comme le papier)
  if (!(g4 > g3 && g4 > g5)) {
    std::printf("FAIL peak_at_4\n");
    ++fails;
  }

  if (fails == 0) std::printf("OK test_diocotron_theory\n");
  return fails == 0 ? 0 : 1;
}
