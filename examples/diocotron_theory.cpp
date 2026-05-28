// Taux de croissance lineaire du diocotron par l'eigensolver radial (host Eigen).
// Reproduit la theorie de Davidson-Felice / Hoffart et al. (fig 5.4) et ecrit
// le spectre gamma_l(l) pour comparaison.
//
// Run : ./build/bin/diocotron_theory [a b Rw] [out.csv]
//   defaut : geometrie du papier (6 8 16), verification gamma_3/4/5.

#include <adc/analysis/diocotron_growth.hpp>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

using namespace adc;

int main(int argc, char** argv) {
  double a = 6, b = 8, Rw = 16;
  std::string out = "diocotron_theory.csv";
  if (argc >= 4) {
    a = std::atof(argv[1]);
    b = std::atof(argv[2]);
    Rw = std::atof(argv[3]);
  }
  if (argc >= 5) out = argv[4];

  std::printf("eigensolver radial diocotron (a=%.3g b=%.3g Rw=%.3g)\n", a, b, Rw);
  std::printf("  reference papier (6,8,16) : g3=0.772 g4=0.911 g5=0.683\n");
  std::printf("  l   gamma/omega_D\n");

  std::ofstream csv(out);
  csv << "# diocotron growth rate (radial eigensolver), a=" << a << " b=" << b
      << " Rw=" << Rw << "\n";
  csv << "l,gamma\n";
  for (int l = 2; l <= 7; ++l) {
    const double g = diocotron_growth_rate(l, a, b, Rw, 1.0, 0.04, 1600);
    std::printf("  %d   %.4f\n", l, g);
    csv << l << ',' << g << '\n';
  }
  std::printf("ecrit %s\n", out.c_str());
  return 0;
}
