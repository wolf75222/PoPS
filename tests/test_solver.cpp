// Verifie la facade COMPILEE (libadc, src/) : les solveurs concrets DiocotronSolver
// et EulerPoissonSolver s'utilisent sans aucun template visible, et reproduisent les
// invariants physiques (masse conservee, quantite de mouvement nulle pour la
// gravite interne). C'est le seul test qui LIE la librairie compilee adc::solver.

#include <adc/solver/diocotron_solver.hpp>
#include <adc/solver/euler_poisson_solver.hpp>
#include <adc/solver/two_fluid_ap_solver.hpp>

#include <cmath>
#include <cstdio>

using namespace adc;

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) { std::printf("FAIL %s\n", w); ++fails; }
  };

  // --- DiocotronSolver : masse conservee, API concrete ---
  {
    DiocotronConfig cfg;
    cfg.n = 64;
    DiocotronSolver s(cfg);
    const double m0 = s.mass();
    for (int k = 0; k < 5; ++k) s.step(0.01);
    std::printf("DiocotronSolver : n=%d masse0=%.6e masse=%.6e t=%.3f\n", s.nx(),
                m0, s.mass(), s.time());
    chk(s.nx() == 64, "diocotron_nx");
    chk(static_cast<int>(s.density().size()) == 64 * 64, "diocotron_density_size");
    chk(std::fabs(s.mass() - m0) < 1e-9, "diocotron_masse_conservee");
  }

  // --- EulerPoissonSolver : masse + quantite de mouvement conservees ---
  {
    EulerPoissonConfig cfg;
    cfg.n = 64;
    EulerPoissonSolver s(cfg);
    const double m0 = s.mass();
    for (int k = 0; k < 5; ++k) s.step(0.004);
    std::printf("EulerPoissonSolver : masse=%.6e energie=%.6e p=(%.2e,%.2e)\n",
                s.mass(), s.energy(), s.total_momentum(0), s.total_momentum(1));
    chk(std::fabs(s.mass() - m0) < 1e-9, "ep_masse_conservee");
    chk(std::fabs(s.total_momentum(0)) < 1e-9 &&
            std::fabs(s.total_momentum(1)) < 1e-9,
        "ep_qte_mouvement_nulle");
    chk(std::isfinite(s.energy()), "ep_energie_finie");
  }

  // --- EulerPoissonSolver, backend FFT (n puissance de 2) : memes invariants ---
  {
    EulerPoissonConfig cfg;
    cfg.n = 64;  // puissance de 2
    cfg.use_fft = true;
    EulerPoissonSolver s(cfg);
    const double m0 = s.mass();
    for (int k = 0; k < 5; ++k) s.step(0.004);
    std::printf("EulerPoissonSolver(FFT) : masse=%.6e p=(%.2e,%.2e)\n", s.mass(),
                s.total_momentum(0), s.total_momentum(1));
    chk(std::fabs(s.mass() - m0) < 1e-9, "ep_fft_masse_conservee");
    chk(std::fabs(s.total_momentum(0)) < 1e-9, "ep_fft_qte_mouvement_nulle");
  }

  // --- TwoFluidAPSolver (facade compilee, GPU-portable) : AP raide ---
  {
    TwoFluidAPConfig cfg;
    cfg.n = 64;
    cfg.omega_pe = 1e3;  // raide
    cfg.omega_pi = 20.0;
    TwoFluidAPSolver s(cfg);
    const double m0e = s.mass_e(), m0i = s.mass_i();
    s.advance(5.0 / 1e3, 200);  // dt*omega_pe = 5 : explicite exploserait
    std::printf("TwoFluidAPSolver : n=%d max|dne|=%.3e max|charge|=%.3e dmasse_e=%.2e\n",
                s.nx(), s.max_dev(), s.max_charge(), std::fabs(s.mass_e() - m0e));
    chk(static_cast<int>(s.density_e().size()) == 64 * 64, "tfap_density_size");
    chk(s.max_dev() < 0.1, "tfap_AP_borne");
    chk(s.max_charge() < 0.1, "tfap_quasi_neutre");
    chk(std::fabs(s.mass_e() - m0e) < 1e-7 && std::fabs(s.mass_i() - m0i) < 1e-7,
        "tfap_masse_conservee");
  }

  if (fails == 0) std::printf("OK test_solver\n");
  return fails == 0 ? 0 : 1;
}
