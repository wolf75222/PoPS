// Deux-fluides isotherme lineaire (model/two_fluid_isothermal.hpp), generalisation
// de LangmuirMode aux deux especes mobiles. On valide :
//   1. dispersion : les deux racines (Langmuir + ion-acoustique) satisfont Vieta
//      (somme/produit) et les limites k->0 ; ion-acoustique non nul (ions mobiles).
//   2. AP : a omega_pe tres grand (raide), l'IMEX (plasma implicite, acoustique
//      explicite) reste BORNE la ou l'explicite EXPLOSE.
//   3. coherence : a frequence resolue, le schema reproduit la branche de Langmuir.

#include <adc/model/two_fluid_isothermal.hpp>

#include <cmath>
#include <cstdio>

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;

// IMEX (explicite acoustique + implicite plasma) ; renvoie max|Ae| et trajectoire
// via le premier passage a zero de Ae (pour mesurer la frequence).
static void run_imex(const TwoFluidLinear& m, double Ae0, double Ai0, double dt,
                     int n, double& maxAe, double& tzero) {
  double Ae = Ae0, Ai = Ai0, Be = 0, Bi = 0, t = 0, prev = Ae0;
  maxAe = std::fabs(Ae0);
  tzero = -1;
  for (int s = 0; s < n; ++s) {
    m.explicit_step(Ae, Ai, Be, Bi, dt);
    m.implicit_solve(Ae, Ai, Be, Bi, dt);
    t += dt;
    if (tzero < 0 && Ae < 0 && prev > 0)
      tzero = (t - dt) + dt * prev / (prev - Ae);
    prev = Ae;
    maxAe = std::fmax(maxAe, std::fabs(Ae));
  }
}

// Euler explicite sur le systeme complet Ä = K A ; renvoie max|Ae|.
static double run_explicit_maxAe(const TwoFluidLinear& m, double dt, int n) {
  const double wpe2 = m.omega_pe * m.omega_pe, wpi2 = m.omega_pi * m.omega_pi;
  double Ae = 1, Ai = 0, Be = 0, Bi = 0, maxAe = 1;
  for (int s = 0; s < n; ++s) {
    const double ae = Ae, ai = Ai, be = Be, bi = Bi;
    Ae = ae + dt * be;
    Ai = ai + dt * bi;
    Be = be + dt * (-(m.cse2k2 + wpe2) * ae + wpe2 * ai);
    Bi = bi + dt * (wpi2 * ae - (m.csi2k2 + wpi2) * ai);
    maxAe = std::fmax(maxAe, std::fabs(Ae));
  }
  return maxAe;
}

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) { std::printf("FAIL %s\n", w); ++fails; }
  };

  // --- 1. dispersion : Vieta + limites + deux branches ---
  {
    TwoFluidLinear m;
    m.omega_pe = 2.0;
    m.omega_pi = 0.5;
    m.cse2k2 = 1.0;
    m.csi2k2 = 0.1;
    double wf, ws;
    m.dispersion(wf, ws);
    const double S = m.cse2k2 + m.csi2k2 + 4.0 + 0.25;          // wf^2 + ws^2
    const double P = m.cse2k2 * m.csi2k2 + 4.0 * m.csi2k2 + 0.25 * m.cse2k2;  // produit
    std::printf("dispersion : w_fast=%.4f w_slow=%.4f | Vieta S=%.4f(%.4f) P=%.4f(%.4f)\n",
                wf, ws, wf * wf + ws * ws, S, wf * wf * ws * ws, P);
    chk(std::fabs(wf * wf + ws * ws - S) < 1e-9, "vieta_somme");
    chk(std::fabs(wf * wf * ws * ws - P) < 1e-9, "vieta_produit");
    chk(wf > ws && ws > 0, "deux_branches_distinctes");

    // limite k -> 0 : Langmuir -> sqrt(wpe^2+wpi^2), ion-acoustique -> 0
    TwoFluidLinear m0 = m;
    m0.cse2k2 = m0.csi2k2 = 0;
    m0.dispersion(wf, ws);
    chk(std::fabs(wf - std::sqrt(4.0 + 0.25)) < 1e-9 && ws < 1e-9, "limite_k0");
  }

  // --- 2. AP : raide -> IMEX borne, explicite explose ---
  {
    TwoFluidLinear m;
    m.omega_pe = 1e3;
    m.omega_pi = 20.0;
    m.cse2k2 = 1.0;
    m.csi2k2 = 0.01;
    const double dt = 0.01;
    double maxAe, tz;
    run_imex(m, 1.0, 0.0, dt, 200, maxAe, tz);
    const double me = run_explicit_maxAe(m, dt, 200);
    std::printf("AP raide (omega_pe=%.0e, dt*omega_pe=%.0f) : IMEX max|Ae|=%.4f | "
                "explicite max|Ae|=%.3e\n",
                m.omega_pe, dt * m.omega_pe, maxAe, me);
    chk(std::isfinite(maxAe) && maxAe < 2.0, "imex_AP_borne");
    chk(!std::isfinite(me) || me > 1e6, "explicite_explose");
  }

  // --- 3. coherence : le schema reproduit la branche de Langmuir (resolu) ---
  {
    TwoFluidLinear m;
    m.omega_pe = 2.0;
    m.omega_pi = 0.5;
    m.cse2k2 = 1.0;
    m.csi2k2 = 0.1;
    double wf, ws;
    m.dispersion(wf, ws);
    // mode propre rapide : A_i/A_e = (cse2k2 + wpe^2 - wf^2)/wpe^2
    const double ratio =
        (m.cse2k2 + m.omega_pe * m.omega_pe - wf * wf) / (m.omega_pe * m.omega_pe);
    const double dt = (2 * kPi / wf) / 400;  // ~400 pas par periode (resolu)
    double maxAe, tz;
    run_imex(m, 1.0, ratio, dt, 400, maxAe, tz);
    const double w_meas = (tz > 0) ? kPi / (2 * tz) : 0.0;
    std::printf("coherence : w_fast theorique=%.4f mesure=%.4f (ecart %.1f%%)\n", wf,
                w_meas, 100 * std::fabs(w_meas - wf) / wf);
    chk(tz > 0 && std::fabs(w_meas - wf) / wf < 0.03, "schema_reproduit_langmuir");
  }

  if (fails == 0) std::printf("OK test_two_fluid\n");
  return fails == 0 ? 0 : 1;
}
