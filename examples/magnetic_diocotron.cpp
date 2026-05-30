// Instabilite diocotron sous le systeme Euler-Poisson MAGNETIQUE COMPLET (Hoffart
// eq 2.4), pas la limite de derive. Bande de charge periodique perturbee au mode k :
// le cisaillement de la derive E x B aux bords de bande enroule la perturbation
// (Kelvin-Helmholtz electrostatique). Ici la dynamique cyclotron est RESOLUE
// (MagneticEulerPoissonCoupler, splitting de Strang) au lieu d'etre supposee
// instantanee comme dans le modele Diocotron.
//
// Le point : le schema est asymptotic-preserving. A grand Omega le pas de temps reste
// gouverne par la derive lente (CFL hydro), PAS par la frequence cyclotron : on tourne
// stablement et le taux de croissance converge vers la limite de derive (M1/M2) quand
// Omega grandit. On ecrit band_amp.csv (t, amplitude du mode k de phi) pour le fit, et
// on reporte la conservation de masse et d'energie + la vitesse max.
//
// Run : ./build/bin/magnetic_diocotron <out> [nc] [nsteps] [Omega] [kmode]

#include <adc/coupling/coupler.hpp>
#include <adc/integrator/magnetic_euler_poisson.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/model/euler_poisson.hpp>
#include <adc/operator/reconstruction.hpp>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;

int main(int argc, char** argv) {
  const std::string out = (argc > 1) ? argv[1] : "mag_dio";
  const int nc = (argc > 2) ? std::atoi(argv[2]) : 128;
  const int nsteps = (argc > 3) ? std::atoi(argv[3]) : 800;
  const double Omega = (argc > 4) ? std::atof(argv[4]) : 20.0;
  const int kmode = (argc > 5) ? std::atoi(argv[5]) : 4;
  std::filesystem::create_directories(out);

  const double L = 1.0, dx = L / nc, dy = dx;
  const double gamma = 5.0 / 3.0, p0 = 0.02;  // froid (la pression doit rester petite)
  const double alpha = 40.0;                  // intensite du couplage electrostatique
  const double y1 = 0.40 * L, y2 = 0.60 * L;  // bande de charge
  const double delta = 0.6;                   // exces de charge dans la bande
  const double eps = 1e-3, kx = 2 * kPi * kmode / L;

  Box2D dom = Box2D::from_extents(nc, nc);
  Geometry geom{dom, 0.0, L, 0.0, L};
  BoxArray ba(std::vector<Box2D>{dom});
  DistributionMapping dm(1, 1);

  // densite : fond 1 + bande lissee [y1,y2] (exces delta), bord perturbe au mode k.
  auto dens = [&](double x, double y) {
    const double yedge = y - eps * std::cos(kx * x);  // ondulation du bord de bande
    const double w = 0.02 * L;                         // epaisseur de transition
    const double s = 0.5 * (std::tanh((yedge - y1) / w) - std::tanh((yedge - y2) / w));
    return 1.0 + delta * s;
  };

  MultiFab U(ba, dm, 4, 2);
  double mean_rho = 0;
  {
    Fab2D& f = U.fab(0);
    const Box2D v = U.box(0);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
        const double rho = dens((i + 0.5) * dx, (j + 0.5) * dy);
        f(i, j, 0) = rho;
        f(i, j, 1) = 0.0;                  // m_x : la derive se developpe seule
        f(i, j, 2) = 0.0;                  // m_y
        f(i, j, 3) = p0 / (gamma - 1);     // E = p/(gamma-1), energie cinetique nulle
        mean_rho += rho;
      }
    mean_rho /= static_cast<double>(nc) * nc;
  }

  EulerPoisson model;
  model.hydro.gamma = gamma;
  model.four_pi_G = alpha;
  model.coupling_sign = -1;     // electrostatique mono-espece (repulsif)
  model.rho0 = mean_rho;        // fond neutralisant : solvabilite periodique (rhs moyen nul)

  BCRec bcU, bcPhi;             // periodique
  MagneticEulerPoissonCoupler<> mag(model, geom, ba, bcU, bcPhi, Omega);

  // initialisation SUR LA VARIETE DE DERIVE (etat pertinent en regime AP) : on resout
  // une fois le champ et on pose m = rho v_ExB, v_ExB = (1/Omega)(-d_y phi, d_x phi).
  // aux du coupler = (phi, +d_x phi, +d_y phi). Le cisaillement de derive agit alors des
  // t = 0 (au lieu de devoir s'etablir), comme dans la limite de derive du modele Diocotron.
  {
    mag.solve_fields(U);
    const ConstArray4 a = mag.aux().fab(0).const_array();
    Fab2D& f = U.fab(0);
    const Box2D v = U.box(0);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
        const double rho = f(i, j, 0);
        const double vx = -a(i, j, 2) / Omega, vy = a(i, j, 1) / Omega;  // v_ExB
        f(i, j, 1) = rho * vx;
        f(i, j, 2) = rho * vy;
        f(i, j, 3) = p0 / (gamma - 1) + 0.5 * rho * (vx * vx + vy * vy);  // E += KE derive
      }
  }

  // amplitude du mode k de phi echantillonne le long du bord de bande y = y1.
  auto mode_amp = [&]() {
    mag.solve_fields(U);
    const ConstArray4 p = mag.phi().fab(0).const_array();
    const int jb = std::min(nc - 1, std::max(0, (int)std::floor(y1 / dy - 0.5)));
    double re = 0, im = 0;
    for (int i = 0; i < nc; ++i) {
      const double x = (i + 0.5) * dx;
      re += p(i, jb, 0) * std::cos(kx * x);
      im += p(i, jb, 0) * std::sin(kx * x);
    }
    return 2.0 * std::sqrt(re * re + im * im) / nc;
  };
  auto total_energy = [&]() { return sum(U, 3); };
  auto max_speed = [&]() {
    const ConstArray4 u = U.fab(0).const_array();
    const Box2D v = U.box(0);
    double s = 0;
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
        const double sp = std::sqrt(u(i, j, 1) * u(i, j, 1) + u(i, j, 2) * u(i, j, 2)) /
                          u(i, j, 0);
        s = std::max(s, sp);
      }
    return s;
  };
  auto dump = [&](int frame) {
    char name[64];
    std::snprintf(name, sizeof(name), "/dens_%04d.txt", frame);
    std::ofstream f(out + name);
    const ConstArray4 u = U.fab(0).const_array();
    for (int j = 0; j < nc; ++j)
      for (int i = 0; i < nc; ++i) f << u(i, j, 0) << (i + 1 < nc ? ' ' : '\n');
  };

  std::ofstream amp(out + "/band_amp.csv");
  amp << "# diocotron magnetique complet nc=" << nc << " Omega=" << Omega
      << " k=" << kmode << " alpha=" << alpha << "\n";
  amp << "t,amplitude\n";

  const double M0 = sum(U, 0), E0 = total_energy();
  const int snap_every = std::max(1, nsteps / 30);
  double t = 0;
  int frame = 0;
  std::printf("diocotron magnetique nc=%d Omega=%.1f k=%d nsteps=%d\n", nc, Omega, kmode,
              nsteps);

  for (int s = 0; s <= nsteps; ++s) {
    amp << t << ',' << mode_amp() << '\n';
    if (s % snap_every == 0) {
      dump(frame++);
      // dEgaz = variation de l'energie du GAZ : NON nulle car le champ travaille sur
      // le gaz (source -m . grad phi), echange physique gaz <-> champ ; la masse, elle,
      // est l'invariant exact.
      std::printf("  s=%5d t=%7.3f a_k=%.4e vmax=%.3f dmasse=%.1e dEgaz=%.1e\n", s, t,
                  mode_amp(), max_speed(), std::fabs(sum(U, 0) - M0),
                  std::fabs(total_energy() - E0));
    }
    if (s == nsteps) break;
    // CFL hydro (son + derive), INDEPENDANTE d'Omega : c'est la propriete AP.
    const double dt = 0.4 * dx / (std::sqrt(gamma * p0) + max_speed() + 1e-6);
    mag.step<Minmod>(U, dt);
    t += dt;
  }
  amp.close();
  std::printf("ecrit %s/band_amp.csv + %d instantanes ; dmasse=%.2e (invariant) "
              "dEgaz=%.2e (echange champ-gaz, physique)\n",
              out.c_str(), frame, std::fabs(sum(U, 0) - M0),
              std::fabs(total_energy() - E0));
  return 0;
}
