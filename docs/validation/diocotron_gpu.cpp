// Run diocotron HyQMOM-15 sur GPU (Kokkos-Cuda, noeud armgpu H100) -- campagne + validation
// device de dense_eig (prepare ADC-181).
//
// La brique modele est EMISE par la DSL (emit_cpp_brick, fichier hyqmom15_brick.hpp genere
// cote Mac depuis le modele VALIDE : flux + vitesses exactes par jacobien autodiff +
// adc::real_eig_minmax + sources electriques) et branchee par le SEAM DE COMPILATION
// adc::add_compiled_model - le meme chemin natif que add_block (assemble_rhs device, halos),
// zero marshaling. L'etat initial est LU en binaire (ic.raw : 15*n*n doubles row-major
// (comp, y, x)) - calcule par le python valide (diocotron_state), jamais re-porte.
//
// Sorties dans --out : snap_<k>.raw (15*n*n + n*n phi, doubles), growth.csv (t, dt, masse,
// |a_l| l=2..6 sur l'anneau), un par-pas leger. Pas de checkpoint HDF5 cote C++ : la
// trajectoire est courte (walltime short) et les snapshots suffisent pour reprendre une
// analyse ; la reprise bit-stable reste le chemin python (CPU).

#include <adc/physics/composite.hpp>
#include <adc/runtime/dsl_block.hpp>
#include <adc/runtime/system.hpp>

#include "hyqmom15_brick.hpp"

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static double arg_d(int argc, char** argv, const char* key, double dflt) {
  for (int i = 1; i + 1 < argc; ++i)
    if (!std::strcmp(argv[i], key)) return std::atof(argv[i + 1]);
  return dflt;
}
static std::string arg_s(int argc, char** argv, const char* key, const char* dflt) {
  for (int i = 1; i + 1 < argc; ++i)
    if (!std::strcmp(argv[i], key)) return argv[i + 1];
  return dflt;
}

int main(int argc, char** argv) {
  const int n = static_cast<int>(arg_d(argc, argv, "--n", 128));
  const double tend = arg_d(argc, argv, "--tend", 4.0);
  const double cfl = arg_d(argc, argv, "--cfl", 0.4);
  const int snap_every = static_cast<int>(arg_d(argc, argv, "--snap-every", 200));
  const int diag_every = static_cast<int>(arg_d(argc, argv, "--diag-every", 5));
  const double max_steps = arg_d(argc, argv, "--max-steps", 1e9);
  const std::string ic = arg_s(argc, argv, "--ic", "ic.raw");
  const std::string out = arg_s(argc, argv, "--out", "out_gpu");
  // rho_background / debye / omega_p sont CAVES dans les briques emises (make_brick_and_ic.py).
  if (n < 1 || snap_every < 1 || diag_every < 1) {
    std::fprintf(stderr, "[gpu] --n/--snap-every/--diag-every doivent etre >= 1\n");
    return 1;
  }

  if (std::system(("mkdir -p " + out).c_str()) != 0) {
    std::fprintf(stderr, "[gpu] mkdir -p %s a echoue\n", out.c_str());
    return 1;
  }

  // --- etat initial : 15*n*n doubles (comp, y, x), produit par make_ic.py (python valide) ---
  std::vector<double> U0(static_cast<std::size_t>(15) * n * n);
  {
    std::ifstream f(ic, std::ios::binary);
    if (!f) { std::fprintf(stderr, "IC introuvable : %s\n", ic.c_str()); return 2; }
    f.read(reinterpret_cast<char*>(U0.data()),
           static_cast<std::streamsize>(U0.size() * sizeof(double)));
    if (!f) { std::fprintf(stderr, "IC tronquee : %s\n", ic.c_str()); return 2; }
  }

  adc::SystemConfig cfg;
  cfg.n = n;
  cfg.L = 1.0;
  cfg.periodic = true;
  adc::System sys(cfg);

  // briques emises par la DSL (flux + vitesses exactes / source Lorentz / rhs de Poisson),
  // assemblees par CompositeModel et branchees par le seam de COMPILATION (chemin natif
  // complet : assemble_rhs device, halos) - le pattern des tests test_dsl_{brick,source,
  // elliptic,compose} du coeur.
  // une brique elliptique PAR n (rho_background cave, depend de la discretisation du
  // scenario) ; Hyp/Src partagees.
  if (n == 128) {
    using Model = adc::CompositeModel<adc_generated::Hyqmom15Hyp, adc_generated::Hyqmom15Src,
                                      adc_generated::Hyqmom15Ell128>;
    adc::add_compiled_model(sys, "mom", Model{}, "none", "hll", "conservative", "explicit");
  } else if (n == 256) {
    using Model = adc::CompositeModel<adc_generated::Hyqmom15Hyp, adc_generated::Hyqmom15Src,
                                      adc_generated::Hyqmom15Ell256>;
    adc::add_compiled_model(sys, "mom", Model{}, "none", "hll", "conservative", "explicit");
  } else {
    std::fprintf(stderr, "n=%d sans brique elliptique emise (128|256)\n", n);
    return 2;
  }
  sys.set_poisson("charge_density", "geometric_mg");

  sys.set_state("mom", U0);
  sys.solve_fields();

  std::FILE* gf = std::fopen((out + "/growth.csv").c_str(), "w");
  if (!gf) {
    std::fprintf(stderr, "[gpu] impossible d'ouvrir %s/growth.csv\n", out.c_str());
    return 1;
  }
  std::fprintf(gf, "step,t,dt,mass,a2,a3,a4,a5,a6\n");

  double t = 0.0;
  long k = 0;
  double mass0 = 0.0;
  for (std::size_t i = 0; i < static_cast<std::size_t>(n) * n; ++i) mass0 += U0[i];

  double dt0 = 0.0;
  while (t < tend && k < static_cast<long>(max_steps)) {
    const double dt = sys.step_cfl(cfl);
    t += dt;
    ++k;
    if (k == 1) dt0 = dt;
    // GARDE dt : sans relaxation par pas (flagrelax=1 du MATLAB), les vitesses exactes
    // explosent quand l'etat approche le bord de realisabilite -> dt s'effondre et le run
    // gele en marquant le temps (observe : n=256 fige a t=0.089). On sort PROPREMENT avec
    // un diagnostic plutot que de bruler le walltime (la projection compilee = ADC-177).
    if (dt < 1e-4 * dt0) {
      std::fprintf(stderr,
                   "[DT_COLLAPSE] pas %ld t=%.6f : dt=%.3e < 1e-4*dt0 (%.3e) -- etat au "
                   "bord de realisabilite, projection requise (ADC-177). Sortie propre.\n",
                   k, t, dt, dt0);
      break;
    }

    if (k % diag_every == 0 || t >= tend) {
      const std::vector<double> U = sys.state_global("mom");
      double mass = 0.0;
      std::complex<double> a[5] = {};
      double ringsum = 0.0;
      bool finite = true;
      for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
          const double m00 = U[static_cast<std::size_t>(j) * n + i];
          if (!std::isfinite(m00)) finite = false;
          mass += m00;
          const double x = -0.5 + (i + 0.5) / n, y = -0.5 + (j + 0.5) / n;
          const double r = std::sqrt(x * x + y * y);
          if (r > 0.30 && r < 0.45) {
            const double th = std::atan2(y, x);
            ringsum += m00;
            for (int l = 0; l < 5; ++l)
              a[l] += m00 * std::exp(std::complex<double>(0.0, -(l + 2.0) * th));
          }
        }
      }
      if (!finite) { std::fprintf(stderr, "[FATAL] M00 non fini au pas %ld\n", k); return 3; }
      std::fprintf(gf, "%ld,%.8f,%.3e,%.15e", k, t, dt, mass);
      for (int l = 0; l < 5; ++l) std::fprintf(gf, ",%.6e", std::abs(a[l]) / ringsum);
      std::fprintf(gf, "\n");
      std::fflush(gf);
    }
    if (k % snap_every == 0 || t >= tend) {
      const std::vector<double> U = sys.state_global("mom");
      const std::vector<double> phi = sys.potential_global();
      char path[512];
      std::snprintf(path, sizeof path, "%s/snap_%06ld.raw", out.c_str(), k);
      std::ofstream f(path, std::ios::binary);
      const double hdr[3] = {static_cast<double>(n), t, static_cast<double>(k)};
      f.write(reinterpret_cast<const char*>(hdr), sizeof hdr);
      f.write(reinterpret_cast<const char*>(U.data()),
              static_cast<std::streamsize>(U.size() * sizeof(double)));
      f.write(reinterpret_cast<const char*>(phi.data()),
              static_cast<std::streamsize>(phi.size() * sizeof(double)));
      std::printf("[snap] pas %ld t=%.5f -> %s\n", k, t, path);
      std::fflush(stdout);
    }
  }

  const std::vector<double> U = sys.state_global("mom");
  double mass = 0.0;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) mass += U[static_cast<std::size_t>(j) * n + i];
  std::printf("[fin] %ld pas, t=%.5f, derive de masse %.2e\n", k, t,
              std::abs(mass - mass0) / mass0);
  std::fclose(gf);
  return 0;
}
