// PARITE MPI du capstone AMR MULTI-BLOCS (PR1). Pendant multi-blocs de test_mpi_amr_compiled_parity :
// DEUX blocs EXPLICITES a schemas DIFFERENTS co-localises sur UNE hierarchie AMR PARTAGEE (Poisson de
// systeme a second membre SOMME q0 n0 + q1 n1), DISTRIBUES sur n_ranks(). Propriete verifiee :
//   (1) CONSISTANCE CROSS-RANG : le grossier etant REPLIQUE, la masse de CHAQUE bloc, la densite de
//       chaque bloc et le potentiel de systeme sont des grandeurs GLOBALES identiques sur tous les
//       rangs (spread max reduit == 0). Un bug de halo / Poisson somme / aux distant le casserait.
//   (2) PARITE AU NB DE RANGS : on imprime des checksums (par bloc + potentiel) ; le script de build
//       relance le MEME binaire en np=1/2/4 et DIFF (np=1 = oracle ; np=2/4 BIT-IDENTIQUES).
//   (3) REDUCTIONS COMPOSITES : le chemin AmrRuntime multi-bloc doit distinguer le grossier
//       replique (deja global sur chaque rang, donc jamais re-somme) du grossier distribue (somme
//       des proprietaires). Pour sum / abs_sum / sum_sq, les deux politiques doivent retrouver la
//       meme grandeur physique qu'un parcours independant de tous les niveaux avec le masque des
//       cellules grossieres couvertes.
//
// Hierarchie FIGEE (regrid_every=0) : multi-blocs PR1 n'a pas de regrid (AmrRuntime ; le regrid
// d'union des tags est une PR ulterieure). On exerce neanmoins le grossier replique + le patch fin
// central multi-patch + le Poisson somme co-localise distribues. Independant du backend (Kokkos
// Serial CI, Kokkos Cuda GH200).
#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/config/model_spec.hpp>
#include <pops/parallel/comm.hpp>  // comm_init, my_rank, n_ranks, all_reduce_*

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

static ModelSpec exb_charge(double q, double B0) {
  ModelSpec s;
  s.transport = "exb";
  s.source = "none";
  s.elliptic = "charge";
  s.q = q;
  s.B0 = B0;
  return s;
}

// creneau lisse a moyenne (offset) nulle, n*n row-major (charge totale solvable en periodique).
static std::vector<double> bump(int n, double amp) {
  std::vector<double> r(static_cast<std::size_t>(n) * n, 1.0);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const double x = (i + 0.5) / n, y = (j + 0.5) / n;
      const double dx = x - 0.5, dy = y - 0.5;
      r[static_cast<std::size_t>(j) * n + i] = 1.0 + amp * std::exp(-(dx * dx + dy * dy) / 0.01);
    }
  // retire l'offset moyen -> Sum q n a moyenne nulle (Poisson periodique solvable).
  double mean = 0;
  for (double v : r)
    mean += v;
  mean /= static_cast<double>(r.size());
  for (double& v : r)
    v += (1.0 - mean);
  return r;
}

struct CompositeReductions {
  double sum = 0.0;
  double abs_sum = 0.0;
  double sum_sq = 0.0;
};

// Independent composite reference for component zero. This deliberately does not use density(): it
// walks the public per-level checkpoint views and applies the AMR covered-cell rule itself. Thus
// abs_sum and sum_sq remain valid when a fine state differs from its coarse average.
static CompositeReductions composite_oracle(AmrSystem& system, const char* block, int base_n,
                                            std::vector<int> selected = {}) {
  CompositeReductions result;
  const int n_levels = system.n_levels();
  if (selected.empty())
    for (int level = 0; level < n_levels; ++level)
      selected.push_back(level);
  const std::vector<PatchBox> boxes = system.patch_boxes();
  for (std::size_t selected_index = 0; selected_index < selected.size(); ++selected_index) {
    const int level = selected[selected_index];
    const int next = selected_index + 1 < selected.size() ? selected[selected_index + 1] : -1;
    const std::size_t width = static_cast<std::size_t>(base_n) << level;
    const std::vector<double> state = system.block_level_state_global(block, level);
    if (state.size() < width * width)
      continue;
    const double dx = 1.0 / static_cast<double>(width);
    const double cell = dx * dx;
    for (std::size_t j = 0; j < width; ++j)
      for (std::size_t i = 0; i < width; ++i) {
        bool covered = false;
        for (const PatchBox& fine : boxes) {
          if (fine.level != next)
            continue;
          const int shift = next - level;
          if (static_cast<int>(i) >= (fine.ilo >> shift) &&
              static_cast<int>(i) <= (fine.ihi >> shift) &&
              static_cast<int>(j) >= (fine.jlo >> shift) &&
              static_cast<int>(j) <= (fine.jhi >> shift)) {
            covered = true;
            break;
          }
        }
        if (covered)
          continue;
        const double value = state[j * width + i];
        result.sum += cell * value;
        result.abs_sum += cell * std::fabs(value);
        result.sum_sq += cell * value * value;
      }
  }
  return result;
}

static CompositeReductions composite_reductions(const AmrSystem& system, const char* block,
                                                const std::vector<int>& levels = {}) {
  return {system.composite_reduce(block, "sum", 0, levels),
          system.composite_reduce(block, "abs_sum", 0, levels),
          system.composite_reduce(block, "sum_sq", 0, levels)};
}

static double composite_reduction_error(const CompositeReductions& actual,
                                        const CompositeReductions& expected) {
  return std::fmax(std::fabs(actual.sum - expected.sum),
                   std::fmax(std::fabs(actual.abs_sum - expected.abs_sum),
                             std::fabs(actual.sum_sq - expected.sum_sq)));
}

struct RunResult {
  std::vector<double> ions;
  std::vector<double> electrons;
  std::vector<double> potential;
  CompositeReductions ions_reduce;
  CompositeReductions electrons_reduce;
  CompositeReductions ions_oracle;
  CompositeReductions electrons_oracle;
  CompositeReductions coarse_reduce;
  CompositeReductions coarse_oracle;
  CompositeReductions fine_reduce;
  CompositeReductions fine_oracle;
  double initial_ions_mass = 0.0;
  double initial_electrons_mass = 0.0;
  double ions_mass = 0.0;
  double electrons_mass = 0.0;
};

static int pops_run_test_mpi_amr_twoblock_parity(int argc, char** argv) {
  comm_init(&argc, &argv);
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#else
  (void)argc;
  (void)argv;
#endif
  const int me = my_rank(), np = n_ranks();
  const int n = 32;
  const double B0 = 1.0, q0 = +1.0, q1 = -1.0;
  const std::vector<double> rho0 = bump(n, 0.40);
  const std::vector<double> rho1 = bump(n, 0.20);

  const double dt = 1e-3;
  const auto run_mode = [&](bool distribute_coarse) {
    AmrSystemConfig cfg;
    cfg.n = n;
    cfg.L = 1.0;
    cfg.periodic = true;
    cfg.regrid_every = 0;  // multi-blocs PR1 : hierarchie FIGEE
    cfg.distribute_coarse = distribute_coarse;
    cfg.coarse_max_grid = distribute_coarse ? n / 2 : 0;

    AmrSystem sys(cfg);
    sys.set_temporal_relations({2}, {1}, {"integral_only"});
    sys.add_block("ions", exb_charge(q0, B0), "none", "rusanov", "conservative", "explicit", 1);
    sys.add_block("electrons", exb_charge(q1, B0), "minmod", "rusanov", "conservative", "explicit",
                  1);  // SCHEMA DIFFERENT
    sys.set_poisson("charge_density", "geometric_mg", "periodic");
    sys.set_density("ions", rho0);
    sys.set_density("electrons", rho1);

    RunResult result;
    result.initial_ions_mass = sys.mass("ions");  // declenche le runtime multi-bloc paresseux
    result.initial_electrons_mass = sys.mass("electrons");
    for (int s = 0; s < 16; ++s)
      sys.step(dt);

#if defined(POPS_HAS_KOKKOS)
    Kokkos::fence();
#endif
    result.ions = sys.density("ions");
    result.electrons = sys.density("electrons");
    result.potential = sys.potential();
    result.ions_mass = sys.mass("ions");
    result.electrons_mass = sys.mass("electrons");
    result.ions_reduce = composite_reductions(sys, "ions");
    result.electrons_reduce = composite_reductions(sys, "electrons");
    result.ions_oracle = composite_oracle(sys, "ions", n);
    result.electrons_oracle = composite_oracle(sys, "electrons", n);
    result.coarse_reduce = composite_reductions(sys, "ions", {0});
    result.coarse_oracle = composite_oracle(sys, "ions", n, {0});
    if (sys.n_levels() > 1) {
      result.fine_reduce = composite_reductions(sys, "ions", {1});
      result.fine_oracle = composite_oracle(sys, "ions", n, {1});
    }
    return result;
  };

  const RunResult replicated = run_mode(false);
  const RunResult distributed = run_mode(true);
  const std::vector<double>& di = replicated.ions;
  const std::vector<double>& de = replicated.electrons;
  const std::vector<double>& phi = replicated.potential;
  const double mi = replicated.ions_mass, mass_e = replicated.electrons_mass;
  const double m0i = replicated.initial_ions_mass, m0e = replicated.initial_electrons_mass;
  const double replicated_ions_reduce_error =
      composite_reduction_error(replicated.ions_reduce, replicated.ions_oracle);
  const double replicated_electrons_reduce_error =
      composite_reduction_error(replicated.electrons_reduce, replicated.electrons_oracle);
  const double distributed_ions_reduce_error =
      composite_reduction_error(distributed.ions_reduce, distributed.ions_oracle);
  const double distributed_electrons_reduce_error =
      composite_reduction_error(distributed.electrons_reduce, distributed.electrons_oracle);
  const double replicated_coarse_error =
      composite_reduction_error(replicated.coarse_reduce, replicated.coarse_oracle);
  const double distributed_coarse_error =
      composite_reduction_error(distributed.coarse_reduce, distributed.coarse_oracle);
  const double replicated_fine_error =
      composite_reduction_error(replicated.fine_reduce, replicated.fine_oracle);
  const double distributed_fine_error =
      composite_reduction_error(distributed.fine_reduce, distributed.fine_oracle);

  auto checksum = [](const std::vector<double>& v) {
    double s = 0;
    for (double x : v)
      s += x * x;
    return s;
  };
  const double ci = checksum(di), ce = checksum(de), cp = checksum(phi);

  // (1) CONSISTANCE CROSS-RANG : grossier replique -> chaque grandeur globale identique sur tout
  // rang. spread = max(max - min) sur les checksums + masses ; == 0 ssi bit-identique cross-rang.
  auto spread = [](double x) { return all_reduce_max(x) - (-all_reduce_max(-x)); };
  const double sp = std::fmax(std::fmax(spread(ci), spread(ce)),
                              std::fmax(spread(cp), std::fmax(spread(mi), spread(mass_e))));

  int fails = 0;
  if (me == 0) {
    std::printf(
        "AMRMB np=%d | mass_ions=%.17e mass_elec=%.17e | csum_ions=%.17e csum_elec=%.17e "
        "csum_phi=%.17e | crossrank_spread=%.3e | composite_rep=(%.3e,%.3e) "
        "composite_dist=(%.3e,%.3e)\n",
        np, mi, mass_e, ci, ce, cp, sp, replicated_ions_reduce_error,
        replicated_electrons_reduce_error, distributed_ions_reduce_error,
        distributed_electrons_reduce_error);
    std::printf("AMRMB conservation: dm_ions=%.3e dm_elec=%.3e\n", std::fabs(mi - m0i),
                std::fabs(mass_e - m0e));
    if (!(di.size() == static_cast<std::size_t>(n) * n)) {
      std::printf("FAIL taille densite\n");
      ++fails;
    }
    if (!(cp > 1e-12)) {
      std::printf("FAIL potentiel trivial (Poisson somme inactif)\n");
      ++fails;
    }
    // masse de CHAQUE bloc conservee (transport conservatif periodique, par bloc).
    if (!(std::fabs(mi - m0i) < 1e-9)) {
      std::printf("FAIL masse ions non conservee\n");
      ++fails;
    }
    if (!(std::fabs(mass_e - m0e) < 1e-9)) {
      std::printf("FAIL masse electrons non conservee\n");
      ++fails;
    }
    // grossier replique : tout bit-identique cross-rang (spread exactement 0).
    if (!(sp == 0.0)) {
      std::printf("FAIL grandeurs non bit-identiques entre rangs\n");
      ++fails;
    }
    // The replicated run must not apply MPI SUM to an already-global level 0. The distributed
    // run must apply it exactly once. Checking all three additive reductions catches both the
    // raw-state and transformed-value paths (abs / square).
    constexpr double kReductionTolerance = 1e-11;
    if (!(replicated_ions_reduce_error < kReductionTolerance &&
          replicated_electrons_reduce_error < kReductionTolerance)) {
      std::printf("FAIL composite_reduce replique multiplie ou perd le grossier\n");
      ++fails;
    }
    if (!(distributed_ions_reduce_error < kReductionTolerance &&
          distributed_electrons_reduce_error < kReductionTolerance)) {
      std::printf("FAIL composite_reduce distribue ne reconstruit pas le grossier\n");
      ++fails;
    }
    if (!(replicated_coarse_error < kReductionTolerance &&
          distributed_coarse_error < kReductionTolerance &&
          replicated_fine_error < kReductionTolerance &&
          distributed_fine_error < kReductionTolerance)) {
      std::printf("FAIL composite_reduce ne respecte pas la selection exacte des niveaux\n");
      ++fails;
    }
    if (fails == 0)
      std::printf(
          "OK test_mpi_amr_twoblock_parity np=%d (multi-blocs AMR : Poisson somme "
          "co-localise, masse par bloc, bit-identique cross-rang)\n",
          np);
  } else {
    (void)sp;
  }
  comm_finalize();
  return fails ? 1 : 0;
}

TEST(test_mpi_amr_twoblock_parity, Runs) {
  EXPECT_EQ(pops::test::RunTestBody(&pops_run_test_mpi_amr_twoblock_parity,
                                    "test_mpi_amr_twoblock_parity"),
            0);
}
