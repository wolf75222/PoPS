// STRONG-SCALING AMR : grossier REPARTI cable dans AmrSystem (deliverable C, perf full-device).
//
// test_mpi_amr_compiled_parity valide la hierarchie AMR + MPI + modele compile avec le grossier
// REPLIQUE (defaut) : le Poisson grossier et le transport grossier sont REDONDANTS sur chaque rang,
// donc le run NE SCALE PAS (cf docs/GPU_RUNTIME_PORT.md phase 10). Ce test exerce le MODE SCALABLE
// (AmrSystemConfig::distribute_coarse=true) : le niveau grossier devient MULTI-BOX (BoxArray::from_domain)
// REPARTI round-robin sur les rangs, le Poisson grossier (GeometricMG multi-box) et le transport
// grossier se DISTRIBUENT. C'est le chemin du strong-scaling AMR.
//
// Ce qu'on verifie (criteres d'HONNETETE du deliverable) :
//   (1) CORRECTION PHYSIQUE : le grossier reparti donne le MEME champ que le grossier replique a
//       l'arrondi pres. On construit DEUX AmrSystem dans le meme binaire (replique = oracle,
//       reparti) avec exactement la meme init et la meme sequence de pas, et on compare la densite
//       grossiere finale. La densite est reconstruite GLOBALEMENT (chaque rang n'a que ses tuiles ->
//       coupler_read_coarse all_reduce les boites disjointes), donc dens.size()==n*n sur chaque rang.
//   (2) MAX CROSS-RANG BIT-IDENTIQUE : cmax (reduction max, INSENSIBLE a l'ordre de sommation) doit
//       etre identique a tous les np. C'est le critere bit-exact que la doc exige pour le reparti
//       (les sommes additives, elles, dependent de l'ordre de reduction FMA quand le grossier est
//       genuinement decoupe -- documente pour #59 ; on ne l'exige donc PAS bit a bit ici).
//   (3) CONSERVATION : masse conservee a l'arrondi (reflux conservatif + all_reduce_sum).
//   (4) GLOBAL ACCESSORS: the replicated coarse is never summed over ranks (which would multiply
//       it by np); the distributed coarse reconstructs the exact global state/potential. The
//       checkpoint views are compared to the independent density()/potential() production reads.
//   (5) MG CONVERGE : phi reste fini et le champ non trivial (pas de divergence du multigrille
//       geometrique sur le grossier multi-box). Couvert par (1) : un MG diverge -> NaN -> echec.
//
// Independant du backend : Kokkos Serial (CI, CPU) et Cuda (GH200). Le script ROMEO relance le MEME
// binaire en np=1/2/4 et diff cmax (bit-identique attendu).
#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/physics/bricks/bricks.hpp>  // CompositeModel, GravityForce, GravityCoupling
#include <pops/physics/fluids/euler.hpp>   // Euler
#include <pops/runtime/amr/bootstrap_transfer_builtins.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>  // add_compiled_model(AmrSystem, ...)
#include <pops/runtime/amr_system.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/parallel/world_communicator.hpp>

#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;
using Model = CompositeModel<Euler, GravityForce, GravityCoupling>;

static bool bootstrap_volume_average_replicates_parent() {
  const Box2D coarse_domain = Box2D::from_extents(4, 4);
  const Box2D fine_domain = coarse_domain.refine(2);
  const BoxArray coarse_boxes({coarse_domain});
  const BoxArray fine_boxes({fine_domain});

  // The bootstrap level-zero contract is one parent copy per rank.  Its rank-local mapping is
  // intentionally different on every rank, whereas the child mapping is globally identical.
  MultiFab coarse(coarse_boxes, DistributionMapping({my_rank()}), 1, 0);
  MultiFab fine(fine_boxes, DistributionMapping(fine_boxes.size(), n_ranks()), 1, 0);
  coarse.set_val(Real(-1));
  fine.set_val(Real(7));

  const auto restriction = runtime::amr::prepare_volume_average();
  restriction.spatial(fine, coarse,
                      runtime::amr::SpatialTransferContext{
                          0, 1, 1,
                          runtime::amr::IndexTransform{{coarse_domain.lo[0], coarse_domain.lo[1]},
                                                       {fine_domain.lo[0], fine_domain.lo[1]},
                                                       {2, 2}},
                          true});

  double local_error = 0.0;
  if (coarse.local_size() != 1) {
    local_error = std::numeric_limits<double>::infinity();
  } else {
    const auto values = coarse.fab(0).const_array();
    for (int j = coarse_domain.lo[1]; j <= coarse_domain.hi[1]; ++j)
      for (int i = coarse_domain.lo[0]; i <= coarse_domain.hi[0]; ++i)
        local_error =
            std::fmax(local_error, std::fabs(static_cast<double>(values(i, j, 0) - Real(7))));
  }
  return all_reduce_max(local_error) == 0.0;
}

static std::vector<double> four_bubbles(int n) {
  std::vector<double> rho(static_cast<std::size_t>(n) * n);
  const double cx[4] = {0.25, 0.75, 0.25, 0.75};
  const double cy[4] = {0.25, 0.25, 0.75, 0.75};
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const double x = (i + 0.5) / n, y = (j + 0.5) / n;
      double r = 1.0;
      for (int b = 0; b < 4; ++b) {
        const double dx = x - cx[b], dy = y - cy[b];
        r += 0.5 * std::exp(-(dx * dx + dy * dy) / 0.004);
      }
      rho[static_cast<std::size_t>(j) * n + i] = r;
    }
  return rho;
}

// Construit un AmrSystem (4 bulles, euler_poisson compile), avance nsteps, rend la densite grossiere
// GLOBALE (n*n) + masse finale + m0. distribute => grossier multi-box reparti (sinon replique).
struct Result {
  std::vector<double> dens;
  std::vector<double> state;
  std::vector<OutputPiece> output_local_pieces;
  std::vector<OutputPiece> output_root_pieces;
  std::vector<double> phi;
  std::vector<double> phi_global;
  double mass, m0;
  int npf;
};

struct OutputPieceCheck {
  long failures = 0;
  long cells = 0;
  double value_error = 0.0;
};

static OutputPieceCheck check_output_pieces(const std::vector<OutputPiece>& pieces,
                                            const std::vector<double>& global, int n,
                                            bool replicated, bool require_local_owner = true) {
  OutputPieceCheck check;
  const std::size_t cells = static_cast<std::size_t>(n) * n;
  if (cells == 0 || global.size() % cells != 0) {
    check.failures = 1;
    return check;
  }
  const int ncomp = static_cast<int>(global.size() / cells);
  for (const OutputPiece& piece : pieces) {
    const int nx = piece.box.ihi - piece.box.ilo + 1;
    const int ny = piece.box.jhi - piece.box.jlo + 1;
    check.failures += piece.box.level != 0;
    check.failures += piece.box.ilo < 0 || piece.box.jlo < 0 || piece.box.ihi >= n ||
                      piece.box.jhi >= n || nx < 1 || ny < 1;
    check.failures += piece.global_box_index < 0;
    check.failures += require_local_owner ? piece.owner_rank != my_rank()
                                          : (piece.owner_rank < 0 || piece.owner_rank >= n_ranks());
    check.failures += piece.replicated != replicated;
    check.failures += piece.ncomp != ncomp;
    const std::size_t expected_size = static_cast<std::size_t>(ncomp) * ny * nx;
    check.failures += piece.values.size() != expected_size;
    if (piece.values.size() != expected_size)
      continue;
    check.cells += static_cast<long>(nx) * ny;
    for (int component = 0; component < ncomp; ++component)
      for (int j = piece.box.jlo; j <= piece.box.jhi; ++j)
        for (int i = piece.box.ilo; i <= piece.box.ihi; ++i) {
          const std::size_t local = static_cast<std::size_t>(component) * ny * nx +
                                    static_cast<std::size_t>(j - piece.box.jlo) * nx +
                                    static_cast<std::size_t>(i - piece.box.ilo);
          const std::size_t full =
              static_cast<std::size_t>(component) * cells + static_cast<std::size_t>(j) * n + i;
          check.value_error =
              std::fmax(check.value_error, std::fabs(piece.values[local] - global[full]));
        }
  }
  return check;
}

static double max_abs_difference(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size())
    return std::numeric_limits<double>::infinity();
  double dmax = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i)
    dmax = std::fmax(dmax, std::fabs(a[i] - b[i]));
  return dmax;
}

static double component_zero_difference(const std::vector<double>& state,
                                        const std::vector<double>& density) {
  if (state.size() < density.size())
    return std::numeric_limits<double>::infinity();
  double dmax = 0.0;
  for (std::size_t i = 0; i < density.size(); ++i)
    dmax = std::fmax(dmax, std::fabs(state[i] - density[i]));
  return dmax;
}

static Result run(int n, int nsteps, double dt, bool distribute) {
  const std::vector<double> rho = four_bubbles(n);
  AmrSystemConfig cfg;
  cfg.n = n;
  cfg.L = 1.0;
  cfg.periodic = true;
  cfg.regrid_every = 4;
  cfg.distribute_coarse = distribute;  // <-- le mode scalable cable dans AmrSystem
  // coarse_max_grid = 0 -> n/2 (decoupage 2x2, le moins agressif pour le MG geometrique).

  AmrSystem sys(cfg);
  add_compiled_model(sys, "gas", Model{Euler{1.4}, GravityForce{}, GravityCoupling{-1.0, 1.0, 1.0}},
                     "minmod", "rusanov", "conservative", "explicit", /*gamma=*/1.4);
  sys.set_poisson("charge_density", "geometric_mg");
  sys.set_refinement(1.2);
  sys.set_density("gas", rho);

  Result R;
  R.m0 = sys.mass();
  for (int s = 0; s < nsteps; ++s)
    sys.step(dt);
#if defined(POPS_HAS_KOKKOS)
  Kokkos::fence();
#endif
  R.dens = sys.density();
  // The density/potential paths reconstruct their global coarse fields independently of the
  // checkpoint accessors below. They are the oracle for the replicated-vs-distributed ownership
  // contract of level_{state,potential}_global(0).
  R.state = sys.level_state_global(0);
  R.output_local_pieces = sys.output_state_local_pieces("gas", 0);
  R.output_root_pieces = sys.output_state_root_pieces(WorldCommunicator::world(), "gas", 0);
  R.phi = sys.potential();
  R.phi_global = sys.level_potential_global(0);
  R.mass = sys.mass();
  R.npf = sys.n_patches();
  return R;
}

static int pops_run_test_mpi_amr_distributed_coarse(int argc, char** argv) {
  comm_init(&argc, &argv);
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#else
  (void)argc;
  (void)argv;
#endif
  const int me = my_rank(), np = n_ranks();
  const int n = 64;
  const int nsteps = 16;
  const double dt = 1e-3;

  const bool bootstrap_restriction_ok = bootstrap_volume_average_replicates_parent();
  const Result rep = run(n, nsteps, dt, /*distribute=*/false);  // oracle : grossier replique
  const Result dis = run(n, nsteps, dt, /*distribute=*/true);   // mode scalable : grossier reparti

  // (1) ecart REPARTI vs REPLIQUE sur la densite grossiere globale (n*n sur chaque rang).
  double dmax = 0;
  if (dis.dens.size() == rep.dens.size())
    for (std::size_t k = 0; k < dis.dens.size(); ++k)
      dmax = std::fmax(dmax, std::fabs(dis.dens[k] - rep.dens[k]));

  // checksums du champ reparti.
  double csum = 0, csumsq = 0, cmax = 0;
  for (double v : dis.dens) {
    csum += v;
    csumsq += v * v;
    const double a = std::fabs(v);
    if (a > cmax)
      cmax = a;
  }
  // (2) cmax cross-rang : max insensible a l'ordre -> doit etre identique sur tous les rangs.
  const double xmax = all_reduce_max(cmax), xmin = -all_reduce_max(-cmax);
  const double cmax_spread = xmax - xmin;
  // dmax reduit sur les rangs (chaque rang a le meme champ global reconstruit, mais on est defensif).
  const double dmax_g = all_reduce_max(dmax);
  const double rep_state_dmax = component_zero_difference(rep.state, rep.dens);
  const double dis_state_dmax = component_zero_difference(dis.state, dis.dens);
  const double rep_phi_dmax = max_abs_difference(rep.phi_global, rep.phi);
  const double dis_phi_dmax = max_abs_difference(dis.phi_global, dis.phi);
  // Full-state / potential parity between ownership policies verifies that the distributed gather
  // exposes the same physical global field rather than a rank-local fragment.
  const double state_mode_dmax = max_abs_difference(rep.state, dis.state);
  const double phi_mode_dmax = max_abs_difference(rep.phi_global, dis.phi_global);
  const OutputPieceCheck rep_output =
      check_output_pieces(rep.output_local_pieces, rep.state, n, true);
  const OutputPieceCheck dis_output =
      check_output_pieces(dis.output_local_pieces, dis.state, n, false);
  const OutputPieceCheck rep_root_output =
      check_output_pieces(rep.output_root_pieces, rep.state, n, true, false);
  const OutputPieceCheck dis_root_output =
      check_output_pieces(dis.output_root_pieces, dis.state, n, false, false);
  const double output_piece_error =
      all_reduce_max(std::fmax(rep_output.value_error, dis_output.value_error));
  const long output_piece_failures = static_cast<long>(
      std::llround(all_reduce_sum(static_cast<double>(rep_output.failures + dis_output.failures))));
  const long distributed_output_cells =
      static_cast<long>(std::llround(all_reduce_sum(static_cast<double>(dis_output.cells))));
  const double replicated_cells_max = all_reduce_max(static_cast<double>(rep_output.cells));
  const double replicated_cells_min = -all_reduce_max(-static_cast<double>(rep_output.cells));
  const double root_output_piece_error =
      all_reduce_max(std::fmax(rep_root_output.value_error, dis_root_output.value_error));
  const long root_output_piece_failures = static_cast<long>(std::llround(
      all_reduce_sum(static_cast<double>(rep_root_output.failures + dis_root_output.failures))));
  const long replicated_root_cells =
      static_cast<long>(std::llround(all_reduce_sum(static_cast<double>(rep_root_output.cells))));
  const long distributed_root_cells =
      static_cast<long>(std::llround(all_reduce_sum(static_cast<double>(dis_root_output.cells))));

  int fails = 0;
  if (me == 0) {
    if (!bootstrap_restriction_ok) {
      std::printf("FAIL volume-average bootstrap absent d'une copie grossiere repliquee\n");
      ++fails;
    }
    std::printf(
        "AMRDIST np=%d distribute_npf=%d replicated_npf=%d | cmax=%.17e | "
        "dist_vs_repl_dmax=%.3e | cmax_crossrank_spread=%.3e | "
        "state_rep=%.3e state_dist=%.3e phi_rep=%.3e phi_dist=%.3e\n",
        np, dis.npf, rep.npf, cmax, dmax_g, cmax_spread, rep_state_dmax, dis_state_dmax,
        rep_phi_dmax, dis_phi_dmax);
#if defined(POPS_HAS_KOKKOS)
    const char* space = Kokkos::DefaultExecutionSpace::name();
#else
    const char* space = "Serial(host)";
#endif
    std::printf(
        "AMRDIST exec=%s | conservation: dm_dist=%.3e dm_repl=%.3e | csum=%.17e csumsq=%.17e\n",
        space, std::fabs(dis.mass - dis.m0), std::fabs(rep.mass - rep.m0), csum, csumsq);

    if (!(dis.dens.size() == static_cast<std::size_t>(n) * n)) {
      std::printf("FAIL densite repartie de mauvaise taille\n");
      ++fails;
    }
    if (!(rep_state_dmax == 0.0 && rep_phi_dmax == 0.0)) {
      std::printf("FAIL les vues globales repliquees ont ete reduites (state=%.3e phi=%.3e)\n",
                  rep_state_dmax, rep_phi_dmax);
      ++fails;
    }
    if (!(dis_state_dmax == 0.0 && dis_phi_dmax == 0.0)) {
      std::printf(
          "FAIL les vues globales distribuees ne reconstituent pas la reference (state=%.3e "
          "phi=%.3e)\n",
          dis_state_dmax, dis_phi_dmax);
      ++fails;
    }
    if (output_piece_failures != 0 || output_piece_error != 0.0 ||
        distributed_output_cells != static_cast<long>(n) * n ||
        replicated_cells_min != static_cast<double>(n * n) ||
        replicated_cells_max != static_cast<double>(n * n)) {
      std::printf(
          "FAIL pieces output AMR invalides (metadata=%ld value=%.3e distributed_cells=%ld "
          "replicated_cells=[%.0f,%.0f])\n",
          output_piece_failures, output_piece_error, distributed_output_cells, replicated_cells_min,
          replicated_cells_max);
      ++fails;
    }
    if (root_output_piece_failures != 0 || root_output_piece_error != 0.0 ||
        replicated_root_cells != static_cast<long>(n) * n ||
        distributed_root_cells != static_cast<long>(n) * n) {
      std::printf(
          "FAIL gather ROOT natif des pieces AMR (metadata=%ld value=%.3e "
          "replicated_cells=%ld distributed_cells=%ld)\n",
          root_output_piece_failures, root_output_piece_error, replicated_root_cells,
          distributed_root_cells);
      ++fails;
    }
    if (!(cmax > 1e-6)) {
      std::printf("FAIL densite repartie triviale\n");
      ++fails;
    }
    if (!std::isfinite(cmax) || !std::isfinite(csum)) {
      std::printf("FAIL champ non fini (MG diverge ?)\n");
      ++fails;
    }
    // (4) MG converge => champ fini ET proche du replique : le grossier reparti doit retrouver le
    // meme physique a l'arrondi pres (la difference vient de l'ordre de reduction du Poisson +
    // transport multi-box, pas d'un schema different). Seuil large mais ferme : un MG qui diverge
    // ou un transport casse exploserait bien au-dela.
    if (!(dmax_g < 1e-9)) {
      std::printf("FAIL reparti != replique au-dela de l'arrondi (dmax=%.3e)\n", dmax_g);
      ++fails;
    }
    if (!(state_mode_dmax < 1e-9 && phi_mode_dmax < 1e-9)) {
      std::printf("FAIL checkpoint global reparti != replique (state=%.3e phi=%.3e)\n",
                  state_mode_dmax, phi_mode_dmax);
      ++fails;
    }
    // (3) conservation des deux modes.
    if (!(std::fabs(dis.mass - dis.m0) < 1e-10)) {
      std::printf("FAIL conservation grossier reparti (dm=%.3e)\n", std::fabs(dis.mass - dis.m0));
      ++fails;
    }
    // (2) cmax bit-identique cross-rang.
    if (!(cmax_spread == 0.0)) {
      std::printf("FAIL cmax non bit-identique entre rangs (spread=%.3e)\n", cmax_spread);
      ++fails;
    }
    if (fails == 0)
      std::printf(
          "OK test_mpi_amr_distributed_coarse np=%d (grossier reparti == replique a "
          "l'arrondi, cmax bit-identique cross-rang, masse conservee)\n",
          np);
  }
  comm_finalize();
  return fails ? 1 : 0;
}

TEST(test_mpi_amr_distributed_coarse, Runs) {
  EXPECT_EQ(pops::test::RunTestBody(&pops_run_test_mpi_amr_distributed_coarse,
                                    "test_mpi_amr_distributed_coarse"),
            0);
}
