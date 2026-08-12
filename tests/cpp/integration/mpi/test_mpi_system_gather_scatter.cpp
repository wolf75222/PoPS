// Verrou de non-regression du bug "finding 8" : acces fab(0) SANS garde local_size dans les
// fermetures rhs_into / advance / max_speed des blocs natifs composes.
//
// System repartit UNE box unique dans l'espace de rangs exact (round-robin), donc a np>1
// un seul rang la possede ; les autres ont local_size()==0. Avant ce fix, les fermetures
// rhs_into / advance / max_speed du chemin add_compiled_model appelaient copy_state(U, ...) /
// write_state(U, ...) sans tester local_size() -> fab(0) hors-bornes -> crash UB silencieux ou
// segfault sur les rangs vides.
//
// Ce test exerce System::add_compiled_model (chemin compile, fermetures type-erased) sous mpirun
// -np {1,2,4} et exige :
//   (a) aucun crash (fermetures no-op sur les rangs sans box) ;
//   (b) resultat INVARIANT au nombre de rangs (la box vit sur rang 0 quel que soit np) ;
//   (c) step() / step_cfl() COLLECTIFS (tous les rangs participent aux all_reduce internes).
//
// RAISONNEMENT COLLECTIF MPI : advance / max_speed gardent leur no-op local sur les rangs vides.
// Le residual, lui, termine par un preflight natif collectif de finitude : eval_rhs() doit donc etre
// appele par TOUS les rangs, comme step(), step_cfl() et mass(). Les rangs vides participent au
// MPI_Allreduce puis renvoient naturellement un buffer local vide ; seul le proprietaire inspecte
// les valeurs. Aucun communicateur factice ni branchement qui saute un collectif n'est permis.
//
// AVANT le fix : segfault / UB a np=2/4 sur les rangs sans box locale (rhs_into / advance /
// max_speed dereferencaient fab(0) inexistant). APRES : np=1/2/4 verts, resultats identiques.

#include <gtest/gtest.h>

#include "explicit_system_program.hpp"
#include "gtest_compat.hpp"
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution.hpp>
#include <pops/mesh/layout/rank_space.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/numerics/spatial/primitives/wave_speed.hpp>
#include <pops/runtime/builders/compiled/dsl_block.hpp>  // add_compiled_model
#include <pops/runtime/builders/compiled/generated_system_block.hpp>
#include <pops/runtime/system.hpp>

#include <pops/parallel/comm.hpp>

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif
#ifdef POPS_HAS_MPI
#include <mpi.h>
#endif

using namespace pops;

namespace pops {

template <int Dim, class Model>
PreparedSystemBlock<Dim> prepare_exact_system_block(
    CompiledSystemBlockPreparation<Dim, Model> request) {
  return prepare_generated_system_block(std::move(request));
}

}  // namespace pops

// Modele scalaire exact-rank sans fournisseur : la voie compilee avance un etat uniforme sans
// introduire de flux ou de source, tout en exercant les fermetures rhs_into / advance / max_speed.
template <int Dim>
struct ScalarModel {
  using Law = nd::ScalarAdvection<Dim>;
  using Schema = typename Law::Schema;
  using State = typename Law::State;
  using Primitive = typename Law::Primitive;
  static constexpr int dimension = Dim;
  static constexpr int n_vars = Law::n_vars;
  static constexpr int n_providers = 0;
  Law law{};

  [[nodiscard]] static constexpr PreparedProviderIdentity provider_identity() noexcept {
    return {"test.mpi-system-gather-scatter.scalar", 1};
  }
  void serialize_exact_parameters(ExactContractBuilder& contract) const {
    for (int axis = 0; axis < Dim; ++axis)
      contract.scalar(law.velocity()[axis]);
  }
  POPS_HD nd::StateConversion<Primitive> recover(const State& state) const {
    return law.recover(state);
  }
  POPS_HD nd::StateConversion<State> make_conservative(const Primitive& primitive) const {
    return law.make_conservative(primitive);
  }
  POPS_HD nd::StateConversionStatus admissibility(const State& state) const {
    return law.admissibility(state);
  }
  template <int Axis>
  POPS_HD State flux(const State& state) const {
    return law.template flux<Axis>(state);
  }
  template <int Axis>
  POPS_HD Real max_wave_speed(const State& state) const {
    return law.template max_wave_speed<Axis>(state);
  }
  template <int Axis>
  POPS_HD void wave_speeds(const State& state, Real& lower, Real& upper) const {
    law.template wave_speeds<Axis>(state, lower, upper);
  }
  POPS_HD State source(const State&, const ProviderValues<0>&) const { return {}; }
  POPS_HD Real elliptic_rhs(const State&) const { return Real(0); }

  static VariableSet conservative_vars() {
    return {VariableKind::Conservative, {"u"}, 1, {VariableRole::Scalar}};
  }
  static VariableSet primitive_vars() {
    return {VariableKind::Primitive, {"u"}, 1, {VariableRole::Scalar}};
  }
};

template <int Dim>
struct DirectDtProbe {
  using Schema = nd::ScalarStateSchema<Dim>;
  using State = typename Schema::Conservative;
  static constexpr int dimension = Dim;
  static constexpr int n_vars = 1;
  Real value;

  POPS_HD Real stability_dt(const State&) const { return value; }
};

static int pops_run_test_mpi_system_gather_scatter(int argc, char** argv) {
  constexpr int Dim = kNativeDimension;
  using NativeSystem = System<Dim>;
  using NativeSystemConfig = SystemConfig<Dim>;
  comm_init(&argc, &argv);
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#endif
  const int me = my_rank(), np = n_ranks();
  long fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("[rank %d/%d] FAIL %s\n", me, np, w);
      ++fails;
    }
  };

  const int n = 16;
  const double rho0 = 1.5, dt = 0.01;
  const int nsteps = 5;
  std::size_t nn = 1;
  for (int axis = 0; axis < Dim; ++axis)
    nn *= static_cast<std::size_t>(n);

  NativeSystemConfig cfg;
  for (int axis = 0; axis < Dim; ++axis) {
    cfg.shape[axis] = n;
    cfg.lower[axis] = Real(0);
    cfg.upper[axis] = Real(1);
    cfg.periodicity[static_cast<std::size_t>(axis)] = true;
  }
  cfg.boxes = {Box<Dim>::from_extents(cfg.shape)};

  NativeSystem sys(cfg);
  // add_compiled_model branche les fermetures natives rhs_into / advance / max_speed : ce sont
  // exactement les sites du finding 8.
  add_compiled_model(sys, "u", ScalarModel<Dim>{}, "none", "rusanov", "conservative", "explicit");
  sys.set_poisson("composite", "cartesian_cg");
  test::install_forward_euler_program(sys);

  // write_global est collectif : chaque rang fournit le meme payload global ordonne. Seul le
  // proprietaire materialise la box; les pairs vides participent neanmoins au contrat collectif.
  sys.set_density("u", std::vector<double>(nn, rho0));
  const bool owns = (me == 0);

  // --- Exerce les 3 chemins du finding 8 sur TOUS les rangs ---
  //
  // (1) step() : appelle solve_fields (collectif) puis s.advance(U, dt, nsub) sur TOUS les rangs.
  //     advance (finding 8, chemin compile) appelle copy_state / write_state sans garde -> crash
  //     hors-bornes sur les rangs vides avant le fix. APRES : no-op sur les rangs vides.
  for (int s = 0; s < nsteps; ++s)
    sys.step(dt);

  // (2) step_cfl(cfl) : appelle s.max_speed(U) sur TOUS les rangs, puis all_reduce_max du dt CFL
  //     (collectif, APRES les max_speed). max_speed (finding 8) appelait copy_state sans garde.
  //     APRES : no-op sur les rangs vides, 0 local -> l'all_reduce prend le vrai max du proprietaire.
  const double dt_cfl = sys.step_cfl(0.5);
  chk(std::isfinite(dt_cfl) && dt_cfl > 0, "dt_cfl_valide");
  // dt_cfl INVARIANT en np : la box vit sur rang 0 quel que soit np ; le max_speed du proprietaire
  // est diffuse par all_reduce_max -> meme dt_cfl sur tous les rangs (seul rang 0 a une vraie vitesse,
  // les autres contribuent 0 a l'all_reduce, sans impact sur le max).
  chk(std::isfinite(dt_cfl), "dt_cfl_fini");

  // (3) eval_rhs() exerce rhs_into (finding 8) : copie U -> applique le residu -> preflight
  //     collectif de finitude -> copie retour. Tous les rangs doivent entrer dans ce contrat ; le
  //     resultat n'est inspecte que par le rang proprietaire.
  const std::vector<double> R = sys.eval_rhs("u");
  if (owns) {
    bool rfin = (R.size() == nn);
    for (double r : R)
      rfin = rfin && std::isfinite(r);
    chk(rfin, "rhs_fini");
  }

  // --- Verification du resultat physique (etat uniforme reste uniforme : transport nul, phi=0) ---
  // Appelee UNIQUEMENT sur le rang proprietaire (densite stockee localement).
  if (owns) {
    const std::vector<double> d = sys.density("u");
    chk(d.size() == nn, "densite_taille");
    bool finite = true;
    double dmin = d[0], dmax = d[0];
    for (double v : d) {
      finite = finite && std::isfinite(v);
      if (v < dmin)
        dmin = v;
      if (v > dmax)
        dmax = v;
    }
    chk(finite, "densite_finie");
    // etat uniforme, phi=0 -> transport nul -> densite INCHANGEE (a la precision machine).
    chk(std::fabs(dmin - rho0) < 1e-10, "densite_min_invariante");
    chk(std::fabs(dmax - rho0) < 1e-10, "densite_max_invariante");
    std::printf("[rank %d/%d] np=%d  rho_min=%.12f  rho_max=%.12f  dt_cfl=%.6e\n", me, np, np, dmin,
                dmax, dt_cfl);
  }

  // mass() est COLLECTIVE (sum -> all_reduce) : TOUS les rangs l'appellent (sinon interblocage).
  // La masse totale = rho0 * n*n INDEPENDANTE de np (la box vit toujours sur rang 0).
  const double mtot = sys.mass("u");
  chk(std::isfinite(mtot), "masse_finie");
  chk(std::fabs(mtot - rho0 * static_cast<double>(nn)) < 1e-9, "masse_conservee");

  // A direct model dt is also a native collective.  Only rank zero owns this one-box field; an
  // invalid value observed there must make every empty peer reject at the same MPI_Allreduce, not
  // return a different step or wait in a later collective.
  Extent<Dim> reduction_extent{};
  for (int axis = 0; axis < Dim; ++axis)
    reduction_extent[axis] = 2;
  const Box<Dim> reduction_box = Box<Dim>::from_extents(reduction_extent);
  const mesh::BoxArray<Dim> reduction_boxes(std::vector<Box<Dim>>{reduction_box});
  Extent<Dim> process_extent{};
  for (int axis = 0; axis < Dim; ++axis)
    process_extent[axis] = axis == 0 ? np : 1;
  const mesh::RankSpace<Dim> rank_space(Index<Dim>{}, process_extent);
  const mesh::Distribution<Dim> reduction_owners =
      mesh::Distribution<Dim>::partitioned(reduction_boxes, rank_space, {rank_space.coordinate(0)});
  MultiFab<Dim> reduction_state(reduction_boxes, reduction_owners, rank_space.coordinate(me), 1,
                                Extent<Dim>{});
  reduction_state.set_val(Real(1));

  bool rejected_invalid_dt = false;
  try {
    (void)nd::min_stability_dt_mf(DirectDtProbe<Dim>{me == 0 ? Real(0) : Real(1)}, reduction_state);
  } catch (const std::domain_error&) {
    rejected_invalid_dt = true;
  }
  chk(rejected_invalid_dt, "stability_dt_invalide_rejetee_collectivement");

  const Real direct_dt = nd::min_stability_dt_mf(DirectDtProbe<Dim>{Real(0.25)}, reduction_state);
  chk(direct_dt == Real(0.25), "stability_dt_valide_diffusee_aux_rangs_vides");

#ifdef POPS_HAS_MPI
  if (np > 1) {
    long g = 0;
    MPI_Allreduce(&fails, &g, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    fails = g;
  }
#endif
  if (me == 0 && fails == 0)
    std::printf("OK test_mpi_system_gather_scatter (np=%d)\n", np);
  comm_finalize();
  return fails == 0 ? 0 : 1;
}

TEST(test_mpi_system_gather_scatter, Runs) {
  EXPECT_EQ(pops::test::RunTestBody(&pops_run_test_mpi_system_gather_scatter,
                                    "test_mpi_system_gather_scatter"),
            0);
}
