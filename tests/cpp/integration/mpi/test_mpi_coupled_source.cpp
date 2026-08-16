// Autorite temporelle explicite avec une source couplee enregistree sous MPI np={1,2,4}.
//
// System repartit UNE box unique en round-robin (DistributionMapping(1, n_ranks())), donc a np>1 un
// seul rang la possede ; les autres ont local_size()==0. Le Program abaisse explicitement l'ionisation
// sur les trois etats. La valeur attendue correspond exactement a UNE application par pas : elle
// verrouille a la fois le chemin MPI du Program et l'absence d'un second replay cache du CoupledSource.
//
// On code l'IONISATION en bytecode (d_t n_e = +k n_e n_g, d_t n_i = +k n_e n_g, d_t n_g = -k n_e n_g) :
//   regs : ne=0, ni=1, ng=2, k=3 (constante)
//   e/i : PushReg(k) PushReg(ne) Mul PushReg(ng) Mul          -> k*ne*ng
//   g   : ... Neg                                              -> -(k*ne*ng)
// Densites UNIFORMES -> transport nul ; la recurrence source doit etre identique sous np=1/2/4.

#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/physics/composition/composite.hpp>
#include <pops/physics/bricks/hyperbolic.hpp>
#include <pops/physics/bricks/source.hpp>  // NoSource
#include <pops/physics/fluids/euler.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/runtime/builders/compiled/dsl_block.hpp>  // add_compiled_model
#include <pops/runtime/builders/compiled/generated_system_block.hpp>
#include <pops/runtime/program/program_context.hpp>
#include <pops/runtime/system.hpp>

#include <pops/coupling/source/coupled_source_program.hpp>  // CsOp (opcodes, miroir Python)
#include <pops/parallel/comm.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif
#ifdef POPS_HAS_MPI
#include <mpi.h>
#endif

using namespace pops;
constexpr int kDim = kNativeDimension;
using NativeSystem = System<kDim>;
using NativeMultiFab = MultiFab<kDim>;

namespace pops {
template <int Dim, class Model>
PreparedSystemBlock<Dim> prepare_exact_system_block(
    CompiledSystemBlockPreparation<Dim, Model> request) {
  return prepare_generated_system_block(std::move(request));
}
}  // namespace pops

struct NoEll {
  template <class State>
  POPS_HD Real rhs(const State&) const {
    return Real(0);
  }  // pas de charge -> phi=0 -> derive nulle
};
using Dens = CompositeModel<EulerND<kDim>, NoSource, NoEll>;

static Dens density_model() {
  return Dens{{}, EulerND<kDim>{Real(1.4)}, NoSource{}, NoEll{}};
}

static std::vector<double> uniform_state(std::size_t cells, double density) {
  std::vector<double> state(static_cast<std::size_t>(kDim + 2) * cells, 0.0);
  std::fill_n(state.begin(), static_cast<std::ptrdiff_t>(cells), density);
  std::fill_n(state.begin() + static_cast<std::ptrdiff_t>((kDim + 1) * cells),
              static_cast<std::ptrdiff_t>(cells), 2.5);
  return state;
}

static void install_ionization_program(NativeSystem& system) {
  system.set_program_block_map({0, 1, 2});
  runtime::program::ProgramContext<kDim> context(&system);
  context.configure_primary_clock("test.clock.macro");
  context.install([context](double step) {
    context.begin_step(step);
    NativeMultiFab& electrons = context.state(0);
    NativeMultiFab& ions = context.state(1);
    NativeMultiFab& neutrals = context.state(2);
    NativeMultiFab& next_electrons = context.scratch_state(100, 0, electrons);
    NativeMultiFab& next_ions = context.scratch_state(101, 0, ions);
    NativeMultiFab& next_neutrals = context.scratch_state(102, 0, neutrals);
    context.lincomb(next_electrons, Real(1), electrons, Real(0), electrons);
    context.lincomb(next_ions, Real(1), ions, Real(0), ions);
    context.lincomb(next_neutrals, Real(1), neutrals, Real(0), neutrals);
    context.apply_coupling_operators(Real(step),
                                     {{0, &next_electrons}, {1, &next_ions}, {2, &next_neutrals}});
    context.commit_many(
        {{&electrons, &next_electrons}, {&ions, &next_ions}, {&neutrals, &next_neutrals}});
  });
  system.set_program_block_map({0, 1, 2});
}

static int run_test_mpi_coupled_source_body(int argc, char** argv) {
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
  const double k = 0.7, dt = 0.01;
  const int nsteps = 25;
  const double ne0 = 0.30, ni0 = 0.10, ng0 = 1.00;

  SystemConfig<kDim> cfg;
  std::size_t cells = 1;
  for (int axis = 0; axis < kDim; ++axis) {
    cfg.shape[axis] = n;
    cfg.lower[axis] = Real(0);
    cfg.upper[axis] = Real(1);
    cfg.periodicity[axis] = true;
    cells *= static_cast<std::size_t>(n);
  }

  NativeSystem sys(cfg);
  sys.install_prepared_boundary_execution_lane(std::make_shared<ExecutionLane>(
      ExecutionLane::duplicate_world_collectively("pops.test.mpi-coupled-source")));
  for (const std::string& block :
       {std::string("electrons"), std::string("ions"), std::string("neutrals")})
    sys.install_block_state_route(block, "test.mpi-coupled-source/" + block + "/state@1");
  add_compiled_model<kDim>(sys, "electrons", density_model(), "none", "rusanov", "conservative",
                           "explicit");
  add_compiled_model<kDim>(sys, "ions", density_model(), "none", "rusanov", "conservative",
                           "explicit");
  add_compiled_model<kDim>(sys, "neutrals", density_model(), "none", "rusanov", "conservative",
                           "explicit");
  sys.set_poisson("composite", "cartesian_cg");

  // Init des densites UNIFORMES sur le rang proprietaire (set_density n'ecrit que la box locale).
  const std::size_t nn = cells;
  const bool owns = (me == 0);
  // Exact field marshaling is collective: every rank authenticates the same global component-major
  // image, while only the owner materializes its local patch.
  sys.set_state("electrons", uniform_state(nn, ne0));
  sys.set_state("ions", uniform_state(nn, ni0));
  sys.set_state("neutrals", uniform_state(nn, ng0));

  // --- Source couplee generique (bytecode ionisation) : appelee sur TOUS les rangs (l'enregistrement
  //     resout des indices, aucun acces par cellule ; l'application itere local_size() -> no-op si vide).
  const std::vector<std::string> in_blocks = {"electrons", "ions", "neutrals"};
  const std::vector<std::string> in_roles = {"density", "density", "density"};
  const std::vector<double> consts = {k};
  const std::vector<std::string> out_blocks = {"electrons", "ions", "neutrals"};
  const std::vector<std::string> out_roles = {"density", "density", "density"};
  const int PUSH = static_cast<int>(CsOp::PushReg), MUL = static_cast<int>(CsOp::Mul),
            NEG = static_cast<int>(CsOp::Neg);
  // e : k*ne*ng ; i : k*ne*ng ; g : -(k*ne*ng). reg ne=0, ng=2, k=3.
  std::vector<int> ops = {PUSH, PUSH, MUL, PUSH, MUL,        // electrons (len 5)
                          PUSH, PUSH, MUL, PUSH, MUL,        // ions      (len 5)
                          PUSH, PUSH, MUL, PUSH, MUL, NEG};  // neutrals  (len 6)
  std::vector<int> args = {3, 0, 0, 2, 0, 3, 0, 0, 2, 0, 3, 0, 0, 2, 0, 0};
  std::vector<int> lens = {5, 5, 6};
  // ADC-214 : la description bytecode est regroupee dans un POD CoupledSourceProgram (initialiseurs
  // designes -> appel auto-documente, plus de liste de vecteurs du meme type intervertibles).
  pops::CoupledSourceProgram prog;
  prog.in_blocks = in_blocks;
  prog.in_roles = in_roles;
  prog.consts = consts;
  prog.out_blocks = out_blocks;
  prog.out_roles = out_roles;
  prog.prog_ops = ops;
  prog.prog_args = args;
  prog.prog_lens = lens;
  sys.add_coupled_source(prog);
  chk(sys.coupled_operators().size() == 1, "coupled_source_metadata_registered");

  const std::size_t pre_probe_couplings = sys.coupled_operators().size();
  bool divergent_preflight_rejected = false;
  try {
    sys.install_prepared_coupling_operator(
        "test.mpi-coupling-invalid-preflight",
        me == 0 ? std::string{} : std::string("test.mpi-coupling/provider@1"),
        CouplingOperatorView{}, [](Real, const std::vector<NativeMultiFab*>&) {});
  } catch (const std::exception&) {
    divergent_preflight_rejected = true;
  }
  chk(divergent_preflight_rejected && sys.coupled_operators().size() == pre_probe_couplings,
      "rank-local prepared coupling failure publishes no partial registry");

  if (np > 1) {
    bool byte_divergence_rejected = false;
    try {
      sys.install_prepared_coupling_operator(
          "test.mpi-coupling-divergent-contract",
          "test.mpi-coupling/provider@1;rank=" + std::to_string(me), CouplingOperatorView{},
          [](Real, const std::vector<NativeMultiFab*>&) {});
    } catch (const std::invalid_argument&) {
      byte_divergence_rejected = true;
    }
    chk(byte_divergence_rejected && sys.coupled_operators().size() == pre_probe_couplings,
        "byte-divergent prepared coupling contract is rejected collectively");
  }

  auto fail_rollback_probe = std::make_shared<bool>(false);
  sys.install_prepared_coupling_operator(
      "test.mpi-coupling-rollback", "test.mpi-coupling-rollback/provider@1;program=reject-v1",
      CouplingOperatorView{},
      [fail_rollback_probe](Real, const std::vector<NativeMultiFab*>& candidates) {
        if (!*fail_rollback_probe)
          return;
        candidates.front()->set_val(Real(-17));
        Kokkos::fence();
        throw std::runtime_error("deliberate prepared coupling rejection");
      });
  bool live_state_rejected = false;
  try {
    sys.apply_coupling_operators(Real(dt),
                                 {&sys.block_state(0), &sys.block_state(1), &sys.block_state(2)});
  } catch (const std::invalid_argument&) {
    live_state_rejected = true;
  } catch (const std::runtime_error&) {
    // Multi-rank preflight reports one collective error after every rank rejects its live carrier.
    live_state_rejected = true;
  }
  chk(live_state_rejected, "accepted_live_states_are_not_coupling_workspace");
  install_ionization_program(sys);

  double ne = ne0, ni = ni0, ng = ng0;
  for (int s = 0; s < nsteps; ++s) {
    sys.step(dt);
    const double rate = k * ne * ng;
    ne += dt * rate;
    ni += dt * rate;
    ng -= dt * rate;
  }

  NativeMultiFab candidate_electrons(sys.block_state(0));
  NativeMultiFab candidate_ions(sys.block_state(1));
  NativeMultiFab candidate_neutrals(sys.block_state(2));
  const Real rollback_electrons = reduce_max_local(candidate_electrons, 0);
  const Real rollback_ions = reduce_max_local(candidate_ions, 0);
  const Real rollback_neutrals = reduce_max_local(candidate_neutrals, 0);
  *fail_rollback_probe = true;
  bool rejected_and_rolled_back = false;
  try {
    sys.apply_coupling_operators(Real(dt),
                                 {&candidate_electrons, &candidate_ions, &candidate_neutrals});
  } catch (const std::runtime_error&) {
    rejected_and_rolled_back = reduce_max_local(candidate_electrons, 0) == rollback_electrons &&
                               reduce_max_local(candidate_ions, 0) == rollback_ions &&
                               reduce_max_local(candidate_neutrals, 0) == rollback_neutrals;
  }
  chk(rejected_and_rolled_back, "prepared coupling failure rolls back every candidate");

  if (owns) {
    const std::vector<double> de = sys.density("electrons");
    const std::vector<double> di = sys.density("ions");
    const std::vector<double> dg = sys.density("neutrals");
    chk(de.size() == nn && di.size() == nn && dg.size() == nn, "density_size");
    double ge = 0, gi = 0, gg = 0, sde = 0, sdi = 0, sdg = 0;
    bool finite = true;
    for (std::size_t q = 0; q < nn; ++q) {
      finite = finite && std::isfinite(de[q]) && std::isfinite(di[q]) && std::isfinite(dg[q]);
      sde += de[q];
      sdi += di[q];
      sdg += dg[q];
    }
    ge = sde / nn;
    gi = sdi / nn;
    gg = sdg / nn;
    chk(finite, "density_finite");
    // etat reste uniforme (transport nul) : min == max == moyenne.
    bool uniform = true;
    for (std::size_t q = 0; q < nn; ++q)
      uniform = uniform && std::fabs(de[q] - ge) < 1e-12 && std::fabs(dg[q] - gg) < 1e-12;
    chk(uniform, "etat_uniforme");
    // UNE recurrence source exactement : un replay implicite du registre natif doublerait le taux.
    chk(std::fabs(ge - ne) < 1e-10, "n_e == explicit Program reference");
    chk(std::fabs(gi - ni) < 1e-10, "n_i == explicit Program reference");
    chk(std::fabs(gg - ng) < 1e-10, "n_g == explicit Program reference");
    chk(ge > ne0 + 1e-6 && gi > ni0 + 1e-6 && gg < ng0 - 1e-6, "e/i croissent, g decroit");
    chk(std::fabs((gi + gg) - (ni0 + ng0)) < 1e-9, "n_i+n_g conserve");
    chk(std::fabs((ge - gi) - (ne0 - ni0)) < 1e-9, "n_e-n_i conserve");
    std::printf("[rank %d/%d] np=%d  n_e=%.8f n_i=%.8f n_g=%.8f\n", me, np, np, ge, gi, gg);
  }

  // mass() est COLLECTIVE (sum -> all_reduce) : appelee par TOUS les rangs (sinon interblocage). La masse
  // lourde (ions + neutres) est conservee a 1e-9 et INVARIANTE en np.
  const double mi = sys.mass("ions"), mg = sys.mass("neutrals");
  chk(std::isfinite(mi) && std::isfinite(mg), "mass_finite");
  chk(std::fabs((mi + mg) - (ni0 + ng0) * static_cast<double>(nn)) < 1e-7,
      "masse_lourde_conservee");

#ifdef POPS_HAS_MPI
  if (np > 1) {
    long g = 0;
    MPI_Allreduce(&fails, &g, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    fails = g;
  }
#endif
  if (me == 0 && fails == 0)
    std::printf("OK test_mpi_coupled_source (np=%d)\n", np);
  return fails == 0 ? 0 : 1;
}

static int pops_run_test_mpi_coupled_source(int argc, char** argv) {
  comm_init(&argc, &argv);
  const int result = run_test_mpi_coupled_source_body(argc, argv);
  comm_finalize();
  return result;
}

TEST(test_mpi_coupled_source, Runs) {
  EXPECT_EQ(pops::test::RunTestBody(&pops_run_test_mpi_coupled_source, "test_mpi_coupled_source"),
            0);
}
