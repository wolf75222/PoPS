// Conformite des classes elliptiques EXISTANTES aux concepts communs formalises
// dans elliptic_interface.hpp (audit D.1). Le test est ESSENTIELLEMENT statique : les
// static_assert ci-dessous echouent A LA COMPILATION si une classe cesse de modeler son
// concept. Les TEST runtime existent parce qu'ils exercent en plus field_postprocess A
// TRAVERS le concept FieldPostProcessor (preuve que la contrainte est appelable, pas
// seulement bien-formee) et revalident quelques bits.
//
// AUCUNE classe elliptique n'est modifiee : ce fichier OBSERVE les contrats deja codes.
// Les concepts sont de la metaprogrammation hote (pas de kernel) : zero incidence device,
// la pile elliptique device-validee reste bit-identique.

#include <gtest/gtest.h>

#include <pops/numerics/elliptic/interface/elliptic_interface.hpp>

#include <pops/numerics/elliptic/interface/elliptic_problem.hpp>  // field_postprocess, FieldPostProcess
#include <pops/numerics/elliptic/interface/elliptic_solver.hpp>  // EllipticSolver
#include <pops/numerics/elliptic/mg/geometric_mg.hpp>            // GeometricMG
#include <pops/numerics/elliptic/poisson/poisson_fft_solver.hpp>  // PoissonFFTSolver, DistributedFFTSolver
#include <pops/numerics/elliptic/polar/polar_poisson_solver.hpp>  // PolarPoissonSolver, PolarEllipticSolver

#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>

#include <cmath>
#include <cstdio>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

using namespace pops;
static constexpr double kPi = 3.14159265358979323846;

static_assert(!std::is_reference_v<decltype(EllipticBuildRequest::geometry)>);
static_assert(!std::is_reference_v<decltype(EllipticBuildRequest::boxes)>);
static_assert(!std::is_reference_v<decltype(EllipticBuildRequest::mapping)>);

// =====================================================================================
// (1) EllipticOperator : role d'operateur (coefficients + geom + bc). GeometricMG le porte.
static_assert(EllipticOperator<GeometricMG>,
              "GeometricMG doit modeler EllipticOperator (role operateur : op_eps/op_coef/op_kappa/"
              "op_eps_y/op_a_xy/op_a_yx/op_mask + geom + bc)");

// GAP DOCUMENTE : les solveurs DIRECTS (FFT, polaire) n'exposent PAS de coefficients
// d'operateur (pas de matvec matrice-libre) -> ils ne modelent PAS EllipticOperator.
// C'est le comportement attendu : seul l'operateur MG porte ce role aujourd'hui.
static_assert(
    !EllipticOperator<PoissonFFTSolver>,
    "PoissonFFTSolver (solveur direct) n'a PAS de role operateur a coefficients : attendu");
static_assert(
    !EllipticOperator<PolarPoissonSolver>,
    "PolarPoissonSolver (solveur direct) n'a PAS de role operateur a coefficients : attendu");

// =====================================================================================
// (2) LinearSolver : solveur ITERATIF a solve(rel_tol, max_iters) -> resultat non void.
static_assert(LinearSolver<GeometricMG>,
              "GeometricMG doit modeler LinearSolver (solve(rel_tol, max_cycles) -> int)");

// Le contrat objet historique (rhs/phi/solve()/residual/geom) reste EllipticSolver pour
// GeometricMG. Le Krylov generique n'est volontairement pas force dans ce concept : son contrat
// final est le protocole prepare explicite (PreparedAffineLinearProblem + KrylovWorkspace +
// solve_prepared_affine), valide exhaustivement par test_generic_krylov.
static_assert(EllipticSolver<GeometricMG>, "GeometricMG modele EllipticSolver");
static_assert(EllipticFactory<DefaultEllipticFactory<GeometricMG>, GeometricMG>,
              "GeometricMG accepte la factory typee de distribution du champ");
static_assert(
    !EllipticFactory<DefaultEllipticFactory<PoissonFFTSolver>, PoissonFFTSolver>,
    "un booleen spectral FFT ne doit jamais modeler implicitement la distribution du champ");
struct ExplicitFftDistributionFactory {
  std::string contract{"pops.test.explicit-fft-factory@1"};

  [[nodiscard]] std::string_view collective_contract() const noexcept { return contract; }

  [[nodiscard]] EllipticOperatorContract expected_operator_contract(
      const EllipticBuildRequest& request) const {
    return PoissonFFTSolver::expected_operator_contract(request, false);
  }

  [[nodiscard]] FieldDistribution materialized_distribution(
      const EllipticBuildRequest&) const noexcept {
    return FieldDistribution::Distributed;
  }

  [[nodiscard]] bool supports(const EllipticBuildRequest& request) const noexcept {
    return request.distribution == FieldDistribution::Distributed && !request.active;
  }

  EllipticFactoryBuildResult<PoissonFFTSolver> build(EllipticBuildRequest request) const noexcept {
    return capture_local_elliptic_factory_build<PoissonFFTSolver>([request = std::move(
                                                                       request)]() mutable {
      if (request.distribution != FieldDistribution::Distributed)
        throw std::invalid_argument("PoissonFFTSolver requires a distributed single-rank field");
      return PoissonFFTSolver(request.geometry, request.boxes, request.boundary,
                              std::move(request.active), false);
    });
  }
};
static_assert(
    EllipticFactory<ExplicitFftDistributionFactory, PoissonFFTSolver>,
    "un backend a options propres doit pouvoir fournir sa factory sans changer le coupler");

struct FaultyFftMaterializationFactory {
  bool declared_spectral = false;
  bool materialized_spectral = false;

  [[nodiscard]] std::string_view collective_contract() const noexcept {
    return "pops.test.faulty-fft-materialization-factory@1";
  }
  [[nodiscard]] EllipticOperatorContract expected_operator_contract(
      const EllipticBuildRequest& request) const {
    return PoissonFFTSolver::expected_operator_contract(request, declared_spectral);
  }
  [[nodiscard]] FieldDistribution materialized_distribution(
      const EllipticBuildRequest&) const noexcept {
    return FieldDistribution::Distributed;
  }
  [[nodiscard]] bool supports(const EllipticBuildRequest& request) const noexcept {
    return request.distribution == FieldDistribution::Distributed;
  }
  EllipticFactoryBuildResult<PoissonFFTSolver> build(EllipticBuildRequest request) const noexcept {
    return capture_local_elliptic_factory_build<PoissonFFTSolver>(
        [request = std::move(request), spectral = materialized_spectral]() mutable {
          return PoissonFFTSolver(request.geometry, request.boxes, request.boundary,
                                  std::move(request.active), spectral);
        });
  }
};
static_assert(EllipticFactory<FaultyFftMaterializationFactory, PoissonFFTSolver>);

struct TestActiveRegionSource {
  [[nodiscard]] static constexpr PreparedProviderIdentity provider_identity() noexcept {
    return {"pops.test.elliptic-interface.active-region", 1};
  }
  void serialize_exact_parameters(ExactContractBuilder& contract) const { contract.scalar(true); }
  [[nodiscard]] bool operator()(Real, Real) const noexcept { return true; }
};

TEST(test_elliptic_interface, build_request_validation_is_pure_and_owns_ghost_contract) {
  const Box2D domain = Box2D::from_extents(8, 8);
  EllipticBuildRequest request{Geometry{domain, 0.0, 1.0, 0.0, 1.0},
                               BoxArray(std::vector<Box2D>{domain}),
                               DistributionMapping(std::vector<int>{0}),
                               BCRec{},
                               {},
                               FieldDistribution::Distributed};
  EXPECT_TRUE(detail::elliptic_build_request_is_valid(request, 0, 2));
  EXPECT_FALSE(detail::elliptic_build_request_is_valid(request, 2, 2));

  request.rhs_ghosts = 1;
  EXPECT_FALSE(detail::elliptic_build_request_is_valid(request, 0, 2));
  request.rhs_ghosts = 0;
  request.phi_ghosts = -1;
  EXPECT_FALSE(detail::elliptic_build_request_is_valid(request, 0, 2));

  request.phi_ghosts = 1;
  request.distribution = FieldDistribution::Replicated;
  request.mapping = DistributionMapping(std::vector<int>{1});
  EXPECT_TRUE(detail::elliptic_build_request_is_valid(request, 1, 2));
  EXPECT_FALSE(detail::elliptic_build_request_is_valid(request, 0, 2));
}

TEST(test_elliptic_interface, postbuild_rejects_backend_that_ignores_physical_boundary) {
  const Box2D domain = Box2D::from_extents(8, 8);
  BCRec boundary;
  boundary.xlo = BCType::Dirichlet;
  boundary.xhi = BCType::Dirichlet;
  EXPECT_THROW(
      (void)make_elliptic_solver<PoissonFFTSolver>({Geometry{domain, 0.0, 1.0, 0.0, 1.0},
                                                    BoxArray(std::vector<Box2D>{domain}),
                                                    DistributionMapping(std::vector<int>{0}),
                                                    boundary,
                                                    {},
                                                    FieldDistribution::Distributed},
                                                   FaultyFftMaterializationFactory{}),
      std::invalid_argument);
}

TEST(test_elliptic_interface, postbuild_rejects_backend_that_ignores_declared_option) {
  const Box2D domain = Box2D::from_extents(8, 8);
  EXPECT_THROW((void)make_elliptic_solver<PoissonFFTSolver>(
                   {Geometry{domain, 0.0, 1.0, 0.0, 1.0},
                    BoxArray(std::vector<Box2D>{domain}),
                    DistributionMapping(std::vector<int>{0}),
                    BCRec{},
                    {},
                    FieldDistribution::Distributed},
                   FaultyFftMaterializationFactory{/*declared_spectral=*/false,
                                                   /*materialized_spectral=*/true}),
               std::invalid_argument);
}

TEST(test_elliptic_interface, postbuild_rejects_backend_that_ignores_active_region) {
  const Box2D domain = Box2D::from_extents(8, 8);
  EXPECT_THROW(
      (void)make_elliptic_solver<PoissonFFTSolver>(
          {Geometry{domain, 0.0, 1.0, 0.0, 1.0}, BoxArray(std::vector<Box2D>{domain}),
           DistributionMapping(std::vector<int>{0}), BCRec{},
           ActiveRegionProvider2D(TestActiveRegionSource{}), FieldDistribution::Distributed},
          FaultyFftMaterializationFactory{}),
      std::invalid_argument);
}

// GAP DOCUMENTE : les solveurs DIRECTS resolvent en une passe, sans tolerance iterative.
// Ils modelent EllipticSolver (cartesien) ou PolarEllipticSolver (polaire) mais PAS
// LinearSolver. On le PROUVE pour verrouiller la frontiere du concept.
static_assert(EllipticSolver<PoissonFFTSolver>,
              "PoissonFFTSolver modele EllipticSolver (cartesien)");
static_assert(EllipticSolver<DistributedFFTSolver>,
              "DistributedFFTSolver modele EllipticSolver (cartesien)");
static_assert(PolarEllipticSolver<PolarPoissonSolver>,
              "PolarPoissonSolver modele PolarEllipticSolver (polaire)");
static_assert(!LinearSolver<PoissonFFTSolver>,
              "PoissonFFTSolver est DIRECT (pas de solve(tol, iters)) : non-LinearSolver attendu");
static_assert(!LinearSolver<DistributedFFTSolver>,
              "DistributedFFTSolver est DIRECT : non-LinearSolver attendu");
static_assert(!LinearSolver<PolarPoissonSolver>,
              "PolarPoissonSolver est DIRECT : non-LinearSolver attendu");

// Le resultat d'arret du solveur objet iteratif est bien NON void.
static_assert(!std::is_same_v<decltype(std::declval<GeometricMG&>().solve(Real(1e-8), 1)), void>,
              "GeometricMG::solve(tol, iters) rend un compte rendu (int), pas void");

// =====================================================================================
// (3) FieldPostProcessor : phi -> aux/grad. field_postprocess (fonction libre) le modele.
// On capture le pointeur de fonction dans un type pour la verification du concept.
using FieldPostProcessFn = void (*)(const MultiFab&, MultiFab&, Real, Real, FieldPostProcess);
static_assert(FieldPostProcessor<FieldPostProcessFn>,
              "field_postprocess (signature (phi, out, cx, cy, spec) -> void) doit modeler "
              "FieldPostProcessor");

// Helper generique CONTRAINT par le concept : ne compile que si pp est un FieldPostProcessor.
// Sert a prouver que le concept est utilisable comme contrainte (pas seulement un predicat).
template <FieldPostProcessor PP>
void apply_pp(PP pp, const MultiFab& phi, MultiFab& out, Real cx, Real cy, FieldPostProcess spec) {
  pp(phi, out, cx, cy, spec);
}

namespace {

// phi connu (1 ghost) periodique, partage par le TEST FieldPostProcessor (identique au temoin de
// test_elliptic_problem).
auto fr(const Geometry& geom, int i, int j) {
  return std::sin(2 * kPi * geom.x_cell(i)) * std::sin(2 * kPi * geom.y_cell(j));
}

}  // namespace

// field_postprocess appele DIRECTEMENT vs via le helper contraint par FieldPostProcessor : memes
// bits. Prouve que la fonction libre traverse la contrainte de concept sans rien changer.
TEST(test_elliptic_interface, field_postprocess_via_concept_is_bit_identical) {
  const int N = 32;
  Box2D dom = Box2D::from_extents(N, N);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  BoxArray ba(std::vector<Box2D>{dom});
  DistributionMapping dm(1, 1);
  BCRec bc;  // periodique

  MultiFab phi(ba, dm, 1, 1);
  {
    Array4 p = phi.fab(0).array();
    const Box2D v = phi.box(0);
    for_each_cell(v, [=] POPS_HD(int i, int j) { p(i, j) = fr(geom, i, j); });
    fill_ghosts(phi, dom, bc);
  }
  const Real cx = Real(1) / (2 * geom.dx());
  const Real cy = Real(1) / (2 * geom.dy());

  const FieldPostProcess spec{FieldPostProcess::GradSign::Plus, true};
  MultiFab direct(ba, dm, 3, 1), via_concept(ba, dm, 3, 1);
  field_postprocess(phi, direct, cx, cy, spec);
  apply_pp(&field_postprocess, phi, via_concept, cx, cy, spec);

  bool bit_eq = true;
  {
    const ConstArray4 ad = direct.fab(0).const_array();
    const ConstArray4 ac = via_concept.fab(0).const_array();
    const Box2D v = direct.box(0);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        for (int c = 0; c < 3; ++c)
          if (ad(i, j, c) != ac(i, j, c))
            bit_eq = false;
  }
  EXPECT_TRUE(bit_eq) << "FieldPostProcessor_via_concept_bit_identique";
}

// Verification runtime legere : GeometricMG (LinearSolver iteratif) resout et son compte rendu
// d'arret est un int positif borne par max_cycles. On ne valide pas la physique (couverte
// ailleurs), seulement le CONTRAT de retour du concept.
TEST(test_elliptic_interface, linear_solver_report_is_bounded_int) {
  const int N = 32;
  Box2D dom = Box2D::from_extents(N, N);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  BoxArray ba(std::vector<Box2D>{dom});
  DistributionMapping dm(1, 1);
  BCRec bc;  // periodique

  GeometricMG mg(geom, ba, bc);
  Array4 f = mg.rhs().fab(0).array();
  const Box2D v = mg.rhs().box(0);
  for_each_cell(v, [=] POPS_HD(int i, int j) { f(i, j) = fr(geom, i, j); });
  mg.phi().set_val(0.0);
  const int cycles = mg.solve(Real(1e-8), 50);  // variante LinearSolver (tol, iters)
  EXPECT_TRUE(cycles >= 0 && cycles <= 50)
      << "LinearSolver_GeometricMG_compte_rendu_borne cycles=" << cycles;
  static_assert(std::is_same_v<decltype(cycles), const int>,
                "GeometricMG::solve(tol, iters) rend int (nombre de V-cycles)");
}
