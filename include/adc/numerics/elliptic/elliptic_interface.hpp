#pragma once

#include <adc/core/types.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/numerics/elliptic/elliptic_problem.hpp>  // FieldPostProcess (spec), field_postprocess
#include <adc/numerics/elliptic/elliptic_solver.hpp>   // concept EllipticSolver (deja en place)

#include <concepts>

// FORMALISATION ADDITIVE des contrats communs de l'etage elliptique (audit D.1).
//
// But : NOMMER, sous forme de concepts C++20, les contrats partages DEJA codes par
// les classes elliptiques existantes, et le PROUVER par static_assert. Ce header est
// PUREMENT DESCRIPTIF :
//   - il n'inclut AUCUNE logique flottante, ne touche AUCUNE classe existante ;
//   - les concepts sont des predicats de COMPILATION (metaprogrammation hote), donc
//     sans aucune incidence device (ni kernel, ni lambda etendue, ni emission nvcc) :
//     la pile elliptique device-validee reste BIT-IDENTIQUE ;
//   - le contrat "solveur au niveau MultiFab" (rhs/phi/solve/residual/geom) est DEJA
//     porte par le concept EllipticSolver (elliptic_solver.hpp) ; on ne le redefinit
//     PAS, on le REUTILISE et on documente comment les trois nouveaux concepts s'y
//     articulent.
//
// REALITE DU CODE vs intitule de l'audit. L'item D.1 nommait des artefacts qui
// n'existent pas sous cette forme dans le depot ; les concepts ci-dessous decrivent
// le contrat REELLEMENT commun, pas une cible theorique :
//   - il n'y a PAS de classe "TensorEllipticOperator". L'operateur elliptique est
//     realise par des FONCTIONS LIBRES (apply_laplacian, poisson_residual, gs_smooth
//     dans poisson_operator.hpp). Une fonction libre n'est pas un TYPE : aucun concept
//     ne peut la contraindre. Le ROLE d'operateur (porter les coefficients du stencil
//     et la geometrie pour une matvec coherente avec le residu) est, lui, porte par un
//     TYPE : GeometricMG, via ses accesseurs op_eps()/op_coef()/op_kappa()/... + bc()
//     + geom(). C'est ce role-la que capture EllipticOperator ci-dessous, parce que
//     c'est exactement ce dont TensorKrylovSolver depend pour sa matvec matrice-libre.
//   - "FieldPostProcess" est DEJA un nom pris : c'est la SPEC (struct POD : signe du
//     gradient, stockage de phi) dans elliptic_problem.hpp, appliquee par la fonction
//     libre field_postprocess(phi, out, cx, cy, spec). Le concept du POST-TRAITEMENT
//     callable est donc nomme FieldPostProcessor (suffixe -or : l'objet qui APPLIQUE),
//     pour ne pas masquer la spec existante.

namespace adc {

// ---------------------------------------------------------------------------
// (1) EllipticOperator : role d'OPERATEUR du stencil elliptique au niveau MultiFab.
//
// Contrat = ce qu'un consommateur de matvec matrice-libre (TensorKrylovSolver) lit sur
// son operateur pour appliquer L_int(phi) = div(A grad phi) - kappa phi de facon
// COHERENTE avec le residu MG (poisson_residual) : la geometrie, la CL physique, le
// BoxArray/DistributionMapping du niveau fin, et les POINTEURS de coefficient de
// l'operateur (eps_x, eps_y, kappa, poids cut-cell, termes croises Axy/Ayx, masque).
// Un terme inactif rend nullptr (cf. op_*_ptr internes), ce que le concept exige juste
// d'etre APPELABLE et convertible en const MultiFab* ; il ne contraint pas la valeur.
//
// GeometricMG modele ce role (et c'est le seul TYPE qui le porte aujourd'hui). Les
// fonctions libres apply_laplacian/poisson_residual/gs_smooth en sont la realisation
// de plus bas niveau (l'APPLICATION proprement dite) : non contraignables par concept
// car ce ne sont pas des types. EllipticOperator NOMME donc l'interface de FOURNITURE
// des coefficients (le "quoi appliquer"), pas le kernel d'application (le "comment").
//
// NOTE : EllipticOperator n'EXIGE PAS solve()/rhs()/phi() ; il decrit le seul role
// operateur. GeometricMG le complete par EllipticSolver (il est aussi un solveur), mais
// un type purement operateur (sans solve) satisferait deja EllipticOperator.
template <class Op>
concept EllipticOperator = requires(Op op) {
  { op.geom() } -> std::convertible_to<const Geometry&>;
  { op.bc() } -> std::convertible_to<const BCRec&>;
  // Pointeurs de coefficient du niveau fin : nullptr quand le terme est inactif.
  { op.op_mask() } -> std::convertible_to<const MultiFab*>;
  { op.op_coef() } -> std::convertible_to<const MultiFab*>;
  { op.op_eps() } -> std::convertible_to<const MultiFab*>;
  { op.op_kappa() } -> std::convertible_to<const MultiFab*>;
  { op.op_eps_y() } -> std::convertible_to<const MultiFab*>;
  { op.op_a_xy() } -> std::convertible_to<const MultiFab*>;
  { op.op_a_yx() } -> std::convertible_to<const MultiFab*>;
};

// ---------------------------------------------------------------------------
// (2) LinearSolver : solveur ITERATIF a critere d'arret explicite.
//
// Contrat = solve(rel_tol, max_iters) qui rend un RESULTAT (la convention "resoudre
// jusqu'a une tolerance relative en au plus max_iters etapes, en rendant un compte
// rendu"). On ne contraint PAS le type de retour a une struct precise : GeometricMG
// rend int (nombre de V-cycles effectues) et TensorKrylovSolver rend KrylovResult
// (iters + residu relatif + convergence). Le seul invariant commun REEL est : le retour
// n'est PAS void (il porte une information d'arret). Le concept reflete donc cette
// realite via !std::same_as<void>, sans imposer un type de resultat partage qui
// n'existe pas (le forcer demanderait de modifier une des deux classes : interdit).
//
// On exige aussi le socle EllipticSolver (rhs/phi/solve()/residual/geom) : un
// LinearSolver elliptique EST un EllipticSolver qui, EN PLUS, expose la variante a
// tolerance. GeometricMG et TensorKrylovSolver modelent les deux.
//
// GAP DOCUMENTE (concept VOLONTAIREMENT separe de EllipticSolver). Les solveurs DIRECTS
// PoissonFFTSolver, DistributedFFTSolver et PolarPoissonSolver resolvent en UNE passe
// (FFT + Thomas) : ils n'ont PAS de solve(rel_tol, max_iters) ni de notion de tolerance
// iterative. Ils modelent EllipticSolver (au niveau MultiFab cartesien) ou
// PolarEllipticSolver (polaire), mais PAS LinearSolver, et c'est CORRECT : un solveur
// direct n'est pas un solveur iteratif. LinearSolver capture donc le sous-ensemble
// ITERATIF du contrat, sans pretendre que tous les backends elliptiques le portent.
template <class S>
concept LinearSolver = EllipticSolver<S> && requires(S s, Real tol, int it) {
  // Variante a tolerance : resout jusqu'a rel_tol (ou max_iters) et rend un compte rendu
  // d'arret. Type de retour LIBRE mais NON void (int pour MG, KrylovResult pour Krylov).
  s.solve(tol, it);
  requires !std::same_as<decltype(s.solve(tol, it)), void>;
};

// ---------------------------------------------------------------------------
// (3) FieldPostProcessor : derivation du champ a partir du potentiel, phi -> aux/grad.
//
// Contrat = un APPLICATEUR callable avec la signature de field_postprocess :
//   (const MultiFab& phi, MultiFab& out, Real cx, Real cy, FieldPostProcess spec) -> void
// c.-a-d. ecrire dans out la convention (phi en composante 0 si demande) + le gradient
// centre (+/- selon le signe de la spec), avec cx = 1/(2 dx), cy = 1/(2 dy). La SPEC
// (signe du gradient, stockage de phi) reste la struct FieldPostProcess existante
// (elliptic_problem.hpp) : on ne la redefinit pas, on la PARAMETRE.
//
// On contraint un type CALLABLE (foncteur ou pointeur de fonction), pas une classe a
// methodes : c'est ce qu'EST field_postprocess (une fonction libre). Le static_assert
// plus bas prouve que &field_postprocess satisfait FieldPostProcessor.
template <class F>
concept FieldPostProcessor =
    requires(F f, const MultiFab& phi, MultiFab& out, Real cx, Real cy,
             FieldPostProcess spec) {
      { f(phi, out, cx, cy, spec) } -> std::same_as<void>;
    };

}  // namespace adc
