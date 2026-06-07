# Backlog recherche / externe (items NON auto-completables)

Scope par workflow multi-agents (juin 2026). Ces items sont de la RECHERCHE numerique ou de l'INTEGRATION
EXTERNE : ils ne se "terminent" pas comme une PR d'agent. Chacun a un etat factuel, un prochain pas concret,
et un critere NO-GO / decision-gate. A reprendre par le proprietaire quand c'est pertinent.

## AP tensorielle sous champ fort -- RECHERCHE

- Etat : le Schur condense (schur_condensation.hpp, condensed_schur_source_stepper.hpp) gere DEJA le champ
  fort INCONDITIONNELLEMENT (inversion 2x2 exacte LorentzEliminator). Il est stable, mais ne neutralise pas
  la croissance des coefficients en omega_c*dt. Un AP tensoriel serait un gain d'EFFICACITE, PAS de stabilite
  (deja acquise). L'AP deux-fluides existe seulement en modif scalaire du RHS (two_fluid_ap.md), pas en
  reformulation d'operateur tensoriel.
- Prochain pas (etude math, pas de code) : expansion asymptotique des eq. Schur condensees dans la limite
  omega_c >> omega_d (analogue two_fluid_ap.md sec.3) ; identifier si un facteur d'echelle adimensionnel
  emerge ; toy 1D (isotherme plan-parallele, B_z uniforme) comparant Schur actuel vs AP-tensoriel ; PR
  seulement si valide.
- NO-GO : si l'AP exige un operateur NON-LOCAL incompatible avec les roles/DSL -> differer indefiniment.

## Perf full-device scaling -- BESOIN ROMEO D'ABORD

- Etat : la grille GROSSIERE est REPLIQUEE par CHOIX (amr_coupler_mp.hpp:224-234 coupler_make_coarse_layout,
  amr_dsl_block.hpp:159-188 ; replicated_coarse=true). Le mode DISTRIBUE existe (distribute_coarse=true) mais
  mesure 3-5x PLUS LENT (705-1403 ms/pas vs 222-278 replique) : le V-cycle MG echange des halos cross-rang a
  chaque niveau grossier (~7 niveaux, fill_boundary latency-bound sur boites 2x2). Le Poisson domine 96-99.9%.
- Prochain pas : profil ROMEO multi-GPU (instrumenter GeometricMG::vcycle_rec avec Kokkos::fence + timing par
  niveau) sur np=2/4 GH200 -> isoler ou se passe le ralentissement 3-5x.
- DECISION GATE : si la latence > 50% du temps grossier -> MG HYBRIDE (distribuer le fin, gather pour le
  bottom-solve) vaut 2-3 semaines ; sinon la replication grossiere est le BON compromis -> clore le lot.

## Integration SAMRAI -- EXTERNE-GROS, DIFFEREE

- Etat : adc a une pile AMR MAISON COMPLETE (Phase 1 hierarchie figee : substeps #175, sources couplees #179,
  IMEX local #184 ; Phase 2 regrid union-tags #199 ; reflux/FluxRegister, multi-bloc, validee CPU Serial/
  OpenMP/MPI np=1/2/4 + GPU Cuda GH200). SAMRAI (LLNL) = framework AMR externe.
- VERDICT : DIFFERER tant que l'AMR maison couvre le chemin science (diocotron polaire convergent en
  resolution via regrid union-tags). Critere de REOUVERTURE : un besoin que l'AMR maison ne couvre PAS
  (ex. MG multi-niveau distribue mature, scaling hero-run multi-noeud) ET un cout d'integration justifie
  (dependance C++, mapping structures, binding Python, maintenance).

## P7-a implicit-total -- RECHERCHE

- Etat : aujourd'hui splitting Lie (explicite-transport SSPRK3 + implicite-source-Schur theta). "implicit-total"
  = schema TOTALEMENT implicite (transport + source), gros chantier numerique (Jacobien complet, solveur
  non-lineaire). P7-b (params runtime DSL) est SEPARE et DOABLE (PR en cours).
- Prochain pas : etude de schema (recherche) ; pas auto-completable. A ne lancer que si le splitting Lie
  ordre 1 devient un facteur limitant mesure (sinon le Strang ordre 2, plus simple, suffit -- cf roadmap).
