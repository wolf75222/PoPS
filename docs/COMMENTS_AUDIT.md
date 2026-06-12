# Audit des commentaires de `adc_cpp`

Date : 2026-06-12.
Base relue : `origin/master` / `ffb9022`.

Perimetre : section Comments du Google C++ Style Guide appliquee a tout le code source du depot :
`include/adc/**/*.hpp`, `python/*.cpp`, `python/adc/*.py`, `tests/**`, `python/tests/**`, `bench/**`,
plus les scripts CMake. Soit ~16 sous-systemes decoupes par couche (coeur/AMR/parallele/physique,
maillage, numerique, couplage, runtime, bindings, tests, bench) et deux balayages transversaux
(TODO et coherence de langue). L'audit porte sur la qualite des commentaires, pas sur la correction
du code : un commentaire est juge fidele, perime, redondant ou absent par rapport au code qu'il decrit.

Methode : relecture integrale par sous-systeme, puis verification croisee de chaque constat non
cosmetique sur piece (le commentaire est confronte ligne a ligne au code qu'il pretend decrire, et a
ses commentaires voisins). Les constats marques `bloquant` sont des commentaires factuellement FAUX,
tous re-verifies une seconde fois. Les chiffres de couverture (`@file`, types documentes) sont des
comptages directs sur le worktree en lecture seule.

Docs lies :
- [`CODEBASE_AUDIT.md`](CODEBASE_AUDIT.md) : audit de maintenabilite, meme en-tete et meme ton.
- [`CODE_DOCUMENTATION_CONVENTION.md`](CODE_DOCUMENTATION_CONVENTION.md) : LA convention de
  commentaires du projet (blocs `///`, `@file`/`@brief`, contrats, invariants threading/MPI/device).
  CODEBASE_AUDIT.md la cite comme reference normative (lignes 15, 542, 643), `.clang-format` s'aligne
  sur son esprit. PROBLEME : ce fichier est present dans le working tree principal mais N'EST PAS
  COMMITE (`git status` le donne `??`). Il est donc ABSENT du commit `ffb9022` que les worktrees
  checkoutent, et tout l'audit ci-dessous a du etre conduit contre la convention DE FAIT (patterns
  dominants + CODEBASE_AUDIT.md) faute de pouvoir lire la norme. Recommandation prioritaire ADC-125 :
  committer `docs/CODE_DOCUMENTATION_CONVENTION.md`, sans quoi le referentiel de tout l'effort de
  documentation reste un lien mort dans l'arbre versionne.
- [`check_docs.py`](check_docs.py) : ne lint que les `.md`, pas les en-tetes `.hpp`.

## 1. Synthese chiffree

Couverture des en-tetes `@file`/`@brief` et constats par severite, par sous-systeme. La severite est
celle apres verification croisee (un `bloquant` initial infirme a la verification est redescendu).

| Sous-systeme | Fichiers / lignes | En-tete @file | Bloq. | Import. | Cosm. |
|---|---|---|---|---|---|
| coeur-amr-parallele-physique | 23 / 2861 | 21/23 | 1 | 5 | 5 |
| maillage | 13 / 1912 | 12/13 | 1 | 1 | 6 |
| numerique-1 (elliptique) | 12 / 3252 | 4/12 Doxygen | 1 | 8 | 7 |
| numerique-2 (spatial/EB) | 10 / 3119 | 8/10 | 2 | 3 | 4 |
| numerique-3 (temps/AMR) | 13 / 2198 | 0/13 Doxygen | 1 | 2 | 7 |
| couplage-1 | 10 / 2499 | 10/10 | 1 | 1 | 8 |
| couplage-2 | 7 / 1850 | 7/7 | 0 | 3 | 4 |
| runtime-1 | 5 / 3163 | 5/5 | 1 | 2 | 2 |
| runtime-2 | 13 / 2855 | 13/13 | 0 | 3 | 3 |
| runtime-3 | 4 / 1380 | 4/4 | 0 | 1 | 5 |
| python-bindings (.cpp) | 3 / 3654 | 0/3 | 2 | 3 | 4 |
| python adc (core) | 3 / 3162 | 3/3 docstring | 0 | 5 | 4 |
| python adc (dsl) | 1 / 4648 | 1/1 docstring | 0 | 1 | 9 |
| tests + bench (transversal) | 246 / 27915 | 245/246 intention | 0 | 0 | 5 |
| TODO + langue (transversal) | tout le depot | n/a | 0 | 1 | 5 |

Total : 10 bloquants (commentaires faux), ~38 importants, ~68 cosmetiques, ~116 constats sur les 13
sous-systemes de code plus deux balayages transversaux qui re-agregent certains constats (TODO,
ilot anglais).

Verdict global : la qualite documentaire est elevee et homogene, nettement au-dessus de la moyenne
d'un projet de cette taille. Les commentaires d'IMPLEMENTATION sont le point fort recurrent : les
pieges reels (deadlock CUDA-IPC des tampons MPI, securite async de l'arene Kokkos, derivation du
terme geometrique polaire, contournements nvcc/EDG par foncteurs nommes, bornes CFL de stabilite,
moyenne harmonique vs arithmetique de la permittivite) sont expliques avec leur justification, sans
paraphrase de l'evident. La dette ne vient pas d'un manque de documentation mais de trois axes :
(1) un noyau dur de 10 commentaires devenus FAUX (comment-rot), qui est la dette dangereuse ;
(2) une migration Doxygen restee a mi-chemin (en-tetes et API publiques documentees en `//` non
extractibles, doubles en-tetes `//`+`///` qui dupliquent la source de verite) ; (3) des references
documentaires fragiles (numeros de ligne codes en dur, chemins de doc perimes).

## 2. Etat des lieux par sous-section Google Comments

### Comment Style
Convention de fait claire et largement appliquee : `///` Doxygen pour l'API, `//` pour l'interne,
`///<` trailing pour les membres, `#pragma once` unanime, aucun bloc `/* */` parasite. Trois ecarts
structurants. (a) Migration Doxygen inachevee : `include/adc/numerics/elliptic` documente 8/12
fichiers par une en-tete prose `//` (fidele mais invisible a Doxygen), `numerics/time` 0/13 en
Doxygen, et plusieurs API publiques sont en `//` (cf. File/Function Comments). (b) Doubles en-tetes
legacy : un bloc `//` paraphrase le `/// @file` dans ~12 fichiers du coeur, 12/13 du maillage, 7/10
de couplage-1, 5/7 de couplage-2 - deux sources de verite a maintenir ensemble. Cas extreme :
`compute_face_fluxes` documente TROIS fois dans `include/adc/numerics/spatial_operator.hpp:548,632,641`.
(c) Ilot de style `/** */` Javadoc dans 4 fichiers `include/adc/physics/` (`euler.hpp:17`,
`advection_diffusion.hpp`, `langmuir.hpp`, `two_fluid_isothermal.hpp`) la ou tout le reste utilise
`///`. Anomalie isolee : un emoji dans `include/adc/runtime/amr_dsl_block.hpp:172`, seul caractere
non-ASCII de ce type du depot.

### File Comments
Couverture globale bonne mais inegale. Absents (demarrent sur `#include`/`namespace`) :
`include/adc/parallel/comm.hpp:1` et `parallel/load_balance.hpp:1` (en-tete `//` seule),
`include/adc/mesh/patch_box.hpp:1`, `include/adc/numerics/time/amr_advance.hpp:1` et
`amr_flux_helpers.hpp:1`, et surtout `python/system.cpp:1` + `python/amr_system.cpp:1` (aucune
en-tete du tout, 1831 et 1192 lignes). Placement non uniforme du `/// @file` (avant vs apres les
includes) dans `numerics-2` (3 fichiers) et `couplage-1` (4 avant / 6 apres). Reference morte :
`docs/CODEBASE_AUDIT.md:15` pointe vers `CODE_DOCUMENTATION_CONVENTION.md`, non commite (cf. en-tete).

### Struct and Class Comments
Quasi tous les types publics non triviaux portent un contrat. Defauts qualitatifs : le concept
`EquationBlockLike` est resume FAUX (`include/adc/core/equation_block.hpp:76`, cf. comment rot) ;
`struct Aux`, canal de couplage central, est decrit par un bloc `//` rattache a une X-macro et non
par un `/// @brief` sur le type (`include/adc/core/state.hpp:102`), donc Doxygen ne lui associe
aucune doc de classe alors que son voisin `StateVec` en a une ; les contraintes de
`DistributedFFTSolver` sont sur- et sous-declarees (`include/adc/numerics/elliptic/poisson_fft_solver.hpp:102`).

### Function Comments
Tres bonne couverture (`@param`/`@return`/`@throws` systematiques sur l'API runtime). Trois familles
de defauts. (a) Enumerations de parametres perimees, le cas le plus frequent : les `@param limiter`
et `@param time` de `include/adc/runtime/system.hpp:79,169,198,199` retardent sur les validateurs
reels (weno5, ssprk3, euler manquants), de meme `include/adc/runtime/amr_system.hpp:203` (`weno5 ;
rusanov` suggere a tort une restriction). (b) Contrats faux : `set_conservative_state`
(`include/adc/runtime/amr_system.hpp:363`, cf. comment rot). (c) Parametres non documentes :
le ctor de `GeometricMG` couvre `active/replicated/cut_cell/levelset` mais omet
`min_coarse/nu1/nu2/nbottom` (`include/adc/numerics/elliptic/geometric_mg.hpp:76`). Cote Python,
`System.add_block`/`AmrSystem.add_block` (`python/adc/__init__.py:1365,2415`), points d'entree
primaires, n'ont aucune docstring alors que `add_equation` est detaillee ; plusieurs `.def` pybind
non triviaux sont exposes sans docstring (`python/bindings.cpp:362` : `step_cfl`, `step_adaptive`,
`dt_hotspot`, `set_poisson`, `variable_names/roles`).

### Variable Comments
Globalement soignee (membres non evidents annotes avec unites). Defauts : `phi_n_` decrit avec un
cycle de vie et une API faux (`include/adc/coupling/condensed_schur_source_stepper.hpp:395`, cf.
comment rot) ; `mg_` annonce un invariant "Dirichlet homogene" que le code ne garantit pas
(`include/adc/numerics/elliptic/composite_fac_poisson.hpp:489`) ; membres de `MGLevel` partiellement
nus (`geometric_mg.hpp:496`, `coef`/`mask` non decodables sur place) ; jetons de configuration
Poisson sans doc par champ (`include/adc/runtime/system_field_solver.hpp:90`).

### Implementation Comments
Point fort du depot, mais c'est aussi la ou vit la dette dangereuse. La majorite des commentaires
substantiels verifies (>120 au total) sont EXACTS. Les defauts sont les commentaires de comment-rot
(section 3) et les references documentaires fragiles : chemins de repertoire perimes apres un
renommage (`integrator/` et `operator/` dans `include/adc/numerics/time/ssprk.hpp:13`,
`implicit_stepper.hpp:31`, `amr_reflux.hpp:25`), renvois `bibliographie sect. 3.3`/`sect. 4.3` qui ne
resolvent vers aucun ancrage (`include/adc/mesh/box_hash.hpp:4,20` + `fill_boundary.hpp:8,101,159`,
5 sites), et numeros de ligne inter-fichiers codes en dur qui derivent a chaque edition du `.md`
cible (`HOFFART_FIDELITY.md ligne 39` desormais vide, cite par `wall_predicate.hpp:41` et
`cut_fraction.hpp:12`). Recommandation transverse : remplacer les numeros de ligne par des ancrages
symboliques (titre de section, nom de fonction).

### Punctuation, Spelling, and Grammar
Coherence quasi totale (cf. section 5). Coquilles isolees : `referenceent`
(`include/adc/core/coupled_system.hpp:40`), `apellent` (`numerics/spatial_operator.hpp:387`),
`Exposes` au lieu d'`Expose` (`numerics/spatial_discretisation.hpp:32`), `Cohrent`
(`runtime/native_loader.hpp:733`), `tolerree` (`python/adc/__init__.py:870`). Phrases agrammaticales
ponctuelles : `for_each.hpp:113` (`acces hote sur sont sans course`), coupure `limite de / Debye`
dans `numerics/time/imex.hpp:11`, auto-correction laissee en place `... non : ...` dans
`python/system.cpp:1511`. Tag `@returns` (anglais) au lieu de `@return` : 2 occurrences
(`amr_coupler_mp.hpp:558`, `amr_system.hpp:439`).

### TODO Comments
Voir section 4. En resume : 20 marqueurs, aucun au format Google, tous des etiquettes de feuille de
route plutot que des taches.

## 3. Comment rot : commentaires FAUX (severite bloquant)

Dette la plus dangereuse : un lecteur qui suit ces commentaires est activement induit en erreur.
Chacun a ete confronte au code et re-verifie.

| Fichier:ligne | Affirmation du commentaire | Realite du code |
|---|---|---|
| `include/adc/core/equation_block.hpp:76` | concept requiert un membre `State` | le concept exige `B::Model` ; aucun alias `State` n'existe (lignes 80-87) |
| `include/adc/mesh/fill_boundary.hpp:242` | MPI recoit des pointeurs UNIFIES (GPUDirect) | `sbuf`/`rbuf` sont en `SharedHostPinnedSpace` (hote epingle), vus HOST par MPI pour eviter CUDA-IPC ; contredit les lignes 118-124 et `core/allocator.hpp` |
| `include/adc/numerics/elliptic/geometric_mg.hpp:425` | durcissement du lissage STICKY entre solves | le code sauve `nu1_/nu2_` (446) et les RESTAURE au retour (454,457) ; le paragraphe voisin (442-444) dit l'inverse |
| `include/adc/numerics/elliptic/polar_tensor_operator.hpp:381` | `solve()` = BiCGStab preconditionne Jacobi | precond par defaut = RadialLine (Thomas radial), selectionnable via `precond_` ; omet le pinning de jauge |
| `include/adc/numerics/spatial_operator_eb.hpp:136` | apertures de face fractionnaires `alpha in [1e-3,1]` entre cellules actives | `cut_distance` rend `h` pour tout voisin actif -> alpha = 1 ; apertures BINAIRES {0,1}, seul `kappa` est fractionnaire |
| `include/adc/numerics/time/amr_reflux.hpp:19` | "version minimale" : sous-cyclage a venir, aux uniforme | `amr_step_2level` IMPLEMENTE deja le sous-cyclage Berger-Oliger (155-174) et lit un aux spatial ; fichier vivant (inclus par `spectral_coupler.hpp`) |
| `include/adc/coupling/condensed_schur_source_stepper.hpp:395` | `phi_n_` alloue au premier `advance_source` | alloue inconditionnellement au ctor (234) ; la methode `advance_source` N'EXISTE PAS (l'API est `step()`, grep depot = 0 autre occurrence) |
| `include/adc/runtime/amr_system.hpp:363` | `set_conservative_state` leve en multi-blocs | la facade `build_multi` thread l'etat aux blocs NATIFS (379) ; seul le chemin compile (.so) rejette (315) |
| `python/bindings.cpp:70` | serie : `my_rank=1`, `n_ranks=0` | `comm.hpp` rend `my_rank()=0`, `n_ranks()=1` ; contredit les docstrings adjacents (73-74) |
| `python/bindings.cpp:240` | descripteurs `set_source_stage` : "Cartesien seulement (polaire : rejet)" | l'etage polaire construit un `PolarCondensedSchurSourceStepper` et honore `bz_aux_component` sans rejet (`system.cpp:1410-1438`) |

Tous sont des vestiges de refactors anterieurs (renommage d'API, correctif pinned-host, vague
multi-blocs, retrait de garde-fou mono-rang). 6/10 sont auto-corrigeables en une ligne ; les 4 autres
demandent une reformulation du contrat.

Note de second rang : quatre constats initialement classes bloquants ont ete redescendus a important
a la verification car le commentaire decrit une intention correcte ou un chemin sans rupture de
comportement : `composite_fac_poisson.hpp:489` (Dirichlet homogene suppose), `elliptic_solver.hpp:12`
(Coupler deja template), `elliptic_problem.hpp:41` (`FieldPostProcess::apply` inexistante, mecanisme
correct documente plus bas), et les enumerations doc-only de `system.hpp` (weno5/ssprk3/euler).

## 4. Inventaire TODO

| Mesure | Valeur |
|---|---|
| Marqueurs TODO | 20 |
| FIXME / XXX / HACK | 0 / 0 / 0 |
| Conformes au format Google `TODO(id):` | 0/20 |
| Repartition | `include/` 9, `tests/` 11, `python` + `bench` + `cmake` + `scripts` 0 |

Aucun TODO ne signale un travail incomplet : ce sont 20 etiquettes de provenance renvoyant a une
numerotation de feuille de route (`TODO 2.3`, `TODO 4`, `TODO 2.2.3`, `TODO 4.3`, `TODO 2.1.1`). Elles
DECRIVENT du comportement deja implemente (`include/adc/coupling/amr_system_coupler.hpp:33,44,62`,
`tests/test_two_species_minimal.cpp`) ou une generalisation future
(`include/adc/numerics/spatial_operator.hpp:560,653`). Deux problemes : la numerotation n'est pas
resolvable depuis le `todo.md` a la racine (renvois orphelins pour un lecteur neuf), et tout scan CI
`grep TODO` remonterait ces 20 hits comme des taches en attente (faux positifs). Le seul renvoi vers
un fichier concret (`composite_fac_poisson.hpp:30` -> `amr_reflux.hpp:20`) est EXACT. A noter aussi
des differes REELS exprimes en prose sans marqueur, donc invisibles a l'outillage
(`amr_diagnostics.hpp:29`, `amr_coupler_mp.hpp:289`, plusieurs `= Phase 4b`).

Recommandation : reserver le token `TODO` au travail incomplet, au format `TODO(ADC-xxx):` ;
remplacer les etiquettes de roadmap par `(jalon 2.3)` ou un renvoi au doc de design ; et marquer les
vrais differes en prose par un `TODO(ADC-xxx)` reperable.

## 5. Coherence de langue

Verdict : francais ECRIT SANS ACCENTS, applique a 99.99 %. Ratio marqueurs FR/EN par repertoire :
`include` 6679/32, `python` 2309/9, `tests` 3202/8, `bench` 157/2. Les rares hits "anglais" sont
presque tous des faux positifs (mots-cles de code en commentaire, titres d'articles cites en
bibliographie, mot latin `via`).

Trois entorses, toutes ponctuelles :
- Prose anglaise authentique : un SEUL ilot, le bloc Doxygen de `max_wave_speed`
  (`include/adc/coupling/amr_coupler_mp.hpp:553-559`), entoure de membres francais. A traduire.
- Accents : 2 lignes dans tout le depot (`bench/scaling_step.cpp:329-330`). A delester.
- ASCII : 1 emoji (`include/adc/runtime/amr_dsl_block.hpp:172`). A remplacer par `ATTENTION :`.

Regle proposee (a inscrire dans `CODE_DOCUMENTATION_CONVENTION.md`) : commentaires en francais sans
accents, ASCII pur, tags Doxygen et identifiants en anglais ; `@return` canonique (pas `@returns`).

## 6. Plan de correction priorise

P0 : commentaires faux (section 3). C'est la seule dette qui trompe activement. 10 corrections,
chacune locale a un fichier. Traitement : soit une PR ciblee unique "corriger le comment-rot
(ADC-125)" qui touche les 10 sites, soit replier chaque correction dans la prochaine PR qui modifie
le fichier concerne. 6/10 sont des one-liners ; les contrats de `set_conservative_state`,
`polar_tensor_operator::solve`, `spatial_operator_eb::eb_face_aperture` et l'en-tete d'`amr_reflux`
demandent une reformulation a relire. A faire en premier, independamment du reste.

P1 : API publique `include/adc/**` et `python/` non extractible ou non documentee. PR ciblees par
sous-systeme (conversion mecanique, contenu deja present) :
- Ajouter `@file`/`@brief` la ou il manque : `parallel/comm.hpp`, `parallel/load_balance.hpp`,
  `mesh/patch_box.hpp`, `numerics/time/amr_advance.hpp` + `amr_flux_helpers.hpp`, `python/system.cpp`,
  `python/amr_system.cpp`.
- Migrer `//` -> `///` sur les API publiques documentees en prose : `numerics/time/*` (12 fichiers,
  dont les helpers `mf_*` et `advance_amr`), checkpoint/restart d'`AmrCouplerMP`
  (`coupling/amr_coupler_mp.hpp:319+`), `runtime/amr_runtime.hpp:279,893`.
- Combler les manques de contrat : `struct Aux` (`core/state.hpp:102`), parametres du ctor
  `GeometricMG`, docstrings de `System.add_block`/`AmrSystem.add_block`, docstrings pybind des `.def`
  non triviaux.

P2 : normalisation TODO + langue (sections 4 et 5). Une PR d'hygiene unique : reclasser les 20
etiquettes de roadmap, traduire l'ilot anglais, retirer les 2 lignes accentuees et l'emoji,
normaliser `@returns` -> `@return`. Committer `CODE_DOCUMENTATION_CONVENTION.md` dans la meme PR pour
ancrer la regle.

P3 : cosmetique, au fil de l'eau. Supprimer les doubles en-tetes `//`+`///` (une source de verite),
uniformiser l'ilot `/** */` de `physics/`, aligner le placement des `@file`, et remplacer les renvois
documentaires fragiles (chemins de repertoire perimes, numeros de ligne inter-fichiers) par des
ancrages symboliques. Ces points n'induisent personne en erreur aujourd'hui mais sont le terreau du
prochain comment-rot : a traiter quand un fichier est de toute facon ouvert.

Note transverse : le maillon manquant de tout l'effort reste `CODE_DOCUMENTATION_CONVENTION.md`. Tant
qu'il n'est pas commite, l'audit "fidelite a la convention projet" s'appuie sur des patterns de fait
et non sur une norme ecrite ; le committer via ADC-125 est la condition prealable a un linting
d'en-tetes reproductible.
