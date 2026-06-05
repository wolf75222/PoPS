# Feuille de route reproduction Hoffart (arXiv:2510.11808)

AUDIT (documentaire, aucune implementation). Recense ce qui MANQUE pour reproduire le
benchmark diocotron de Hoffart, Maier, Shadid, Tomas (arXiv:2510.11808, Section 5.3 : taux de
croissance de l'instabilite diocotron d'une colonne creuse, dans la limite de derive
`omega_d << omega_p << omega_c`), et classe chaque manque dans l'un de 4 paniers.

Sources lues pour cet audit :
- `docs/ALGORITHMS.md` (briques numeriques : sections 1-3 FV, 9-12 elliptique + cut-cell,
  13-16 AMR, 19 DSL JIT/AOT) ;
- `docs/ARCHITECTURE.md` (sections 2-4 couches, 7 elliptique, 8 AMR distribue, 10 frontiere
  lib/application) ;
- `docs/GPU_RUNTIME_PORT.md` (phases 8-11 : limite nvcc des lambdas etendues, strong-scaling
  AMR negatif) ;
- `docs/BIBLIOGRAPHY.md` section 3 (entree Hoffart) et `docs/archive/two_fluid_ap.md`
  (note de methode du schema AP deux-fluides) ;
- `todo.md` section 6 (M1/M2/M2b Hoffart) + sections 1-2 (aux/EPM) ;
- cas `adc_cases/diocotron/{run.py,README.md,band_instability.py}`,
  `adc_cases/diocotron_amr/run.py`, `adc_cases/two_fluid_ap/`, `adc_cases/cases_manifest.toml` ;
- bindings : `python/system.cpp`, `python/amr_system.cpp`, `python/bindings.cpp`,
  `python/adc/__init__.py`, `python/adc/dsl.py`, `include/adc/numerics/elliptic/geometric_mg.hpp`.

## Etat de reproduction (factuel)

Deux niveaux dans Hoffart (Section 5.3) :

1. **Cible analytique** (probleme aux valeurs propres radial de Petri/Davidson-Felice).
   REPRODUITE a 3 chiffres en numpy cote `adc_cases/diocotron/run.py` :
   `gamma_3 = 0.772`, `gamma_4 = 0.912`, `gamma_5 = 0.687` (cf. README du cas). Hors perimetre
   du coeur : c'est de la verification analytique pure numpy.
2. **Taux numerique mesure par `adc`** : pipeline complet (composition `ExB` + `BackgroundDensity`,
   Poisson de systeme a paroi conductrice circulaire `wall="circle"`, mesure du mode `l` de
   `phi` par FFT azimutale, ajustement `exp(gamma t)`). Tourne, capture l'instabilite (croissance
   exponentielle, classement des modes correct, `l=4` dominant), mais SOUS-ESTIME le taux :
   `l=3 -22 %`, `l=4 -27 %`, `l=5 -5 %` (n=192, Minmod ordre 2). `todo.md` section 6 : M1
   "limite par la diffusion numerique du bord d'anneau", M2/M2b "AMR sur le bord d'anneau triple
   le taux a base egale". Le balayage ordre x resolution etend desormais cet axe jusqu'a O5
   (WENO5-Z + SSPRK3) et jusqu'a n=512 ; voir la lecture par mode dans la section "verrou" ci-dessous.

A ce stade, l'ecart ressemble davantage a une limite numerique/structurelle du schema FV
cartesien qu'a un bug isole, mais le niveau exact du plancher reste a confirmer. Le candidat
identifie est le **bord d'anneau cartesien**.

## Le verrou structurel : bord d'anneau cartesien

La capacite cut-cell Shortley-Weller (`docs/ALGORITHMS.md` section 12) vit UNIQUEMENT dans
`include/adc/numerics/elliptic/geometric_mg.hpp` : elle place la paroi conductrice circulaire
Dirichlet a sa position REELLE pour le solveur de POISSON. Mais le transport hyperbolique
(`numerics/spatial_operator.hpp`, `numerics/numerical_flux.hpp`, `numerics/reconstruction.hpp`)
n'a AUCUNE notion de bord embedded : l'anneau de charge est advecte sur la grille cartesienne
pleine. Le predicat de paroi (`runtime/wall_predicate.hpp`, `python/system.cpp::wall_active`)
n'alimente que l'operateur elliptique, jamais le flux. Le gradient radial net de l'anneau est
donc diffuse par le schema FV cartesien, ce qui amortit le taux de croissance de facon
l-dependante (les modes a plus courte longueur d'onde, l=4, paient le plus). Monter en
resolution referme partiellement l'ecart mais ne change pas la nature du verrou.

MESURE (`diocotron/SWEEP_RESULTS.md` cote adc_cases). Le balayage ordre x resolution chiffre la
part diffusion vs structurel par mode. Il couvre maintenant l'axe haut ordre O5 = WENO5-Z + SSPRK3
(atteignable depuis Python depuis adc_cpp #88, cf. Panier 1) et la haute resolution n=384/512 (job
ROMEO x64cpu). L'axe ordre reel est donc `{O1 none, O2 minmod, O2 vanleer, O5 weno5}`. But de l'axe
O5 : eclairer la question laissee ouverte a O2 - le residu l-dependant est-il de la diffusion
(refermable par l'ordre) ou un plancher structurel du bord d'anneau cartesien ? Lecture par mode
(les %err detailles, les fenetres de fit et les reserves sont dans `SWEEP_RESULTS.md`, source de
verite) :

- **l = 3 (le signal le plus PROPRE, fenetre de fit homogene a tout n)** : l'`|%err|` se referme
  d'abord nettement avec l'ordre et la resolution, puis a O5 il APLATIT autour de -9 % a haute
  resolution (-10.3 % a n=256, -8.6 % a n=384, -8.8 % a n=512 : un cran plat dans le bruit de
  mesure). Ce n'est pas le comportement d'une diffusion qui s'epuise, c'est le candidat le plus
  CREDIBLE a un residu structurel.
- **l = 4 (le mode-cle)** : a basse resolution O5 tombe a ~ -4 % (n=128, n=256), ce que la lecture
  O2 prenait pour "diffusion presque epuisee" ; mais a n=384/512 il REMONTE vers ~ -9/-10 %. Reserve
  majeure : ces deux points haute resolution ont une fenetre de fit qui s'ouvre tot (t0 = 6.3 et
  5.4, comme le point n=192 deja ecarte), donc ils sous-lisent probablement la pente. On NE peut
  donc PAS conclure fortement sur l=4 : on peut seulement dire que le -4 % ne se reproduit a aucune
  des deux resolutions superieures.
- **l = 5** : deja proche de la cible des O2 a n=192 ; petit residu de signe variable (quelques %),
  ni l'ordre ni la haute resolution n'y font apparaitre de plancher.

CONCLUSION PRUDENTE (a confirmer, ne constitue PAS une preuve definitive). L'axe O5 + haute
resolution AFFAIBLIT l'hypothese "tout l'ecart etait de la diffusion d'ordre 2" : a l'ordre 5, sur
le mode le mieux mesure (l=3), le residu ne continue pas de se refermer mais semble plafonner. Les
DONNEES SUGGERENT un plancher residuel l-dependant de l'ordre de ~9-10 % a l'ordre 5 (contre ~12 %
vus a O2), probablement lie au bord d'anneau cartesien / paroi de transport, RESTE A CONFIRMER. Deux
limites empechent d'en faire un chiffre ferme : (1) le plateau l=3 ne tient pour l'instant que sur
un seul cran plat n=384 -> n=512 (un n=768/1024 ou deux horizons `t_end` excluraient une convergence
tres lente) ; (2) les points l=4 haute resolution sont biaises par leur fenetre de fit precoce
(diagnostic de fenetre robuste a prevoir avant de chiffrer un plancher l=4). Ce candidat structurel
reste l'argument quantitatif pour la PR-A "transport-wall", desormais avec une taille plausible
revisee a ~9-10 % a l'ordre 5.

## Classification des manques (4 paniers)

### Panier 1 : deja possible avec l'API ACTUELLE (a lancer / regler)

Capacites cablees et exposees, suffisantes pour pousser plus loin sans nouveau code.

- **Montee en RESOLUTION et en ORDRE** : reglage pur (le cas diocotron tourne deja a n variable).
  C'est la voie M3 de `todo.md` section 6, et le balayage resolution x ordre est fait (cf.
  `SWEEP_RESULTS.md`). La montee en ORDRE WENO5-Z + SSPRK3 est desormais atteignable depuis Python
  (adc_cpp #88) : `adc.Spatial(limiter="weno5")` (raccourci `weno5=True`) selectionne la
  reconstruction WENO5-Z dans `make_block`, et `adc.Explicit(method="ssprk3")` (raccourci
  `ssprk3=True`) l'integrateur SSPRK3, par le chemin natif `add_block`. Le defaut reste inchange
  (Minmod / SSPRK2, bit-identique au pre-#88). Seule limite : le chemin natif `add_block` expose
  WENO5 ; les chemins `.so` AOT/JIT (`add_compiled_block`) allouent 2 ghosts et rejettent `"weno5"`
  (cf. Panier 2 / locks infra). Le balayage couvre donc `{O1, O2-minmod, O2-vanleer, O5 weno5}`,
  jusqu'a n=512.
- **Paroi conductrice circulaire sur Poisson** : `wall="circle"` + `wall_radius` est cable sur
  `System` (`python/bindings.cpp:97`) ET sur `AmrSystem` (`python/bindings.cpp:193`,
  `python/amr_system.cpp:78`). Le cut-cell elliptique est valide (MMS ordre 2, multi-box, MPI ;
  `docs/ALGORITHMS.md` section 12). Rien a ecrire pour la geometrie d'anneau de Petri.
- **AMR sur le bord d'anneau** : `adc.AmrSystem` + `set_refinement(threshold)` tourne et
  conserve la masse (cas `adc_cases/diocotron_amr/run.py`). M2/M2b de `todo.md` notent que
  l'AMR triple le taux a base egale. Pousser le raffinement / le nombre de niveaux est un
  reglage de config.
- **Diagnostic de taux** : la chaine mesure (FFT azimutale du mode `l` de `phi`, ajustement de
  la phase lineaire) est entierement en place cote `adc_cases`.

### Etat des chemins d'execution GPU / MPI (production)

Statut factuel des chemins production (briques natives, pas DSL), independant de la PR-A :

- **`System` production CPU** : valide (ctest serie ; pipeline diocotron tourne).
- **`AmrSystem` production CPU** : valide.
- **`System` GPU production np=1** : valide sur GH200 (adc_cpp #97). #97 corrige le segfault device
  des kernels elliptique/maillage (lambdas etendues premiere-instanciees depuis une TU externe ->
  foncteurs nommes, codegen device robuste sous nvcc) ; parite Cuda vs Serial `dmax_abs` ~ 1e-13
  sur `solve_fields`, `compute-sanitizer` propre.
- **`System::solve_fields` MPI CPU np=1/2/4** : valide (adc_cpp #99). #99 corrige le segfault hote
  du post-traitement par cellule (`fab(0)` sans garde `local_size()` sur les rangs sans box) ;
  resultat bit-invariant au nombre de rangs (`test_mpi_system_solve_fields_np{1,2,4}`, joue en CI MPI).
- **device-MPI production (GPU multi-rang)** : RESTE A VALIDER separement (adc_cpp #100, suivi).

Ces chemins ne sont PAS sur le chemin critique de la cible analytique ni du sweep diocotron (CPU),
mais ils conditionnent la montee en resolution multi-GPU evoquee au Panier 4.

### Panier 2 : facade DSL de production `m.compile(backend=...)`

Le DSL symbolique existe et compile (JIT IModel + AOT natif) ; ce qui manque est la
consolidation en facade de production, pas la machinerie.

- **Facade `compile` unifiee** : `python/adc/dsl.py` expose `compile_so` (JIT, backend "jit"),
  `compile_aot` (AOT, backend "compile") et `compile_or_jit(mode=...)`. C'est la facade visee
  `m.compile(backend=...)`, mais elle reste prototype/experimentale : le cas DSL `dsl_euler` est
  marque `category = "experimental", ci = false` dans `cases_manifest.toml`. Aucun cas diocotron
  ne passe par le DSL aujourd'hui (les compositions vont par `models.diocotron(...)`, briques
  natives).
- **Limite device connue** : la recette device-clean (lambda etendue -> foncteur nomme, codegen
  device robuste sous nvcc) couvre maintenant le transport (`block_builder.hpp`, adc_cpp #64) ET
  les kernels elliptique/maillage de `solve_fields` (#97), d'ou la validation GPU np=1 ci-dessus.
  Le chemin `add_compiled_model` / WENO5 sur `.so` reste a part : `add_compiled_block` alloue 2
  ghosts (rejette `"weno5"`) et le pilotage device de bout en bout du modele compile n'est pas
  consolide. Reproduire Hoffart NE depend pas du DSL (les briques natives suffisent) ; ce panier
  n'est requis que si l'on veut piloter le modele magnetise complet en formules depuis Python
  plutot qu'en composant des briques.

### Panier 3 : domaine-disque FV / capacite de paroi (vrai domaine circulaire, pas bord cartesien)

C'est le panier qui leve le VERROU structurel. Aucune de ces capacites n'existe aujourd'hui.

- **Bord embedded cote TRANSPORT** : porter la notion de cut-cell / paroi reflechissante du
  solveur elliptique (`geometric_mg.hpp`) vers l'operateur hyperbolique (`spatial_operator.hpp`)
  pour que l'anneau de charge ne soit plus advecte sur une grille cartesienne pleine. C'est le
  manque qui explique le sous-taux l-dependant. `docs/ARCHITECTURE.md` section 12 (comparaison
  AMReX) note d'ailleurs un Laplacien EB "en escalier" cote operateur, le cut-cell n'etant que
  pour le bord COURBE elliptique.
- **Maillage circulaire / coordonnees adaptees** : alternative au cartesien embedded, un domaine
  reellement disque (coordonnees polaires ou maillage cut-cell complet) ; non present (le coeur
  est cartesien adaptatif, `docs/ARCHITECTURE.md` section 1).

Tant que ce panier n'est pas traite, la reproduction QUANTITATIVE fine du taux numerique reste
bornee par la diffusion du bord cartesien (constat M1, `todo.md` section 6).

### Panier 4 : AMR multi-bloc avance ou EPM avance

Capacites partiellement presentes mais incompletes pour un usage Hoffart pousse.

- **AMR multi-bloc / multi-niveau a parite `System`** : `AmrSystem` est MONO-bloc, explicite,
  SANS reconstruction primitive ni flux de Roe (`docs/ARCHITECTURE.md` section 8 : "AmrSystem
  n'est PAS a parite avec System"). Un diocotron AMR a haute resolution + ordre eleve sur
  plusieurs niveaux demande de faire porter le meme `EquationBlock` par le moteur AMR.
- **Strong-scaling AMR full-device** : le grossier reparti (`replicated_coarse=false`) est cable
  mais NEGATIF a l'echelle testee (`docs/GPU_RUNTIME_PORT.md` phase 11). Requis seulement pour de
  tres grandes resolutions multi-GPU, pas pour la cible Section 5.3.
- **EPM avance** : l'operateur elliptique etendu (eps(x), Helmholtz/ecrante, anisotrope) est fait
  et valide device (`docs/ALGORITHMS.md` section 11, `todo.md` section 2), mais le branchement
  `EllipticProblem` -> stencil par la fabrique additive reste DESCRIPTIF, et le decoupage Schur
  EPM est differe (`docs/ARCHITECTURE.md` section 7, `todo.md` section 7). Non requis pour le
  diocotron de derive (Poisson pur + paroi), pertinent si l'on vise le modele deux-fluides
  magnetise complet.
- **Modele magnetise complet** : le schema AP deux-fluides (`adc_cases/two_fluid_ap/`,
  `docs/archive/two_fluid_ap.md`) porte la rotation cyclotron exacte (section 6 de la note) mais
  PAS encore le couplage `E x B` + diamagnetique inhomogene au transport. C'est l'extension
  Hoffart au-dela de la limite de derive.

## Plan ordonne

### FAIT (a date)

- **WENO5-Z / SSPRK3 atteignables depuis Python** (adc_cpp #88) : `adc.Spatial(limiter="weno5")` +
  `adc.Explicit(method="ssprk3")` via le chemin natif `add_block`, defaut inchange.
- **Balayage ordre x resolution etendu a O5 et a n=384/512** (cf. `SWEEP_RESULTS.md`) : ordre
  `{O1, O2 minmod, O2 vanleer, O5 weno5}`, jusqu'a n=512 (haute resolution sur ROMEO x64cpu).
  Lecture : l=3 plafonne ~ -9 % a O5 haute resolution (candidat structurel le plus propre) ; l=4 ne
  reproduit pas son -4 % basse resolution mais ses points haute resolution sont biaises par leur
  fenetre de fit ; l=5 deja a la cible (cf. conclusion prudente section verrou).
- **GPU `System` production np=1** valide sur GH200 (adc_cpp #97).
- **`solve_fields` MPI CPU np=1/2/4** valide (adc_cpp #99).

### Prochain VERROU scientifique (le seul qui leve le sous-taux structurel)

- **Panier 3 - bord de transport / domaine-disque / bord embedded** : porter un bord embedded /
  paroi cote transport (ou un domaine reellement disque) pour que l'anneau de charge ne soit plus
  advecte sur une grille cartesienne pleine. C'est la seule voie qui adresse le candidat plancher
  structurel ~9-10 % mis en evidence par le sweep O5. Chantier le plus lourd et le plus payant pour
  le taux numerique. Le sweep n'en est PAS une preuve : il SUGGERE le candidat et reste a confirmer
  (n=768/1024 ou deux horizons `t_end` pour l=3 ; diagnostic de fenetre robuste pour l=4).

### Prochains verrous d'infrastructure (peuvent atterrir en parallele)

- **Validation device-MPI production** (GPU multi-rang) : adc_cpp #100 (suivi). Prerequis a une
  montee en resolution multi-GPU.
- **WENO5 sur `CompiledModel` / `.so`** : les chemins AOT/JIT allouent 2 ghosts et rejettent
  `"weno5"` ; etendre le stencil 5 points au chemin compile pour aligner DSL et `add_block`.
- **Ergonomie `compile()` / cache** : consolider `m.compile(backend=...)` (Panier 2) + re-router
  `add_compiled_model` sur les foncteurs nommes device-clean. PAS sur le chemin critique de la
  reproduction (briques natives suffisent), mais utile pour piloter le modele magnetise en formules.
- **Panier 4 selon l'ambition** : parite `AmrSystem` <-> `System` (recon primitive + Roe +
  multi-bloc) pour pousser l'AMR a haute resolution ; puis modele magnetise complet
  (`two_fluid_ap` couple au transport) si l'on sort de la limite de derive.

## Resume du verrou

Reproduire la CIBLE analytique de Hoffart est fait (numpy, 3 chiffres). Reproduire le taux
NUMERIQUE a parite demande de lever le bord d'anneau cartesien (panier 3) : aujourd'hui le
cut-cell ne sert que Poisson, le transport reste cartesien, d'ou un sous-taux l-dependant que la
resolution attenue sans supprimer. Le balayage etendu a O5 (WENO5-Z + SSPRK3, atteignable depuis
Python depuis adc_cpp #88) et a la haute resolution n=384/512 AFFAIBLIT l'hypothese "tout l'ecart
etait de la diffusion d'ordre 2" : sur le mode le mieux mesure (l=3), le residu O5 ne se referme
plus mais plafonne autour de -9 %. Les donnees SUGGERENT donc un plancher structurel candidat de
l'ordre de ~9-10 % a l'ordre 5, RESTE A CONFIRMER (un seul cran plat sur l=3 ; fenetre de fit
precoce biaisant l=4) - PAS une preuve definitive, mais l'argument quantitatif pour la PR-A
"transport-wall". Le pilotage WENO5-Z/SSPRK3 depuis Python n'est donc plus un verrou (fait, #88) ;
le verrou restant est bien le bord de transport (panier 3).
