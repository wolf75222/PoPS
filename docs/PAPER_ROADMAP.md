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
   le taux a base egale".

L'ecart restant N'EST PAS un bug : c'est l'ecart attendu d'un schema FV d'ordre modere sur le
bord d'anneau, l-dependant et structurel. Le verrou identifie est le **bord d'anneau
cartesien**.

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
resolution ou en ordre (WENO5-Z + SSPRK3) referme partiellement l'ecart mais ne change pas la
nature du verrou.

## Classification des manques (4 paniers)

### Panier 1 : deja possible avec l'API ACTUELLE (a lancer / regler)

Capacites cablees et exposees, suffisantes pour pousser plus loin sans nouveau code.

- **Montee en ORDRE du transport** : `adc.Spatial` expose deja WENO5-Z (`Weno5`) et SSPRK3
  (`docs/ALGORITHMS.md` sections 3-4) ; le cas diocotron actuel tourne en Minmod/SSPRK2. Lancer
  le balayage ordre + resolution est un reglage, pas un developpement. C'est la voie M3 ouverte
  de `todo.md` section 6 ("montee en resolution / convergence vers le taux analytique").
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

### Panier 2 : facade DSL de production `m.compile(backend=...)`

Le DSL symbolique existe et compile (JIT IModel + AOT natif) ; ce qui manque est la
consolidation en facade de production, pas la machinerie.

- **Facade `compile` unifiee** : `python/adc/dsl.py` expose `compile_so` (JIT, backend "jit"),
  `compile_aot` (AOT, backend "compile") et `compile_or_jit(mode=...)`. C'est la facade visee
  `m.compile(backend=...)`, mais elle reste prototype/experimentale : le cas DSL `dsl_euler` est
  marque `category = "experimental", ci = false` dans `cases_manifest.toml`. Aucun cas diocotron
  ne passe par le DSL aujourd'hui (les compositions vont par `models.diocotron(...)`, briques
  natives).
- **Limite device connue** : `System::add_compiled_model` a LAMBDAS ETENDUES segfaute sur Cuda
  (`docs/GPU_RUNTIME_PORT.md` phase 8 et round 2). Le contournement device-clean (foncteurs
  nommes `block_builder.hpp`) est valide sur GH200 (phase 9, limites device (a) et (b) levees,
  `todo.md` section 4), mais le chemin `add_compiled_model` Python n'est PAS encore re-route
  dessus de bout en bout. Reproduire Hoffart NE depend pas du DSL (les briques natives
  suffisent) ; ce panier n'est requis que si l'on veut piloter le modele magnetise complet en
  formules depuis Python plutot qu'en composant des briques.

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

1. **Panier 1 d'abord** : balayage ordre (WENO5-Z + SSPRK3) x resolution sur le cas diocotron
   existant, et AMR plus agressif sur le bord d'anneau. Chiffre la part de l'ecart imputable a
   la diffusion (resolution/ordre) vs au verrou structurel (bord cartesien). Aucun code coeur.
2. **Panier 3 ensuite** : c'est la seule voie qui leve le verrou. Porter un bord embedded /
   paroi cote transport (ou un domaine disque) pour que l'anneau ne soit plus diffuse par la
   grille cartesienne. C'est le chantier le plus lourd et le plus payant pour le taux numerique.
3. **Panier 4 selon l'ambition** : parite `AmrSystem` <-> `System` (recon primitive + Roe +
   multi-bloc) pour pousser l'AMR a haute resolution ; puis modele magnetise complet
   (`two_fluid_ap` couple au transport) si l'on sort de la limite de derive.
4. **Panier 2 transverse, optionnel** : consolider `m.compile(backend=...)` + re-router
   `add_compiled_model` sur les foncteurs nommes device-clean. N'est PAS sur le chemin critique
   de la reproduction (les briques natives suffisent), mais utile pour piloter le modele
   magnetise en formules.

## Resume du verrou

Reproduire la CIBLE analytique de Hoffart est fait (numpy, 3 chiffres). Reproduire le taux
NUMERIQUE a parite demande de lever le bord d'anneau cartesien (panier 3) : aujourd'hui le
cut-cell ne sert que Poisson, le transport reste cartesien, d'ou un sous-taux l-dependant
structurel que la resolution et l'ordre (panier 1) attenuent sans supprimer.
