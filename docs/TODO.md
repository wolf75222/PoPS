# adc — Tâches : du `PhysicalModel` au système multi-espèces

Liste unique et actionnable de ce qui est **fait** et de ce qui **reste**, d'après la
discussion tuteur et les incréments. Pas un solveur de production : on pose un **squelette
testable**. `[x]` = fait et vert ; `[ ]` = à faire.

---

## 0. Posture (cadre du tuteur)

- Squelette d'abord, pas le solveur final ; tester le niveau d'abstraction sur des cas simples.
- Abstraction et architecture **avant** la structure de données (elle se change plus tard).
- Performance **après** stabilisation (« j'optimiserai quand le code sera propre et figé »).
- Validé **par un utilisateur** (Sacha peut-il décrire son cas sans toucher AMR/MPI/GPU ?), pas par le compilateur.
- `PhysicalModel` est validé : on **ajoute** le niveau au-dessus, on ne le remplace pas.
- Diffusion = un flux de plus, pas une nouvelle couche.
- Deux dépôts : `adc_cpp` = cœur générique, `adc_cases` = modèles / façades / Python / exemples.

---

## 1. Fait (état actuel, tout vert)

### Architecture / dépôts
- [x] Split cœur/applications : `adc_cpp` (moteur, zéro modèle) ↔ `adc_cases` (modèles, façades, Python) via FetchContent (`adc::adc`). Poussés.
- [x] `adc_cpp` HEAD ne contient **aucune** application (vérifié) ; README recentré « bibliothèque cœur ».

### Abstractions par bloc (le niveau manquant — squelette en place)
- [x] `core/equation_block.hpp` : `EquationBlock<Model, Spatial, Time>` = state + modèle + discrétisation spatiale + politique temps + **BC par bloc**.
- [x] `core/coupled_system.hpp` : `CoupledSystem<Blocks...>` (variadic), `for_each_block`, `block<I>()`. **→ design data = `tuple<Blocks...>` (MultiFab par bloc) retenu pour le squelette.**
- [x] `integrator/time_integrator.hpp` : `TimePolicy<Method, Treatment, Substeps>` + `ExplicitTime` / `ImplicitTime` / `IMEXTime` / `PrescribedTime` ; tags `SSPRK2`/`SSPRK3` ; `UserTimeIntegrator`.
- [x] `integrator/scheduler.hpp` : `advance_subcycled(system, dt, advance_block)` lit les sous-pas par bloc.
- [x] `coupling/elliptic_rhs.hpp` : `SingleModelEllipticRhs` (mono) + `TwoFieldChargeDensityRhs` / `TwoBlockChargeDensityRhs` (RHS = q0·n0 + q1·n1, lit plusieurs blocs).
- [x] `coupling/system_coupler.hpp` : `SystemCoupler` mono-niveau — RHS elliptique global, blocs explicites avancés par le cœur, **blocs implicites/IMEX délégués à un callback** (point de branchement Newton/linéaire/collisions), sous-pas par bloc.
- [x] Tests `test_system_abstraction`, `test_system_coupler` (électrons implicite délégué + ions SSPRK3, sous-pas, RHS système) verts.

### Briques sélectionnables
- [x] `SpatialDiscretisation<Limiter, NumericalFlux>` + `Coupler::step<Disc, TimePolicy>` (flux + intégrateur + sous-cyclage) sur grille uniforme.
- [x] `SpatialDiscretisation` câblée en **AMR** (`AmrCoupler/MP::step<Disc>`, MUSCL conservatif avec 2 ghosts).
- [x] Solveur elliptique choisi à l'exécution (façade diocotron MG/FFT, pattern `variant`).

### Corrections
- [x] **AMR applique `model.source`** (le chemin AMR l'ignorait — bug) + test `test_amr_source`.
- [x] Diffusion = flux : trait `DiffusiveModel` (`+ν∆U` dans `assemble_rhs`) + modèle `AdvectionDiffusion` (grille uniforme).
- [x] `SpectralCoupler` appelle `model.elliptic_rhs` au lieu de coder `alpha*(u−n_i0)` du diocotron en dur.

---

## 2. À faire — compléter le squelette multi-espèces

**Ordre conseillé** : (1) `PoissonRhsAssembler` N-blocs → (2) `CoupledSource` → (3) vraie
interface implicite `ImplicitSolver`/`IMEXStepper` → (4) **exemple C++ minimal** (électrons
implicites + ions explicites, rhs = n_i − n_e, sans Python) → (5) `AmrSystemCoupler` →
(6) API Python dans `adc_cases`. On valide à chaque étape par un test d'intégration (§2.5).

### 2.1 Couplage et RHS
- [ ] **`PoissonRhsAssembler` / `ChargeDensityRhs<Blocks...>` à N blocs** : généraliser
  `TwoBlockChargeDensityRhs` (q0·n0+q1·n1) à N espèces `f = Σ_s q_s n_s` (somme sur
  `for_each_block`), avec **charges, composantes densité et signes configurables** par bloc.
- [ ] **`CoupledSource` (sources inter-espèces)** : distinguer `source` locale du modèle et
  une couche de **couplage** qui lit plusieurs blocs : `S_e(U_e, U_i, phi)`,
  `S_i(U_i, U_e, phi)` (collisions, échange q/m). Aujourd'hui `source(u,aux)` ne voit que le
  bloc local.
- [ ] **BC par bloc réellement appliquées** : `EquationBlock::bc` existe ; vérifier/forcer son usage dans `SystemCoupler` (remplissage des halos par bloc), pas une BC globale unique.

### 2.2 Temps : implicite / IMEX réellement exécutés
- [ ] **Vraie interface implicite** : aujourd'hui `SystemCoupler::step` exécute l'**explicite**
  (SSPRK2/3) et **délègue** l'implicite/IMEX à un simple callback. Définir un **contrat clair**
  `ImplicitSolver<Block>` / `IMEXStepper<Block>` (au lieu du callback nu), et fournir **un
  IMEX par défaut** (réutiliser `integrator/imex.hpp` / `two_fluid_ap`) pour un cas plasma
  raide sans que l'utilisateur écrive Newton.
- [ ] **IMEX partiel** : traiter implicitement un **sous-ensemble** des variables d'un bloc (trait `which_implicit()` sur le modèle), pas tout le bloc.
- [ ] **Sous-cyclage temporel par espèce** : déjà exprimé par `substeps` ; vérifier la cohérence du couplage Poisson quand les blocs ont des `dt` différents (re-résoudre φ à quelle cadence ?).

### 2.3 AMR pour le système : `AmrSystemCoupler`
- [ ] `SystemCoupler` est **mono-niveau**. Créer `AmrSystemCoupler` qui porte
  `EquationBlock / CoupledSystem` sur AMR : niveaux 0/1/2, **sous-cyclage AMR**, **reflux**,
  **average_down**, **Poisson par niveau**, chaque bloc avancé par `advance_amr<Disc_bloc>`
  avec son schéma. (L'AMR est déjà un orchestrateur : `fill_ghosts → subcycle → average_down
  → reflux → regrid`.) Note : `AmrCoupler`/`AmrCouplerMP` restent mono-modèle aujourd'hui.

### 2.4 Cas de validation (squelette testable)
- [ ] **Exemple C++ minimal SANS Python** (pour valider l'archi utilisateur) : électrons
  implicites + ions explicites + `rhs Poisson = n_i − n_e`, via `CoupledSystem` +
  `SystemCoupler`. C'est le test « est-ce qu'un utilisateur peut construire son cas ? ».
- [ ] **électrons Euler + ions Euler isothermes + Poisson** (cas canonique deux-espèces) — modèles dans `adc_cases`.
- [ ] **diocotron à ions mobiles** (les ions deviennent un 2e bloc ; réutiliser le diocotron existant pour les électrons).
- [ ] Garde : masse conservée par bloc, RHS = q_i n_i − q_e n_e correct, comparaison à une référence simple.

### 2.5 Tests d'intégration à ajouter
- [ ] `CoupledSystem` + **Poisson RHS non nul** (au-delà du `ZeroSystemRhs` actuel).
- [ ] `SystemCoupler` avec **deux blocs explicites différents** (schémas/sous-pas distincts).
- [ ] `SystemCoupler` avec **bloc implicite** branché sur le callback (déjà amorcé par `test_system_coupler`, à durcir).
- [ ] `AmrSystemCoupler` quand il existera (conservation par niveau, reflux).

---

## 3. À faire — API Python de composition (dans `adc_cases`, APRÈS l'exemple C++)

But : Python **compose**, ne calcule pas cellule par cellule. Les chaînes sélectionnent
des briques C++ compilées. À ne faire qu'**après** l'exemple C++ minimal (§2.4) qui valide
l'architecture utilisateur.

- [ ] **Modèles physiques prêts à composer** (dans `adc_cases`) : `ElectronEuler`, `IonEuler`,
  `Diocotron`, `TwoFluid`… exposés pour servir de blocs.
- [ ] Façade compilée `MultiSpeciesSolver(vector<SpeciesConfig>)` instanciant un `CoupledSystem` + `SystemCoupler` (PIMPL, comme les solveurs existants).
- [ ] Bindings pybind11 : `adc.Mesh2D(...)`, `adc.Simulation(mesh, backend=...)`, `sim.add_equation(name, model, spatial, time, bc)`, `sim.add_poisson(rhs, solver)`, `sim.run(t_end, cfl, output)`.
- [ ] `model="electron_euler"`, `flux="hllc"`, `time="imex"` ↔ tags C++ ; jamais de callback `flux(u)` Python dans le hot path.
- [ ] (avancé) exposer un `PhysicalModel` écrit en C++ par l'utilisateur, puis composable depuis Python.

---

## 4. À faire — follow-ups cœur (nettoyages / cohérence)

- [ ] **Diffusion sur AMR** : la porter comme **flux de face diffusif** (pas un Laplacien direct), sinon le reflux AMR ne la voit pas → non conservatif aux interfaces coarse-fine. (`compute_face_fluxes` doit recevoir `geom` et ajouter `−ν(u_R−u_L)/h`.)
- [ ] **Contrat `Aux`** : trancher. Soit `Aux` est fixe (`phi, grad_x, grad_y`) et on retire `typename M::Aux` du concept ; soit on généralise et `load_aux<Model>` construit `typename Model::Aux`. Aujourd'hui le contrat promet plus que le code ne donne.
- [ ] **`SpectralCoupler::max_drift_speed`** : encore `/model_.B0` (diagnostic). Le généraliser via `model.max_wave_speed` (attention : change le `dt`, donc la trajectoire → re-valider, pas bit-identique).
- [ ] **`Coupler` fourre-tout** : extraire l'orchestrateur (le `SystemCoupler` va dans ce sens) ; dédupliquer les diagnostics `AmrCouplerMP::mass()`/`max_drift_speed()` qui réimplémentent `amr_diagnostics.hpp`. Nom : `Coupler` est historique = *Assembler/Simulation Driver* (à acter, pas urgent).

---

## 5. Décision à acter au tableau (avec Sacha)

- **Structure de données du système** : le squelette a retenu `tuple<Blocks...>` (un
  `MultiFab` par bloc, `EquationBlock::state` = pointeur). Alternative : `StateVec<N_total>`
  empilé (un bloc mémoire contigu, offsets par espèce — meilleure localité, plus complexe).
  À confirmer/infléchir : c'est la décision qui pèse sur la perf (point 0 : structure de
  données = plus tard, mais la décision d'interface se prend maintenant).

---

## 6. Lexique (pour les slides — définir avant de jeter les sigles)

| Terme | Sens court |
|---|---|
| `BoxArray` | découpage du domaine en blocs (boîtes) |
| `MultiFab` | les champs `U` stockés sur ces blocs (collection distribuée) |
| `BCRec` | conditions aux limites d'un champ |
| `aux` | variable auxiliaire transportée (`phi, grad phi`) |
| seam | couture où vit le parallélisme (`for_each_cell`, `comm`) |
| `EquationBlock` | state + modèle + méthode spatiale + politique temps + BC (un bloc) |
| `CoupledSystem` | plusieurs `EquationBlock` |
| `SystemCoupler` | orchestrateur : RHS elliptique global + avance chaque bloc |

---

## 7. Synthèse (phrase tableau)

> `adc` sait prendre une **loi physique locale** (`PhysicalModel`) et la faire tourner sur
> un maillage avec Poisson, AMR, MPI et GPU. Le niveau d'**assemblage multi-blocs** est
> désormais en place en squelette : `EquationBlock` (state + modèle + schéma spatial +
> politique temps + BC), `CoupledSystem` (plusieurs blocs), `SystemCoupler` (RHS elliptique
> global, explicite avancé par le cœur, implicite délégué, sous-pas par bloc). Reste à le
> remplir : RHS à N espèces, source de couplage, implicite/IMEX par défaut, AMR du système,
> et l'API Python de composition. Le `PhysicalModel` décrit une équation, le `CoupledSystem`
> un système, le `Scheduler` l'ordre d'exécution ; le cœur garantit AMR / MPI / GPU.
