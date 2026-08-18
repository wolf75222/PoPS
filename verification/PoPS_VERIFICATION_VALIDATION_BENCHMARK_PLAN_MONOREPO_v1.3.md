# Plan de vérification, validation et benchmark de PoPS

**Statut :** proposition normative prête à implémenter  
**Version :** 1.3  
**Date :** 17 août 2026  
**Périmètre matériel :** au maximum deux nœuds par campagne  
**Dépôt concerné :** `wolf75222/PoPS` (monorepo unique)  
**Révision du dépôt examinée :** `0a18620d2fc7bed8f4ec60792f48982230f4c10d` (`master`, 16 août 2026)  
**Révision 1.3 :** conversion complète en architecture monorepo, alignement sur le pipeline public exact-rank de PoPS et sur les autorités actuelles du code : `pops.analytic`, `pops.diagnostics`, `pops.amr`, `tests/test_manifest.toml`, `benchmarks/manifest.toml` et `examples/final/`. Tous les cas, oracles, scripts, schémas et rapports sont rattachés au même commit PoPS.

---

## Sommaire

1. Objet, architecture du monorepo et niveau de preuve
2. Contrat des cas, manifests, sorties et provenance
3. Métriques, normes et critères d’acceptation
4. Configurations numériques communes
5. Transport, Euler, Poisson et couplages plasma
6. Intégration temporelle, splitting et régimes raides
7. Campagnes AMR
8. Chocs, géométrie et robustesse
9. MPI, OpenMP, GPU, restart et reproductibilité
10. Matrice de couverture du solveur
11. Benchmarks de performance
12. Matrices CPU/GPU et campagnes limitées à deux nœuds
13. CI, rapports, diagnostic des pannes et roadmap
14. Sources, Definition of Done et annexes de traçabilité

---

## 1. Objet du document

Ce document définit la suite intégrée nécessaire pour vérifier PoPS comme solveur générique de systèmes hyperboliques, elliptiques et couplés sur maillages uniformes et AMR. Il précise :

- les cas scientifiques à implémenter ;
- l’oracle utilisé par chaque cas : solution analytique, solution manufacturée, relation de dispersion, invariant, symétrie, solution exacte de Riemann ou référence publiée ;
- les composants du solveur réellement exercés ;
- les variantes uniformes, AMR, MPI et espaces d’exécution Kokkos ;
- les campagnes de convergence spatiale et temporelle ;
- les diagnostics obligatoires ;
- les critères de réussite ;
- les campagnes de performance et de scaling limitées à deux nœuds ;
- l’organisation des dossiers, des manifests, des sorties et des rapports dans un dépôt unique.

La suite décrite ici ne remplace pas les tests unitaires et d’intégration de `tests/`. Elle vise des calculs complets, exécutés par le vrai pipeline public PoPS, capables de révéler des défauts qui n’apparaissent qu’après composition de plusieurs briques : flux, reconstruction, ghost cells, frontières, AMR, solveur de Poisson, couplage, intégrateur temporel, MPI, Kokkos, restart et sorties.

### 1.1 Alignement sur le monorepo actuel

La révision de référence du dépôt possède les autorités suivantes :

| Autorité actuelle | Emplacement | Conséquence pour la suite |
|---|---|---|
| Cœur C++20 exact-rank | `include/pops/` et `src/` | Aucun cas scientifique nommé ne doit être codé dans le cœur générique. |
| API publique et codegen | `python/pops/` | Les cas passent par `Case -> validate -> resolve(layout=...) -> compile -> bind -> run`. |
| Expressions analytiques | `python/pops/analytic/` | Les profils, prédicats et sources manufacturées utilisent des arbres immuables et callback-free lorsqu’ils sont exprimables par cette API. |
| Diagnostics natifs typés | `python/pops/diagnostics/` | `Norm`, `Integral`, `MinMax`, `Balance`, `ConservationCheck` et `StepChangeNorm` décrivent des réductions exécutées par C++/Kokkos/MPI. |
| Autorité AMR publique | `python/pops/amr/authoring.py` | Hiérarchie, regrid, relations d’horloge, exécution subcyclée/synchrone et distribution sont déclarés par des objets publics uniques. |
| Acceptation exécutable finale | `examples/final/` | Ces scripts prouvent le contrat public final ; ils ne remplacent pas les études scientifiques de convergence. |
| Tests rapides | `tests/cpp/`, `tests/python/`, `tests/gates/`, `tests/test_manifest.toml` | Les régressions minimales et les refus déterministes restent dans `tests/`. |
| Performance | `benchmarks/`, `benchmarks/manifest.toml`, `benchmarks/romeo/` | Les chronométrages restent séparés des preuves scientifiques. |
| Schémas machine-readable | `schemas/` | Les schémas de manifest, métriques et provenance de vérification sont ajoutés ici. |
| Orchestration du dépôt | `scripts/` et `.github/workflows/` | Le runner scientifique et sa sélection CI suivent les conventions existantes du monorepo. |

Un module natif PoPS compile exactement une dimension spatiale, choisie par `POPS_NATIVE_DIM=1`, `2` ou `3`. Il n’existe ni commutation de dimension au runtime, ni représentation padded de rang fixe. Une campagne multi-dimensionnelle agrège donc plusieurs builds authentifiés distincts.

Le parallélisme on-node passe exclusivement par Kokkos. Les configurations « Serial », « OpenMP » et « GPU » désignent respectivement des espaces d’exécution Kokkos Serial, Kokkos OpenMP et Kokkos Cuda/HIP/SYCL construits dans l’installation de Kokkos. MPI est une couche distribuée optionnelle ; il ne constitue pas un remplacement de Kokkos.

### 1.2 Logique reprise des grandes suites AMR

Le plan reprend la logique employée dans des codes comme H-AMR, AMReX/Castro, FLASH et Athena :

- ondes linéaires lisses pour mesurer l’ordre, la phase et la dissipation ;
- problèmes de Riemann pour les chocs et les contacts ;
- explosions décentrées pour les symétries, les coordonnées et le raffinement dynamique ;
- solutions manufacturées pour exercer tous les termes ;
- comparaison uniforme fine/AMR pour isoler les interfaces coarse-fine ;
- cas plasma à dispersion analytique pour le couplage hyperbolique–elliptique.

PoPS ajoute des gates spécifiques à son architecture actuelle : champ recalculé aux stages du `Program`, Poisson composite AMR, handles owner-qualified, Strang/IMEX/Schur, multi-espèces, parité modèle fourni/codegen, artefacts natifs exact-rank, restart transactionnel et campagnes Kokkos/MPI/GPU.

## 2. Décision d’architecture du monorepo

### 2.1 Répartition normative

La vérification scientifique devient une capacité du dépôt unique `PoPS`. Elle ne doit toutefois être fusionnée ni avec les tests rapides, ni avec les exemples d’API, ni avec le harness de performance. La séparation est fonctionnelle à l’intérieur du monorepo :

```text
PoPS/
├── include/pops/                         # cœur C++ public et générique
├── src/                                  # implémentations natives centrales
├── python/pops/                          # paquet public installé
│   ├── analytic/                        # expressions analytiques immuables, callback-free
│   ├── diagnostics/                     # réductions natives typées
│   └── amr/                             # autorités publiques AMR
├── examples/
│   └── final/                            # cibles d’acceptation du contrat public
│
├── verification/                         # nouvelle suite scientifique, non installée dans la wheel
│   ├── README.md
│   ├── __init__.py
│   ├── manifest.toml                     # source de vérité des campagnes scientifiques
│   ├── pops_verify/                      # package privé au dépôt, jamais dans le hot path
│   │   ├── __init__.py
│   │   ├── campaign.py
│   │   ├── case_contract.py
│   │   ├── reference_errors.py
│   │   ├── cell_averages.py
│   │   ├── convergence.py
│   │   ├── conservation.py
│   │   ├── phase.py
│   │   ├── symmetry.py
│   │   ├── interface_error.py
│   │   ├── provenance.py
│   │   ├── capabilities.py
│   │   └── report.py
│   ├── cases/
│   │   ├── transport/
│   │   ├── euler/
│   │   ├── poisson/
│   │   ├── euler_poisson/
│   │   ├── multifluid/
│   │   ├── time_integration/
│   │   ├── amr/
│   │   ├── robustness/
│   │   ├── geometry/
│   │   └── infrastructure/
│   ├── reference/                        # petites références traçables uniquement
│   └── machines/                         # profils de lancement scientifique
│       ├── local.toml
│       ├── romeo2025_x64cpu.toml
│       └── romeo2025_gh200.toml
│
├── schemas/
│   ├── verification_manifest.v1.json
│   ├── verification_metrics.v1.json
│   ├── verification_provenance.v1.json
│   └── verification_report.v1.json
│
├── scripts/
│   ├── run_verification.py               # point d’entrée stable
│   ├── check_verification_manifest.py
│   └── render_verification_report.py
│
├── tests/
│   ├── cpp/
│   ├── python/
│   │   └── integration/verification/     # smokes PR très courts seulement
│   ├── gates/
│   └── test_manifest.toml
│
├── benchmarks/
│   ├── README.md
│   ├── manifest.toml
│   ├── src/
│   ├── include/
│   └── romeo/
│
├── docs/
│   └── verification/
│       ├── VERIFICATION_VALIDATION_BENCHMARK_PLAN.md
│       ├── ACCEPTANCE_CRITERIA.md
│       ├── METRICS_AND_NORMS.md
│       └── TWO_NODE_SCALING.md
│
└── build/
    └── verification/                     # résultats générés sous la racine déjà git-ignorée
```

Le répertoire `verification/` est importable depuis la racine du checkout pour l’orchestration, mais n’entre pas dans `wheel.packages`, qui reste limité à `python/pops`. Aucune API de vérification n’est donc ajoutée au paquet utilisateur par accident.

### 2.2 Rôle de chaque emplacement

| Emplacement | Contenu autorisé | Contenu interdit |
|---|---|---|
| `include/pops/`, `src/` | Algorithmes génériques, runtime, contrats natifs | Cas nommés, paramètres de benchmark scientifique, références publiées |
| `python/pops/analytic/` | Expressions analytiques sérialisables pour données initiales et MMS | Lecture de sorties ou orchestration de campagne |
| `python/pops/diagnostics/` | Réductions natives attachées au `ConsumerGraph`, avec contrats MPI/AMR | Comparaison à un oracle externe ou calcul de pentes multi-runs |
| `python/pops/amr/` | Autorité publique de hiérarchie, tagging, regrid, transfert, clocks et patch layout | Politique AMR parallèle inventée par le runner scientifique |
| `python/pops/` | API publique typée, validation/résolution/codegen/bind/runtime | Runner de campagne, tolérances propres à un cas, figures de validation |
| `examples/final/` | Scripts autonomes prouvant le lifecycle public et ses obligations d’acceptation | Matrices de convergence lourdes, scaling, répertoire de cas scientifiques |
| `tests/` | Tests unitaires, intégrations courtes, régressions isolées, refus déterministes, gates d’architecture | Campagnes multi-résolution lourdes et résultats de publication |
| `verification/cases/` | Cas complets avec équations, oracle, configurations, analyse et README | Boucles numériques Python remplaçant le runtime PoPS |
| `verification/pops_verify/reference_errors.py` | Erreurs par rapport aux solutions exactes, moyennes analytiques de cellules, phase, fréquence et agrégation des ordres | Réimplémentation des réductions natives déjà fournies par `pops.diagnostics` |
| `verification/pops_verify/` | Agrégation des erreurs externes, provenance, sélection de campagnes et génération de rapports | Réduction d’état déjà fournie par `pops.diagnostics`, flux, reconstruction ou solveur exécuté dans le hot path |
| `benchmarks/` | Mesures de performance reproductibles du moteur, kernels, solveurs, AMR, I/O | Cas présentés comme validation physique ou preuve d’ordre à eux seuls |
| `schemas/` | Contrats JSON versionnés du manifest, des métriques, de la provenance et du rapport | Données de run ou valeurs machine spécifiques |
| `scripts/` | Points d’entrée et vérifications du monorepo | Logique scientifique du cas ou calcul cellule par cellule |
| `docs/verification/` | Contrat normatif, métriques, seuils, politique de campagne | Champs HDF5, snapshots et résultats volumineux |
| `build/verification/` | HDF5, CSV, JSONL, figures et rapports générés | Références prétendument officielles sans provenance |

`/build/` est déjà git-ignoré ; `build/verification/` réutilise cette politique sans ajouter une seconde racine de sorties. Les seules sorties susceptibles d’être versionnées sont de petits fichiers de référence ou des figures de reproduction établies, accompagnés d’une provenance machine-readable.

### 2.3 Principe d’exécution

Le code de cas peut :

1. construire un `pops.Case` et un `pops.Program` typés ;
2. construire les données analytiques avec `pops.analytic` ou les profils publics de `pops.lib.initial` ;
3. attacher au `ConsumerGraph` les diagnostics natifs de `pops.diagnostics` ;
4. appeler le pipeline public `validate -> resolve(layout=...) -> compile -> bind` ;
5. lancer `pops.run(...)` sur l’artefact exact-rank sélectionné ;
6. rouvrir les sorties scientifiques et les checkpoints ;
7. calculer uniquement les erreurs dépendant d’un oracle externe et produire les rapports hors du runtime chronométré.

Le code de cas ne doit pas :

- importer une façade native privée pour contourner le pipeline public ;
- exécuter une boucle Python par cellule dans le calcul utilisé comme preuve du solveur ;
- reconstruire une deuxième hiérarchie AMR ou une deuxième autorité temporelle ;
- modifier silencieusement une capacité non supportée ;
- dépendre d’un autre dépôt pour être exécutable.

Les intégrales, extrema, normes d’état, variations de pas et bilans acceptés utilisent les descripteurs publics de `pops.diagnostics` lorsqu’ils existent. Le post-traitement repo-local ne remplace pas ces réductions : il calcule les moyennes analytiques de cellule, les erreurs à l’oracle, la phase, la fréquence, la symétrie, l’erreur coarse-fine et les pentes entre plusieurs runs. Les solutions analytiques, sources manufacturées et conditions initiales peuvent être exprimées avec `pops.analytic`, générées hors ligne, puis abaissées par les protocoles publics ou compilées par le codegen. Le post-traitement Python intervient après la synchronisation et hors de toute région de performance.

## 3. Vocabulaire et niveau de preuve

### 3.1 Vérification du code

La vérification du code répond à la question : **les équations discrètes et les algorithmes annoncés sont-ils implémentés correctement ?**

Elle repose principalement sur :

- solutions analytiques ;
- méthode des solutions manufacturées ;
- ordre de convergence observé ;
- solutions exactes de problèmes de Riemann ;
- relations de dispersion linéaires ;
- invariants exacts ;
- symétries imposées par le problème.

### 3.2 Vérification de la solution

La vérification de la solution quantifie, pour un calcul donné :

- l’erreur de discrétisation ;
- l’erreur du solveur elliptique ;
- l’erreur temporelle ;
- l’erreur liée à l’AMR ;
- la sensibilité au maillage, au pas de temps et au placement des patches.

### 3.3 Validation physique

Un cas n’est une validation physique que s’il est comparé à une mesure expérimentale, une solution théorique reconnue dans le même régime ou un résultat publié clairement identifié. Une conservation de masse ou une convergence d’ordre deux ne constitue pas à elle seule une validation physique.

### 3.4 Régression

Une régression compare une version courante à un résultat de référence. Elle peut détecter un changement, mais ne prouve pas que la référence était correcte. Toute régression doit être adossée à un oracle ou à une justification scientifique indépendante.

### 3.5 Benchmark de performance

Un benchmark de performance mesure un coût, un débit ou une efficacité parallèle. Il ne prouve pas la correction scientifique. Toute mesure de performance doit être précédée d’une validation numérique exécutée hors de la région chronométrée.

---

## 4. Contrat obligatoire de chaque cas

Chaque dossier sous `verification/cases/` commence par un `README.md` contenant un bloc de contrat. Le manifest scientifique est autonome ; il ne réutilise pas le catalogue généré des tests rapides.

| Champ | Contenu obligatoire |
|---|---|
| Identifiant | Identifiant stable, par exemple `TR-01` ou `CP-02` |
| `verification_kind` | `code-verification`, `solution-verification`, `physical-validation`, `robustness` ou `infrastructure` |
| `evidence_status` | `required`, `capability-gated`, `reproduction-candidate` ou `established-reproduction` |
| Équations | Système réellement résolu, variables conservées et sources |
| Oracle | Formule analytique, MMS, dispersion, invariant, référence publiée ou solution numérique convergée |
| Domaine et frontières | Dimension, bornes, périodique/Dirichlet/Neumann/réfléchissant |
| Paramètres | Valeurs, unités ou mention `dimensionless` |
| Dimensions natives | Valeurs de `POPS_NATIVE_DIM` requises ; une exécution n’en utilise qu’une |
| Capacités requises | Cartésien/polaire, uniforme/AMR, nombre total de niveaux, subcycling, Poisson, MPI, HDF5, espace Kokkos |
| Configurations | Résolutions, blocs, CFL, intégrateur, flux, reconstruction, AMR, placement des interfaces |
| Diagnostics | Normes, conservation, phase, fréquence, symétrie, résidu, positivité, coût |
| Seuils | Tolérances et ordre minimal, avec justification |
| Prouve | Ce que les assertions établissent effectivement |
| Ne prouve pas | Ce qui reste hors périmètre |
| Ressources | Résolutions, rangs MPI, threads, GPU, nœuds, mémoire et temps maximal |
| Provenance | SHA unique du monorepo, état dirty, version PoPS, digests des catalogues/headers/artefact natif, compilateur, Kokkos, MPI, machine et job Slurm |

Structure type :

```text
verification/cases/euler_poisson/langmuir_cold/
├── README.md
├── run.py
├── exact.py
├── analyze.py
├── case.toml
├── configs/
│   ├── uniform.toml
│   ├── spatial_convergence.toml
│   ├── temporal_convergence.toml
│   ├── amr_static.toml
│   ├── amr_subcycling.toml
│   ├── amr_dynamic.toml
│   └── mpi.toml
└── reference/
    └── README.md
```

`run.py` expose au minimum une construction déterministe du cas et un point d’entrée pilotable par le runner. `exact.py` ne lit jamais la sortie PoPS pour fabriquer sa référence. `analyze.py` ne modifie jamais les champs de simulation.

Les données de `reference/` restent petites et traçables. Les champs complets, snapshots et figures générés vont dans `build/verification/<case-id>/<run-id>/`.

## 5. Manifest scientifique du monorepo

Le fichier source de vérité est `verification/manifest.toml`. Ses schémas sont versionnés sous `schemas/`.

Les quatre contrats machine-readable sont :

```text
schemas/verification_manifest.v1.json
schemas/verification_metrics.v1.json
schemas/verification_provenance.v1.json
schemas/verification_report.v1.json
```

Le manifest est validé avant toute compilation. Chaque `metrics.json` et `provenance.json` est validé avant agrégation, puis le rapport final est validé avant publication. Une évolution incompatible exige une nouvelle version de schéma.

```toml
schema = "pops.verification.manifest.v1"
repository = "wolf75222/PoPS"
max_nodes = 2

[current_capabilities]
exact_native_dimension = true
cartesian_system_runtime = true
polar_system_runtime = false
amr_total_levels_baseline = 3
amr_refinement_ratios_baseline = [2, 2]
hdf5_requires_mpi = true

[[case]]
id = "CP-02"
path = "verification/cases/euler_poisson/langmuir_cold/run.py"
name = "Cold Langmuir wave"
verification_kind = "code-verification"
evidence_status = "required"
physics = ["continuity", "momentum", "poisson", "electrostatic_source"]
oracle = "linear_eigenmode_and_closed_form"
native_dimensions = [1, 2]
execution_spaces = ["KokkosSerial", "KokkosOpenMP", "KokkosCuda"]
mpi_modes = ["off", "on"]
suites = ["pr", "nightly", "weekly", "release", "two_node"]
requires = [
  "public_case_pipeline",
  "cartesian_layout",
  "poisson",
  "field_at_program_stage",
]

[case.resources.pr]
nodes = 1
mpi_ranks = 1
omp_threads = 1
resolutions = [32, 64, 128]

[case.resources.two_node]
nodes = [1, 2]
mpi_ranks_per_node = [1, 2, 4]
gpus_per_node = [1, 2, 4]
max_wall_seconds = 3600

[case.acceptance]
spatial_order_min = 1.8
temporal_order_min = 1.8
poisson_relative_residual_max = 1.0e-10
finite = true
charge_conservation = true
```

### 5.1 Découverte des capacités

Le manifest indique ce que le cas demande ; il ne prétend pas que le build courant le fournit. Avant de planifier un run, le runner interroge le paquet installé et conserve :

- le rapport de `pops.doctor()` ;
- la dimension de la feuille native authentifiée ;
- l’espace d’exécution Kokkos ;
- la présence de MPI et, si demandé, du writer HDF5 collectif ;
- les capacités du provider AMR : dimension, nombre total de niveaux, ratios, regrid transactionnel et événements de lifecycle ;
- les routes elliptiques et temporelles effectivement résolues.

Un cas `required` échoue si une capacité annoncée par la release manque. Un cas `capability-gated` est marqué `not-supported` avec l’évidence exacte ; il ne peut pas être compté comme réussi ni remplacé par une configuration plus faible.

### 5.2 Refus fail-closed du runner

Le runner refuse immédiatement :

- `nodes > 2` ;
- une dimension du cas différente de `POPS_NATIVE_DIM` de l’artefact chargé ;
- une configuration GPU avec plusieurs rangs visibles sur le même GPU sans demande explicite ;
- une sursouscription CPU ;
- HDF5 natif sans MPI ;
- une résolution de convergence ne contenant que deux points ;
- une référence sans provenance ;
- une assertion d’ordre deux sur une solution contenant un choc ;
- un nombre de niveaux AMR supérieur à la capacité authentifiée du provider courant ;
- une exécution polaire intégrée lorsque le runtime public ne fournit qu’un layout cartésien ;
- un fallback vers une extension native d’une autre dimension ou vers un chemin privé.

La classification scientifique est portée par `verification_kind` et `evidence_status`. Elle reste indépendante des markers pytest et des labels CTest utilisés dans `tests/`.

## 6. Sorties et provenance

Chaque run produit au minimum :

```text
build/verification/<case-id>/<run-id>/
├── resolved_case.json
├── provenance.json
├── metrics.json
├── metrics.jsonl
├── fields.h5
├── stdout.log
├── stderr.log
└── analysis/
    ├── summary.md
    ├── errors.csv
    ├── convergence.csv
    ├── conservation.csv
    └── figures/
```

### 6.1 Schéma universel de métriques obligatoires

Chaque exécution doit écrire un objet `pops.verification.metrics.v1`, validé par `schemas/verification_metrics.v1.json`, contenant les mêmes champs, même lorsque certains diagnostics ne s’appliquent pas au cas. L’absence silencieuse d’un champ est interdite. Une grandeur non applicable vaut `null` et possède une justification dans `not_applicable_reason`. Une grandeur applicable mais non calculée constitue un échec du runner.

Champs obligatoires :

| Groupe | Champs obligatoires |
|---|---|
| Erreurs | $L^1$, $L^2$, $L^\infty$ par variable ; ordre observé par variable ou `null` pour un run isolé |
| Conservation | masse totale ; moment total par composante ; énergie totale ; charge totale ; énergie électrostatique |
| Poisson | résidu ; défaut de Gauss $\|\nabla\cdot E-\rho_q/\epsilon_0\|$ |
| Extrema | minimum de densité ; minimum de pression |
| Symétrie | erreur de symétrie déclarée par le cas |
| AMR | erreur près des interfaces coarse-fine ; erreur loin des interfaces ; cellules feuilles ; patches ; regriddings |
| Temps | temps de ghost fill ; temps de Poisson ; temps de refluxing |

Règles d’applicabilité :

- sans équation de densité ou de pression, `rho_min` ou `p_min` vaut `null` avec justification ;
- sans Poisson, le temps Poisson vaut `0`, tandis que résidu, défaut de Gauss et énergie électrostatique valent `null` ;
- sans AMR, `leaf_cells` reste renseigné avec le nombre de cellules actives, `regrid_count=0`, `patch_count` reste renseigné, et les erreurs coarse-fine valent `null` ;
- sans référence analytique, les normes analytiques valent `null`, mais les invariants, symétries, extrema et métriques de robustesse restent obligatoires ;
- l’ordre observé est produit par l’agrégateur de campagne à partir de plusieurs runs et recopié dans le rapport de convergence ;
- les grandeurs conservées sont enregistrées au minimum à l’état initial, à l’état final et sous forme de dérive maximale sur l’intervalle.

Exemple minimal :

```json
{
  "schema": "pops.verification.metrics.v1",
  "case_id": "CP-02",
  "errors": {
    "density": {"l1": 0.0, "l2": 0.0, "linf": 0.0, "observed_order": null},
    "velocity_x": {"l1": 0.0, "l2": 0.0, "linf": 0.0, "observed_order": null},
    "electric_field_x": {"l1": 0.0, "l2": 0.0, "linf": 0.0, "observed_order": null}
  },
  "conservation": {
    "mass_total": {"initial": 0.0, "final": 0.0, "max_relative_drift": 0.0},
    "momentum_total": {"initial": [0.0], "final": [0.0], "max_relative_drift": 0.0},
    "energy_total": {"initial": 0.0, "final": 0.0, "max_relative_drift": 0.0},
    "charge_total": {"initial": 0.0, "final": 0.0, "max_absolute_drift": 0.0},
    "electrostatic_energy": {"initial": 0.0, "final": 0.0, "max_relative_drift": 0.0}
  },
  "poisson": {
    "residual_l2": 0.0,
    "gauss_defect_l2": 0.0
  },
  "extrema": {"rho_min": 0.0, "p_min": null},
  "symmetry": {"error": null},
  "amr": {
    "interface_error": null,
    "bulk_error": 0.0,
    "leaf_cells": 128,
    "patch_count": 1,
    "regrid_count": 0
  },
  "not_applicable_reason": {
    "errors.*.observed_order": "single-resolution run",
    "extrema.p_min": "cold-fluid model has no pressure variable",
    "symmetry.error": "one-dimensional mode",
    "amr.interface_error": "uniform-grid configuration"
  },
  "timings_seconds": {
    "ghost_fill": 0.0,
    "poisson": 0.0,
    "reflux": 0.0
  }
}
```

Le schéma complet doit également accepter des séries temporelles ou des fichiers CSV associés sans remplacer les résumés obligatoires de `metrics.json`.

### 6.2 Provenance minimale

`provenance.json` est validé par `schemas/verification_provenance.v1.json`.

Exemple de provenance :

```json
{
  "schema": "pops.verification.provenance.v1",
  "case_id": "CP-02",
  "repository": "wolf75222/PoPS",
  "repository_sha": "<sha>",
  "repository_dirty": false,
  "pops_version": "1.0.0",
  "component_catalog_digest": "<sha256>",
  "native_header_signature": "<sha256>",
  "native_variant_manifest_digest": "<sha256>",
  "doctor_ok": true,
  "date_utc": "2026-08-17T00:00:00Z",
  "compiler": "GCC 13.x",
  "build_type": "Release",
  "precision": "float64",
  "pops_native_dim": 2,
  "kokkos_execution_space": "OpenMP",
  "mpi_enabled": true,
  "mpi_library": "MPICH or OpenMPI, exact version",
  "mpi_thread_level_requested": "MPI_THREAD_MULTIPLE",
  "mpi_thread_level_provided": "MPI_THREAD_MULTIPLE",
  "hdf5_collective_enabled": true,
  "nodes": 1,
  "mpi_ranks": 4,
  "omp_threads_per_rank": 48,
  "gpus": 0,
  "hostname": "<host>",
  "slurm_job_id": "<id>",
  "dimension": 2,
  "resolution": [128, 128],
  "block_size": [32, 32],
  "amr_total_levels": 2,
  "refinement_ratio": 2,
  "subcycling": true,
  "time_program": "SSPRK2",
  "cfl": 0.4,
  "final_time": 1.0
}
```

Le SHA du monorepo est l’unique révision source. Les digests supplémentaires authentifient les éléments susceptibles de différer malgré un même checkout apparent : catalogue de composants généré, ensemble des headers publics et feuille native exact-rank réellement chargée. Le rapport de doctor et l’inventaire des devices sont archivés avec la provenance.

### 6.3 Politique de conservation des résultats

Le rapport ne doit jamais se limiter à « pass/fail ». Il doit conserver les valeurs mesurées afin de distinguer :

- un échec brutal ;
- une perte progressive d’ordre ;
- une dérive faible mais cumulative ;
- une dépendance aux frontières de blocs ;
- une variation propre à un backend ;
- une régression de performance masquée par le bruit.

---

## 7. Métriques communes

### 7.0 Autorité de calcul des diagnostics

Deux couches sont strictement séparées :

1. **État numérique accepté.** `pops.diagnostics.Norm`, `Integral`, `MinMax`, `StepChangeNorm`, `ConservationCheck` et `Balance` décrivent les réductions exécutées nativement par C++/Kokkos/MPI. Le `ConsumerGraph` est l’autorité de leur cadence et de leur publication.
2. **Erreur par rapport à un oracle.** `verification/pops_verify/reference_errors.py` rouvre les sorties acceptées, retire les cellules coarse couvertes et les répliques, calcule les moyennes analytiques de cellule puis agrège erreurs, phases, fréquences, symétries et ordres entre plusieurs runs.

Lorsqu’un `BalanceLedger` est disponible, `Balance` est l’autorité du bilan discret accepté : variation de stockage, flux sortant, sources, reflux et projection. Reconstituer ce bilan uniquement depuis deux snapshots est interdit comme preuve principale.

### 7.1 Normes sur maillage uniforme

Pour une variable scalaire ou une composante de l’état :

$$
L^1 = \frac{\sum_i V_i |U_i-U_i^{\mathrm{exact}}|}{\sum_i V_i},
$$

$$
L^2 = \left(\frac{\sum_i V_i |U_i-U_i^{\mathrm{exact}}|^2}{\sum_i V_i}\right)^{1/2},
$$

$$
L^\infty = \max_i |U_i-U_i^{\mathrm{exact}}|.
$$

### 7.2 Normes AMR

Les normes AMR sont calculées uniquement sur les cellules feuilles. Une cellule grossière couverte par un niveau fin ne doit jamais être comptée une seconde fois.

$$
L^1_{\mathrm{AMR}} =
\frac{\sum_{i\in\mathrm{leaf}}V_i |U_i-U_i^{\mathrm{exact}}|}
{\sum_{i\in\mathrm{leaf}}V_i}.
$$

Le masque de couverture utilisé pour les normes doit être le même que celui utilisé pour les bilans conservatifs.

### 7.3 Comparaison volumes finis

Lorsque PoPS stocke des moyennes de cellule, la référence doit être la moyenne analytique sur la cellule :

$$
\bar U_i^{\mathrm{exact}} = \frac{1}{V_i}\int_{V_i} U^{\mathrm{exact}}(\mathbf{x},t)\,dV.
$$

Évaluer uniquement la solution au centre de cellule peut créer une erreur artificielle d’ordre deux et fausser la pente mesurée.

### 7.4 Ordre observé

Pour deux résolutions successives de ratio deux :

$$
p_{\mathrm{obs}} = \frac{\log(E_h/E_{h/2})}{\log 2}.
$$

Une conclusion d’ordre nécessite au moins quatre résolutions, idéalement cinq. Les deux derniers intervalles de raffinement servent au gate principal.

### 7.5 Bilans conservatifs

Pour une quantité conservative $Q$ :

$$
\delta Q(t)=Q(t)-Q(0)-\int_0^t S_Q\,dt
+\int_0^t\int_{\partial\Omega}F_Q\cdot n\,dA\,dt.
$$

Les bilans doivent inclure les sources et les flux physiques aux frontières. Une comparaison brute $Q(t)-Q(0)$ n’est correcte que pour des frontières et des sources compatibles.

### 7.6 Diagnostics Poisson

Pour $A_h\phi_h=f_h$ :

- résidu algébrique : $r_h=f_h-A_h\phi_h$ ;
- erreur de potentiel : $\|\phi_h-\phi\|$ ;
- erreur de champ : $\|-\nabla_h\phi_h-E\|$ ;
- défaut de Gauss : $\|\nabla_h\cdot E_h-\rho_q/\epsilon_0\|$ ;
- discontinuité de flux normal près des interfaces coarse-fine ;
- nombre de V-cycles, itérations et réduction de résidu par cycle.

Un faible résidu ne prouve pas que l’opérateur discret, les frontières ou le signe physique sont corrects.

### 7.7 Phase, amplitude et contamination modale

Pour une onde :

- fréquence numérique $\omega_{\mathrm{num}}$ ;
- erreur relative de fréquence ;
- vitesse de phase ;
- amplitude du mode principal ;
- perte d’amplitude par période ;
- énergie injectée dans les autres modes propres ;
- contenu harmonique, par exemple $|\widehat U_{2k}|/|\widehat U_k|$.

Pour l’onde de Langmuir canonique, rapporter explicitement :

$$
E_{\omega}=\frac{|\omega_{\mathrm{num}}-\omega_{pe}|}{\omega_{pe}},
\qquad
H_2=\frac{|\widehat E_{2k}|}{|\widehat E_k|}.
$$

La fréquence doit être estimée par au moins deux méthodes parmi FFT temporelle, ajustement de phase et passages à zéro. Un désaccord supérieur à l’erreur temporelle attendue doit être signalé.

### 7.8 Symétrie

Exemples :

$$
E_{xy}=\frac{\|U(x,y)-U(y,x)\|_2}{\|U\|_2},
$$

$$
E_{\mathrm{radial}}=
\frac{\max_\theta R(\theta)-\min_\theta R(\theta)}{\langle R\rangle}.
$$

### 7.9 Erreur près des interfaces AMR

Définir une bande de $m$ cellules fines autour de chaque interface coarse-fine $\Gamma_{cf}$ :

$$
E_{cf}=\max_{d(\mathbf{x},\Gamma_{cf})<mh_f}|U_h-U^{\mathrm{exact}}|.
$$

Rapporter séparément :

- erreur dans la bande d’interface ;
- erreur loin de l’interface ;
- rapport $E_{cf}/E_{bulk}$ ;
- position du maximum d’erreur.

### 7.10 Énergie totale des systèmes Euler–Poisson

Lorsque le modèle possède le budget électrostatique standard, diagnostiquer séparément l’énergie fluide et l’énergie de champ, puis leur somme :

$$
\mathcal{E}_{\mathrm{tot}}=
\int_{\Omega}
\left(
\mathcal{E}_{\mathrm{fluides}}+
\frac{\epsilon_0}{2}|E|^2
\right)\,dV.
$$

Si le modèle, le nondimensionnement ou les frontières introduisent un autre terme de potentiel, le README doit écrire le budget exact utilisé. Il est interdit de présenter l’énergie fluide seule comme énergie totale du système couplé.

---

## 8. Critères d’acceptation généraux

### 8.1 Cas lisses d’ordre deux

Le critère par défaut est :

- erreurs décroissantes sur toutes les résolutions ;
- $p_{\mathrm{obs}}\geq 1.8$ sur les deux derniers intervalles ;
- aucune chute systématique d’ordre avec AMR ;
- absence de plateau prématuré non expliqué par l’arrondi ou la tolérance elliptique.

La plage 1.9 à 2.1 est la cible. Le seuil 1.8 laisse une marge aux effets pré-asymptotiques sans accepter un schéma effectivement d’ordre un.

### 8.2 Intégrateurs temporels

| Intégrateur annoncé | Ordre minimal observé |
|---|---:|
| Forward Euler | 0.9 |
| RK2 / SSPRK2 | 1.8 |
| Strang | 1.8 |
| RK3 / SSPRK3 | 2.7 |
| Méthode implicite annoncée d’ordre deux | 1.8 |

### 8.3 Poisson

- l’erreur de discrétisation sur $\phi$ doit suivre l’ordre annoncé ;
- l’erreur sur $E=-\nabla\phi$ doit être testée séparément ;
- l’erreur algébrique estimée en resserrant la tolérance doit être inférieure à 1 % de l’erreur de discrétisation ;
- le solveur ne doit pas s’arrêter uniquement parce que le résidu absolu est petit sur un problème mal normalisé ;
- les cas de Neumann doivent retirer ou fixer explicitement le mode constant.

### 8.4 Conservation

Les tolérances sont dimensionnées par :

$$
\mathrm{tol}(Q)=\max(\mathrm{abs\_tol},\mathrm{rel\_tol}\,Q_{scale},C\,\epsilon_{mach}\,N_{updates}).
$$

Valeurs initiales proposées en double précision :

- uniforme périodique, CPU : erreur relative cible $10^{-12}$ à $10^{-11}$ ;
- AMR, GPU, restart ou réductions MPI : gate initial $10^{-10}$, à resserrer après caractérisation ;
- aucune dérive monotone non expliquée par les flux ou sources physiques.

Une tolérance fixe universelle est interdite pour des grandeurs proches de zéro.

### 8.5 Positivité et robustesse

- aucun NaN ou Inf ;
- densité et pression au-dessus du plancher physique déclaré ;
- aucune valeur silencieusement clampée sans compteur ;
- le nombre et l’emplacement des interventions de positivité sont enregistrés ;
- un cas de choc ne reçoit pas de gate d’ordre deux global.

### 8.6 Parité backend

- même ordre de convergence ;
- mêmes invariants et même régime physique ;
- différence champ-à-champ compatible avec les réassociations flottantes ;
- égalité bit à bit exigée uniquement lorsque le backend et le mode déterministe la garantissent explicitement.

### 8.7 Performance

Aucun seuil absolu en millisecondes n’est portable entre machines. Sur une machine fixe et dédiée :

- au moins cinq échantillons valides ;
- warmups exclus ;
- temps du rang le plus lent ;
- médiane, MAD, p10, p90 et moyenne tronquée ;
- campagne baseline/candidate entrelacée, de préférence `A B B A` ;
- régression à examiner au-delà de 10 % si le signal dépasse clairement le bruit ;
- validation numérique exécutée hors de la région chronométrée.

---

## 9. Configurations numériques communes

### 9.0 Capacités courantes et capability gates

Le catalogue décrit la cible scientifique complète, mais le runner doit distinguer les capacités déjà qualifiées du monorepo et les extensions futures.

| Capacité | Baseline courante du dépôt | Politique du plan |
|---|---|---|
| Dimension | Une seule dimension native par artefact, `1`, `2` ou `3` | Builds et rapports séparés, agrégation ensuite |
| Exécution on-node | Kokkos obligatoire : Serial, OpenMP ou accélérateur selon l’installation | Aucun backend hôte alternatif |
| Distribution | MPI optionnel | Les variantes MPI utilisent le même espace Kokkos on-node |
| HDF5 natif | Writer collectif, donc MPI requis | Refus pré-runtime en configuration sérielle |
| Runtime de géométrie | `System`/`AmrSystem` cartésiens exact-rank | Les campagnes polaires intégrées sont `capability-gated` |
| Hiérarchie AMR de production | Trois niveaux totaux dans la cible finale d’advection, avec ratios `(2, 2)` et relations temporelles 2:1 explicites | Deux niveaux restent le smoke minimal ; quatre niveaux ou ratios anisotropes sont `capability-gated` |
| Contrat d’authoring AMR | Peut représenter des transitions supplémentaires et des providers externes | Ne vaut pas preuve que le provider natif courant les exécute |
| Temps | `Program` est l’unique autorité temporelle publique | Aucun stepper privé parallèle dans les cas de vérification |

La baseline doit être re-détectée à chaque SHA. Si le provider natif gagne une capacité, le manifest peut promouvoir le cas correspondant de `capability-gated` à `required` dans une modification revue séparément.

### 9.1 Matrice de builds exact-rank

Une exécution charge un seul artefact natif exact-rank. Les campagnes multi-dimensionnelles utilisent des répertoires et des environnements isolés, par exemple :

```text
build/verification/dim1-serial/
build/verification/dim2-serial/
build/verification/dim3-serial/
build/verification/dim2-mpi/
build/verification/dim3-cuda-mpi/
```

Commandes CPU de référence :

```bash
bash scripts/build_python.sh --dim 1
bash scripts/build_python.sh --dim 2
bash scripts/build_python.sh --dim 3
bash scripts/build_python.sh --dim 2 --mpi
```

Pour un build GPU, le profil machine fournit l’installation Kokkos accélérée et les variables CMake nécessaires. Après chaque build :

1. exécuter `pops.doctor()` ;
2. vérifier la dimension native ;
3. enregistrer le manifest de variante et son digest ;
4. vérifier Serial/MPI/HDF5/device selon le profil ;
5. interdire toute extension native de secours à la racine du paquet.

### 9.2 Configuration canonique de référence

Les variantes étendues du catalogue ne remplacent pas les configurations canoniques suivantes. Chaque cas concerné conserve au moins une configuration portant le label `canonical`, exécutée sans modification silencieuse. Toute adaptation est enregistrée sous un autre identifiant.

#### Advection sinusoïdale canonique

Sur $[0,1]^d$ périodique :

$$
q(\mathbf{x},0)=1+\varepsilon\sin\left[2\pi(x+2y+3z)\right],
\qquad \varepsilon=10^{-2},
$$

$$
\mathbf{a}=(1,1,1),
\qquad T=1.
$$

Après $T=1$, chaque coordonnée est translatée d’une période et le champ revient exactement à son état initial. En 1D et 2D, utiliser la restriction naturelle de la formule aux coordonnées présentes.

#### Série de convergence canonique

```text
N = 16, 32, 64, 128, 256
```

Cette série est obligatoire pour les cas lisses lorsque le coût et la mémoire le permettent. Les séries dimensionnelles plus grandes de la section 9.3 sont des extensions, pas des remplacements. Une affirmation d’ordre repose sur au moins quatre points, idéalement les cinq points canoniques.

#### Décomposition et interfaces canoniques

```text
block_size = 8, 16, 32, 64
interface_x = 0.25, 0.2578125, 0.375
MPI 2D layouts for 4 ranks = 1×4, 2×2, 4×1
MPI 3D layouts for 4 ranks = 1×1×4, 1×2×2, 4×1×1
```

Les layouts allongés et équilibrés sont exécutés avec la même résolution globale, le même pas de temps et le même état initial.

#### Subcycling canonique

```text
global_dt
subcycling_ratio_2
subcycling_ratio_4          # uniquement si annoncé par le provider
amr_subcycling_no_reflux    # contrôle négatif de développement uniquement
```

Le ratio 4 est exécuté uniquement si l’architecture publique le permet ; son absence est enregistrée comme capacité non supportée, jamais remplacée silencieusement par le ratio 2. La configuration sans reflux doit échouer nettement et ne constitue jamais une option utilisateur de production.

#### Stress test de regridding canonique

```text
refine/coarsen cycles = 256
extended stress        = 512
```

La trajectoire du patch est prescrite analytiquement. Le stress test mesure la dérive cumulative et ne remplace pas la campagne de convergence avec regridding physiquement motivé.

#### Répétitions ondulatoires canoniques

Les ondes lisses sont analysées après 1, 2 et 4 périodes. Les cas de stabilité longue peuvent ajouter 20 périodes, mais le résultat à une période reste la baseline de comparaison.

### 9.3 Résolutions de convergence

| Dimension | Série standard | Série étendue |
|---|---|---|
| 1D | 16, 32, 64, 128, 256 | 512, 1024 ou 2048 |
| 2D | $16^2$, $32^2$, $64^2$, $128^2$, $256^2$ | jusqu’à $512^2$ |
| 3D | $16^3$, $32^3$, $64^3$, $128^3$ | $256^3$ si le coût le permet |

Les résolutions très faibles servent à détecter le régime pré-asymptotique. Le gate d’ordre porte sur les résolutions les plus fines encore au-dessus du plancher d’arrondi.

### 9.4 Séparation spatial/temps

**Ordre spatial :**

- utiliser un intégrateur temporel d’ordre supérieur au schéma spatial ; ou
- imposer $\Delta t\propto h^2$ pour un schéma spatial d’ordre deux ; ou
- vérifier par une étude préalable que l’erreur temporelle est au moins dix fois plus faible.

**Ordre temporel :**

- fixer une grille suffisamment fine ;
- utiliser $\Delta t$, $\Delta t/2$, $\Delta t/4$, $\Delta t/8$, $\Delta t/16$ ;
- confirmer que le résultat ne change presque plus lorsque la grille est encore raffinée.

**Ordre global :**

- garder un CFL constant, donc $\Delta t\propto h$ ;
- mesurer l’ordre du pipeline public complet tel qu’utilisé en production.

### 9.5 Matrice AMR standard

| Code | Configuration | Statut courant |
|---|---|---|
| `U-C` | Grille uniforme coarse | requis |
| `U-F` | Grille uniforme à la résolution fine équivalente | requis |
| `A-S0` | Deux niveaux totaux, patch statique, sans subcycling | requis |
| `A-S2` | Deux niveaux totaux, subcycling ratio 2 | requis si le cas subcycle |
| `A-DP` | Patch dynamique prescrit par la solution exacte | requis |
| `A-DT` | Raffinement dynamique par tagging numérique | requis |
| `A-2L` | Niveaux totaux `0+1`, ratio 2 | configuration minimale obligatoire |
| `A-3L` | Niveaux totaux `0+1+2`, ratios 2 puis 2 | baseline d’acceptation publique actuelle |
| `A-4L` | Niveaux totaux `0+1+2+3` | extension future |
| `A-ANISO` | Ratio anisotrope | capability-gated |
| `A-CF` | Sweep du placement des interfaces coarse-fine | requis |
| `A-RG` | Sweep de fréquence de regridding | requis |

Paramètres par défaut :

- ratio de raffinement baseline : 2 ;
- deux niveaux totaux pour les smokes et l’isolation d’une seule interface ;
- trois niveaux totaux pour la baseline d’acceptation complète ;
- quatre niveaux totaux et davantage uniquement après découverte positive de capacité ;
- subcycling : off puis on ;
- tailles de blocs 2D : 8, 16, 32, 64 cellules par direction ;
- tailles de blocs 3D : 8, 16, 32, 64 cellules par direction, sous réserve mémoire ;
- bande d’analyse coarse-fine : quatre cellules fines de chaque côté ;
- regridding : tous les 1, 2, 4, 8 et 16 pas pour les campagnes dédiées.

### 9.6 Placement des interfaces

Pour les solutions trigonométriques ou ondulatoires, placer successivement l’interface AMR :

- sur un maximum de la variable ;
- sur un zéro ;
- sur un maximum de gradient ;
- sur un maximum de courbure ;
- à une position non alignée avec une période ou une cellule remarquable.

### 9.7 Schémas à couvrir

La suite sélectionne, selon les capacités réellement exposées par PoPS :

- Forward Euler ;
- SSPRK2 ;
- SSPRK3 ;
- Strang ;
- IMEX ;
- Schur ou opérateur implicite condensé ;
- Rusanov ;
- HLL ;
- HLLC ;
- Roe ;
- minmod ;
- Van Leer ;
- WENO5-Z.

Chaque méthode n’est pas exécutée sur tous les cas. Une matrice de couverture minimale est donnée plus loin. Les noms du rapport doivent correspondre aux routes résolues et au catalogue de composants du SHA testé, pas à un alias choisi par le script.

## 10. Catalogue synthétique

Les identifiants suivants sont stables et doivent être utilisés dans les manifests, les noms de jobs et les rapports.

| Domaine | Identifiants | Objectif principal |
|---|---|---|
| Transport lisse et ghost cells | `TR-01` à `TR-07` | Ordre, halos, interfaces de blocs, advection multidimensionnelle |
| Euler lisse | `EU-01` à `EU-06` | Modes propres, ordre, flux, reconstruction, symétries |
| Poisson | `PO-01` à `PO-07` | Opérateur elliptique, frontières, gradient, AMR, FFT/GMG |
| Euler–Poisson et plasma | `CP-01` à `CP-12` | Couplage état–charge–champ–source, dispersion, signe, énergie |
| Temps, splitting et sources | `TM-01` à `TM-08` | Ordre temporel, stages RK, Strang, IMEX, collisions, multirate |
| Campagnes AMR transverses | `AM-01` à `AM-12` | Prolongation, restriction, subcycling, reflux, regrid, synchronisation |
| Chocs et robustesse | `RB-01` à `RB-09` | Positivité, Riemann, chocs forts, symétrie, AMR dynamique |
| Géométrie | `GE-01` à `GE-06` | Coordonnées polaires, axe, rotation, modes radiaux |
| Infrastructure | `IF-01` à `IF-10` | MPI, OpenMP, GPU, restart, I/O, DSL/native, compilateurs |
| Performance | `PF-01` à `PF-12` | Débit, scaling, coût des sous-systèmes et du pipeline complet |

Priorités :

- **P0 :** indispensable avant de présenter PoPS comme solveur vérifié ;
- **P1 :** nécessaire pour une publication ou une release scientifique majeure ;
- **P2 :** étendu ou conditionnel à une capacité particulière.

---

## 11. Transport lisse, ghost cells et interfaces de blocs

### TR-01 — Advection sinusoïdale périodique 1D, 2D et 3D

**Priorité : P0**

Équation :

$$
\partial_t q+\mathbf{a}\cdot\nabla q=0.
$$

Condition initiale proposée :

$$
q(\mathbf{x},0)=q_0+\epsilon\sin\left(2\pi(k_xx+k_yy+k_zz)\right),
$$

avec $q_0=1$, $\epsilon=10^{-2}$ et un vecteur d’onde non dégénéré, par exemple $(k_x,k_y,k_z)=(1,2,3)$. La solution exacte est une translation :

$$
q(\mathbf{x},t)=q(\mathbf{x}-\mathbf{a}t,0).
$$

La configuration canonique utilise $\mathbf a=(1,1,1)$ et $T=1$ sur $[0,1]^d$ périodique. Elle est exécutée aux résolutions $N=16,32,64,128,256$. Les variantes avec d’autres vitesses ou un temps sans retour périodique sont autorisées en complément et sont alors comparées directement à la translation analytique.

**Variantes obligatoires :**

- 1D : propagation positive et négative ;
- 2D : axe $x$, axe $y$, diagonale $(1,1)$, direction non symétrique $(1,0.37)$ ;
- 3D : axes, diagonale et direction oblique ;
- `U-C`, `U-F`, `A-S0`, `A-S2`, `A-DP`, `A-DT` ;
- tailles de blocs 8, 16, 32 et 64 ;
- un, deux et quatre passages complets sur les frontières périodiques ;
- permutations d’axes ;
- 1, 2, 4, 8 puis davantage de rangs MPI dans la campagne infrastructure.

**Composants testés :**

- reconstruction et flux d’advection ;
- intégrateur temporel ;
- ghost fill périodique ;
- échanges same-level et MPI ;
- coins et arêtes de halos ;
- interpolation coarse-fine spatiale et temporelle ;
- subcycling ;
- restriction et reflux pour la forme conservative ;
- invariance aux axes et au découpage des blocs.

**Diagnostics :**

- $L^1$, $L^2$, $L^\infty$ ;
- ordre spatial, temporel et global ;
- erreur de phase ;
- perte d’amplitude ;
- conservation de $\int q\,dV$ ;
- $E_{cf}$ et $E_{bulk}$ ;
- spectre de l’erreur ;
- position du maximum d’erreur.

**Acceptation :** ordre global au moins 1.8 pour une configuration annoncée d’ordre deux ; pas de perte d’ordre en AMR ; conservation compatible avec la tolérance déclarée ; aucun pic fixe aux frontières de blocs.

**Signature de bug :** un ordre correct sur grille uniforme et proche de un en AMR indique en priorité un ghost fill coarse-fine, une interpolation temporelle ou une synchronisation incorrecte.

---

### TR-02 — Impulsion gaussienne transportée

**Priorité : P0**

$$
q(\mathbf{x},0)=q_0+A\exp\left[-\frac{\|\mathbf{x}-\mathbf{x}_0\|^2}{2\sigma^2}\right].
$$

La solution exacte est la translation de la gaussienne. Ce cas complète `TR-01` en introduisant une structure localisée et un spectre de nombres d’onde plus large.

**Variantes :**

- centre initial non aligné avec les centres de blocs ;
- trajectoire traversant une face, une arête et un coin de bloc ;
- patch AMR fixe ;
- patch AMR prescrit qui suit exactement la gaussienne ;
- tagging automatique par gradient puis par erreur estimée ;
- cycles répétés de raffinement et déraffinement.

**Diagnostics :**

- normes globales ;
- position du barycentre ;
- position et valeur du maximum ;
- variance et élargissement numérique ;
- masse ;
- saut instantané d’erreur avant/après chaque regrid ;
- nombre de cellules fines utilisées.

**Composants ciblés :** prolongation, restriction, regrid, tagging, déplacement des patches, diffusion numérique et conservation cumulative.

**Acceptation :** le barycentre converge à l’ordre attendu ; la masse ne dérive pas au rythme des regrids ; l’erreur immédiatement après regrid décroît avec $h$ au même ordre que le schéma.

---

### TR-03 — Vortex réversible de transport de type AMReX SingleVortex

**Priorité : P1**

Utiliser un champ de vitesse incompressible dépendant du temps qui déforme un profil scalaire jusqu’à une déformation maximale, puis inverse le mouvement et ramène exactement la solution à son état initial au temps final.

**Oracle :** état initial au temps de retour.

**Intérêt :** une simple translation conserve la forme du stencil rencontré. Le vortex réversible étire la solution, crée de forts gradients lisses, traverse de nombreuses interfaces de blocs et teste la capacité de l’AMR à suivre une structure déformée.

**Variantes :**

- 2D obligatoire, 3D étendu ;
- disque lisse, cosinus compact ou gaussienne ;
- grille uniforme fine ;
- AMR dynamique suivant le gradient ;
- AMR dynamique suivant une estimation d’erreur ;
- subcycling off/on ;
- regrid tous les 1, 2, 4 et 8 pas.

**Diagnostics :** retour $L^1/L^2/L^\infty$, masse, extrema, épaisseur de l’interface, fraction de cellules fines, coût par rapport à la grille uniforme fine.

**Acceptation :** conservation, convergence de l’erreur de retour et absence de cicatrice visible sur les anciennes frontières de patches.

**Référence externe :** tutoriels officiels AMReX `Advection_AmrCore` / `SingleVortex`.

---

### TR-04 — Traversée face–arête–coin

**Priorité : P0**

Ce cas utilise `TR-02` ou une onde compacte et construit trois géométries de blocs où la structure traverse successivement :

1. une face entre deux blocs ;
2. une arête commune à quatre blocs en 3D ;
3. un coin commun à huit blocs en 3D.

Les paramètres physiques, la résolution et le pas de temps restent identiques. Seul le placement de la décomposition change.

**Composants ciblés :** remplissage des coins de ghost cells, ordre des communications, halos multidimensionnels, stencils diagonaux, indices et strides.

**Diagnostics :** différence champ-à-champ entre les trois placements, $L^\infty$, carte d’erreur, erreur dans une bande autour de la jonction.

**Acceptation :** les différences entre placements convergent avec le même ordre que l’erreur physique. Un défaut qui reste attaché à une arête ou à un coin est un échec même si la norme globale reste faible.

---

### TR-05 — Translation des frontières de blocs

**Priorité : P0**

Le problème physique est inchangé, mais la décomposition est translatée de 1, 2, 3 puis 5 cellules. Exemples d’interfaces : $x=0.25$, $x=0.2578125$, $x=0.375$.

**Composants ciblés :** ghost cells same-level, stencils incomplets, communication MPI, hypothèses d’alignement implicites.

**Diagnostics :** norme de la différence entre décompositions, corrélation spatiale entre l’erreur et les frontières de blocs, ordre de convergence de cette différence.

**Acceptation :** aucune structure d’erreur d’amplitude non convergente ne doit suivre la frontière déplacée.

---

### TR-06 — Permutation et rotation des axes

**Priorité : P0**

Exécuter le même cas sous les transformations :

- $x\leftrightarrow y$ ;
- $x\leftrightarrow z$ ;
- $y\leftrightarrow z$ ;
- réflexion $x\mapsto -x$ ;
- rotation de $90^\circ$ ;
- rotation de $45^\circ$ lorsque l’oracle permet une comparaison.

**Composants ciblés :** indices directionnels, strides, signes de normales, conditions aux limites, ordre des sweeps, reconstruction directionnelle.

**Acceptation :** les erreurs et invariants sont équivalents à l’arrondi près pour les permutations exactes ; les rotations non alignées conservent le même ordre et une amplitude d’erreur cohérente.

---

### TR-07 — Créneau ou disque discontinu transporté

**Priorité : P1**

**Oracle :** translation exacte d’un profil discontinu.

Ce test ne mesure pas l’ordre deux global. Il vérifie :

- absence d’oscillations non physiques ;
- comportement des limiteurs ;
- conservation ;
- diffusion de l’interface ;
- traversée coarse-fine ;
- stabilité lors des regrids.

**Diagnostics :** total variation, overshoot, undershoot, largeur numérique de l’interface, masse, position de l’interface et profil moyen.

**Acceptation :** aucun overshoot incompatible avec le limiter annoncé, conservation, position correcte et convergence de la largeur d’interface vers zéro.

---

## 12. Euler compressible : cas lisses et ordre de convergence

### EU-01 — Modes propres linéaires d’Euler

**Priorité : P0**

Autour d’un état uniforme $\bar U$, calculer l’éigensystème du Jacobien dans la direction $\mathbf n$ :

$$
A_{\mathbf n}r_m=\lambda_m r_m.
$$

Initialiser un mode propre :

$$
U(\mathbf{x},0)=\bar U+\epsilon\,r_m\sin(\mathbf{k}\cdot\mathbf{x}),
$$

avec $\epsilon$ entre $10^{-7}$ et $10^{-5}$. La solution linéarisée est :

$$
U(\mathbf{x},t)=\bar U+\epsilon\,r_m
\sin(\mathbf{k}\cdot\mathbf{x}-\lambda_m|\mathbf{k}|t).
$$

**Modes obligatoires :**

- acoustique gauche ;
- entropie/contact ;
- acoustique droite.

**Variantes :**

- 1D selon $x$, $y$, $z$ ;
- 2D diagonale et oblique ;
- 3D oblique ;
- un puis plusieurs retours périodiques ;
- HLL, HLLC et Rusanov lorsque compatibles ;
- reconstruction primitive puis conservative si les deux chemins existent ;
- `U-F`, `A-S0`, `A-S2`, `A-CF`.

**Diagnostics :** normes par variable, phase, amplitude, contamination dans les autres vecteurs propres, conservation, ordre spatial/temporel.

**Composants testés :** conversion primitive/conservative, eigenvalues, solveur de Riemann, reconstruction, énergie, multidimensionnel, AMR et temps.

**Acceptation :** ordre attendu pour chaque mode ; pas de contamination modale d’ordre inférieur ; courbes comparables sous permutation des axes.

**Référence externe :** suite de linear waves d’Athena/Athena++.

---

### EU-02 — Vortex isentropique advecté

**Priorité : P0**

Utiliser la solution classique d’un vortex isentropique lisse superposé à un écoulement uniforme. La solution exacte à l’instant $t$ est le vortex initial translaté par la vitesse de fond.

**Variantes obligatoires :**

- vortex stationnaire ;
- vitesses de fond $(1,0)$, $(0,1)$, $(1,1)$ et $(1,0.37)$ ;
- centre décalé par rapport aux blocs ;
- 2D uniforme ;
- AMR statique ;
- AMR dynamique ;
- passage par un coin de bloc ;
- plusieurs nombres de Mach de fond.

**Diagnostics :** erreurs sur $\rho$, $p$, $u$, $v$ et $E$ ; déplacement du centre ; vorticité maximale ; entropie ; conservation ; symétrie radiale ; ordre.

**Composants testés :** toutes les équations Euler, flux multidimensionnels, énergie, primitives/conservées, dissipation, quasi-invariance galiléenne, AMR dynamique.

**Acceptation :** ordre global annoncé, absence de vitesse radiale parasite non convergente, conservation et déplacement exact du centre.

**Référence externe :** cas isentropic vortex de FLASH et littérature de vérification CFD.

---

### EU-03 — Solution manufacturée Euler complète

**Priorité : P0**

Choisir des champs positifs, non séparables et dépendants du temps, par exemple :

$$
\rho=2+0.1\sin(2\pi(x+y-t)),
$$

$$
u=0.3+0.1\cos(2\pi(x-t)),
$$

$$
v=-0.2+0.1\sin(2\pi(y+t)),
$$

$$
p=1+0.1\cos(2\pi(x-y+t)).
$$

Construire $U=(\rho,\rho u,\rho v,E)$ puis générer symboliquement :

$$
S_{\mathrm{MMS}}=\partial_tU+\nabla\cdot F(U).
$$

**Règle d’implémentation :** les expressions finales sont générées hors ligne, versionnées sous forme lisible et compilées par le chemin PoPS. Aucun callback Python par cellule n’est autorisé.

**Variantes :**

- 1D, 2D et 3D ;
- périodique ;
- Dirichlet exact ;
- AMR statique et dynamique prescrite ;
- sources évaluées aux temps de stage corrects ;
- Forward Euler, SSPRK2 et SSPRK3.

**Diagnostics :** normes sur toutes les composantes, ordre spatial et temporel, erreur de source, conservation corrigée des sources, erreur coarse-fine.

**Composants testés :** flux, divergence, énergie, sources dépendantes du temps, temps de stage, conditions physiques, ghost cells, AMR.

**Acceptation :** ordre annoncé sur chaque composante. Une seule composante d’ordre inférieur constitue un échec.

---

### EU-04 — Onde acoustique stationnaire avec frontières réfléchissantes

**Priorité : P1**

Construire une onde acoustique linéaire compatible avec des parois réfléchissantes, par exemple avec vitesse normale nulle aux bords et pression/densité en cosinus.

**Oracle :** solution linéaire analytique dans une cavité.

**Composants testés :** ghost cells de réflexion, signe de vitesse normale, coins de frontières, phase temporelle, énergie acoustique.

**Variantes :** 1D, rectangle 2D, frontières alignées et patch AMR touchant une frontière physique.

**Acceptation :** ordre attendu ; absence de fuite de masse ; fréquence correcte ; pas de couche d’erreur d’ordre un au mur.

---

### EU-05 — Vortex stationnaire de Gresho

**Priorité : P1**

Utiliser une version compressible clairement définie, ou une solution manufacturée, où le gradient de pression équilibre la force centrifuge.

**Oracle :** état stationnaire.

**Diagnostics :** vitesse tangentielle, vitesse radiale parasite, pression, moment angulaire, symétrie, dissipation après plusieurs temps de rotation.

**Composants testés :** équilibre stationnaire, symétrie, diffusion, conservation du moment angulaire, sensibilité aux blocs et à l’AMR.

**Acceptation :** vitesse radiale parasite convergente ; profil tangent conservé avec l’ordre attendu ; aucune empreinte non convergente des patches.

---

### EU-06 — Préservation exacte d’un écoulement uniforme

**Priorité : P0**

Exécuter un vrai cas Euler complet avec état uniforme, vitesse non nulle, frontières périodiques, plusieurs blocs, plusieurs rangs et plusieurs regrids AMR forcés.

Ce n’est pas un test unitaire : il traverse l’assemblage complet, les sorties, le temps, les échanges, le regrid et le restart.

**Oracle :** état initial exact.

**Acceptation :** état uniforme conservé à l’arrondi ; aucun bruit produit aux frontières de blocs, aux interfaces coarse-fine ou lors du restart.

**Bugs ciblés :** ghost non initialisé, flux incohérents, erreur de métrique, prolongation non constante, source parasite, mauvaise conversion d’état.

---

## 13. Solveur de Poisson et champs elliptiques

### PO-01 — Poisson périodique trigonométrique

**Priorité : P0**

Choisir en 3D :

$$
\phi(x,y,z)=\sin(2\pi x)\sin(4\pi y)\cos(2\pi z).
$$

Alors :

$$
-\Delta\phi=\left[(2\pi)^2+(4\pi)^2+(2\pi)^2\right]\phi.
$$

Réduire naturellement en 1D et 2D.

**Variantes :**

- 1D, 2D, 3D ;
- GMG uniforme ;
- FFT périodique ;
- AMR composite ;
- deux niveaux totaux pour le cas minimal ;
- trois niveaux totaux pour la baseline complète ;
- plusieurs placements d’interface ;
- plusieurs tolérances du solveur.

**Diagnostics :** erreurs de $\phi$, de chaque composante de $E=-\nabla\phi$, défaut de Gauss, résidu, V-cycles, ordre, erreur coarse-fine.

**Acceptation :** ordre au moins 1.8 sur $\phi$ et sur $E$ si le gradient annoncé est d’ordre deux ; FFT et GMG convergent vers la même solution discrète uniforme après retrait du mode moyen.

---

### PO-02 — Poisson avec Dirichlet non homogène

**Priorité : P0**

Exemple :

$$
\phi(x,y)=e^x\sin(2\pi y)+x^2y,
$$

avec $f=-\Delta\phi$ calculé analytiquement et valeurs exactes imposées sur toutes les frontières.

**Composants testés :** ghost cells physiques, signe de l’opérateur, coins, positionnement cell-centered/node-centered, stencil près des bords, AMR touchant la frontière.

**Variantes :** domaine carré puis rectangle, patch intérieur puis patch touchant un bord et un coin.

**Acceptation :** ordre global et ordre dans une bande de quatre cellules près des frontières ; pas de réduction à l’ordre un au bord.

---

### PO-03 — Poisson avec Neumann et espace nul

**Priorité : P0**

Choisir :

$$
\phi(x,y)=\cos(2\pi x)+\cos(4\pi y),
$$

qui donne une dérivée normale nulle sur le bord de $[0,1]^2$.

La solution est définie à une constante près. Comparer :

$$
\phi_h-\langle\phi_h\rangle
$$

à la référence recentrée.

**Composants testés :** mode nul, compatibilité du second membre, réduction globale MPI, normalisation, frontières Neumann.

**Acceptation :** convergence après retrait de la moyenne ; détection explicite d’un second membre incompatible ; absence de dérive arbitraire du mode constant.

---

### PO-04 — Huang–Greengard sur AMR

**Priorité : P1**

Adapter le test Huang–Greengard utilisé par FLASH : somme de sources lisses localisées, potentiel analytique connu ou calculable, raffinement concentré autour de plusieurs structures.

**Objectif :** exercer un vrai problème multiscale avec plusieurs patches, interfaces non triviales et plusieurs niveaux.

**Diagnostics :** erreurs potentiel/champ, convergence, V-cycles, résidu par niveau, flux normal coarse-fine, dépendance au placement des patches, coût par cellule feuille.

**Acceptation :** ordre attendu, convergence robuste du multigrille et absence de source ou force parasite localisée aux interfaces.

---

### PO-05 — Oracle croisé FFT contre multigrille géométrique

**Priorité : P0**

Sur grille uniforme périodique, résoudre exactement le même second membre avec :

- le solveur spectral FFT ;
- le multigrille géométrique.

**Oracle principal :** solution analytique de `PO-01`. L’accord FFT/GMG est un oracle secondaire, jamais l’unique preuve.

**Diagnostics :** différence champ-à-champ après harmonisation du mode moyen, erreurs analytiques, résidus, coût, invariance MPI.

**Acceptation :** mêmes ordres ; différence compatible avec l’erreur de discrétisation du GMG et les conventions de gradient ; aucune divergence de signe.

---

### PO-06 — Ordre du gradient et placement coarse-fine

**Priorité : P0**

Réutiliser `PO-01` avec des interfaces placées sur :

- maximum de $\phi$ ;
- zéro de $\phi$ ;
- maximum de $|\nabla\phi|$ ;
- maximum de $|\Delta\phi|$.

**Objectif :** dissocier la qualité du solveur de potentiel et celle de la reconstruction du champ.

**Diagnostics :** ordre de $\phi$, ordre de $E$, composante normale et tangentielle de $E$, saut coarse-fine, force appliquée à un état test.

**Acceptation :** aucune configuration ne doit réduire systématiquement $E$ à l’ordre un si PoPS annonce un champ d’ordre deux.

---

### PO-07 — Sensibilité à la tolérance elliptique

**Priorité : P0**

Pour chaque résolution, répéter le solve avec plusieurs tolérances, par exemple $10^{-6}$, $10^{-8}$, $10^{-10}$, $10^{-12}$ en relatif, sous réserve de la normalisation interne.

**Objectif :** séparer erreur algébrique et erreur de discrétisation.

**Diagnostics :** erreur analytique, résidu, itérations et coût en fonction de la tolérance.

**Acceptation :** l’erreur se stabilise sur le plateau de discrétisation ; le plateau décroît avec la résolution ; la tolérance de production est assez stricte pour ne pas polluer les courbes d’ordre.

---

## 14. Couplage Euler–Poisson, plasma et systèmes multi-fluides

### CP-01 — Solution manufacturée Euler–Poisson complète

**Priorité : P0**

Ce cas est le test central du chemin couplé :

```text
état conservé
→ densité de charge
→ assemblage du second membre elliptique
→ résolution de Poisson
→ calcul de phi et E
→ lecture du champ par les sources
→ mise à jour de l’état à chaque stage temporel
```

Choisir un potentiel analytique dépendant de l’espace et du temps, par exemple :

$$
\phi(x,t)=A\cos(kx-\omega t).
$$

Déduire une densité de charge compatible avec la convention exacte de PoPS :

$$
\mathcal{D}\phi=\rho_q(U).
$$

La configuration électrostatique canonique 1D conserve également la relation fermée suivante. Pour des ions fixes, des électrons de charge $q_e=-e$ et la convention :

$$
-\epsilon_0\,\partial_{xx}\phi=e(n_i-n_e),
$$

le choix

$$
\phi(x,t)=A\cos(kx-\omega t)
$$

donne :

$$
n_e(x,t)=n_i-\frac{\epsilon_0k^2}{e}\,\phi(x,t).
$$

Cette expression est un oracle canonique obligatoire. Si l’opérateur elliptique résolu par PoPS emploie une autre convention de signe ou de normalisation, le README du cas doit dériver la transformation exacte entre cette convention et le système résolu ; il est interdit de corriger le signe a posteriori dans le script d’analyse.

Choisir ensuite $u_e(x,t)$ et $p_e(x,t)$ positifs et non triviaux, puis calculer les sources manufacturées de continuité, de moment et d’énergie. La densité, la vitesse et la pression sont choisies positives. Les sources manufacturées sont ajoutées dans les équations hyperboliques afin que l’état analytique soit solution exacte. Pour le premier niveau de preuve, aucune source artificielle ne doit être ajoutée dans Poisson : le potentiel et la charge doivent satisfaire réellement l’équation elliptique choisie.

**Variantes :**

- 1D et 2D ;
- périodique puis Dirichlet ;
- uniforme ;
- AMR statique ;
- AMR dynamique prescrite ;
- SSPRK2 et SSPRK3 ;
- champ recalculé à chaque stage puis contrôle négatif de développement avec champ gelé ;
- une et deux espèces.

**Diagnostics :** erreurs sur toutes les variables fluides, $\phi$, $E$, charge, force $q n E$, énergie fluide et énergie de champ, résidu Poisson, défaut de Gauss, ordre spatial/temporel/global.

**Acceptation :** ordre annoncé pour chaque variable et pour le champ ; aucune réduction d’ordre lorsque le couplage est activé ; force correcte en signe, phase et amplitude.

**Bugs détectés :** mauvais signe de charge, $E=+\nabla\phi$ au lieu de $-\nabla\phi$, champ évalué au mauvais temps, oubli d’un stage RK, source d’énergie incohérente, mauvaise interpolation du champ entre niveaux AMR.

---

### CP-02 — Onde de Langmuir froide

**Priorité : P0**

Utiliser un fluide électronique froid linéarisé autour d’un équilibre uniforme avec ions immobiles. La fréquence plasma est :

$$
\omega_{pe}=\sqrt{\frac{n_0e^2}{m_e\epsilon_0}},
$$

ou son équivalent nondimensionné dans PoPS.

Deux oracles complémentaires sont obligatoires.

**Oracle principal générique :** construire le vecteur propre du système PoPS linéarisé selon les conventions de signe réellement résolues :

$$
\delta U(x,t)=\Re\left[r\,e^{ikx-i\omega_{pe}t}\right].
$$

Cet oracle reste valable lorsque les variables, normalisations ou conventions changent.

**Oracle secondaire fermé 1D :** sous les conventions $q_e=-e$, ions fixes et $\partial_xE=e(n_i-n_e)/\epsilon_0$, utiliser :

$$
n_e(x,t)=n_0+A\cos(kx)\cos(\omega_{pe}t),
$$

$$
u_e(x,t)=\frac{A\omega_{pe}}{n_0k}\sin(kx)\sin(\omega_{pe}t),
$$

$$
E(x,t)=-\frac{eA}{\epsilon_0k}\sin(kx)\cos(\omega_{pe}t),
$$

et, après fixation de la jauge,

$$
\phi(x,t)=-\frac{eA}{\epsilon_0k^2}\cos(kx)\cos(\omega_{pe}t).
$$

Les signes doivent être redérivés dans le README à partir du contrat exact de PoPS. Le cas doit vérifier que l’eigenmode générique et l’oracle fermé décrivent la même solution après conversion des conventions ; une divergence de signe constitue un échec du couplage.

**Variantes :**

- amplitudes $10^{-8}$, $10^{-6}$, $10^{-4}$ pour confirmer le régime linéaire ;
- nombres d’onde 1, 2, 4 et 8 ;
- une, cinq, vingt périodes ;
- grille uniforme ;
- AMR statique avec interface sur maximum puis zéro de $E$ ;
- subcycling ;
- 1D puis onde oblique 2D ;
- SSPRK2, SSPRK3 et schéma couplé de production.

**Diagnostics :** fréquence, phase, amplitude, harmoniques, erreur sur $n$, $u$, $\phi$, $E$, défaut de Gauss, charge totale, énergie cinétique et électrostatique, ordre temporel. Les sorties obligatoires incluent $E_{\omega}$ et $H_2$ définis en section 7.7.

**Acceptation :** fréquence convergente vers $\omega_{pe}$ à l’ordre attendu ; phase correcte ; absence de dérive de charge ; échange d’énergie cohérent ; maintien de l’ordre avec AMR.

**Référence externe :** les suites WarpX utilisent des ondes de Langmuir comme cas de vérification plasma. PoPS doit cependant comparer à son propre modèle fluide, non à la dynamique cinétique complète de WarpX.

---

### CP-03 — Onde de Langmuir chaude et relation de dispersion

**Priorité : P0**

Pour une fermeture fluide donnée, la dispersion linéaire prend typiquement la forme :

$$
\omega^2=\omega_{pe}^2+c_e^2k^2,
$$

avec un coefficient $c_e$ dépendant de la fermeture isotherme, polytropique ou énergétique réellement implémentée.

**Méthode :** dériver automatiquement la matrice linéarisée du modèle PoPS, calculer ses valeurs propres, initialiser un eigenmode et mesurer $\omega_{\mathrm{num}}(k)$.

**Sweep :** $kL/(2\pi)=1,2,4,8$, avec au moins 16 à 32 cellules par longueur d’onde sur les points utilisés pour le gate.

**Diagnostics :** courbe de dispersion, erreur relative de fréquence, amortissement numérique, contamination modale, ordre spatial et temporel.

**Acceptation :** la courbe numérique converge vers la dispersion du modèle exact. Une seule longueur d’onde n’est pas suffisante pour valider le couplage pression–Poisson.

---

### CP-04 — Onde électrostatique oblique 2D et 3D

**Priorité : P1**

Construire un eigenmode avec $\mathbf{k}$ non aligné, par exemple $(1,2)$ ou $(1,2,3)$. Le champ électrique, la vitesse et la densité possèdent plusieurs composantes non nulles.

**Composants ciblés :** gradient multidimensionnel, divergence de charge, coins de ghost cells, couplage des composantes, AMR face/arête/coin.

**Variantes :** permutations d’axes, plusieurs découpages MPI, patch AMR traversé par le front de phase.

**Acceptation :** fréquence et phase indépendantes de l’orientation à l’ordre de discrétisation près ; pas de composante transverse parasite non convergente.

---

### CP-05 — Générateur générique d’eigenmodes multi-fluides

**Priorité : P0**

Pour le modèle exact assemblé par PoPS, linéariser autour d’un équilibre uniforme et passer en Fourier :

$$
\partial_t\widehat U=M(k)\widehat U.
$$

Calculer :

$$
M(k)r_j=\lambda_jr_j,
$$

puis initialiser :

$$
U(x,0)=\bar U+\epsilon\Re(r_je^{ikx}).
$$

La référence est :

$$
U(x,t)=\bar U+\epsilon\Re(r_je^{ikx+\lambda_jt}).
$$

**Modes à couvrir selon le modèle :** acoustiques électroniques et ioniques, plasma, ion-acoustic, modes amortis, modes collisionnels, modes magnétisés.

**Objectif :** éviter de coder à la main une formule différente pour chaque nouvelle composition. Le générateur devient l’oracle commun des systèmes linéarisables.

**Acceptation :** chaque mode annoncé est propagé ou amorti avec la valeur propre correcte ; contamination des autres modes convergente ; même résultat pour le chemin natif et le chemin DSL.

---

### CP-06 — Onde ion-acoustique

**Priorité : P1, conditionnelle au modèle**

Construire l’eigenmode correspondant à l’onde ion-acoustique du modèle deux-fluides ou ions + électrons choisi. La dispersion et la vitesse dépendent des températures, masses, charges et fermetures utilisées.

**Composants testés :** deux espèces, échelles de temps différentes, système Poisson commun, neutralité de fond, multirate éventuel.

**Diagnostics :** fréquence, vitesse de phase, phase relative entre espèces, quasi-neutralité, énergie, ordre.

**Acceptation :** accord avec l’eigenvalue du système linéarisé exact, pas avec une formule de manuel utilisant d’autres hypothèses.

---

### CP-07 — Équilibre pression–champ électrique

**Priorité : P0**

Construire un état stationnaire satisfaisant :

$$
\nabla p_s=q_sn_sE,
\qquad E=-\nabla\phi.
$$

Pour une espèce isotherme $p_s=T_sn_s$, choisir une densité positive $n_s(x)$ et déduire :

$$
\phi(x)=-\frac{T_s}{q_s}\ln n_s(x)+C,
$$

avec la densité de fond ou la charge fixe nécessaire pour satisfaire Poisson.

**Oracle :** $u_s=0$ et état stationnaire.

**Variantes :** uniforme, AMR, interface coarse-fine dans une zone de fort gradient, plusieurs pas de temps et plusieurs intégrateurs.

**Diagnostics :** vitesse parasite maximale, dérive de densité, défaut d’équilibre force–pression, énergie, erreur coarse-fine.

**Acceptation :** vitesse parasite convergente vers zéro ; aucun jet produit à l’interface AMR ; ordre de préservation cohérent avec la discrétisation.

**Bugs détectés :** champ calculé au mauvais temps, force non centrée, signe de charge, gradient pression/champ évalués à des positions incompatibles.

---

### CP-08 — Accélération sous champ électrique uniforme

**Priorité : P0**

Utiliser une configuration où le champ uniforme est connu et où la dynamique spatiale est absente ou exactement uniforme. Pour une espèce de charge $q$ et masse $m$ :

$$
\frac{du}{dt}=\frac{q}{m}E_0,
$$

$$
u(t)=u_0+\frac{qE_0}{m}t.
$$

L’énergie cinétique possède également une expression exacte.

**Objectif :** isoler le signe et le centrage temporel de la source électrique dans un run complet.

**Variantes :** charge positive/négative, Forward Euler, RK2, RK3, source explicite et implicite si disponible.

**Acceptation :** ordre temporel annoncé sur vitesse et énergie ; accélérations opposées pour charges opposées ; aucun changement de densité pour l’état uniforme périodique.

---

### CP-09 — Écran de Debye linéarisé

**Priorité : P2, conditionnelle à la fermeture**

Lorsque le modèle possède une fermeture compatible avec une réponse de Boltzmann linéarisée, utiliser une perturbation de charge fixe et comparer le potentiel à une solution de type Helmholtz :

$$
(-\Delta+\lambda_D^{-2})\phi=f.
$$

**Objectif :** tester le couplage entre réponse de densité et champ dans un régime stationnaire.

Ce cas ne doit pas être activé si les équations PoPS ne conduisent pas réellement à cette limite.

---

### CP-10 — Modes de Jeans stables et instables

**Priorité : P1**

Pour un Euler auto-gravitant linéarisé, la dispersion typique est :

$$
\omega^2=c_s^2k^2-4\pi G\rho_0,
$$

avec les conventions exactes de nondimensionnement de PoPS.

**Deux régimes :**

- $k>k_J$ : onde stable ;
- $k<k_J$ : croissance exponentielle.

**Diagnostics :** fréquence ou taux de croissance, phase densité/vitesse/potentiel, signe de l’accélération, conservation de masse, dépendance à l’amplitude.

**Acceptation :** bon signe physique et convergence vers la valeur propre linéaire. Ce test distingue une gravité attractive d’un couplage plasma répulsif en changeant uniquement le contrat de signe.

---

### CP-11 — Mode diocotron linéaire

**Priorité : P1, catégorie reproduction/validation selon la référence**

Réutiliser le cas diocotron existant, mais formaliser une campagne dédiée :

- taux de croissance analytique ou publié ;
- plusieurs nombres de mode ;
- régime linéaire démontré par un sweep d’amplitude ;
- uniforme puis AMR ;
- plusieurs résolutions ;
- potentiel et vitesse $E\times B$ diagnostiqués ;
- symétrie polaire et position des patches.

**Diagnostics :** taux de croissance, erreur relative, masse, spectre modal, symétrie, coût AMR/uniforme.

**Acceptation :** le niveau de preuve dépend de l’oracle. Une correspondance d’invariant seule reste une validation de code, pas une reproduction quantitative du papier.

---

### CP-12 — Annulation de charge et équilibre multi-espèces

**Priorité : P0**

Initialiser des espèces dont la charge totale s’annule analytiquement :

$$
\sum_s q_sn_s=0,
$$

avec états uniformes ou profils compensés. Le potentiel doit rester constant et $E=0$ après fixation de la jauge.

**Variantes :** deux puis trois espèces, masses très différentes, AMR, MPI, ordre différent d’ajout des espèces, DSL/native.

**Composants testés :** accumulation du second membre Poisson, ordre des réductions, index d’espèce, cancellation, mode moyen, champs owner-qualified.

**Acceptation :** champ compatible avec zéro et aucun moment parasite. Une erreur qui dépend de l’ordre d’ajout des espèces signale une accumulation ou un binding incorrect.

---

## 15. Intégration temporelle, splitting, collisions et régimes raides

### TM-01 — Convergence temporelle pure

**Priorité : P0**

Utiliser successivement :

- `TR-01` ;
- `EU-01` ;
- `CP-02` ;
- `TM-03`.

Fixer une grille très fine puis exécuter $\Delta t$, $\Delta t/2$, $\Delta t/4$, $\Delta t/8$, $\Delta t/16$.

**Objectif :** vérifier séparément chaque intégrateur sur transport, Euler, Poisson couplé et source locale.

**Acceptation :** seuils de la section 8.2. Un RK3 d’ordre trois sur advection mais d’ordre un sur Langmuir indique que le champ ou la source n’est pas recalculé à chaque stage.

---

### TM-02 — Strang avec opérateurs non commutatifs

**Priorité : P0**

Utiliser un système à deux composantes :

$$
\partial_t q+A\partial_xq=Bq,
$$

avec :

$$
A=\begin{pmatrix}a_1&0\\0&a_2\end{pmatrix},
\qquad
B=\begin{pmatrix}-\nu&\nu\\\nu&-\nu\end{pmatrix},
\qquad a_1\neq a_2.
$$

Ainsi $AB\neq BA$. Pour un mode de Fourier :

$$
q(x,t)=\Re\left[e^{ikx}e^{(-ikA+B)t}\widehat q_0\right].
$$

La matrice exponentielle $2\times2$ fournit l’oracle exact.

**Variantes :** Lie transport→collision, Lie collision→transport, Strang T/2–C–T/2, Strang C/2–T–C/2, uniforme et AMR.

**Acceptation :** Lie ordre un ; Strang ordre deux. Un problème où les opérateurs commutent est interdit pour ce gate, car il pourrait masquer un mauvais ordonnancement.

---

### TM-03 — Relaxation collisionnelle exacte à deux espèces

**Priorité : P0**

$$
\rho_1\frac{du_1}{dt}=K(u_2-u_1),
\qquad
\rho_2\frac{du_2}{dt}=K(u_1-u_2).
$$

La vitesse barycentrique :

$$
V=\frac{\rho_1u_1+\rho_2u_2}{\rho_1+\rho_2}
$$

reste constante et la différence :

$$
w(t)=w(0)e^{-\lambda t},
\qquad
\lambda=K\left(\frac{1}{\rho_1}+\frac{1}{\rho_2}\right).
$$

**Régimes :** $\lambda\Delta t\ll1$, $\lambda\Delta t\approx1$, $\lambda\Delta t\gg1$.

**Diagnostics :** ordre temporel, conservation du moment total, monotonie, overshoot, stabilité, coût implicite.

**Acceptation :** bon taux exponentiel ; conservation ; absence d’oscillation non physique ; stabilité du chemin implicite/IMEX dans le régime raide annoncé.

---

### TM-04 — Rotation de Larmor ou oscillateur magnétisé exact

**Priorité : P1**

Pour un champ magnétique uniforme et sans gradients spatiaux, la vitesse transverse suit une rotation exacte :

$$
\frac{d}{dt}
\begin{pmatrix}u_x\\u_y\end{pmatrix}
=
\begin{pmatrix}0&\omega_c\\-\omega_c&0\end{pmatrix}
\begin{pmatrix}u_x\\u_y\end{pmatrix}.
$$

La solution est donnée par une matrice de rotation.

**Variantes :** explicite, implicite, Schur, IMEX ; $\omega_c\Delta t$ de $10^{-2}$ à $10^2$ ; charges opposées.

**Diagnostics :** phase, norme de vitesse, énergie, ordre, stabilité, limite raide.

**Acceptation :** phase et amplitude correctes ; conservation de la norme pour le schéma qui l’annonce ; stabilité sans faux amortissement incontrôlé dans le domaine de conception.

---

### TM-05 — Limite asymptotic-preserving

**Priorité : P1, conditionnelle au schéma AP**

Choisir un problème possédant une limite réduite connue lorsque $\epsilon\to0$ ou $\omega_p\Delta t\gg1$. Exécuter une suite $\epsilon=1,10^{-1},10^{-2},10^{-3},10^{-4}$ à pas de temps macroscopique fixé.

**Diagnostics :** erreur par rapport à la limite, stabilité, coût, conservation, dépendance à $\epsilon$.

**Acceptation :** solution convergeant vers la limite sans contrainte explicite résiduelle sur le pas de temps ; comparaison à une référence explicite résolue sur les petites échelles pour quelques points.

---

### TM-06 — Multirate et sous-pas d’espèces

**Priorité : P1**

Utiliser deux sous-systèmes avec vitesses caractéristiques différentes et oracle linéaire `CP-05` ou `TM-02`.

**Sweep :** ratios de sous-pas 1, 2, 4, 8 et 16 ; ordre inverse des blocs ; synchronisation à des temps intermédiaires.

**Diagnostics :** ordre global, phase de chaque espèce, conservation échangée, coût, erreur à la synchronisation.

**Acceptation :** maintien de l’ordre annoncé ; aucun biais dépendant de l’ordre d’enregistrement des blocs ; bilan des échanges fermé.

---

### TM-07 — Recalcul du champ à chaque stage RK

**Priorité : P0**

Ce gate est dérivé de `CP-01` et `CP-02`. Instrumenter le runtime pour enregistrer :

- temps de chaque stage ;
- état utilisé pour former la charge ;
- identifiant de la résolution elliptique ;
- temps associé au champ lu par la source.

**Acceptation :** une résolution ou mise à jour de champ existe à chaque stage requis par le programme temporel ; l’ordre temporel du cas couplé correspond à l’intégrateur annoncé.

Le test ne doit pas dépendre uniquement du nombre d’appels : la convergence temporelle reste la preuve principale.

---

### TM-08 — Symétrie et réversibilité de l’ordonnancement

**Priorité : P1**

Pour un système réversible sans dissipation physique, avancer de $T$, inverser les vitesses ou le signe temporel selon le modèle, puis revenir à zéro.

**Objectif :** détecter un ordre d’opérateurs asymétrique, une source évaluée au mauvais instant ou un état intermédiaire non restauré.

**Diagnostics :** erreur de retour, énergie, ordre en $\Delta t$, différences entre permutations de programme.

**Acceptation :** erreur de retour convergente à l’ordre du programme symétrique ; pas de biais linéaire en $\Delta t$ pour Strang.

---

## 16. Campagnes AMR transverses

Les cas `AM-*` ne définissent pas toujours une nouvelle physique. Ils imposent une campagne AMR normalisée sur des oracles déjà définis. Cette organisation évite de dupliquer les équations tout en garantissant que chaque mécanisme AMR est explicitement couvert.

### AM-01 — Onde traversant une interface coarse-fine statique

**Priorité : P0**

Base : `TR-01`, `EU-01` puis `CP-02`.

L’onde doit :

1. partir dans une zone coarse ;
2. entrer dans une zone fine ;
3. traverser entièrement le patch ;
4. ressortir vers le coarse ;
5. franchir une frontière périodique ;
6. recommencer plusieurs fois.

**Comparaisons :** `U-C`, `U-F`, `A-S0`, `A-S2`, subcycling ratio 4 si supporté, puis `amr_subcycling_no_reflux` comme contrôle négatif de développement.

**Diagnostics :** ordre global, phase avant/après interface, réflexion numérique, transmission, conservation, $E_{cf}$, erreur après chaque traversée.

**Acceptation :** pas de réflexion coarse-fine d’ordre inférieur ; maintien de l’ordre ; conservation ; même vitesse de phase dans les zones coarse et fine à l’erreur de discrétisation près.

---

### AM-02 — Patch mobile prescrit par la solution exacte

**Priorité : P0**

Base : `TR-02` ou `EU-02`.

Le patch suit une trajectoire calculée à partir de la solution exacte, indépendamment de la solution numérique. Cette configuration sépare l’erreur de regrid de l’erreur du critère de tagging.

Deux campagnes sont obligatoires :

1. une campagne de convergence avec un nombre de regrids fixé par la trajectoire physique ;
2. une campagne de stress avec **256 cycles refine/coarsen**, étendue à **512 cycles** pour les releases, à résolution et solution lisse fixées.

Le stress test doit enregistrer l’erreur et les invariants en fonction du numéro de cycle afin de révéler une dérive faible mais cumulative. Il ne doit pas être mélangé aux mesures de performance ni utilisé seul comme preuve d’ordre.

**Mesures à chaque regrid :**

- erreur immédiatement avant ;
- erreur immédiatement après ;
- variation des invariants ;
- volume raffiné/créé/supprimé ;
- coût de prolongation, restriction et migration.

**Acceptation :** saut d’erreur convergent ; aucune dérive cumulative de masse ; résultat comparable à `U-F` dans la région fine ; sur 256 puis 512 cycles, aucune dérive linéaire non expliquée des invariants et aucune croissance d’erreur indépendante de la résolution.

---

### AM-03 — Raffinement dynamique piloté par tagging

**Priorité : P0**

Base : `TR-02`, `TR-03`, `EU-02`, `RB-05`.

Tester séparément :

- seuil sur gradient ;
- seuil sur seconde différence ;
- estimateur d’erreur, s’il existe ;
- buffers de 1, 2, 4 et 8 cellules ;
- hystérésis de raffinement/déraffinement ;
- clustering et tailles minimales/maximales de patches.

**Diagnostics :** couverture de la région d’intérêt, taux de faux positifs/faux négatifs par rapport à une région exacte ou de référence, fraction raffinée, nombre de patches, oscillation du tagging, erreur/coût.

**Acceptation :** le tagging ne doit pas modifier l’ordre de la solution ; aucune oscillation refine/coarsen à chaque pas sans hystérésis justifiée ; les structures à suivre restent dans la zone fine avec le buffer annoncé.

---

### AM-04 — Subcycling temporel

**Priorité : P0**

Même maillage et même temps final :

- pas global sans subcycling ;
- subcycling ratio 2 ;
- subcycling ratio 4 si l’architecture le permet ;
- plusieurs niveaux imbriqués.

**Bases :** `TR-01`, `EU-01`, `PO-01` couplé via `CP-01`, puis `CP-02`.

**Composants ciblés :** interpolation temporelle coarse→fine, synchronisation des niveaux, flux registers, champ elliptique aux temps intermédiaires.

**Acceptation :** ordre temporel global conservé ; aucune dérive de conservation ; solution subcyclée convergeant vers la solution sans subcycling.

**Signature de bug :** ordre deux sans subcycling et ordre un avec subcycling indique en priorité un état coarse interpolé au mauvais temps ou une synchronisation de champ incorrecte.

---

### AM-05 — Fréquence de regridding

**Priorité : P1**

Exécuter le même cas avec regrid tous les 1, 2, 4, 8 et 16 pas.

**Diagnostics :** erreur, conservation, nombre de patches, cellules feuilles, coût de regrid, coût total, amplitude des sauts de solution.

**Acceptation :** aucune dérive systématique proportionnelle au nombre de regrids ; l’erreur physique doit tendre vers une limite lorsque la fréquence devient suffisante.

---

### AM-06 — Nombre total de niveaux et capacité du provider

**Priorité : P0 jusqu’à trois niveaux totaux ; P1 capability-gated au-delà**

Base : `TR-01`, `PO-01`, `CP-02`.

Configurations :

- uniforme, niveau 0 seul ;
- `A-2L` : niveaux totaux `0+1`, ratio 2, smoke minimal d’une interface ;
- `A-3L` : niveaux totaux `0+1+2`, ratios `(2, 2)`, baseline d’acceptation publique actuelle ;
- `A-4L` : niveaux totaux `0+1+2+3`, capability-gated ;
- ratios anisotropes uniquement sous capability gate explicite.

Pour chaque configuration acceptée, les structures traversent successivement toutes les interfaces et un coin où la topologie des patches change. Une configuration non supportée doit échouer avant artefact avec l’évidence `requested_level_count`/`supported_level_count`; le runner ne réduit jamais le nombre de niveaux.

**Acceptation :** conservation globale, ordre attendu dans chaque région, pas d’accumulation d’erreur à chaque niveau, résidu elliptique contrôlé sur la hiérarchie composite et refus déterministe des niveaux non supportés.

---

### AM-07 — AMR contre grille uniforme fine équivalente

**Priorité : P0**

Comparer :

- grille uniforme à $h$ ;
- AMR base $h$ avec zone locale $h/2$ ;
- uniforme $h/2$ partout.

Dans la zone couverte par le niveau fin, comparer directement l’AMR à la grille uniforme fine. Dans la zone coarse, comparer à la référence analytique.

**Acceptation :** même ordre ; amplitude d’erreur de la zone fine comparable à l’uniforme fine ; pas de saut visible ou de couche persistante à l’interface.

---

### AM-08 — Sweep de placement des interfaces

**Priorité : P0**

Base : `TR-01`, `PO-01`, `EU-01`, `CP-02`.

Déplacer le patch par fractions de longueur d’onde et par incréments d’une cellule coarse. Le résultat doit couvrir maxima, minima, zéros et forts gradients.

**Diagnostics :** enveloppe min/max de l’erreur sur tous les placements, position du pire cas, ordre du pire cas.

**Acceptation :** le pire placement conserve l’ordre minimal. Le rapport erreur max/min ne doit pas diverger avec le raffinement.

---

### AM-09 — Conservation et reflux intégrés

**Priorité : P0**

Base : advection conservative, Euler lisse puis Sod.

Le calcul doit faire traverser une quantité conservative par une interface coarse-fine pendant de nombreux pas et plusieurs regrids.

**Diagnostics :** masse, moment, énergie, flux intégrés coarse et somme espace-temps des flux fins, correction de reflux appliquée, bilan avant/après synchronisation.

**Acceptation :** bilan global fermé ; aucune dent de scie synchronisée avec les pas coarse ; correction de reflux cohérente avec le défaut de flux mesuré.

**Contrôle négatif réservé au développement :** compiler une variante instrumentée où le reflux est désactivé. Le test doit alors échouer nettement. Cette variante ne constitue pas une configuration utilisateur et ne doit pas rester activable silencieusement en production.

---

### AM-10 — Poisson composite multilevel

**Priorité : P0**

Base : `PO-01`, `PO-04`, `PO-06`.

**Configurations :** deux niveaux totaux pour le smoke minimal ; trois niveaux totaux pour la baseline complète ; quatre niveaux totaux uniquement sous capability gate ; patches imbriqués et disjoints, source localisée traversant une interface, patch touchant une frontière physique.

**Diagnostics :** résidu composite, résidu par niveau, erreur de potentiel/champ, flux coarse-fine, nombre de cycles, invariance au nombre de rangs.

**Acceptation :** convergence multigrille robuste et ordre du potentiel/champ. Une résolution indépendante par niveau sans synchronisation correcte doit être détectée par le défaut de flux et de Gauss.

---

### AM-11 — Synchronisation Euler–Poisson sur AMR

**Priorité : P0**

Base : `CP-01`, `CP-02`, `CP-07`.

Vérifier explicitement :

- charge composée uniquement sur cellules feuilles ;
- restriction cohérente des densités ;
- champ composite cohérent ;
- source appliquée au bon niveau et au bon temps ;
- correction après reflux ;
- re-solve elliptique après synchronisation lorsque le programme le requiert.

**Acceptation :** ordre global du cas couplé, défaut de Gauss contrôlé, absence de force parasite coarse-fine, conservation de charge et d’énergie selon le modèle.

---

### AM-12 — Symétrie et forme des patches

**Priorité : P1**

Base : vortex isentropique, Sedov décentré, onde radiale et diocotron.

Faire varier :

- aspect ratio des patches ;
- orientation des découpages ;
- taille maximale de bloc ;
- ordre de clustering ;
- décomposition MPI.

**Diagnostics :** erreurs de symétrie, harmonique cartésienne d’ordre quatre, position des patches, corrélation entre patch map et erreur.

**Acceptation :** aucune structure physique ne doit être déterminée par la forme des patches au-delà de l’erreur de discrétisation convergente.

---

## 17. Chocs, discontinuités, positivité et robustesse

Les cas `RB-*` ne reçoivent pas de gate d’ordre deux global. Ils vérifient la bonne solution faible, la conservation, la positivité, la symétrie et la robustesse des interfaces AMR.

### RB-01 — Sod avec traversée d’interface AMR

**Priorité : P0**

Paramètres classiques, $\gamma=1.4$ :

$$
(\rho,u,p)_L=(1,0,1),
\qquad
(\rho,u,p)_R=(0.125,0,0.1).
$$

Temps final standard : $t=0.2$ sur $[0,1]$, à ajuster seulement avec justification.

Placer la discontinuité à une position non alignée, par exemple $x_0=0.37$. Construire le patch pour que le choc, le contact et la raréfaction rencontrent des interfaces coarse-fine à des instants différents.

**Variantes :** axes $x/y/z$, diagonale 2D, HLL/HLLC/Rusanov, uniforme et AMR dynamique.

**Oracle :** solveur exact de Riemann avec moyennes de cellules.

**Diagnostics :** $L^1$, position du choc/contact, états intermédiaires, conservation, minima de $\rho$ et $p$, overshoot, erreur ajoutée lors de chaque traversée.

**Acceptation :** bonne position à une ou deux cellules fines près selon résolution, conservation, positivité, absence d’oscillations croissantes.

**Référence externe :** suite de vérification Castro.

---

### RB-02 — Double raréfaction et quasi-vide

**Priorité : P0**

$$
(\rho,u,p)_L=(1,-2,0.4),
\qquad
(\rho,u,p)_R=(1,2,0.4),
$$

avec $\gamma=1.4$ et temps final standard proche de 0.15.

**Objectif :** tester positivité, énergie interne, solveurs de Riemann et comportement près du vide.

**Diagnostics :** densité/pression minimales, nombre de corrections de positivité, profil exact, conservation, dépendance au CFL.

**Acceptation :** aucune valeur non physique ; corrections explicites et rares ; convergence vers la solution exacte.

---

### RB-03 — Choc très fort

**Priorité : P1**

Configuration Castro courante :

$$
(\rho,u,p)_L=(1,0,1000),
\qquad
(\rho,u,p)_R=(1,0,0.01),
$$

avec $\gamma=1.4$ et temps final proche de 0.012.

**Objectif :** grands rapports de pression, séparation choc/contact, robustesse HLL/HLLC, AMR autour du choc.

**Acceptation :** positivité, bonne vitesse de choc, conservation et absence d’échec du solveur de Riemann.

---

### RB-04 — Shu–Osher

**Priorité : P1**

Faire interagir un choc avec une perturbation sinusoïdale de densité. Utiliser les paramètres standards de la suite FLASH ou de la littérature, documentés sans modification silencieuse.

**Oracle :** référence uniforme très fine et, si possible, comparaison avec un code reconnu.

**Diagnostics :** profil de densité derrière le choc, amplitude et spectre des oscillations, position du choc, conservation, coût AMR.

**Composants testés :** résolution des petites structures, limiter, WENO, HLL/HLLC, tagging.

**Acceptation :** position correcte, convergence vers la référence et absence de destruction excessive du train d’ondes.

---

### RB-05 — Sedov décentré 2D et 3D

**Priorité : P0**

Explosion forte dans un milieu uniforme, centre décalé par rapport au domaine, aux cellules et aux blocs.

La loi autosimilaire du rayon est :

$$
R(t)=\xi\left(\frac{Et^2}{\rho_0}\right)^{1/(d+2)},
$$

avec la constante $\xi$ correspondant précisément à $\gamma$, la dimension et la convention d’énergie utilisées.

**Variantes :** cylindrique 2D, sphérique 3D, uniforme fine, AMR dynamique, différents placements du centre.

**Diagnostics :** rayon moyen, rayon selon l’angle, profils radiaux, énergie, symétrie, nombre de cellules fines, patch map.

**Acceptation :** rayon et profils convergents ; conservation d’énergie ; anisotropie décroissante ; aucune asymétrie attachée au découpage MPI.

**Référence externe :** configurations Sedov de Castro et FLASH.

---

### RB-06 — Noh 1D, 2D et 3D

**Priorité : P1**

Le problème de Noh possède une solution analytique autosimilaire avec choc convergent et fort échauffement au centre. Il est particulièrement sensible au wall heating et aux erreurs géométriques.

**Variantes :** planar, cylindrique et sphérique ; centre aligné puis décalé ; AMR dynamique.

**Diagnostics :** position du choc, densité post-choc, erreur près du centre, symétrie, conservation, plancher de pression.

**Acceptation :** bonne position du choc et convergence des états ; artefact central quantifié et décroissant.

---

### RB-07 — Implosion de Liska–Wendroff

**Priorité : P1**

Domaine $[0,0.3]^2$, frontières réfléchissantes. États initiaux standards :

- pour $x+y>0.15$ : $\rho=1$, $p=1$ ;
- sinon : $\rho=0.125$, $p=0.14$ ;
- vitesses nulles.

**Objectif :** symétrie exacte par rapport à $x=y$, contacts, jets diagonaux, dépendance aux layouts MPI.

**Diagnostics :** erreur de symétrie, longueur/largeur du jet, conservation, différence entre décompositions 1×N, N×1 et carrées.

**Acceptation :** symétrie compatible avec l’arrondi et la discrétisation ; aucune orientation privilégiée par MPI.

**Référence externe :** test officiel Athena de l’implosion.

---

### RB-08 — Double Mach Reflection

**Priorité : P2**

Utiliser la configuration Woodward–Colella standard : choc Mach 10 incliné, paroi réfléchissante et temps final documenté.

**Objectif :** choc oblique, interactions choc–paroi, jet dense, frontières complexes, AMR dynamique.

**Oracle :** géométrie et contours de référence à haute résolution, pas solution analytique complète.

**Diagnostics :** position du Mach stem, angle, longueur du jet, contours de densité, conservation, coût.

**Acceptation :** morphologie convergente avec la résolution et cohérente avec les références publiées ; pas de prétention d’ordre global.

---

### RB-09 — Blast waves de Woodward–Colella en 1D

**Priorité : P1**

Utiliser le problème standard à deux explosions fortes et parois réfléchissantes.

**Objectif :** interactions multiples de chocs et contacts, robustesse longue, frontières réfléchissantes, résolution des structures fines.

**Oracle :** solution convergée haute résolution et références publiées.

**Diagnostics :** profils de densité/pression/vitesse, positions des fronts, conservation, positivité et dépendance au limiter.

---

## 18. Symétries radiales cartésiennes et extensions polaires

Le runtime public exact-rank de la révision examinée exécute les `System` et `AmrSystem` cartésiens. Les descripteurs de maillage polaire et plusieurs composants numériques polaires existent, mais ils ne constituent pas encore un pipeline public intégré équivalent au cartésien. La politique est donc :

- les tests radiaux construits sur une grille cartésienne restent exécutables et obligatoires ;
- les tests d’algorithmes polaires isolés restent dans `tests/cpp/` tant qu’ils ne passent pas par un `Case` public ;
- les cas intégrés polaires sont enregistrés `capability-gated` ;
- la release vérifie le refus explicite du layout polaire non supporté au lieu de le substituer silencieusement par un cartésien.

### GE-01 — Poisson manufacturé radial en coordonnées polaires

**Priorité : P2, capability-gated tant que le runtime polaire public est absent**

Choisir une solution régulière, par exemple :

$$
\phi(r,\theta)=r^m\cos(m\theta)
$$

sur un anneau excluant l’axe, ou une fonction de Bessel compatible avec les frontières. Déduire analytiquement le Laplacien polaire.

Pour un domaine contenant l’axe, choisir une fonction dont la régularité en $r=0$ est démontrée.

**Composants testés après activation :** métriques, volumes de cellules, aires de faces, facteur $1/r$, conditions à l’axe, gradient radial et azimutal, Poisson AMR.

**Acceptation actuelle :** refus de résolution documenté pour une exécution intégrée non supportée ; tests C++ isolés des opérateurs polaires maintenus. **Acceptation future :** ordre attendu sur $\phi$ et $E$, sans singularité ni réduction d’ordre à l’axe.

---

### GE-02 — Rotation solide exacte d’un scalaire

**Priorité : P2, capability-gated**

Avec une vitesse azimutale $v_\theta=\Omega r$, un profil est tourné rigidement. Après $T=2\pi/\Omega$, il revient exactement à son état initial.

**Variantes futures :** anneau gaussien, perturbation azimutale, centre inclus/exclu, uniforme et AMR.

**Diagnostics :** erreur de retour, masse pondérée par le volume polaire, phase azimutale, symétrie, erreur à la couture périodique en $\theta$.

---

### GE-03 — Onde acoustique radiale en cartésien

**Priorité : P1, exécutable actuellement**

Initialiser une perturbation de faible amplitude radialement symétrique dans un domaine cartésien. Utiliser une solution 1D radiale haute résolution ou une représentation analytique par fonctions de Bessel lorsque les frontières le permettent.

**Objectif :** tester anisotropie cartésienne, propagation radiale, dispersion, réflexion, coins de patches et différence axes/diagonales sans dépendre d’un choc.

**Diagnostics :** fréquence, profil radial, phase, erreur angulaire, ordre, harmonique cartésienne d’ordre quatre et corrélation avec la carte des patches.

**Acceptation :** fréquence et profil convergents ; anisotropie décroissante ; aucune empreinte non convergente des frontières MPI ou coarse-fine.

---

### GE-04 — Même oracle radial en cartésien et en polaire

**Priorité : P2, capability-gated pour la branche polaire**

Choisir un problème radial possédant un oracle analytique, par exemple Poisson avec source gaussienne radiale ou onde acoustique faible. La branche cartésienne est exécutée dès maintenant ; la branche polaire n’entre dans le gate qu’après activation du runtime public correspondant.

**Oracle principal :** solution analytique. L’accord cartésien/polaire reste secondaire.

**Acceptation future :** les deux géométries convergent vers le même oracle et non simplement l’une vers l’autre.

---

### GE-05 — Régularité à l’axe et conservation des volumes polaires

**Priorité : P2, capability-gated**

Après activation du runtime polaire, exécuter un état constant et un état radial régulier avec plusieurs blocs touchant l’axe, AMR et MPI.

**Acceptation actuelle :** les descripteurs et opérateurs isolés refusent ou traitent l’axe conformément à leur contrat. **Acceptation future intégrée :** état constant conservé, flux radial nul à l’axe et aucun terme $1/r$ évalué singulièrement.

---

### GE-06 — Diocotron cartésien : modes, symétrie et AMR

**Priorité : P1**

Le cas diocotron est formulé sur le domaine cartésien public avec une densité annulaire. Cette campagne complète `CP-11` avec :

- plusieurs modes azimutaux ;
- anneaux déplacés par rapport aux patches ;
- deux niveaux AMR totaux pour le smoke minimal ;
- trois niveaux totaux pour la baseline complète ;
- métriques de symétrie polaire calculées en post-traitement ;
- comparaison uniforme fine/AMR ;
- décompositions MPI cartésiennes allongées et équilibrées.

Une implémentation future sur maillage polaire reçoit un identifiant de configuration distinct et ne remplace pas la baseline cartésienne.

**Acceptation :** taux de croissance et spectre modal cohérents, masse conservée et absence d’empreinte des patches dans les modes non excités.

## 19. Infrastructure, parallélisme et reproductibilité

### IF-01 — Invariance au découpage MPI

**Priorité : P0**

Exécuter au minimum `TR-01`, `PO-01`, `CP-02`, `AM-10`, `EU-02` et `RB-07` avec :

- 1, 2, 4, 8, 16 rangs ;
- pour quatre rangs en 2D : layouts canoniques `1×4`, `2×2`, `4×1` ;
- pour quatre rangs en 3D : `1×1×4`, `1×2×2`, `4×1×1` ;
- décompositions allongées et équilibrées supplémentaires ;
- plusieurs ordres d’assignation des blocs ;
- un puis deux nœuds pour la campagne dédiée.

**Diagnostics :** normes analytiques, conservation, symétrie, différence champ-à-champ, position des erreurs, temps maximum de rang.

**Acceptation :** même ordre et mêmes invariants ; aucune structure alignée sur une frontière MPI ; différence numérique expliquée par l’ordre des réductions.

---

### IF-02 — Invariance au nombre de threads Kokkos OpenMP

**Priorité : P0**

Sur un rang, puis sur plusieurs rangs, utiliser 1, 2, 4, 8, 16, 32, 48, 96 et jusqu’au nombre de cœurs physiques disponibles.

**Diagnostics :** résultats, ordre, conservation, déterminisme, races détectées par builds dédiés, performance.

**Acceptation :** résultat scientifique inchangé à la tolérance ; aucune dépendance aux threads ; pas de race ou réduction non initialisée.

---

### IF-03 — Parité des espaces Kokkos et de MPI

**Priorité : P0**

Cas minimaux : `TR-01`, `EU-01`, `PO-01`, `CP-02`, `AM-01`, `AM-10`.

**Comparaisons :**

- Kokkos Serial contre Kokkos OpenMP ;
- Kokkos OpenMP sur un rang contre le même espace Kokkos composé avec MPI ;
- CPU contre GPU ;
- GPU un nœud contre GPU deux nœuds.

**Acceptation :** mêmes ordres et invariants ; différences compatibles avec l’arithmétique flottante ; aucune divergence de branche physique ou de tagging inexpliquée.

---

### IF-04 — Checkpoint et restart

**Priorité : P0**

Comparer :

```text
run continu jusqu’à T
```

à :

```text
run jusqu’à T/2
checkpoint
restart
run jusqu’à T
```

**Variantes obligatoires :**

- uniforme ;
- AMR statique ;
- AMR dynamique ;
- subcycling ;
- Poisson couplé ;
- restart avec même nombre de rangs ;
- restart avec nombre de rangs différent ;
- CPU→CPU et, si le format le permet, CPU→GPU.

**Diagnostics :** différence finale, hiérarchie AMR, temps de niveau, flux registers, compteurs RK, état du solveur elliptique, invariants.

**Acceptation :** même-backend et même-layout bitwise si le mode déterministe le garantit ; sinon équivalence à l’arrondi. Changer le nombre de rangs ne doit pas modifier le résultat physique.

---

### IF-05 — Invariance à la cadence de sortie

**Priorité : P1**

Exécuter le même cas avec sorties :

- désactivées ;
- chaque pas ;
- tous les 2, 10, 50 pas ;
- Catalyst/live visualization activé si disponible.

**Objectif :** détecter un fence manquant, une mutation de buffer, un alias, un ordre de communication modifié ou un champ lu avant synchronisation.

**Acceptation :** même résultat scientifique ; le coût I/O est mesuré séparément.

---

### IF-06 — Mode déterministe et réductions

**Priorité : P1**

Lorsque PoPS expose un mode déterministe :

- répéter exactement le même run plusieurs fois ;
- changer le scheduling des threads ;
- comparer les réductions locales et MPI ;
- conserver le même layout pour le gate bitwise.

**Acceptation :** bitwise uniquement dans le contrat déterministe déclaré. Hors de ce contrat, utiliser des tolérances et vérifier l’absence de divergence amplifiée.

---

### IF-07 — Parité natif, DSL et hybride

**Priorité : P0**

Pour un même modèle mathématique, construire :

- chemin natif ;
- chemin DSL compilé ;
- composition hybride.

Cas : advection, Euler, Euler–Poisson, multi-espèces et une source magnétique.

**Diagnostics :** état initial, flux ponctuels hors run, état final, ordres, champs, invariants, code généré et paramètres résolus.

**Acceptation :** égalité bit à bit lorsque les expressions et l’ordre des opérations sont réellement identiques ; sinon différence au niveau de l’arrondi et mêmes courbes de convergence.

---

### IF-08 — Compilateurs, modes de build et spécialisation native exacte

**Priorité : P1**

Matrice minimale :

- GCC Release ;
- Clang Release ;
- Kokkos Serial et Kokkos OpenMP ;
- Kokkos CUDA/HIP/SYCL disponible ;
- Debug/sanitizers pour petits cas ;
- `POPS_NATIVE_DIM=1`, `2` et `3` dans des builds séparés ;
- build MPI séparé ;
- options vectorisation rapides uniquement si leur contrat flottant est documenté.

Pour chaque artefact, exécuter `pops.doctor()`, vérifier le manifest de variante et tenter un contrôle négatif où un cas d’une autre dimension est présenté à l’artefact. Ce contrôle doit échouer avant exécution native.

**Acceptation :** mêmes résultats scientifiques sur les compilateurs et espaces annoncés ; aucune dépendance à un comportement indéfini ; aucune sélection implicite de dimension ; aucune extension native de fallback ; les builds rapides ne changent pas l’ordre ou les invariants au-delà de leur contrat flottant.

---

### IF-09 — Précision flottante

**Priorité : P2, si plusieurs précisions sont supportées**

Comparer float32 et float64 sur cas lisses. Le but n’est pas d’exiger les mêmes derniers chiffres, mais de vérifier :

- le même régime de convergence avant le plateau d’arrondi ;
- un plateau cohérent avec la précision ;
- des tolérances de solveur adaptées ;
- aucune instabilité propre à une accumulation insuffisante.

---

### IF-10 — I/O HDF5 et relecture distribuée

**Priorité : P1**

Écrire puis relire un état uniforme, un état analytique lisse, une hiérarchie AMR et un cas multi-espèces.

**Variantes :** 1→N rangs, N→1 rang, N→M rangs, ordre de blocs modifié.

**Diagnostics :** métadonnées, hiérarchie, précision, ordre des composantes, champs owner-qualified, checksums scientifiques.

**Acceptation :** état relu identique ou équivalent au contrat ; aucune perte de niveau, de ghost policy, de temps ou de paramètres physiques.

---

## 20. Matrice de couverture des composants du solveur

Légende : `P` = test principal, `S` = couverture secondaire.

| Composant PoPS | Tests principaux | Tests secondaires |
|---|---|---|
| Reconstruction lisse | `TR-01`, `EU-01`, `EU-02`, `EU-03` | `CP-02`, `CP-03` |
| Limiteurs sur discontinuités | `TR-07`, `RB-01`, `RB-04` | `RB-03`, `RB-09` |
| HLL/HLLC/Rusanov | `EU-01`, `RB-01`, `RB-02`, `RB-03` | `RB-04`, `RB-05` |
| Conversion primitive/conservative | `EU-01`, `EU-02`, `EU-03` | `RB-01`, `CP-01` |
| Énergie Euler | `EU-02`, `EU-03`, `RB-05` | `CP-01`, `CP-10` |
| Ghost cells same-level | `TR-04`, `TR-05`, `IF-01` | `EU-06`, `RB-07` |
| Coins et arêtes de halos | `TR-04`, `CP-04` | `AM-12` |
| Frontières périodiques | `TR-01`, `EU-01`, `PO-01` | `CP-02` |
| Frontières réfléchissantes | `EU-04`, `RB-07`, `RB-09` | `RB-08` |
| Dirichlet | `PO-02`, `CP-01` | `AM-10` |
| Neumann et jauge | `PO-03` | `GE-01` |
| Prolongation | `AM-02`, `AM-06`, `EU-06` | `TR-02` |
| Restriction/average-down | `AM-02`, `AM-09`, `AM-11` | `CP-12` |
| Reflux | `AM-09`, `RB-01` | `TR-01`, `RB-05` |
| Subcycling | `AM-04`, `AM-01` | `CP-02`, `AM-11` |
| Regrid/clustering | `AM-02`, `AM-03`, `AM-05` | `TR-03`, `RB-05` |
| Tagging | `AM-03` | `TR-03`, `RB-04` |
| Poisson uniforme | `PO-01`, `PO-02`, `PO-03` | `CP-01` |
| FFT | `PO-05` | `CP-02` |
| Multigrille composite | `PO-04`, `AM-10` | `AM-11` |
| Gradient du potentiel | `PO-06`, `CP-01` | `CP-07` |
| Défaut de Gauss | `PO-01`, `CP-01`, `CP-02` | `AM-11` |
| Signe du couplage | `CP-07`, `CP-08`, `CP-10` | `CP-02` |
| Champ à chaque stage RK | `TM-07`, `CP-01`, `CP-02` | `TM-01` |
| Strang | `TM-02`, `TM-08` | `TM-03` |
| IMEX/AP | `TM-03`, `TM-05` | `TM-04` |
| Multirate/substeps | `TM-06`, `CP-06` | `CP-05` |
| Multi-espèces | `CP-05`, `CP-06`, `CP-12` | `TM-03` |
| Champs magnétiques/sources Lorentz | `TM-04`, `CP-05` | `IF-07` |
| Géométrie polaire intégrée | `GE-01`, `GE-02`, `GE-05` après activation de capacité | `GE-04`, `GE-06` |
| MPI | `IF-01` | tous les cas P0 |
| Kokkos OpenMP | `IF-02` | tous les cas P0 |
| Kokkos accélérateur | `IF-03` | tous les cas P0 compatibles |
| Restart | `IF-04` | `CP-02`, `AM-11` |
| I/O | `IF-05`, `IF-10` | performance `PF-10` |
| DSL/codegen | `IF-07` | `EU-03`, `CP-01`, `CP-05` |
| Spécialisation native exacte et manifest installé | `IF-08` | toutes les campagnes multi-dimensionnelles |

### 20.1 Gate de couverture

Une release scientifique ne peut pas être déclarée complète si une ligne du tableau ne possède aucun test principal exécuté sur le backend concerné.

### 20.2 Couverture par dimension

- tous les algorithmes génériques : 1D, 2D et 3D lorsque le code les annonce ;
- AMR face/arête/coin : 3D obligatoire ;
- géométrie polaire : 2D obligatoire ;
- cas lourds de choc : au moins une dimension multidimensionnelle ;
- chaque spécialisation compilée doit produire son propre rapport de couverture.

---

## 21. Benchmarks de performance du cœur PoPS

Les cas `PF-*` appartiennent à `benchmarks/`. Ils utilisent le vrai moteur, produisent des mesures JSONL et exécutent leurs contrôles numériques hors de la région chronométrée.

### PF-01 — Arithmétique MultiFab et halo périodique

**Statut : existant à conserver**

Mesurer :

- opérations `saxpy`/`lincomb` et chemins alias-safe ;
- `fill_boundary` périodique ;
- coût local, communication et fence ;
- débit en cellules/s et octets/s ;
- dépendance à la taille des blocs, aux ghost layers, aux rangs et aux threads.

**Configurations :** 1D/2D/3D, blocs 8 à 128, ghost width 1 à la largeur maximale utilisée par les schémas, CPU/GPU, 1 à 2 nœuds.

---

### PF-02 — Multigrille scalaire

**Statut : existant à conserver**

Cas manufacturé Dirichlet utilisant le vrai `GeometricMG<Dim>`.

Mesurer :

- temps total ;
- temps par V-cycle ;
- nombre de cycles ;
- lissage ;
- restriction/prolongation ;
- résidu ;
- inconnues résolues/s ;
- efficacité parallèle ;
- coût de la tolérance.

Le benchmark échoue numériquement si le résidu ou l’erreur analytique n’est pas conforme avant d’enregistrer la performance.

---

### PF-03 — RHS volumes finis d’advection

**Priorité : P0 performance**

Mesurer le vrai chemin : reconstruction, flux de faces, divergence et mise à jour. Séparer :

- intérieur de bloc ;
- ghost fill ;
- faces MPI ;
- kernel total.

**Paramètres :** ordre de reconstruction, largeur de halo, dimension, taille de bloc, nombre de composantes.

---

### PF-04 — Pas Euler complet

**Priorité : P0 performance**

Mesurer un pas complet Euler avec conversion primitives/conservées, reconstruction, solveur de Riemann, divergence et intégrateur.

**Variantes :** HLL, HLLC, Rusanov ; minmod/Van Leer/WENO5 ; 1D/2D/3D ; 4 ou 5 variables selon dimension ; uniformes puis AMR statique.

**Métriques :** cellules-feuilles mises à jour/s, inconnues mises à jour/s, temps par stage, coût de reconstruction, coût du Riemann, bandwidth mémoire estimée.

---

### PF-05 — Poisson composite AMR

**Priorité : P0 performance**

Mesurer le solveur elliptique sur une hiérarchie AMR réaliste :

- deux niveaux totaux pour le point minimal ;
- trois niveaux totaux pour la baseline complète ;
- fractions raffinées 5 %, 20 %, 50 % ;
- patches petits/nombreux puis grands/compacts ;
- tolérances représentatives ;
- même charge globale par rang pour le weak scaling.

**Métriques :** temps total, V-cycles, résidu/cycle, lissage, communications, coarse solve, charge imbalance, inconnues feuilles/s.

---

### PF-06 — Pas Euler–Poisson complet

**Priorité : P0 performance**

Chronométrer le pipeline de production :

```text
halo état
→ RHS hyperbolique
→ charge
→ Poisson
→ gradient/champ
→ source
→ stage RK ou programme couplé
→ synchronisation AMR
```

Rapporter le temps de chaque segment et la fraction du pas passée dans Poisson, les halos, le calcul local et la synchronisation.

---

### PF-07 — Regrid et clustering

**Priorité : P1 performance**

Mesurer séparément :

- calcul du tagging ;
- clustering Berger–Rigoutsos ou équivalent ;
- création/destruction des patches ;
- prolongation ;
- restriction ;
- migration MPI ;
- reconstruction des métadonnées et voisinages.

**Paramètres :** nombre de tags, fragmentation, buffer, taille min/max de patch, niveaux, rangs.

---

### PF-08 — Reflux et synchronisation AMR

**Priorité : P1 performance**

Mesurer :

- accumulation dans les flux registers ;
- communication associée ;
- correction ;
- average-down ;
- coût par face coarse-fine ;
- dépendance à la fraction d’interface.

La validation conservative `AM-09` est exécutée avant la mesure.

---

### PF-09 — Load balancing et migration

**Priorité : P1 performance**

Construire des distributions de coût : uniforme, localisée, deux hotspots, hotspot mobile. Comparer les politiques disponibles.

**Métriques :**

- temps max/moyen par rang ;
- coefficient de variation ;
- cellules pondérées par rang ;
- volume migré ;
- coût de migration ;
- temps nécessaire pour amortir un rebalance.

---

### PF-10 — Checkpoint et HDF5 parallèle

**Priorité : P1 performance**

Séparer :

- écriture état uniforme ;
- écriture hiérarchie AMR ;
- écriture multi-espèces et champs ;
- lecture/restart ;
- temps de métadonnées ;
- débit agrégé ;
- taille de fichier par cellule feuille.

Les runs sans I/O ne doivent jamais inclure une sortie implicite dans la région chronométrée.

---

### PF-11 — AMR dynamique end-to-end

**Priorité : P0 performance**

Utiliser `TR-03`, `RB-05` ou une charge mobile réaliste. Mesurer au moins 50 pas après warmup, avec plusieurs regrids.

**Métriques :** temps/pas, débit normalisé par cellule feuille, coût regrid amorti, coût Poisson, coût halo, nombre de patches, efficacité de raffinement, imbalance.

---

### PF-12 — État à grand nombre de moments

**Priorité : P1 performance**

Utiliser HyQMOM15 ou un état synthétique possédant le même nombre de composantes et des flux comparables.

**Objectif :** vérifier que les conclusions tirées d’Euler à 4–5 variables restent valides pour un modèle large en mémoire et en calcul.

**Métriques :** inconnues mises à jour/s, octets/cellule, occupation GPU, pression registres, taille de bloc optimale, coût des eigenvalues et du projecteur.

### 21.13 Entrée de manifest performance proposée

Exemple pour `PF-06` dans `benchmarks/manifest.toml` :

```toml
[[benchmark]]
id = "PF-06"
name = "euler_poisson_step"
kind = "end_to_end"
dimensions = [2, 3]
execution_spaces = ["KokkosSerial", "KokkosOpenMP", "KokkosCuda"]
mpi_modes = ["off", "on"]
max_nodes = 2
requires_validation = "CP-02"

[benchmark.measurement]
warmups = 10
samples = 10
steps_per_sample = 50
rank_time = "maximum"
statistics = ["median", "mad", "p10", "p90", "trimmed_mean"]

[benchmark.metrics]
items = [
  "leaf_cell_updates_per_second",
  "unknown_updates_per_second",
  "time_halo",
  "time_hyperbolic_rhs",
  "time_poisson",
  "time_field_gradient",
  "time_source",
  "time_reflux",
  "time_total"
]
```

Les paramètres de machine ne doivent pas être codés dans le benchmark. Le harness de performance conserve ses profils et scripts dans `benchmarks/manifest.toml` et `benchmarks/romeo/`. Les campagnes scientifiques utilisent séparément `verification/machines/*.toml` afin de ne pas coupler leurs résolutions aux microbenchmarks.

Un profil machine contient uniquement les ressources et commandes de lancement : cœurs/nœud, GPU/nœud, mémoire, contraintes Slurm, environnement Kokkos/MPI, binding et chemins de résultats. Les tailles scientifiques restent dans le cas ou la campagne.

---

## 22. Méthodologie de performance

### 22.1 Région chronométrée

Chaque région mesurée doit être encadrée par :

- un fence Kokkos avant ;
- une barrière MPI avant lorsque la mesure compare les rangs ;
- le kernel ou le segment exact ;
- un fence Kokkos après ;
- une barrière ou réduction du temps maximum après.

Le temps enregistré est le maximum sur les rangs. Une moyenne des temps de rang peut masquer un déséquilibre réel.

### 22.2 Warmup et échantillons

Configuration standard :

- 5 à 10 warmups ;
- au moins 10 mesures pour les microbenchmarks ;
- au moins 5 répétitions indépendantes pour les cas end-to-end ;
- au moins 50 pas mesurés par répétition pour les petits pas ;
- médiane comme statistique principale ;
- MAD et p10/p90 pour le bruit.

### 22.3 Débits normalisés

Rapporter :

$$
R_{cell}=\frac{N_{leaf\ cells}\times N_{steps}}{T},
$$

$$
R_{unknown}=\frac{N_{leaf\ cells}\times N_{components}\times N_{stages}}{T}.
$$

Pour Poisson :

$$
R_{elliptic}=\frac{N_{active\ unknowns}\times N_{Vcycles}}{T}
$$

ou une métrique équivalente clairement documentée.

### 22.4 Strong scaling

Problème global fixe. Pour $p$ ressources :

$$
S_p=\frac{T_1}{T_p},
\qquad
E_p=\frac{T_1}{pT_p}.
$$

Si la baseline une ressource ne tient pas en mémoire, utiliser la plus petite baseline commune et indiquer explicitement $p_0$ :

$$
E_{p|p_0}=\frac{p_0T_{p_0}}{pT_p}.
$$

### 22.5 Weak scaling

Charge par rang ou par GPU fixe. Rapporter :

$$
E^{weak}_p=\frac{T_1}{T_p}.
$$

La charge doit être définie en cellules feuilles, nombre de blocs, composantes, stages et fréquence de regrid. « Même résolution par rang » est insuffisant si la topologie AMR change.

### 22.6 Mesure à erreur égale

Pour comparer uniforme et AMR, rapporter aussi :

- coût pour atteindre une erreur $L^1$ donnée ;
- mémoire pour atteindre cette erreur ;
- nombre de cellules feuilles ;
- temps de Poisson ;
- fraction du domaine raffinée.

Une accélération AMR sans contrôle d’erreur ne constitue pas un résultat utile.

### 22.7 Taille minimale par rang

Pour éviter un benchmark dominé par le lancement et les métadonnées :

- au moins 4 blocs actifs par rang ;
- cible de 8 à 16 blocs par rang ;
- au moins plusieurs dizaines de milliers de cellules par rang en 3D ;
- sur GPU, assez de blocs pour saturer le device ;
- rapporter les cas où le nombre de blocs/rang tombe sous quatre.

---

## 23. Matériel cible et limite stricte de deux nœuds

### 23.1 Règle générale

Aucun manifest, script Slurm ou campagne ne doit demander plus de deux nœuds.

Le runner doit vérifier :

```text
requested_nodes <= 2
SLURM_NNODES <= 2, si la variable existe
number_of_unique_hosts <= 2 après lancement
```

Toute violation arrête le run avant compilation lourde ou calcul.

### 23.2 ROMEO CPU

La configuration publique actuelle de ROMEO indique des nœuds CPU équipés de deux AMD EPYC 9654 de 96 cœurs, soit **192 cœurs physiques par nœud**. Deux nœuds donnent donc au maximum :

- 384 cœurs physiques ;
- jusqu’à 384 rangs MPI en mode un rang par cœur ;
- aucune sursouscription dans les campagnes de référence.

### 23.3 ROMEO accéléré GH200

La configuration publique actuelle indique **quatre superchips GH200 par nœud**, soit :

- 4 GPU par nœud ;
- 8 GPU au maximum sur deux nœuds ;
- 288 cœurs Grace Arm par nœud ;
- 72 cœurs hôte associés à chaque GH200 ;
- 96 GiB de mémoire GPU par device selon la documentation de la partition.

La configuration GPU de référence est **un rang MPI par GPU**. Un rang ne doit voir qu’un device assigné par le scheduler.

### 23.4 Principe de pinning

CPU :

```bash
export OMP_PLACES=cores
export OMP_PROC_BIND=close
srun --cpu-bind=cores ...
```

GPU :

- un rang par GPU ;
- réservation CPU compatible avec la localité du superchip ;
- device visible unique par rang ;
- validation fail-fast des UUID/device ordinals ;
- pas de partage de GPU dans la baseline.

Tester `OMP_PROC_BIND=spread` une fois dans la campagne d’affinité peut être utile, mais `close` reste la baseline hybride tant que la topologie mesurée ne justifie pas l’inverse.

---

## 24. Matrices CPU : rangs MPI et threads Kokkos OpenMP

### 24.1 Scaling Kokkos OpenMP sur un nœud CPU

Un processus MPI ou lancement sans MPI, avec :

```text
OMP_NUM_THREADS = 1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 192
```

Cas : `PF-03`, `PF-04`, `PF-05`, `PF-06`.

Ce sweep mesure le scaling des kernels sur un nœud. Il ne doit pas être confondu avec le meilleur layout hybride.

### 24.2 MPI pur

Kokkos OpenMP fixé à un thread.

**Un nœud :**

```text
MPI ranks = 1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 192
OMP threads/rank = 1
```

**Deux nœuds équilibrés :**

```text
MPI ranks total = 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 384
OMP threads/rank = 1
```

À deux nœuds, le nombre de rangs par nœud doit être identique sauf campagne spécifique de déséquilibre.

### 24.3 Décomposition hybride à occupation complète

| Rangs MPI par nœud | Threads Kokkos OpenMP par rang | Cœurs utilisés par nœud | Rangs totaux sur 2 nœuds |
|---:|---:|---:|---:|
| 1 | 192 | 192 | 2 |
| 2 | 96 | 192 | 4 |
| 4 | 48 | 192 | 8 |
| 8 | 24 | 192 | 16 |
| 16 | 12 | 192 | 32 |
| 24 | 8 | 192 | 48 |
| 32 | 6 | 192 | 64 |
| 48 | 4 | 192 | 96 |
| 64 | 3 | 192 | 128 |
| 96 | 2 | 192 | 192 |
| 192 | 1 | 192 | 384 |

**Sous-ensemble primaire pour limiter le coût :**

```text
1×192, 4×48, 8×24, 16×12, 32×6, 48×4, 96×2, 192×1 par nœud
```

Le sweep complet est réservé aux releases ou à une investigation de topologie.

### 24.4 Campaignes CPU obligatoires

1. **Thread scaling, un rang, un nœud.**
2. **MPI scaling, un thread, un nœud.**
3. **Layouts hybrides à 192 cœurs, un nœud.**
4. **Layouts hybrides à 384 cœurs, deux nœuds.**
5. **Strong scaling de 1 à 2 nœuds avec le meilleur layout identifié.**
6. **Weak scaling 1→2 nœuds avec cellules feuilles/rang constantes.**
7. **Sensibilité à la taille de bloc pour les trois meilleurs layouts.**

---

## 25. Matrice GPU : un à huit GPU

### 25.1 Scaling principal

| Nœuds | GPU utilisés | Rangs MPI totaux | Rangs par nœud | GPU par rang |
|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 1 | 1 |
| 1 | 2 | 2 | 2 | 1 |
| 1 | 4 | 4 | 4 | 1 |
| 2 | 8 | 8 | 4 | 1 |

Pour isoler l’effet réseau, ajouter si le scheduler le permet :

| Nœuds | GPU utilisés | But |
|---:|---:|---|
| 2 | 2 | 1 GPU par nœud, latence inter-nœud |
| 2 | 4 | 2 GPU par nœud, étape intermédiaire |

### 25.2 Threads hôte par rang GPU

La baseline de calcul GPU garde :

```text
MPI ranks = number of GPUs
OMP_NUM_THREADS = 1
```

Une campagne séparée teste :

```text
OMP_NUM_THREADS = 1, 2, 4, 8, 16
```

avec un rang par GPU. Les 72 cœurs hôte associés à chaque GH200 peuvent être réservés pour la localité sans nécessairement être tous utilisés comme threads OpenMP. L’utilisation de 32, 64 ou 72 threads hôte n’est ajoutée que si le profiling montre un chemin significativement exécuté sur CPU.

### 25.3 Règles GPU

- un seul device visible par rang ;
- pas de MPS ou partage de GPU dans la baseline ;
- mêmes clocks/power policy lorsque la plateforme permet de les enregistrer ;
- température et état du GPU enregistrés lorsque disponibles ;
- mémoire HBM maximale et pic alloué enregistrés ;
- warmup suffisant pour compiler/cacher les kernels et stabiliser les clocks ;
- validation scientifique avant timing.

### 25.4 Campaignes GPU obligatoires

1. strong scaling 1→2→4→8 GPU ;
2. weak scaling 1→2→4→8 GPU ;
3. tailles de blocs 16³, 32³, 64³ et éventuellement 128³ ;
4. un rang/GPU avec threads hôte 1, 4, 8 et 16 ;
5. uniformes puis AMR statique ;
6. AMR dynamique end-to-end ;
7. Poisson seul puis Euler–Poisson complet ;
8. CPU contre GPU à erreur scientifique égale.

---

## 26. Tailles de problèmes pour les campagnes à deux nœuds

Les tailles exactes doivent être calibrées une fois par machine et versionnées dans un profil machine. Les règles sont plus importantes qu’un nombre universel.

### 26.1 Strong scaling

Le problème fixe doit :

- tenir sur la plus petite configuration de la courbe ;
- durer assez longtemps pour dépasser le bruit de lancement ;
- utiliser au moins 30 % de la mémoire de la plus petite configuration sans dépasser 70 % ;
- produire au moins 0,2 à 1 seconde par pas sur la baseline, selon le cas ;
- contenir assez de blocs pour conserver au moins quatre blocs par rang à la plus grande configuration.

Points de départ à calibrer :

| Cas | CPU, 1 nœud | GPU, 1 device |
|---|---|---|
| Euler 2D uniforme | environ $4096^2$ | environ $8192^2$ si mémoire et temps adaptés |
| Euler 3D uniforme | environ $256^3$ | $384^3$ à $512^3$ selon le nombre de tableaux |
| Poisson 3D | environ $256^3$ | $384^3$ ou davantage après calibration |
| AMR 3D | 20 à 40 millions de cellules feuilles | 40 à 100 millions de cellules feuilles |

Ces valeurs sont des points de départ, pas des seuils normatifs. Le `provenance.json` doit conserver la taille réellement utilisée et la mémoire mesurée.

### 26.2 Weak scaling CPU

Choisir une charge par rang, par exemple :

- 8 à 16 blocs de $32^3$ par rang ; ou
- un nombre fixe de cellules feuilles et de faces coarse-fine par rang.

Exécuter 1 nœud puis 2 nœuds avec le même layout rangs/threads par nœud.

### 26.3 Weak scaling GPU

Choisir une charge par GPU représentant 30 à 60 % de la HBM, après prise en compte de tous les tableaux temporaires et niveaux AMR. Conserver :

- même nombre de cellules feuilles/GPU ;
- même nombre de blocs/GPU ;
- même fraction raffinée ;
- même nombre de V-cycles ;
- même fréquence de regrid.

### 26.4 Campagne de taille de bloc

Pour chaque backend, tester au minimum :

- CPU 3D : 16³, 32³, 64³ ;
- GPU 3D : 16³, 32³, 64³, 128³ si supporté ;
- 2D : 16², 32², 64², 128² ;
- nombre de ghost layers correspondant au schéma.

Rapporter simultanément performance et efficacité de raffinement. Un bloc optimal pour le kernel peut être mauvais pour l’AMR ou le load balancing.

---

## 27. Templates Slurm limités à deux nœuds

Les noms de partitions et de comptes restent des paramètres de site. Les scripts ci-dessous définissent le contrat de ressources.

### 27.1 CPU hybride, deux nœuds, 8 rangs par nœud et 24 threads

```bash
#!/usr/bin/env bash
#SBATCH --job-name=pops-cpu-2n
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=24
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --account=<project>
#SBATCH --constraint=x64cpu
#SBATCH --mem=<memory_per_node>

set -euo pipefail

if [[ "${SLURM_NNODES:-1}" -gt 2 ]]; then
  echo "PoPS verification policy: more than two nodes is forbidden" >&2
  exit 2
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export KOKKOS_NUM_THREADS="${OMP_NUM_THREADS}"

romeo_load_x64cpu_env
# Charger ensuite la toolchain PoPS/MPI versionnée par le profil machine.

srun --cpu-bind=cores \
  build/benchmarks/bin/pops_benchmark \
  --case=euler_poisson_step \
  --output="results-${SLURM_JOB_ID}.jsonl"
```

### 27.2 MPI pur, deux nœuds

```bash
#!/usr/bin/env bash
#SBATCH --job-name=pops-mpi-2n
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=192
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --account=<project>
#SBATCH --constraint=x64cpu
#SBATCH --mem=<memory_per_node>

set -euo pipefail
[[ "${SLURM_NNODES:-1}" -le 2 ]]

romeo_load_x64cpu_env

export OMP_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close

srun --cpu-bind=cores \
  build/benchmarks/bin/pops_benchmark \
  --case=euler_poisson_step \
  --output="results-${SLURM_JOB_ID}.jsonl"
```

Cette configuration à 384 rangs est un endpoint, pas la seule mesure. Le script de campagne doit lancer les points intermédiaires équilibrés.

### 27.3 GPU, deux nœuds, quatre GPU par nœud

```bash
#!/usr/bin/env bash
#SBATCH --job-name=pops-gpu-2n
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --account=<project>
#SBATCH --constraint=armgpu
#SBATCH --mem=<memory_per_node>

set -euo pipefail

if [[ "${SLURM_NNODES:-1}" -gt 2 ]]; then
  echo "PoPS verification policy: more than two nodes is forbidden" >&2
  exit 2
fi

export OMP_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export CUDA_DEVICE_ORDER=PCI_BUS_ID

romeo_load_armgpu_env
# Charger ensuite NVHPC/HPC-X, CUDA, Kokkos et PoPS selon le profil machine versionné.

srun --cpu-bind=cores --gpus-per-task=1 \
  build/benchmarks/bin/pops_benchmark \
  --case=euler_poisson_step \
  --require-one-device-per-rank \
  --output="results-${SLURM_JOB_ID}.jsonl"
```

Le flag exact de binding GPU peut varier selon la version Slurm du site. Le runtime doit néanmoins vérifier qu’un rang voit exactement le device qui lui est assigné et enregistrer l’inventaire.

---

## 28. Métriques de scaling obligatoires

Chaque point de scaling produit :

| Groupe | Métriques |
|---|---|
| Ressources | nœuds, hôtes, rangs, threads/rang, GPU, device/rang, cœurs utilisés |
| Taille | cellules base, cellules feuilles, blocs, niveaux, composantes, ghost width |
| Temps | total, pas, stage, halo, reconstruction, Riemann, Poisson, regrid, reflux, I/O |
| Débit | cellules/s, inconnues/s, faces/s, inconnues Poisson/s |
| MPI | octets envoyés/reçus, temps collectifs, temps point-à-point, imbalance |
| AMR | fraction raffinée, faces coarse-fine, patches, regrids, migration |
| Poisson | V-cycles, itérations, réduction de résidu, temps coarse solve |
| GPU | HBM utilisée, débit mémoire estimé, occupation si disponible, temps kernels |
| Qualité | erreur analytique, conservation, résidu, symétrie, positivité |
| Statistiques | warmups, échantillons, médiane, MAD, p10, p90, trimmed mean |

### 28.1 Indicateurs spécifiques deux nœuds

Rapporter :

$$
E_{2n}^{strong}=\frac{T_{1n}}{2T_{2n}},
$$

pour une occupation équivalente par nœud, et :

$$
E_{2n}^{weak}=\frac{T_{1n}}{T_{2n}}.
$$

Une efficacité basse n’est pas automatiquement un échec scientifique. Elle déclenche une analyse par segment : halo, collectives, multigrille coarse solve, load balance, nombre de blocs/rang et synchronisations.

### 28.2 Seuils de performance

- pas de gate portable sur une efficacité parallèle absolue ;
- objectif informatif initial : efficacité strong scaling 1→2 nœuds supérieure à 60–70 % pour les cas suffisamment gros ;
- toute valeur inférieure doit être expliquée, pas masquée ;
- regression gate fixe-hardware : médiane plus lente de plus de 10 % et signal supérieur à deux MAD ;
- amélioration revendiquée uniquement avec validation numérique identique et campagne entrelacée.

---

## 29. Niveaux de campagne et intégration continue

### 29.1 Intégration au système de tests existant

`verification/manifest.toml` reste la seule source des paramètres scientifiques. L’intégration avec le système existant se fait par trois ponts minces :

1. `scripts/run_verification.py` planifie et exécute les cas ;
2. quelques smokes sous `tests/python/integration/verification/` appellent le runner avec un profil réduit ;
3. la CI route les changements de `verification/**`, `schemas/verification_*`, `scripts/run_verification.py`, `include/**`, `src/**` et `python/pops/**` vers les lanes concernées.

`tests/test_manifest.toml` peut référencer les smokes, mais ne duplique jamais les résolutions, oracles ou seuils scientifiques. Les campagnes lourdes ne sont pas importées par pytest pendant la collecte normale.

### 29.2 Suite `pr`

**But :** détecter rapidement une rupture majeure sur chaque pull request.

**Ressources :**

- un nœud ;
- Kokkos Serial ou OpenMP avec 1 à 4 threads ;
- éventuellement MPI 2 rangs pour un petit sous-ensemble ;
- artefact exact-rank Dim1 ou Dim2 selon le cas ;
- aucun scaling ;
- résolutions typiques 32, 64, 128 ;
- temps final réduit mais physiquement significatif.

**Cas minimum :**

- `TR-01` 1D/2D ;
- `EU-01` un mode acoustique et un mode entropie ;
- `EU-06` ;
- `PO-01` 2D ;
- `PO-03` ;
- `CP-01` petit ;
- `CP-02` une période ;
- `TM-02` ;
- `TM-03` ;
- `AM-01` petit sur deux niveaux totaux ;
- `AM-09` petit ;
- `IF-04` restart minimal ;
- `IF-08` authentification de la feuille native.

Le gate PR vérifie des tendances et des invariants. La preuve complète d’ordre reste nightly/release.

### 29.3 Suite `nightly`

**But :** convergence complète sur un nœud.

**Ressources :**

- Kokkos Serial/OpenMP ;
- MPI 2 et 4 rangs ;
- un GPU si le runner en dispose ;
- quatre ou cinq résolutions ;
- AMR à deux niveaux totaux ;
- builds Dim1, Dim2 et Dim3 séparés pour les cas applicables.

**Cas :** tous les P0 lisses, Poisson, couplage, temps et principaux cas AMR supportés.

### 29.4 Suite `weekly`

**But :** variantes coûteuses et invariance structurelle.

Contenu :

- tous les modes propres ;
- 3D via un artefact Dim3 dédié ;
- trois niveaux AMR totaux dans la baseline ; quatre niveaux et davantage uniquement si le provider les annonce, sinon preuve de refus ;
- sweep de placement coarse-fine ;
- fréquence de regrid ;
- layouts MPI ;
- nombres de threads ;
- parité CPU/GPU ;
- restart avec changement de rangs ;
- cas de choc P0/P1 ;
- onde radiale cartésienne ;
- extensions polaires uniquement si la capacité runtime est active.

### 29.5 Suite `two_node`

**But :** communication inter-nœud et scaling limité à deux nœuds.

Contenu :

- `IF-01` sur cas lisses et AMR ;
- strong scaling CPU 1→2 nœuds ;
- weak scaling CPU 1→2 nœuds ;
- GPU 1→2→4→8 devices ;
- Poisson composite ;
- Euler–Poisson complet ;
- AMR dynamique ;
- restart N→M rangs ;
- I/O parallèle.

Cette suite est planifiée ou manuelle sur allocation dédiée. Elle ne doit pas être lancée automatiquement sur un runner partagé.

### 29.6 Suite `release`

**But :** produire le rapport scientifique versionné d’une release.

Elle comprend :

- tous les P0 applicables aux capacités annoncées ;
- tous les P1 supportés par la release ;
- tous les espaces Kokkos annoncés et leurs variantes MPI ;
- toutes les dimensions annoncées, chacune avec son artefact authentifié ;
- campagnes d’ordre complètes ;
- rapports AMR sur le nombre de niveaux réellement supporté ;
- parité CPU/GPU ;
- deux nœuds ;
- figures et tables finales ;
- archive de provenance ;
- liste explicite des P2 et capability gates non exécutés.

Une capacité absente n’est pas transformée en succès. Le rapport distingue `pass`, `fail`, `not-supported` et `not-run`.

### 29.7 Suite `performance-regression`

**But :** comparer baseline et candidate sur matériel fixe.

- builds Release propres ;
- mêmes dépendances et même dimension native ;
- même allocation exclusive ;
- séquence ABBA ;
- validation hors timing ;
- analyse médiane/MAD ;
- aucune mise à jour automatique de baseline après échec.

## 30. Mapping minimal des cas vers les suites

| Cas | PR | Nightly | Weekly | Two-node | Release |
|---|:---:|:---:|:---:|:---:|:---:|
| `TR-01` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `TR-02` | petit | ✓ | ✓ | ✓ | ✓ |
| `TR-03` |  | ✓ | ✓ | perf | ✓ |
| `TR-04` | 2D | 3D | ✓ | ✓ | ✓ |
| `TR-05` |  | ✓ | ✓ |  | ✓ |
| `TR-06` | partiel | ✓ | ✓ |  | ✓ |
| `EU-01` | partiel | ✓ | ✓ | ✓ | ✓ |
| `EU-02` | petit | ✓ | ✓ | ✓ | ✓ |
| `EU-03` | 2D | ✓ | ✓ |  | ✓ |
| `EU-04` |  | ✓ | ✓ |  | ✓ |
| `EU-05` |  |  | ✓ |  | ✓ |
| `EU-06` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `PO-01` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `PO-02` | petit | ✓ | ✓ |  | ✓ |
| `PO-03` | ✓ | ✓ | ✓ |  | ✓ |
| `PO-04` |  |  | ✓ | ✓ | ✓ |
| `PO-05` |  | ✓ | ✓ | ✓ | ✓ |
| `PO-06` |  | ✓ | ✓ |  | ✓ |
| `PO-07` |  | ✓ | ✓ |  | ✓ |
| `CP-01` | petit | ✓ | ✓ | ✓ | ✓ |
| `CP-02` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `CP-03` |  | ✓ | ✓ |  | ✓ |
| `CP-04` |  |  | ✓ | ✓ | ✓ |
| `CP-05` | partiel | ✓ | ✓ |  | ✓ |
| `CP-07` | petit | ✓ | ✓ |  | ✓ |
| `CP-08` | ✓ | ✓ | ✓ |  | ✓ |
| `CP-10` |  | ✓ | ✓ |  | ✓ |
| `CP-12` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `TM-01` | partiel | ✓ | ✓ |  | ✓ |
| `TM-02` | ✓ | ✓ | ✓ |  | ✓ |
| `TM-03` | ✓ | ✓ | ✓ |  | ✓ |
| `TM-04` |  | ✓ | ✓ |  | ✓ |
| `TM-05` |  |  | ✓ |  | si AP publié |
| `TM-06` |  | ✓ | ✓ |  | ✓ |
| `AM-01` | petit | ✓ | ✓ | ✓ | ✓ |
| `AM-02` |  | ✓ | ✓ |  | ✓ |
| `AM-03` |  | ✓ | ✓ | ✓ | ✓ |
| `AM-04` | petit | ✓ | ✓ | ✓ | ✓ |
| `AM-06` |  | 2 niveaux totaux | 3 niveaux totaux | 3 niveaux + scaling | ✓ |
| `AM-09` | petit | ✓ | ✓ | ✓ | ✓ |
| `AM-10` |  | ✓ | ✓ | ✓ | ✓ |
| `AM-11` |  | ✓ | ✓ | ✓ | ✓ |
| `RB-01` | petit | ✓ | ✓ |  | ✓ |
| `RB-02` |  | ✓ | ✓ |  | ✓ |
| `RB-05` |  | 2D | 2D/3D | perf | ✓ |
| `RB-07` |  |  | ✓ | ✓ | ✓ |
| `GE-03`, `GE-06` | selon cas | ✓ | ✓ | partiel | ✓ |
| `GE-01`, `GE-02`, `GE-04`, `GE-05` | refus | capability | capability |  | capability |
| `IF-*` | partiel | partiel | ✓ | ✓ | ✓ |
| `PF-*` |  | smoke | ✓ | ✓ | ✓ |

---

## 31. Rapport automatique attendu

Le rapport et son `summary.json` sont validés par `schemas/verification_report.v1.json` avant publication.

La commande de campagne proposée :

```bash
python scripts/run_verification.py \
  --suite release \
  --dimensions 1,2,3 \
  --max-nodes 2 \
  --output build/verification/release-<sha>
```

produit :

```text
report/
├── REPORT.md
├── summary.json
├── coverage.csv
├── failures.csv
├── convergence/
│   ├── spatial_orders.csv
│   ├── temporal_orders.csv
│   └── plots/
├── amr/
│   ├── interface_errors.csv
│   ├── conservation.csv
│   ├── patch_efficiency.csv
│   └── plots/
├── poisson/
│   ├── residual_vs_error.csv
│   ├── vcycles.csv
│   └── plots/
├── coupling/
│   ├── dispersion.csv
│   ├── energy.csv
│   └── plots/
├── infrastructure/
│   ├── mpi_parity.csv
│   ├── backend_parity.csv
│   ├── restart.csv
│   └── plots/
└── performance/
    ├── strong_scaling.csv
    ├── weak_scaling.csv
    ├── stage_breakdown.csv
    └── plots/
```

### 31.1 Première page du rapport

La première page doit répondre directement :

1. quels composants sont couverts ;
2. quels backends et dimensions ont été testés ;
3. quels cas échouent ;
4. quels ordres ont été mesurés ;
5. si AMR conserve l’ordre et les invariants ;
6. si Poisson et le champ convergent ;
7. si le couplage conserve la phase, le signe et l’énergie attendus ;
8. si les résultats dépendent des rangs, threads ou GPU ;
9. quelles performances ont été mesurées sur un et deux nœuds ;
10. ce qui n’a pas été testé.

### 31.2 Figures obligatoires

- log-log erreur/résolution avec pentes de référence ;
- erreur temporelle/pas de temps ;
- erreur près/loin des interfaces AMR ;
- conservation en fonction du temps ;
- potentiel et champ, erreur et résidu ;
- dispersion $\omega(k)$ ;
- phase/amplitude Langmuir ;
- symétrie des cas radiaux ;
- patch maps AMR ;
- strong et weak scaling ;
- breakdown du temps par composant ;
- débit à erreur égale uniforme/AMR.

### 31.3 Politique des figures

- les figures exploratoires restent dans `build/verification/` ;
- seules les reproductions établies peuvent versionner des figures canoniques ;
- chaque figure versionnée possède une provenance machine-readable ;
- aucune figure ne remplace une assertion quantitative ;
- une carte visuellement symétrique sans métrique de symétrie n’est pas une preuve.

---

## 32. Signatures de panne et diagnostic probable

| Observation | Causes probables à examiner en premier |
|---|---|
| Ordre 2 uniforme, ordre 1 AMR | Prolongation, ghost coarse-fine, interpolation temporelle, subcycling |
| Pics fixes aux frontières de blocs | Ghost fill, largeur de halo, stencil lisant une valeur non initialisée |
| Pics uniquement aux coins/arêtes | Remplissage diagonal, ordre des communications, index 3D |
| Masse dérive uniquement avec AMR | Reflux, average-down, cellules coarse couvertes comptées deux fois |
| Dents de scie à chaque synchronisation coarse | Flux register ou ordre reflux/average-down |
| Saut d’erreur à chaque regrid | Prolongation/restriction, initialisation de nouveaux patches |
| Résultat dépend de la fréquence de regrid | Dérive cumulative de projection, tagging instable, état non conservé |
| $\phi$ ordre 2 mais $E$ ordre 1 | Gradient, interpolation de champ, raccord coarse-fine |
| Résidu Poisson faible mais solution fausse | Frontières, signe, jauge, second membre, opérateur discret |
| Poisson uniforme correct, AMR incorrect | Opérateur composite, flux coarse-fine, masques de couverture |
| FFT et GMG diffèrent par une constante | Jauge/mode moyen, probablement normal si non fixé |
| FFT et GMG diffèrent en gradient | Signe, normalisation, positionnement ou gradient |
| Charge correcte, accélération opposée | Signe de $q$, convention $E=-\nabla\phi$, binding du champ |
| Fréquence Langmuir correcte, amplitude fausse | Dissipation, source d’énergie, time-centering |
| Langmuir dérive en énergie | Champ au mauvais stage, travail $J\cdot E$, splitting |
| RK3 devient ordre 1 avec Poisson | Champ non recalculé à chaque stage ou source figée |
| Strang devient ordre 1 | Ordre d’opérateurs, sous-opérateur d’ordre insuffisant, temps de stage |
| Collision ne conserve pas le moment | Signes de sources inter-espèces, masses ou densités |
| Régime AP devient instable lorsque $\epsilon\to0$ | Mauvaise limite, solve implicite incomplet, pas de temps encore contraint |
| Résultat change avec nombre de rangs | Halo MPI, réduction, race, ordre d’accumulation amplifié |
| Structure alignée avec frontière MPI | Échange incomplet, mauvais voisin, buffer non synchronisé |
| Résultat change avec threads | Race, réduction non déterministe mal tolérée, scratch partagé |
| CPU et GPU ont des ordres différents | Branche backend différente, précision, fence, réduction ou kernel manquant |
| Restart diverge immédiatement | Ghost non reconstruits, temps de niveau, flux register ou stage perdus |
| Restart diverge seulement après regrid | Métadonnées AMR/tagging non restaurées |
| Sortie fréquente change le résultat | Fence, alias de buffer, I/O lisant ou modifiant un état en cours |
| Asymétrie $x/y$ | Indice directionnel, stride, sweep, signe de normale |
| Asymétrie suit les patches | AMR, clustering, ghost coarse-fine |
| Sedov présente une harmonique d’ordre quatre fixe | Empreinte cartésienne non convergente, patches ou flux directionnels |
| Implosion dépend du layout MPI | Ghost, ordre de sweep, réduction ou load balance modifiant le calcul |
| Performance chute à deux nœuds uniquement | Taille locale trop petite, collectives, coarse solve, réseau, imbalance |
| GPU sous-utilisé avec petits blocs | Pas assez de travail par kernel, launch overhead, fragmentation AMR |
| Poisson domine soudainement | Tolérance trop stricte, mauvais initial guess, coarse solve, perte de convergence MG |

### 32.1 Procédure de triage

Lorsqu’un cas échoue :

1. conserver tous les artefacts et la provenance ;
2. reproduire en Serial ;
3. comparer uniforme/AMR ;
4. désactiver le subcycling sans changer le maillage ;
5. déplacer les frontières de blocs ;
6. réduire à un seul mode propre ;
7. resserrer la tolérance Poisson ;
8. comparer CPU/GPU seulement après avoir isolé la physique ;
9. localiser l’erreur dans le temps, pas uniquement à $t_f$ ;
10. ajouter un test du cœur dans `tests/` une fois le défaut isolé.

La campagne scientifique détecte le défaut ; le test du cœur ajouté après correction empêche sa réintroduction à faible coût.

---

## 33. Tests à ne pas utiliser comme preuve principale

Les approches suivantes peuvent servir de diagnostic secondaire, mais ne suffisent pas :

- comparer deux solveurs PoPS entre eux sans oracle analytique ;
- comparer uniquement une image à une image publiée ;
- utiliser un golden produit par une version antérieure sans justification ;
- conclure à l’ordre deux avec deux résolutions ;
- utiliser le résidu Poisson comme seule métrique ;
- exiger l’ordre deux sur Sod, Sedov ou Shu–Osher ;
- annoncer une validation physique à partir d’une conservation ;
- comparer CPU/GPU bitwise sans mode déterministe ;
- mesurer une performance sur un seul run ou un nœud partagé ;
- chronométrer compilation, I/O ou warmup sans l’indiquer ;
- calculer les normes AMR en comptant les cellules coarse couvertes ;
- utiliser des valeurs analytiques aux centres lorsque les inconnues sont des moyennes de cellules ;
- exécuter un schéma numérique de référence en Python dans le hot path et attribuer sa performance à PoPS ;
- désactiver silencieusement une capacité non disponible et déclarer le cas réussi.

---

## 34. Ordre d’implémentation

### Phase 0 — Infrastructure commune

**Livrables :**

- `verification/manifest.toml` ;
- `verification/pops_verify/` avec `reference_errors.py` pour les comparaisons aux oracles ;
- `schemas/verification_manifest.v1.json` ;
- `schemas/verification_metrics.v1.json` ;
- `schemas/verification_provenance.v1.json` ;
- `schemas/verification_report.v1.json` ;
- `scripts/run_verification.py` et `scripts/check_verification_manifest.py` ;
- intégration de la sélection aux autorités existantes `scripts/ci_select_tests.py` et `.github/workflows/ci.yml` ;
- runner de campagne avec `max_nodes=2` et contrôle exact de `POPS_NATIVE_DIM` ;
- attachement des diagnostics natifs `pops.diagnostics` au `ConsumerGraph` ;
- provenance ;
- normes leaf-only pour l’erreur à l’oracle ;
- calcul des moyennes analytiques de cellules ;
- convergence ;
- conservation ;
- phase/fréquence ;
- symétrie ;
- erreur coarse-fine ;
- rapport Markdown/CSV/JSON ;
- conventions de dossier et README.

**Critère de sortie :** un cas factice analytique peut produire un rapport complet et refuser une campagne de plus de deux nœuds.

### Phase 1 — Noyau lisse uniforme

**Cas :**

- `TR-01` ;
- `TR-02` ;
- `EU-01` ;
- `EU-02` ;
- `EU-03` ;
- `PO-01`, `PO-02`, `PO-03`, `PO-07` ;
- `TM-01`.

**Critère de sortie :** ordres spatiaux et temporels conformes sur Kokkos Serial/OpenMP avec des artefacts Dim1/Dim2 distincts, puis Dim3 pour les opérateurs génériques.

### Phase 2 — Couplage Poisson

**Cas :**

- `CP-01` ;
- `CP-02` ;
- `CP-03` ;
- `CP-07` ;
- `CP-08` ;
- `CP-12` ;
- `TM-07`.

**Critère de sortie :** ordre couplé, fréquence Langmuir, champ et force corrects, énergie diagnostiquée, stages RK prouvés.

### Phase 3 — AMR complet

**Cas :**

- `AM-01`, `AM-02`, `AM-04`, `AM-06`, `AM-07`, `AM-08`, `AM-09`, `AM-10`, `AM-11` ;
- `TR-04`, `TR-05` ;
- `PO-06`.

**Critère de sortie :** ordre maintenu en AMR, conservation par reflux, Poisson composite, subcycling et regrid sans dérive.

### Phase 4 — Sources, multi-fluides et raideur

**Cas :**

- `CP-05`, `CP-06` ;
- `TM-02`, `TM-03`, `TM-04`, `TM-05`, `TM-06`, `TM-08`.

**Critère de sortie :** eigenmodes génériques, Strang ordre deux, collisions exactes, limite raide et multirate.

### Phase 5 — Robustesse et géométrie

**Cas :**

- `RB-01`, `RB-02`, `RB-03`, `RB-04`, `RB-05`, `RB-06`, `RB-07`, `RB-09` ;
- `GE-03` et `GE-06` sur le runtime cartésien ;
- `GE-01`, `GE-02`, `GE-04`, `GE-05` après activation de la capacité polaire.

**Critère de sortie :** positivité, chocs corrects, symétries et cas radiaux cartésiens ; les extensions polaires restent explicitement capability-gated tant que le runtime public ne les exécute pas.

### Phase 6 — Infrastructure multi-backend

**Cas :** `IF-01` à `IF-10`, avec `IF-08` exécuté pour chaque dimension native annoncée.

**Critère de sortie :** résultats scientifiques indépendants des rangs/threads, parité des espaces Kokkos, restart et I/O fiables, codegen/modèles fournis cohérents, et artefacts exact-rank authentifiés sans fallback.

### Phase 7 — Performance et deux nœuds

**Cas :** `PF-01` à `PF-12`.

**Critère de sortie :** rapports strong/weak scaling CPU et GPU, breakdown du pas couplé, AMR dynamique, baseline de release et aucune campagne au-delà de deux nœuds.

---

## 35. Priorité de développement concrète

Ordre recommandé des premières pull requests :

1. outils communs de normes, provenance et rapport ;
2. `TR-01` advection sinusoïdale ;
3. `PO-01` Poisson périodique ;
4. `EU-01` modes linéaires Euler ;
5. `CP-02` Langmuir froid ;
6. `TM-01` ordre temporel ;
7. `AM-01` traversée coarse-fine ;
8. `AM-09` reflux ;
9. `AM-10` Poisson composite ;
10. `CP-01` MMS Euler–Poisson ;
11. `EU-02` vortex isentropique ;
12. `AM-02` patch mobile prescrit ;
13. `IF-01` MPI layouts ;
14. `IF-03` CPU/GPU ;
15. `IF-04` restart ;
16. `RB-01` Sod ;
17. `RB-05` Sedov ;
18. performance `PF-03` à `PF-06` ;
19. scaling un nœud ;
20. scaling deux nœuds.

Cet ordre maximise la capacité de diagnostic : chaque nouvelle couche est ajoutée après qu’un oracle plus simple a vérifié la couche précédente.

### 35.1 Ensemble scientifique prioritaire canonique de 25 tests

Cette liste est conservée telle quelle comme baseline scientifique. Elle ne remplace pas l’ordre technique des pull requests ci-dessus.

| No | Nom canonique | Cas PoPS correspondants |
|---:|---|---|
| 01 | `01_advection_sine_oblique_3d` | `TR-01`, complété par `TR-04` et `AM-01` |
| 02 | `02_advection_gaussian_dynamic_amr` | `TR-02`, `AM-02`, `AM-03` |
| 03 | `03_euler_linear_eigenmodes` | `EU-01` |
| 04 | `04_euler_isentropic_vortex` | `EU-02` |
| 05 | `05_euler_manufactured_solution` | `EU-03` |
| 06 | `06_block_face_edge_corner_crossing` | `TR-04` |
| 07 | `07_axis_permutation` | `TR-06` |
| 08 | `08_poisson_periodic_mms` | `PO-01`, `PO-06`, `PO-07` |
| 09 | `09_poisson_dirichlet_mms` | `PO-02` |
| 10 | `10_poisson_neumann_nullspace` | `PO-03` |
| 11 | `11_poisson_huang_greengard_amr` | `PO-04`, `AM-10` |
| 12 | `12_euler_poisson_mms` | `CP-01`, `TM-07`, `AM-11` |
| 13 | `13_langmuir_cold` | `CP-02` |
| 14 | `14_langmuir_warm_dispersion` | `CP-03` |
| 15 | `15_multifluid_linear_eigenmodes` | `CP-05` |
| 16 | `16_electrostatic_equilibrium` | `CP-07` |
| 17 | `17_temporal_order_rk` | `TM-01`, `TM-07` |
| 18 | `18_noncommuting_strang_splitting` | `TM-02` |
| 19 | `19_collision_relaxation` | `TM-03` |
| 20 | `20_amr_subcycling_convergence` | `AM-04`, complété par `AM-01` |
| 21 | `21_sod_interface_crossing` | `RB-01`, `AM-09` |
| 22 | `22_sedov_offcenter_symmetry` | `RB-05`, `AM-12` |
| 23 | `23_mpi_layout_invariance` | `IF-01` |
| 24 | `24_cpu_gpu_invariance` | `IF-02`, `IF-03` |
| 25 | `25_checkpoint_restart` | `IF-04` |

Les cas 01 à 20 fournissent les preuves quantitatives d’ordre, de couplage et d’AMR. Les cas 21 à 25 ciblent la robustesse, la symétrie, le parallélisme et la restauration de l’état interne.

---

## 36. Definition of Done d’un cas

Un cas n’est terminé que si :

- le `README.md` contient le contrat complet ;
- les équations et conventions de signe sont explicites ;
- l’oracle est indépendant du résultat PoPS ;
- les conditions initiales sont reproductibles ;
- les moyennes de cellules sont correctement traitées ;
- les normes leaf-only sont utilisées en AMR ;
- au moins quatre résolutions sont disponibles pour une affirmation d’ordre ;
- l’ordre spatial et temporel sont séparés ;
- les tolérances sont justifiées ;
- les variantes uniformes et AMR prévues sont exécutables ;
- la provenance est complète ;
- le manifest, `metrics.json`, `provenance.json` et le rapport passent leurs schémas versionnés ;
- les réductions d’état disponibles utilisent `pops.diagnostics`, tandis que les erreurs à l’oracle utilisent uniquement la couche repo-local ;
- les expressions analytiques compatibles utilisent `pops.analytic` ou un profil public équivalent, sans callback par cellule ;
- `metrics.json` contient tous les champs du schéma universel, avec `null` et justification pour les diagnostics non applicables ;
- toute configuration `canonical` conserve exactement les paramètres de la section 9.2 ;
- un cas AMR de regrid possède la campagne de stress 256 cycles, et 512 cycles lorsqu’il entre dans une release ;
- les sorties vont dans `build/verification/`, sous la racine `/build/` déjà git-ignorée ;
- le cas apparaît dans le manifest ;
- la suite CI correspondante est définie ;
- les erreurs typiques sont documentées ;
- le rapport indique « prouve » et « ne prouve pas » ;
- aucune boucle Python par cellule ne remplace le runtime PoPS ;
- le cas échoue réellement lorsqu’un contrôle négatif ciblé est activé en développement ;
- le cas passe sur les dimensions natives, espaces Kokkos et modes MPI qu’il prétend couvrir ;
- un cas capability-gated produit une preuve de capacité ou de refus, sans fallback ;
- le cas est exécutable depuis le checkout du monorepo sans dépendance à un dépôt voisin.

---

## 37. Definition of Done de la suite PoPS

La suite globale est considérée prête pour une release scientifique lorsque :

1. tous les tests P0 applicables passent ;
2. chaque méthode annoncée possède au moins un test d’ordre ou un oracle quantitatif ;
3. chaque dimension annoncée est couverte ;
4. l’ordre deux est démontré sur grille uniforme et AMR pour les cas lisses ;
5. Poisson est vérifié sur potentiel et champ, pas seulement sur résidu ;
6. Euler–Poisson est vérifié par MMS et par eigenmode physique ;
7. les schémas temporels atteignent leur ordre annoncé ;
8. Strang est testé avec des opérateurs non commutatifs ;
9. le reflux ferme les bilans AMR ;
10. le subcycling conserve l’ordre ;
11. les chocs restent positifs et conservatifs ;
12. MPI et les espaces Kokkos CPU/GPU produisent les mêmes conclusions scientifiques ;
13. restart et I/O ne modifient pas la solution ;
14. les campagnes un et deux nœuds sont archivées ;
15. les limites, cas non supportés et P2 non exécutés sont listés ;
16. le rapport de release contient le SHA unique du monorepo, les digests des catalogues/headers/artefacts natifs et tous les paramètres nécessaires à une reproduction indépendante ;
17. les 32 tests de l’annexe A possèdent chacun au moins un cas exécutable ou une justification explicite de non-applicabilité ;
18. les 25 tests prioritaires canoniques de la section 35.1 sont conservés dans le manifest de release ;
19. le schéma universel de métriques est validé automatiquement avant agrégation du rapport ;
20. chaque dimension annoncée a été testée avec une feuille native exact-rank distincte et authentifiée ;
21. toutes les références, commandes et chemins du plan restent internes au monorepo PoPS ;
22. `verification/manifest.toml`, `tests/test_manifest.toml` et `benchmarks/manifest.toml` conservent des responsabilités distinctes sans autorité dupliquée ;
23. la sélection PR, nightly, weekly, release et two-node est reproductible depuis les manifests versionnés et enregistrée dans la provenance ;
24. le solveur, les cas, les oracles, les scripts, les schémas et le rapport final sont tous rattachés au même SHA du monorepo.

---

## 38. Sources et suites de référence

Les adaptations doivent citer la version exacte, les paramètres repris et les écarts introduits.

### Cahier source conservé

- `Fichier markdown(20260817-102141).md collé` : spécification initiale contenant les 32 tests intégrés, les diagnostics obligatoires pour chaque benchmark et la liste prioritaire de 25 cas. L’annexe A en assure la traçabilité complète.

### Monorepo PoPS examiné

- dépôt : <https://github.com/wolf75222/PoPS> ;
- révision de référence : `0a18620d2fc7bed8f4ec60792f48982230f4c10d` ;
- `CMakeLists.txt` : dimension native exacte, Kokkos obligatoire, MPI optionnel et HDF5 collectif dépendant de MPI ;
- `pyproject.toml` : seule la source `python/pops` est installée dans la wheel ;
- `python/pops/analytic/` : arbres d’expressions analytiques immuables et callback-free ;
- `python/pops/diagnostics/` : réductions natives typées, y compris bilans et conservation ;
- `python/pops/amr/authoring.py` : `AMRHierarchy`, `AMRRegrid`, `AMRClockRelation`, `AMRExecution` et `PatchLayout` ;
- `examples/final/README.md` : cibles d’acceptation exécutables du pipeline public ;
- `examples/final/EXEMPLE_SPEC_FINALE_ADVECTION_SCALAIRE_COMPLET.py` : hiérarchie publique à trois niveaux, ratios `(2, 2)`, subcycling explicite, diagnostics natifs et restart strict ;
- `tests/test_manifest.toml` et `tests/gates/` : catalogue et gates des tests rapides ;
- `tests/python/integration/amr/test_amr_program_reflux.py` : conservation par reflux, regrid réel et parité de programmes RK sur une hiérarchie à deux niveaux ;
- `benchmarks/README.md` et `benchmarks/manifest.toml` : harness de performance existant ;
- `docs/ARCHITECTURE.md` : couches runtime, exact-rank, AMR et limites de géométrie.

Ces fichiers sont la base de l’organisation monorepo de la révision 1.3. Toute évolution future des capacités doit être détectée par le runner et accompagnée d’une mise à jour explicite du manifest.

### AMR et conservation

- H-AMR, framework AMR relativiste/MHD : <https://arxiv.org/abs/1912.10192>
- Berger et Colella, *Local Adaptive Mesh Refinement for Shock Hydrodynamics* : <https://doi.org/10.1016/0021-9991(89)90035-1>
- AMReX AMR tutorials, `Advection_AmrCore`, `SingleVortex` : <https://amrex-codes.github.io/amrex/tutorials_html/AMR_Tutorial.html>
- AMReX `AmrCore` documentation : <https://amrex-codes.github.io/amrex/docs_html/AmrCore.html>
- AMReX linear solvers : <https://amrex-codes.github.io/amrex/docs_html/LinearSolvers.html>

### Euler et chocs

- Castro verification suite : <https://amrex-astro.github.io/Castro/docs/Verification.html>
- Castro problem setups : <https://amrex-astro.github.io/Castro/docs/Problem_Setups.html>
- FLASH supplied test problems : <https://flash.rochester.edu/site/flashcode/user_support/flash_ug_devel/node190.html>
- FLASH User Guide : <https://flash.rochester.edu/site/flashcode/user_support/flash_ug_devel/>
- Athena linear waves : <https://www.astro.princeton.edu/~jstone/Athena/tests/linear-waves/linear-waves.html>
- Athena implosion : <https://www.astro.princeton.edu/~jstone/Athena/tests/implode/Implode.html>
- Athena Double Mach Reflection : <https://www.astro.princeton.edu/~jstone/Athena/tests/dmr/dmr.html>

### Plasma et couplage

- WarpX examples et tests Langmuir : <https://warpx.readthedocs.io/en/latest/usage/examples.html>

### Solutions manufacturées

- Patrick J. Roache / Christopher J. Roy, revue sur la méthode des solutions manufacturées et la vérification de code ;
- ASME, *Code Verification by the Method of Manufactured Solutions* : <https://doi.org/10.1115/1.1436090>
- FDA, présentation de la Method of Manufactured Solutions : <https://www.fda.gov/science-research/about-science-research-fda/method-manufactured-solutions-mms>

### Proxy-applications et performance

- Mantevo MiniAMR : <https://mantevo.org/downloads/>
- ROMEO, description matérielle officielle : <https://romeo.univ-reims.fr/documentation/ressources/description_mat%C3%A9rielle_des_ressources/>
- ROMEO, allocation et visibilité des GPU : <https://romeo.univ-reims.fr/documentation/ressources/romeo_2025/utiliser_des_gpu/>
- ROMEO, options Slurm et contraintes d’architecture : <https://romeo.univ-reims.fr/documentation/ressources/romeo_2025/lancer_un_calcul/>

### Règle de citation

Pour chaque adaptation, le README du cas doit préciser :

- source ;
- version ou date de consultation ;
- paramètres identiques ;
- paramètres modifiés ;
- raison de l’adaptation ;
- nature de l’oracle ;
- ce qui est directement comparable ;
- ce qui ne l’est pas.

---

## 39. Résultat attendu

À terme, PoPS doit pouvoir produire automatiquement un rapport où chaque capacité annoncée est reliée à :

- au moins un oracle quantitatif ;
- une courbe de convergence lorsque la solution est lisse ;
- une vérification AMR équivalente ;
- une preuve de conservation ou d’équilibre ;
- une campagne backend/MPI/restart ;
- une mesure de coût ;
- une liste explicite de limites.

Le résultat recherché n’est pas une collection de cas qui « tournent ». C’est une chaîne de preuves falsifiables permettant de localiser une erreur dans le transport, les ghost cells, l’AMR, Poisson, le couplage, le temps, les sources, les géométries ou le parallélisme.

---

## Annexe A — Traçabilité intégrale des 32 tests de la spécification source

Cette annexe garantit qu’aucun test de la spécification initiale n’est perdu dans le catalogue plus détaillé. Une correspondance multiple signifie que le test source a été séparé en plusieurs campagnes orthogonales afin d’isoler les causes de panne.

| N° source | Test source | Cas du présent plan | Paramètre ou preuve canonique conservé |
|---:|---|---|---|
| 1 | Advection sinusoïdale oblique 3D | `TR-01`, `TR-04`, `AM-01` | $\mathbf a=(1,1,1)$, $T=1$, face/arête/coin/coarse-fine/périodique |
| 2 | Impulsion gaussienne transportée | `TR-02`, `AM-02`, `AM-03` | translation exacte, barycentre, largeur, patch prescrit puis tagging |
| 3 | Modes propres linéaires d’Euler | `EU-01` | modes acoustiques gauche/droite et entropie, axial/diagonal/oblique |
| 4 | Vortex isentropique advecté | `EU-02` | vitesses $(1,0)$, $(0,1)$, $(1,1)$, $(1,0.37)$ |
| 5 | Solution manufacturée Euler complète | `EU-03` | toutes composantes, sources dépendantes du temps, uniforme et AMR |
| 6 | Déplacement des frontières de blocs | `TR-05` | blocs 8/16/32/64, interfaces 0.25/0.2578125/0.375 |
| 7 | Traversée face–arête–coin | `TR-04` | même physique et même résolution, seul le placement change |
| 8 | Permutation des axes | `TR-06`, `IF-01` | $x\leftrightarrow y$, $x\leftrightarrow z$, $y\leftrightarrow z$ |
| 9 | Réflexion et rotation | `TR-06`, `AM-12` | réflexions, rotations 90° et 45°, erreur de symétrie |
| 10 | Poisson périodique trigonométrique | `PO-01`, `PO-06`, `PO-07` | erreur sur $\phi$, $E$, résidu et placement coarse-fine |
| 11 | Poisson Dirichlet non homogène | `PO-02` | $\phi=e^x\sin(2\pi y)+x^2y$ et valeurs exactes aux bords |
| 12 | Poisson Neumann et espace nul | `PO-03` | recentrage de la moyenne et rejet du second membre incompatible |
| 13 | Huang–Greengard Poisson AMR | `PO-04`, `AM-10` | sources localisées, plusieurs niveaux, flux coarse-fine |
| 14 | Solution manufacturée Euler–Poisson | `CP-01`, `TM-07`, `AM-11` | relation fermée charge–potentiel, champ à chaque stage, AMR composite |
| 15 | Onde de Langmuir froide | `CP-02` | eigenmode générique et formules fermées $n_e,u_e,E,\phi$ |
| 16 | Onde de Langmuir chaude | `CP-03` | dispersion $\omega^2=\omega_{pe}^2+c_e^2k^2$, $k=1,2,4,8$ |
| 17 | Eigenmodes multi-fluides | `CP-05` | matrice $M(k)$ et solution $e^{\lambda_jt}$ |
| 18 | Équilibre pression–champ | `CP-07` | $\nabla p_s=q_sn_sE$, vitesse parasite vers zéro |
| 19 | Convergence temporelle pure | `TM-01` | $\Delta t$ à $\Delta t/16$, FE/RK2/RK3/Strang |
| 20 | Splitting non commutatif | `TM-02` | $AB\neq BA$, Lie ordre 1, Strang ordre 2 |
| 21 | Relaxation collisionnelle exacte | `TM-03` | décroissance exponentielle et moment barycentrique constant |
| 22 | Onde traversant coarse-fine | `AM-01`, `AM-09` | coarse→fine→coarse, contrôle négatif sans reflux réservé au développement |
| 23 | Raffinement mobile prescrit | `AM-02` | mesures avant/après regrid, 256 puis 512 cycles de stress |
| 24 | AMR contre uniforme fine | `AM-07` | comparaison locale à $h/2$ dans la région fine |
| 25 | Subcycling temporel | `AM-04` | pas global, ratios 2 et 4 |
| 26 | Vortex de Gresho | `EU-05` | équilibre stationnaire, vitesse radiale parasite, moment angulaire |
| 27 | Sedov décentré | `RB-05`, `AM-12` | $R(\theta)$, anisotropie et harmonique cartésienne d’ordre quatre |
| 28 | Sod traversant une interface AMR | `RB-01`, `AM-09` | choc/contact/raréfaction traversent des interfaces distinctes |
| 29 | Onde acoustique radiale | `GE-03`, `GE-04` | oracle radial/Bessel ou référence 1D fine, axes contre diagonales |
| 30 | Invariance au découpage MPI | `IF-01` | 1/2/4/8/16 rangs, layouts `1×4`, `2×2`, `4×1` |
| 31 | CPU contre GPU | `IF-02`, `IF-03` | mêmes diagnostics, threads et tailles de blocs variés |
| 32 | Restart exact | `IF-04` | continu contre checkpoint/restart, AMR/Poisson/subcycling/N→M rangs |

### A.1 Gate de traçabilité

Le runner de release lit cette table depuis une représentation machine-readable et échoue si :

- un numéro source n’a plus de cas associé ;
- un cas associé a été supprimé du manifest sans justification ;
- une configuration canonique a été remplacée par une variante non équivalente ;
- les sorties obligatoires du schéma de métriques ne sont pas disponibles pour le cas.

## Annexe B — Configuration minimale machine-readable

Le manifest doit pouvoir représenter explicitement les paramètres canoniques :

```toml
[repository]
name = "wolf75222/PoPS"
single_repository = true
require_clean_worktree_for_release = true

[build]
exact_native_dimension = true
native_dimensions = [1, 2, 3]
require_doctor = true
require_native_variant_digest = true
on_node_backend = "Kokkos"
mpi_optional = true
hdf5_requires_mpi = true

[current_capability_gates]
cartesian_system_runtime = true
polar_system_runtime = false
amr_total_levels_baseline = 3
amr_refinement_ratios_baseline = [2, 2]

[canonical]
source_suite_version = "2026-08-17-monorepo-v1.3"
convergence_resolutions = [16, 32, 64, 128, 256]
block_sizes = [8, 16, 32, 64]
max_nodes = 2

[canonical.advection_sine]
domain = [0.0, 1.0]
wave_numbers = [1, 2, 3]
velocity = [1.0, 1.0, 1.0]
epsilon = 1.0e-2
final_time = 1.0
period_counts = [1, 2, 4]

[canonical.block_interfaces]
x_positions = [0.25, 0.2578125, 0.375]

[canonical.subcycling]
ratios = [1, 2, 4]

[canonical.regrid_stress]
cycles = [256, 512]

[canonical.mpi_layouts_2d]
layouts = ["1x4", "2x2", "4x1"]
```

Toute valeur modifiée crée une configuration dérivée avec un nouvel identifiant ; elle ne peut pas écraser la baseline canonique.

