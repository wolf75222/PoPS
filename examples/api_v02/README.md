# PoPS : réception de l'algèbre API 0.2

Ces exemples construisent des `Case`, `Model` et `Program` PoPS réels. Les équations
Python deviennent le C++ des artifacts chargés par le runtime PoPS/Kokkos/MPI.
Les références NumPy sont des contrôles après exécution ; elles ne calculent
jamais la trajectoire publiée par PoPS.

Le profil public est **`0.2-experimental`**. Les 28 identifiants de modèles et
les 12 identifiants de méthodes sélectionnent des contrats explicites dans
`profile_contracts.py` : équations, composants requis, paramètres, hypothèses
et récepteurs numériques. Les profils réduits antérieurs sont nommés séparément.
`require_complete=True` exige tous les composants du contrat sélectionné et
vérifie leurs sources Python ; cette option ne demande pas une preuve de
validité dans tous les régimes physiques. La réception native, les garanties
mathématiques et la CI conservent chacune leurs propres preuves.

## Écrire et exécuter les équations

```python
import pops
from examples.api_v02.models_M15_M28 import build_model

spec = build_model("M18", order=3)
plan = pops.resolve(pops.validate(spec.case), layout=spec.layout)
artifact = pops.compile(plan)
artifact.verify()
context = pops.ExecutionContext.mpi_world(artifact)
runtime = pops.bind(
    artifact, initial_values=dict(spec.initial_values),
    resources={"execution_context": context},
)
report = pops.run(runtime, t_end=spec.t_end, max_steps=10000, console=False)
metrics = spec.invariants(runtime)  # Collectif si plusieurs rangs participent.
assert metrics["passed"]
```

Le constructeur M18 écrit la quadrature, la Hessienne, les itérations de Newton,
les résidus de fermeture, le transport spatial et les étages RK4 en Python.
Aucun identifiant M18 n'est transmis au cœur pour choisir des équations.

Les constructions ajoutées ou renforcées réutilisent le compilateur existant :

| Construction | Exemple exécutable | Contrat |
|---|---|---|
| `Program.product`, `fold`, `iterate`, `capture`, `match`, `require_success` | `models_M18_entropy.py`, `method_krylov.py`, `algebra_checks.py` | Produits, régions et budgets natifs ; résultats en erreur opaques. |
| `range_set`, `index`, `at`, `map`, `scan`, `reduce`, `gather` | `tests/python/integration/runtime/test_api_v02_finite_arrays_native.py` | Cardinalité native bornée ; indices liés à une instance ; tableaux immuables répliqués. |
| `StencilFlux` | `method_interval_transport.py`, `method_euler_correction.py` | Corps d'équations fermé, lectures bornées, paramètres scalaires SSA, un flux partagé par face. |
| `ProgramRealization`, `Realize`, `Use` | `method_interval_transport.py`, `method_flux_correction.py` | Occurrences physiques conservées ; sorties effectivement consommées ; pas de faux taux nul pour fournir un stockage. |
| `Contribution`, `Delivery`, `Increment`, `Interval`, `HistoryInterval` | `methods_T01_T12.py`, `models_M22_M25_M28.py` | Valeurs livrées et intervalle authentifiés ; historique distinct de la tentative courante. |
| `cell_field`, `cell_mean`, opérateurs spatiaux et interactions | `models_M15_M28.py`, `models_M20_M21.py` | Support, représentation, propriétaire et point d'évaluation explicites. |
| `FiberAxis`, `FiberRepresentation` | `models_M15_M28.py`, `models_M20_M21.py` | Axes, nœuds, poids, frontières et bijection des degrés de liberté retenus. |
| `GraphFunction`, `graph`, `call` | `graph_function_checks.py` | Arguments et captures typés, insertion SSA, effets inférés ; pas de fonction Python appelée dans le noyau. |
| `Variant`, `Result`, `match` | `variant_checks.py`, `terminal_failure_checks.py` | Payload structural actif ; échec terminal conservé en octets avec son schéma et les fragments des rangs. |
| `Representation`, `ScientificType`, `Claim` | `representation_checks.py` | Représentations et hypothèses scientifiques explicites, indépendantes de la réception. |
| `Problem`, `Algorithm`, `Accuracy`, `solve` | `relation_newton_krylov.py` | Contrôle du résidu original, des contraintes et de la jauge après le solve. |
| `jvp`, `vjp`, `Pairings`, `implicit_jvp`, `implicit_vjp` | `algebra_checks.py`, `implicit_relation_checks.py` | Transformations de graphes admis et relations tangentes/adjointes ; dualités et budgets explicites. |
| `ContinuationSchema`, `Candidate`, `Assess`, `accept`, `Advance` | `continuation_checks.py` | Essai privé, autorité de publication, rollback et checkpoint structuraux dans le profil Uniform. |
| `Distributed`, `local_map`, `requests`, `import_values`, CSR | `distributed_program_checks.py` | Ownership contigu, requêtes et échanges MPI bornés ; calcul local généré à partir des équations Python. |

`Program.rate(name, expressions, of=state)` donne le type de taux à une
production pointwise réellement définie par ses expressions. Il ne doit pas
effacer les obligations d'un dérivatif spatial capturé. Le flux spatial reste
une dépendance explicite de sa mise à jour et de son registre d'échanges.

## Campagne native reproductible

Employer l'environnement `pops` du worktree et les scripts du dépôt :

```sh
bash scripts/setup_env.sh                 # Une fois dans ce worktree.
env -u PYTHONPATH bash scripts/build_python.sh --dim 2 --mpi
env -u PYTHONPATH bash scripts/build_python.sh --dim 1 --mpi

env -u PYTHONPATH OMP_NUM_THREADS=1 KOKKOS_NUM_THREADS=1 \
  python -m examples.api_v02.campaign --output /chemin/neuf/reception-api02
```

Configurer auparavant la toolchain et le transport MPI propres à la machine.
Le `python` de cette commande doit être celui de l'environnement `pops` installé,
sans surcharge `PYTHONPATH`. Les deux dimensions doivent être installées ensemble :
M15 utilise Dim1 ; M28 multidimensionnel et les autres profils utilisent Dim2. La campagne est séquentielle,
avec un processus indépendant et un watchdog par profil.

Le reçu conserve les équations sources, le C++ généré, les bibliothèques et leurs
hashes, le paquet Python, les variantes natives, les headers source/installés,
les états calculés, les métriques et la sortie réelle des processus. Il refuse
un succès si l'installation, les sources scientifiques ou le diff du worktree
changent pendant la campagne. Un test réussi avant `MPI_Finalize` mais suivi
d'un code de sortie non nul n'est pas une réception réussie.

Les tests supplémentaires sous `tests/python/integration/runtime/` vérifient
les contre-exemples, les variantes, les refus, la reprise et le multirang. Leurs
reçus sont séparés : ni une émission C++, ni une compilation seule, ni les tests
source ne se substituent à leur exécution native.

## Compositions structurales de la révision 14

Les mécanismes ci-dessous passent par le même `Program`, son C++ généré et
`System::step`. Leurs états appartiennent à la continuation native ; un tableau
NumPy ne remplace ni leur calcul, ni leur publication. Les fichiers sont des
témoins exécutables, dont la présence seule ne constitue pas une réception.

| Composition | Corps Python | Vérification discriminante |
|---|---|---|
| `BundleSchema`, `Remap`, instances d'ensembles et histoire dépendante | `bundle_remap_v14.py`, `finite_bdf2_v14.py`, `coarsening_moments_v14.py` | Transférer géométrie, champs, cartes et histoire ; refuser domaine périmé, capacité dépassée et coarsening incompatible avec les moments conservés. |
| Continuation complète, deux enfants, `DecisionAssess`, reprise et transformation | `multisupport_children_v14.py`, `decision_assess_v14.py`, `localized_event_v14.py` | Recontrôler chaque candidat transformé, préserver l'enveloppe acceptée après refus parent, reconstruire l'état complet à la frontière effectivement atteinte. |
| Solve spatial par graphes finis et remplacement de contributions | `scientific_spatial_v14.py`, `scientific_krylov_v14.py`, `hierarchy_flux_v14.py` | Newton/CG/GMRES construits en Python, résidu original, hiérarchie à deux barrières, intégrales de flux et GCL ; faux solve refusé. |
| Régions et valeurs distribuées structurales | `distributed_algebra_v14.py`, `bundle_repartition_v14.py`, `distributed_gc_v14.py` | Réduction, scan, scatter, tableaux imbriqués, rang vide, repartition, checkpoint et réutilisation d'un budget de versions. |
| Dérivées de tableaux dynamiques et de relations entières | `array_derivative_v14.py`, `distributed_derivative_v14.py`, `finite_iteration_ad_v14.py`, `implicit_array_v14.py`, `implicit_domain_v14.py` | Dualités pondérées, transposées des accès, domaine exact des données compatibles, distinction trace finie/solution implicite et refus des surfaces non différentiables. |
| Actions à contrat d'erreur et nouvelle structure scientifique | `action_checks_v14.py`, `extension_closure_v14.py` | `phi1` sans inverse d'une matrice singulière ; loi `H(q)=q+q³` et méthode construites sans branche scientifique dans le cœur. |

Les tableaux s'obtiennent par `Program.range_set/map/at`, avec `IndexFamily`
pour une famille dynamique, et par `Distributed` pour l'ownership MPI.
Le schéma de `Bundle` conserve les rôles fixes tandis que tailles, cartes et
connectivités varient dans leurs capacités. Une continuation peut contenir ces
valeurs sans déclarer de faux bloc de champs. L'adaptateur `Uniform` ne limite
pas la topologie représentée par les données du Bundle à une grille uniforme.

Le sens scientifique est vérifié par `ScientificMap(..., strict=True)`,
`Realize(..., strict=True)` et le schéma scientifique de la continuation.
L'analyse porte sur les opérations et les valeurs réellement retenues, avec
unités par composante, supports, variance et représentation. Les anciennes
annotations déclaratives portent le mode `legacy_annotations` ; elles ne
constituent pas une preuve de typage scientifique strict.

`artifact.source_maps` expose les correspondances des
opérations et des `Use` vers les plages du C++ émis. Chaque carte conserve le
hash de son source ; les alias sans instruction gardent leur liaison et son
origine. La campagne archive également `program.source_map.json`. Une carte
source n'est pas une trace d'exécution : le reçu de la bibliothèque chargée et
les contrôles de l'état natif restent nécessaires.

## Contrats et limites de réception

* `GraphFunction/call` construit des régions fermées, pas du Python arbitraire.
  Les capacités, effets et échecs font partie du contrat ; une opération sans
  réalisation ou un contrôle collectif non admissible est refusé.
  Les parcours structurels et codecs sont itératifs. Les budgets déclarent
  la mémoire, les cardinalités et les versions vivantes ; ils ne définissent
  pas une liste de dimensions ou de modèles physiques acceptés.
* Les maps scalaires finies et distribuées émettent des noyaux Kokkos dans le
  runtime PoPS. L'identité du backend, l'exécution GPU et les mesures MPI sont
  reçues séparément. Le coût d'une dérivée construite par colonnes reste celui
  de cet algorithme, sauf substitution générique explicitement qualifiée.
* Le remappage et la repartition déclarés se font sur le communicateur courant.
  Le checkpoint préserve ses fragments et identités ; la reprise avec un autre
  nombre de rangs et la récupération après panne matérielle MPI ne sont pas
  promises par cette réception.
* Les anciens opérateurs `Field/MultiFab`, registres de flux et `StencilFlux`
  conservent leurs restrictions propres. Une composition générale s'écrit par
  les valeurs finies et les graphes inspectables ci-dessus ; elle ne contourne
  pas un refus en appelant une routine physique opaque.
* Les tableaux dérivés gardent leurs instances. Les changements de topologie,
  points de branche non différentiables et relations singulières exigent un
  domaine, une sélection ou une règle explicite. Les relations surdéterminées
  vérifient toutes leurs équations et la compatibilité des données ; une
  pseudoinverse arbitraire ne fournit pas une sélection implicite.
* `Realize` vérifie la livraison des valeurs effectivement consommées et leurs
  occurrences. Cela ne prouve pas automatiquement consistance, stabilité,
  propriété AP, entropie ou convergence. Les hypothèses analytiques et les
  contrôles numériques restent distincts dans les claims.

Les 40 profils du corpus et les réceptions croisées des mécanismes répondent
à deux questions différentes. Leurs résultats doivent être reliés aux hashes
exacts des sources, du paquet, du SDK et des bibliothèques chargées. Une
qualification de toutes les variantes d'une famille physique ne découle ni de
la fermeture du langage, ni de la réussite d'un profil fini.


## Publication des inventaires et dépendances courantes

`Increment.apply` et `Increment.pack` conservent dans le graphe un contrat de
provenance vérifié à la construction, au rechargement et avant émission C++.
La publication exige que chaque destinataire retenu provienne de l'application
du même ledger, dans la même branche et la même invocation de région. Remplacer
une contribution exige donc de recalculer ses consommateurs publiés. Un ancien
étage conservé comme historique reste licite. Un Product construit librement
ne constitue pas un ledger certifié.

Pour une accumulation non linéaire, `pops.inventory_decode.checked_inventory_decode`
reçoit les graphes `encode`, `decode`, `inventory_domain`, `admissible` et
`selection`. La bibliothèque réévalue le même `encode` sur le résultat, contrôle
le résidu original et les tolérances, puis retourne un `Result`. Si le résidu
est toléré, le ledger final le déclare une fois comme défaut avec sa livraison
identité ; il ne réapplique pas ce défaut à la valeur décodée. `InventoryBinding`
relie explicitement le destinataire d'inventaire au chemin publié et à cette
conversion. Les lois et les algorithmes restent des corps Python fermés.

`pops.current_dependencies.CurrentDependency` déclare une équation finie reliant
un chemin retenu aux chemins du même état courant. La déclaration appartient au
`ContinuationSchema` et ne peut disparaître au candidat suivant. Son graphe
sert à produire la valeur et à vérifier sa provenance. Il ne déduit pas une
relation physique du nom d'un champ ; une relation implicite reste un problème
avec sa réalisation et son contrôle explicites.

Voir `inventory_publication_v15.py` pour les constructions complètes, leurs
historiques, l'évaluation du bilan et les refus d'inversion ou de branche.
Les tests natifs associés contrôlent aussi les checkpoints et le rollback.
Leur présence ne vaut réception que lorsque l'artefact effectivement chargé,
les états et la sortie des processus sont consignés pour le même gel.

Le typage scientifique suit également les valeurs dérivées sans annotation de
sortie : extraire `ScientificValue.value` ne supprime pas les unités. Le chargement
d'un GraphFunction fini ne fabrique pas d'unités pour ses paramètres ; un contrat
scientifique strict est revérifié avec les significations de ses entrées. Une
conversion de représentation est déclarée et liée à son corps exact.
