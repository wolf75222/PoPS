# Spécification technique finale - architecture operator-first de PoPS

<!-- markdownlint-disable MD013 -->

## 1. Statut et portée

Ce document est le contrat normatif de l'interface Python et de sa jonction au coeur C++ de PoPS.
Il décrit l'architecture livrée, pas une API de migration et pas un catalogue de fonctions envisagées.
Le code, les schémas versionnés, les quatre exemples finaux et la gate de release authentifiée doivent
rester conformes à ce document.

La surface racine est volontairement petite :

```python
pops.Model
pops.Program
pops.Case
pops.validate
pops.inspect
pops.explain
pops.resolve
pops.compile
pops.bind
pops.run
pops.__version__
```

Tout autre concept public est importé depuis son module thématique. Les moteurs natifs `System` et
`AmrSystem`, les enregistrements `BindInputs` et `InstallPlan`, et les fonctions d'installation sont
des détails internes. Ils ne constituent ni une seconde API, ni une voie de secours.

### 1.1 Objectifs

PoPS doit permettre de :

- décrire la physique avec des états, flux, sources, champs et taux typés ;
- choisir séparément discrétisation spatiale, frontières, maillage, temps, solveurs et sorties ;
- écrire un programme temporel explicite, implicite, IMEX ou multirate avec les mêmes primitives ;
- instancier plusieurs fois un modèle sans ambiguïté de handles ;
- refuser une capacité absente avant la première mutation native ;
- inspecter les décisions, identités, capacités et erreurs de chaque phase ;
- ajouter des composants scientifiques par de petites interfaces, sans branche centrale par classe ;
- exécuter les kernels de production en C++20/Kokkos, avec MPI quand la plateforme l'annonce.

### 1.2 Non-objectifs

Cette version ne promet pas :

- de compatibilité avec les anciennes façades impératives ou stringly typed ;
- d'exécuter un callback Python dans une boucle numérique native ;
- de sélectionner silencieusement un autre algorithme lorsqu'un choix est indisponible ;
- une algèbre d'unités physiques ; il n'existe pas de module public `pops.units` ;
- un backend natif pour toute dimension, tout ratio AMR ou toute géométrie concevable ;
- qu'un protocole générique implique automatiquement qu'un provider installé sait l'exécuter.

Les unités opaques ne sont pas une extension tolérée : `Model.state(..., units=...)` refuse toute
valeur non nulle. Un futur système d'unités devra être typé et participer à la validation, à l'identité,
à la conversion, au lowering et aux rapports de bout en bout.

## 2. Principes non négociables

1. Une chaîne publique unique va de l'authoring à l'exécution.
2. Un concept a une autorité unique ; une BC, un transfert ou un paramètre n'est jamais enregistré deux fois.
3. Une string peut nommer un objet utilisateur, jamais sélectionner une sémantique PoPS.
4. Un objet déjà déclaré est référencé par un handle qualifié, pas par son nom local.
5. Les choix scientifiques sont explicites ; seuls les calculs exactement dérivables sont automatiques.
6. Un preset de `pops.lib.*` construit les mêmes objets et le même graphe que l'écriture manuelle.
7. Un descripteur accepté est entièrement abaissé, prouvé dérivé, ou refusé.
8. Un pas est transactionnel : rejet et exception restaurent états, historiques, horloges et effets.
9. Une sortie, un diagnostic ou un checkpoint n'est publié qu'après acceptation.
10. Une extension déclare de petites facettes et des capacités ; les phases centrales ne testent pas sa classe concrète.
11. Les limites du provider sont des capacités vérifiées, jamais des limites cachées de l'IR.
12. Aucun état intermédiaire de migration, alias historique ou fallback n'appartient au contrat final.

## 3. Couches et dépendances

La direction des dépendances est :

```text
authoring scientifique
    Model / Domain / Frame / Grid / DiscretizationPlan / Program / Case
        -> validation et snapshots immuables
        -> résolution exigences/capacités et LayoutPlan
        -> lowering et artefact compilé
        -> bind des valeurs/ressources
        -> RuntimeInstance
        -> moteur C++ interne / Kokkos / MPI
```

Les responsabilités sont strictes :

| Couche | Autorité | Ne décide pas |
| --- | --- | --- |
| `Model` | états, rôles, paramètres, flux, sources, taux, champs physiques | maillage, solveur, cadence, runtime |
| domaine/frame | géométrie, axes, régions, frontières topologiques | équations, discrétisation |
| `DiscretizationPlan` | réalisation des taux et BC de transport | programme de temps, transfert AMR |
| layout | placement, hiérarchie, tagging, regrid, transfert, exécution AMR | physique |
| `Program` | appels d'opérateurs, stages, solves, sync, historique, commit | stockage et backend |
| `Case` | instanciation et assemblage des autorités | réinterprétation de leur contenu |
| résolution | preuve de cohérence et sélection exacte des providers | substitution algorithmique |
| runtime | exécution de l'artefact et publication transactionnelle | invention de métadonnées scientifiques |

`pops.*` fournit les protocoles de construction. `pops.lib.*` fournit des implémentations et
compositions configurables qui retournent ces mêmes types publics.

### 3.1 Fournisseur de modèle pour la compilation

Un bloc de `Case` entre en compilation par le protocole public
`pops.codegen.CompilerLowerable`. Son unique méthode
`__pops_compiler_lowering__()` retourne un `pops.codegen.CompilerLowering` exact avec :

- `emit_model`, l'émetteur compilable qui valide ses dépendances par `check()` ;
- `source_module`, un `pops.model.Module` exact, autorité canonique de l'IR et de son identité ;
- `facade`, la valeur d'authoring à citer dans les diagnostics.

`pops.model.Module` et les façades physiques implémentent ce protocole. Un fournisseur tiers le
fait de la même manière : il délègue son IR à un `Module` et son émission à un émetteur explicite.
Les phases centrales ne sélectionnent jamais un fournisseur par `isinstance` sur sa classe ; une
méthode absente, un retour non exact, un émetteur incomplet ou une autre autorité IR sont refusés
avant la compilation. Il n'existe pas de fallback par attributs `check` ou `module`.

## 4. Modèle de données Python

### 4.1 Handle et Expr sont deux familles distinctes

Un `Handle` est :

- immuable ;
- hashable ;
- comparable avec une égalité Python booléenne stable ;
- identifié par version de schéma, owner path, kind et identifiant local ;
- qualifiable par une instance de bloc sans perdre la référence à sa déclaration.

Un `Expr` est :

- immuable et transitivement gelé ;
- non hashable ;
- composé par les opérateurs arithmétiques et symboliques ;
- impossible à convertir implicitement en booléen.

```python
if grad(u) > threshold:
    ...
```

échoue immédiatement. Les comparaisons chaînées, `and`, `or` et `not` Python ne sont pas une syntaxe
de graphe. Les combinateurs symboliques typés doivent être utilisés. L'égalité booléenne des handles
reste sûre pour les dictionnaires et registres.

Le passage d'un handle lisible à une expression est explicite : `ValueExpr(handle)`,
`model.value(param_handle)` ou `case.value(param_handle)` selon l'autorité.

### 4.2 Ownership et qualification

Chaque déclaration possède un `OwnerPath`. Avant validation, le chemin peut porter une capacité
d'authoring ; après gel il est canonique et sérialisable. Un `Case.block(...)` crée une instance et
`block[declaration]` produit le handle qualifié correspondant.

```python
model = pops.Model("transport", frame=frame)
U = model.state("U", components=("u",), space=CellState(frame=frame))

case = pops.Case("two_instances")
left = case.block("left", model=model)
right = case.block("right", model=model)

left_u = left[U]
right_u = right[U]
assert left_u != right_u
```

Diagnostics, sorties, historiques, AMR et bind utilisent les handles qualifiés. Une string locale ou
un handle d'un autre owner est refusé.

### 4.3 Builders, snapshots et identités

Les builders (`Model`, `Program`, `Case`, builders de layout) sont mutables uniquement pendant
l'authoring. `pops.validate(case)` ferme le graphe et gèle transitivement le `Case`. Toute mutation
ultérieure est refusée.

Les identités utilisent une sérialisation canonique versionnée et des domaines séparés. Elles couvrent
notamment le graphe du programme, le plan de layout, les composants, le graphe des consommateurs, la
stratégie de pas et les ressources transactionnelles. Les labels de présentation ne doivent pas
invalider une identité scientifique ; toute donnée qui change le comportement doit l'invalider.

## 5. Authoring physique

### 5.1 États, rôles, paramètres et expressions

```python
model = pops.Model("scalar_advection", frame=frame)
U = model.state(
    "U",
    components=("u",),
    representation=Conservative(),
    space=CellState(frame=frame),
)
(u,) = U

a_x_h = model.param(RuntimeParam("a_x", default=1.0, domain=Positive()))
a_x = model.value(a_x_h)
```

`Model.state` déclare un espace d'état complet. `Model.species` est la variante multi-espèces et
produit des espaces distincts de taille arbitraire. Les rôles (`Density`, `Momentum(axis=...)`, etc.)
sont des objets typés ; ils ne sont ni inférés par position ni confondus avec des unités.

Les paramètres ont des kinds fermés (`RuntimeParam`, `ConstParam`, `DerivedParam`), des domaines
typés et une phase d'utilisation vérifiée. Une valeur structurelle ne peut pas être transformée en
paramètre runtime pour contourner compilation ou allocation.

### 5.2 Flux, sources, champs et taux

Le flux physique, l'équation d'évolution et sa discrétisation sont trois autorités différentes :

```python
F = model.flux(
    "advection_flux",
    frame=frame,
    state=U,
    components={frame.x: (a_x * u,), frame.y: (a_y * u,)},
    waves={frame.x: (a_x,), frame.y: (a_y,)},
)
A = model.rate("advection_rate", equation=ddt(U) == -div(F))

fv = FiniteVolume(
    flux=F,
    variables=variables.Conservative(U),
    reconstruction=reconstruction.MUSCL(limiters.VanLeer()),
    riemann=riemann.ScalarUpwind(velocity=velocity),
)
disc.rates.add(A, fv)
```

La validation prouve que le flux demandé par la réalisation numérique est celui référencé par le taux.
Un ordre formel, une profondeur de halo ou une propriété CFL déjà définis par les composants ne sont
pas répétés dans l'API de haut niveau ; ils sont dérivés de leurs manifests et rapportés.

Les champs couplés séparent pareillement :

- `FieldOperator` : équation, inconnue, providers physiques et outputs dérivés ;
- `FieldDiscretization` : stencil, BC, solver, nullspace et gauge ;
- appel dans `Program` : instant logique et politique d'échec.

Un solve périodique singulier exige un contrat de nullspace et, lorsqu'une valeur unique est consommée,
une gauge. Le runtime ne corrige jamais silencieusement un second membre incompatible.

## 6. Domaine, maillage et layouts

### 6.1 Domaine et frame

```python
domain = Rectangle(
    "unit_square",
    lower=(0.0, 0.0),
    upper=(1.0, 1.0),
    boundaries=RectangleBoundaryNames(
        x_min="inlet_x", x_max="outlet_x",
        y_min="inlet_y", y_max="outlet_y",
    ),
).tag("fluid")
frame = domain.frame(Cartesian2D())
grid = CartesianGrid(frame=frame, cells=(128, 128))
```

Les frontières sont des handles topologiques issus du frame (`frame.boundaries.x_min`, etc.). Les
noms personnalisés sont des labels ; orientation, côté, périodicité et connexions restent typés.

Un `Case` ne possède pas son layout. Après validation, `case.layout_subjects()` expose l'ensemble
immuable des blocs, états et champs à assigner. Un `LayoutPlan` associe explicitement ces sujets à un
ou plusieurs layouts et porte les mappings/synchronisations nécessaires.

### 6.2 Autorité AMR

Un layout AMR agrège cinq facettes :

- `AMRHierarchy` : niveaux et ratios par transition ;
- `AMRTagging` : graphe de prédicats, décisions, hystérésis et conflits ;
- `AMRRegrid` : cadence et règle de reconstruction ;
- `AMRTransfer` : politique par espace/état ;
- `AMRExecution` : relation temporelle entre niveaux.

```python
layout = AMR(
    grid=grid,
    hierarchy=AMRHierarchy(max_levels=..., ratios=(...)),
    tagging=tagging,
    regrid=AMRRegrid(schedule=every(5, clock=T.clock)),
    transfer=transfer,
    execution=AMRExecution.subcycled(),
)
```

Le transfert appartient au layout et n'est pas ajouté une seconde fois au `DiscretizationPlan`.
Les seuils de tagging sont des paramètres du `Case` et sont donc résolus/bindés comme toute autre
valeur. Une expression telle que `norm(grad(ValueExpr(block[U]))) > case.value(threshold)` est
résolue dans un contexte discret explicite ; elle n'invente pas un gradient continu exécutable.

Le plan normalisé conserve chaque ratio de transition et le raffinement cumulé de chaque niveau.
Un provider natif peut annoncer un sous-ensemble, par exemple Cartesian2D et transitions isotropes
2:1. Il doit alors refuser toute autre demande pendant la résolution ou le bind, avec ses capacités
exactes. Le coeur de planification ne remplace pas la demande par ce sous-ensemble.

Les critères booléens et les politiques de transfert sont des protocoles authentifiés ouverts. Une
nouvelle implémentation fournit données canoniques, requirements/capabilities et lowering ; elle ne
nécessite pas un `isinstance` ajouté à chaque phase centrale.

### 6.3 Frontières et conditions initiales

Une BC de transport est enregistrée une fois dans `DiscretizationPlan.boundaries`. Une BC de champ
appartient à son `FieldDiscretization`. Une interface multibloc et une frontière coarse/fine sont des
ports distincts. Le graphe de producteurs de ghosts prouve la couverture, la profondeur, le temps et
les dépendances de chaque région avant exécution.

Une condition initiale associe : handle d'état qualifié, donnée, projection et éventuellement preuve
de reprojection AMR. Pour un bootstrap AMR, la projection analytique, prolongation et restriction sont
des choix explicites. Le bind refuse deux autorités concurrentes (`initial_state` et plan IC AMR).

## 7. Programme de temps

### 7.1 Langage générique

`pops.Program` est un builder de graphe SSA, pas une boucle Python. Les opérateurs principaux sont :

- `state`, `value`, appel d'opérateur et appel de champ ;
- `solve(problem, solver=...)` puis consommation explicite de l'outcome ;
- `keep_history` et lecture qualifiée de l'historique ;
- `subcycle` et `synchronize` pour les domaines d'horloge ;
- contrôle structuré typé ;
- `commit` et `commit_many` ;
- `step_strategy` pour le contrôleur et le contrat transactionnel.

Écriture SSPRK2 normative :

```python
T = pops.Program("SSPRK2")
q = T.state(block[U])
k0 = T.value("ssprk2_k_0", A(q.n), at=stage_0)
q_stage = T.value("ssprk2_U1", q.n + T.dt * k0, at=stage_1)
k1 = T.value("ssprk2_k_1", A(q_stage), at=stage_1)
q_next = T.value(
    "ssprk2_step",
    q.n + T.dt * Fraction(1, 2) * k0 + T.dt * Fraction(1, 2) * k1,
    at=q.next.point,
)
T.commit(q.next, q_next)
T.step_strategy(AdaptiveCFL(cfl=0.45, max_dt=1.0e-2))
```

`pops.lib.time.SSPRK2(block[U], rate=A)` existe comme factory, mais son graphe canonique doit être
identique à cette écriture. Le nom du preset ne sélectionne aucune branche runtime.

### 7.2 Explicite, implicite et IMEX

Un appel explicite évalue un opérateur à un `TimePoint` ou `StagePoint` exact. Un solve implicite
sépare le problème mathématique du solveur :

```python
result = T.solve(
    LocalLinear(operator=L, rhs=b, fields=field_context),
    solver=DenseLU(),
    name="local_linear_stage",
).consume(action=RejectAttempt())
```

Pour un résidu non linéaire local, `LocalResidual` est résolu avec `LocalNewton`. Pour un couplage
multi-états à l'étape suivante, `CoupledImplicitEuler` reçoit le taux couplé, les prédicteurs et les
points qualifiés, puis un `LocalNewton`. Les problèmes linéaires de champ utilisent
`pops.linalg.LinearProblem` et un solveur typé (`GMRES`, `CG`, etc.). Tolérance, budget, stratégie et
préconditionneur appartiennent au solveur, jamais au résidu.

Un outcome fallible doit être consommé par une action adaptée à sa phase (`RejectAttempt`, `FailRun`,
etc.) avant que sa valeur puisse contribuer à un commit ou un effet.

Un schéma IMEX/ARK porte les abscisses exactes de chaque partition dans ses `StagePoint`. Les
coefficients sont rationnels/exacts lorsqu'ils le sont mathématiquement. Un certificat d'ordre ou SSP
est dérivé du tableau/graphe ; l'utilisateur ne répète pas `order=2` si le tableau l'établit.

### 7.3 Horloges, multirate et synchronisation

Une horloge est un handle qualifié. Un sous-cycle crée un domaine d'horloge enfant avec sa relation de
pas. Une lecture cross-clock est interdite sans `synchronize` et sans politique d'échantillonnage
explicite. Les schedules ciblent leur horloge ; ils ne lisent jamais implicitement le macro-step global.

`Program.subcycle(state, clock=child, within=parent, count=N, body_fn=...)` est un contrôle structuré :
`child` et `parent` sont distinctes, `N` est strictement positif, le corps conserve le state qualifié et
voit exactement `parent_dt / N`. Les sous-cycles imbriqués composent leurs ratios. L'entrée et la sortie
du domaine enfant restent deux appels explicites à `synchronize`. `SampleAndHold()` est le premier
provider natif ; tout autre provider sans lowering déclaré est refusé avant publication de l'artefact.

`Program.temporal_manifest()` authentifie la clock primaire, les ratios et parents, les points de
synchronisation, les schedules/caches et, pour chaque historique, owner, state, espace, clock, lag,
taille de ring, interpolation, domaine de validité et politique de checkpoint. Toute clock non primaire
doit avoir une route unique vers la clock primaire. Un historique AMR sur clock enfant exige une
capacité composée AMR-level/clock et un provider de dense output ; sans cette preuve il est refusé, il
n'est jamais exécuté à une fausse cadence macro.

Le restart temporel schema v2 sauvegarde le manifeste exact, les clocks qualifiées et leurs
compteurs/phases acceptés, les cursors de sous-cycles/schedules/synchronisations, la validité des
historiques et caches, l'event queue, les statistiques et l'état du contrôleur. Il n'est sérialisable
qu'à un point accepté et entièrement synchronisé. Une clock attendue mais absente fait échouer le
restart et les consommateurs ; elle n'est jamais reconstruite depuis `macro_step`. Les anciens schemas
sont des entrées de migration offline, pas des branches du runtime.

### 7.4 Transaction d'un pas

La transaction couvre au minimum : états, champs provisoires, histories, clocks, caches, solve outcomes,
flux ledgers, regrid planifié et effets consommateurs. Le protocole est :

```text
begin attempt -> stages/solves/sync -> guards -> prepare effects
    -> accept: commit atomique puis publication
    -> reject/error: rollback complet, aucune publication
```

`FixedDt` est transactionnel au même titre qu'un contrôleur adaptatif. `pops.run` refuse `strategy=` et
`cfl=` : la stratégie appartient au `Program`, les contrôles d'exécution restent numériques.

## 8. Case et cycle de vie public

### 8.1 Assemblage

```python
case = pops.Case("case")
block = case.block("tracer", model=model)
case.numerics(disc, block=block)
case.initials.add(initial_condition)
case.program(T)
case.consumers(consumer_graph)
```

`states=` sur `case.block` est uniquement une sélection explicite d'un sous-ensemble déclaré ; il est
omis lorsque tout l'espace du modèle est instancié. Les sorties, diagnostics et AMR utilisent ensuite
`block[U]`, pas `U` non qualifié.

### 8.2 Phases exactes

```python
validated = pops.validate(case)
resolved = pops.resolve(
    validated,
    layout=layout_plan,
    layout_providers=providers,
    backend=Production(),
    platform=platform_manifest,
    compile_options={"debug": False},
)
artifact = pops.compile(resolved)
simulation = pops.bind(
    artifact,
    initial_state=initial_state,
    params=params,
    aux=aux,
    resources=resources,
    initial_values=initial_values,
)
report = pops.run(simulation, t_end=1.0, max_steps=100_000, output_dir=output_dir)
```

Contrat des phases :

| Phase | Entrée | Sortie | Interdictions |
| --- | --- | --- | --- |
| `validate` | exact `Case` mutable | même `Case` gelé | import natif, mutation scientifique |
| `resolve` | `Case` gelé + autorités typées | plan résolu immuable | tableaux runtime, fallback |
| `compile` | exact plan résolu | artefact authentifié | réinterprétation des choix |
| `bind` | artefact + cinq familles de valeurs | instance runtime | changement de structure/algorithme |
| `run` | instance bindée + contrôles numériques | rapport d'exécution | authoring ou sélection de stratégie |

Les seules options de compilation sont celles acceptées par `pops.resolve(..., compile_options=...)` :
`so_path`, `force`, `cxx`, `include`, `std` et `debug`. Le backend est une autorité séparée. Il n'existe
pas de `CompileConfig` public, de `strict=True`, de `sim.run`, ni de `RejectOldManifest`.

`pops.bind` accepte exactement cinq familles : `initial_state`, `params`, `aux`, `resources` et
`initial_values`. L'enregistrement interne qui les authentifie n'est pas importé par l'utilisateur.

### 8.3 Quatre catégories d'implicite

Toute valeur effective appartient à une catégorie rapportée :

1. **Dérivation exacte** : conséquence unique des objets choisis (ordre d'un tableau, halo d'un stencil).
2. **Défaut unique documenté** : valeur neutre/sûre ayant une seule signification dans ce contrat.
3. **Choix scientifique explicite** : solveur, limiteur, BC, projection, stratégie, tolérance, transfert.
4. **Heuristique explicitement demandée** : autotuning ou sélection selon un objectif fourni.

Une ambiguïté entre plusieurs choix valides est une erreur. Une heuristique n'est jamais activée par
omission. Tous les défauts et dérivations entrent dans les rapports et, s'ils changent le comportement,
dans l'identité sémantique.

## 9. Consommateurs, sorties et restart

`ConsumerGraph` est l'unique autorité des effets acceptés : `ScientificOutput`, `Checkpoint` et
diagnostics planifiés. Chaque consommateur déclare schedule, handles qualifiés, sélection de niveaux,
format, cible déterministe et comportement d'échec.

Les formats livrés sont des descripteurs (`HDF5`, `NPZ`, `ParaView`) abaissés vers des writers réels.
Les tests de release rouvrent leurs fichiers indépendamment avec les readers publics ; l'existence du
fichier seule n'est pas une preuve.

Un checkpoint strict conserve au minimum : identités du plan/programme/composants/consumer graph,
états, champs matériels requis, histories, clocks/schedules, contrôleur, hiérarchie AMR, cursors des
consommateurs et contrat de plateforme. Un restart refuse toute divergence non autorisée. La garantie
bit-identique est prouvée par continuation indépendante, pas par comparaison du manifest seul.

## 10. Extension et C++

### 10.1 Petites interfaces

Une famille extensible expose des facettes minimales, par exemple : validation, données sémantiques,
requirements, capabilities, lowering et inspection. Une classe n'implémente que les facettes utiles.
Les agrégats (`AMR`, plan numérique, pack de providers) authentifient ces protocoles ; ils ne dépendent
pas des classes de `pops.lib`.

Une extension scientifique est recevable si :

- elle possède un identifiant namespacé et une version ;
- ses données comportementales sont canoniques et couvertes par le digest sémantique ;
- ses besoins/capacités et effets sont fermés et validés ;
- son point d'entrée natif est déclaré dans un manifest ;
- son lowering est total pour la route acceptée ;
- un test externe l'ajoute sans modifier les passes centrales.

Une nouvelle famille de layout, de noeud Program, de centering fondamental, de ressource transactionnelle
ou d'ABI est une extension du coeur et exige une évolution versionnée du contrat.

### 10.2 Manifests et catalogue

`schemas/component_catalog.v2.json` est l'autorité des composants builtin. Le générateur produit les
IDs/routes Python et C++ ; `--check` interdit leur dérive. `ComponentManifest` couvre signature, ports,
paramètres, interfaces, requirements, capabilities, effets, layouts, clocks, déterminisme, précision,
restart et points d'entrée.

Les champs sémantiques inconnus, capacités sans preuve, collisions d'identité et entry points manquants
sont refusés. Un vieux manifest n'est pas « réparé » silencieusement.

### 10.3 Performance

Les kernels chauds sont C++20, device-callable et exécutés par Kokkos. Les vues sont triviales et ne
font ni allocation, ni réflexion Python, ni polymorphisme dynamique par cellule. La communication MPI,
les fences, espaces mémoire et streams sont des ressources planifiées. Les rapports permettent
d'attribuer allocations, transferts, halo exchanges, solves, regrids et sorties.

Le contrat générique n'autorise pas une abstraction coûteuse dans la boucle chaude : la composition
haut niveau est résolue/compilée avant exécution et les décisions statiques sont abaissées en types,
tables compactes ou code généré.

## 11. Erreurs et refus

PoPS promet les catégories et les preuves d'erreur, pas une liste fictive de classes d'exception. Une
erreur doit indiquer : phase, chemin qualifié, code/catégorie stable, demande, capacité observée,
alternatives explicites lorsqu'elles existent, et provenance source.

Cas obligatoirement refusés :

- `Expr` utilisé comme booléen Python ;
- handle non qualifié ou owner incompatible dans un `Case` multi-instance ;
- descripteur/string sélectionnant une sémantique ;
- BC, transfert, layout, programme ou paramètre avec deux autorités ;
- champ périodique incompatible avec nullspace/gauge ;
- solve outcome non consommé ;
- lecture cross-clock sans synchronisation ;
- historique ou clock attendu absent du restart ;
- ratio/dimension/layout non supporté par le provider installé ;
- champ sémantique accepté mais non abaissé ;
- dépassement de capacité qui serait autrement tronqué ;
- erreur de begin/stage/commit d'une transaction laissant un état partiel ;
- sortie publiée depuis un pas rejeté ;
- unité opaque sur la route d'état publique.

`pops.inspect(obj)` produit une vue structurée sans importer arbitrairement le runtime natif.
`pops.explain(obj)` rend une explication orientée utilisateur à partir des mêmes données ; il ne
recalcule pas un second diagnostic.

## 12. Capacités livrées et limites explicites

La release est conforme uniquement pour les lignes prouvées par la matrice native et les exemples :

- layouts Uniform et AMR structuré sur les routes annoncées ;
- physique conservative hyperbolique, sources locales/couplées et champs elliptiques couplés ;
- programmes explicites, solves locaux/linéaires, IMEX et sous-cycles/synchronisations abaissables ;
- HDF5, NPZ, ParaView et checkpoint/restart transactionnels ;
- Kokkos comme backend on-node, MPI lorsque compilé et authentifié ;
- packages C++ externes conformes au manifest et à l'ABI courants.

Les dimensions, ratios, nombres de niveaux, géométries, solveurs et combinaisons device réellement
exécutables sont lus dans les manifests/capability reports. Le provider AMR livré peut être plus étroit
que l'IR (notamment Cartesian2D et ratio isotrope 2:1) ; cette restriction est un refus pré-run, pas une
normalisation du plan.

Les maillages non structurés, mobiles/déformables ou changeant de topologie, de nouvelles familles de
stockage, la complétion 3D de toutes les routes et une algèbre d'unités demandent de futurs contrats du
coeur. Ils ne sont pas simulés par des placeholders publics.

## 13. Exemples exécutables normatifs

Quatre scripts sont des tests d'acceptation, pas des esquisses :

1. `examples/final/EXEMPLE_SPEC_FINALE_ADVECTION_SCALAIRE_COMPLET.py` : flux conservatif, SSPRK2
   manuel/preset, AMR, sorties, checkpoint et continuation bit-identique ;
2. `examples/final/EXEMPLE_SPEC_FINALE_MULTIPHYSIQUE_CORE.py` : deux fluides, champ elliptique,
   couplage implicite, layouts qualifiés et restart ;
3. `examples/final/EXEMPLE_SPEC_FINALE_ADVECTION_IMEX_AMR.py` : partitions explicite/implicite,
   points de stage exacts, sous-cycles AMR et synchronisation ;
4. `examples/final/EXEMPLE_SPEC_FINALE_15_MOMENTS_HYQMOM.py` : modèle 15 moments, closures et
   opérateurs génériques sans branche de scénario dans le coeur.

Chaque script doit :

- utiliser exclusivement le cycle de vie public ;
- sortir avec un code nul depuis le package installé ;
- produire et rouvrir ses artefacts réels ;
- exercer le restart strict lorsqu'il le déclare ;
- échouer sans fallback si une capacité nécessaire manque ;
- ne pas importer une classe interne pour remplacer un trou d'API.

## 14. Gate de conformance finale

Une release ne peut être déclarée conforme que si une seule gate groupée produit une evidence JSON
liée au commit, à la version du package et au digest du release contract. L'evidence est générée depuis
les retours de commandes et ne contient pas de booléens fournis à la main.

La gate couvre :

1. `scripts/setup_env.sh` puis `scripts/build_python.sh` ;
2. `pops.runtime.doctor.doctor()` sur le package installé ;
3. signature/code-signing des extensions lorsque la plateforme l'exige ;
4. suite Python complète et tests C++ via CTest ;
5. les quatre exemples exacts ;
6. réouverture HDF5/NPZ/ParaView et restart strict indépendant ;
7. génération/catalogue/schémas sans dérive ;
8. `docs/check_docs.py` ;
9. `git diff --check` et worktree de release propre.

`scripts/release_preflight.py --release` refuse une evidence incomplète, issue d'un autre commit ou
d'un autre digest. Les tests conditionnellement ignorés ne peuvent couvrir une exigence finale ; seuls
les lanes de toolchain explicitement non disponibles peuvent être conditionnels.

## 15. Décisions finales

- L'interface est objet, operator-first, proche des équations et organisée par responsabilités.
- `Handle` et `Expr` restent séparés pour respecter le data model Python.
- Le flux physique explicite et `DiscretizationPlan` organisé par familles sont conservés.
- Le programme temporel explicite est la norme ; `pops.lib.time` apporte des factories équivalentes.
- Le domaine produit des handles de frontières typés ; les labels ne portent pas la sémantique.
- Les BC appartiennent au plan numérique ; les transferts appartiennent au layout AMR.
- Le `Case` instancie et qualifie, tandis que le `LayoutPlan` matérialise séparément.
- L'implicite utilise un problème/résidu complet et un solveur séparé.
- L'AMR conserve ses ratios et raffinements sans les compresser dans une constante globale.
- Le multirate est fondé sur des clocks qualifiées et des synchronisations explicites.
- Tous les pas et effets sont transactionnels.
- La généricité vient de petites interfaces et de manifests, pas de branches par classes.
- Les limites natives sont exposées comme capacités et refus propres.
- `pops.units` reste hors périmètre jusqu'à l'existence d'un vrai système typé.
- Aucun alias historique, fallback silencieux ou promesse non exécutable ne fait partie de la cible.

## 16. Sources de vérité associées

- `README.md` : installation et premier parcours public ;
- `docs/ARCHITECTURE.md` : architecture C++ et conventions de maillage ;
- `docs/VERSIONING.md` : surfaces versionnées et politique de rupture ;
- `docs/design/native-capability-matrix.md` : capacités providers/plateformes ;
- `docs/design/consumer_graph_transaction_contract.md` : effets acceptés et rollback ;
- `docs/design/temporal-execution-contract.md` : clocks, sous-cycles et restart temporel v2 ;
- `docs/design/external-component-packages.md` : extension C++ externe ;
- `schemas/release_contract.v1.json` : versions de schémas, ABI et matrice supportée ;
- `schemas/component_catalog.v2.json` : composants builtin et routes natives ;
- les quatre scripts de `examples/final/` : conformance utilisateur exécutable.

Toute divergence entre ce document et une gate exécutable est un défaut à corriger. Une fonctionnalité
non prouvée est limitée ou refusée ; elle n'est pas documentée comme livrée.
