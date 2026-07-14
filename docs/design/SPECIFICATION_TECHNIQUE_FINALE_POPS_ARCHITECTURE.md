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

Les unités opaques ne sont pas une extension tolérée. Il n'existe ni module `pops.units`, ni
descripteur public d'unité. `Model.state(..., units=...)` refuse toute valeur autre que `None` ;
`Module.state_space(..., units=...)` et `Module.field_space(..., units=...)` n'acceptent que `None` par
composante. Le refus précède l'identité et le lowering. Les espaces livrés sont explicitement sans
unité ; aucune string telle que `"kg/m3"` ou `"V/m"` n'est conservée comme métadonnée décorative.

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

La surface publique de `pops.codegen` est exactement `Production`, `CompilerLowerable` et
`CompilerLowering`. Les plans résolus, artefacts compilés, enregistrements d'installation et helpers
de validation restent internes ; ils ne constituent pas un second cycle de vie public.

### 3.2 Métadonnées exactes du modèle compilé

Après compilation, chaque modèle attaché à un bloc fournit ses faits d'artefact par l'unique protocole
structurel interne `ArtifactModelMetadataProvider`, c'est-à-dire la méthode
`__pops_artifact_model_metadata__()`. Elle retourne un dictionnaire exact de schéma v1 avec les seules
clés suivantes :

```text
schema_version, state_spaces, cons_names, n_vars,
params, aux_names, n_aux, capabilities
```

Cette projection est fail-closed : aucune lecture opportuniste d'attribut, aucun compte fabriqué et
aucun fallback vers le premier modèle ne sont admis. `n_vars` égale exactement la taille de
`cons_names`, `n_aux` couvre au moins tous les `aux_names`, les capacités associent des noms non vides
à des booléens exacts, et la route `state_spaces` doit être identique à celle du bloc résolu. Le runtime
natif livré exige ici exactement un espace d'état nommé par bloc. Cette interface sert aux rapports,
au calcul mémoire et aux contrôles de bind ; elle ne réintroduit pas une autorité d'authoring.

### 3.3 Contrat natif obligatoire du module Program

Chaque bibliothèque `Program` exporte une seule famille complète de métadonnées qualifiées : identité
du registre de routes, opérateurs `(owner, name, kind, signature, requirements)`, espaces d'état et
espaces de champ avec leur owner. Tous les compteurs et accesseurs sont obligatoires, y compris pour
une table vide. Une valeur vide, un doublon qualifié, un JSON de requirements mal formé, un symbole
absent ou un registre de routes différent refuse l'artefact avant l'appel de son installer ; un ancien
module n'est jamais exécuté en sautant l'introspection.

`System` et `AmrSystem` appliquent les mêmes contrôles de requirements sur toutes les plateformes :
instances de blocs, solveur de champ et champs auxiliaires fournis. En AMR, `B_z` exige une donnée
installée avant le `Program`; `T_e` est refusé tant qu'aucun provider AMR typé ne l'implémente. Aucun
canal auxiliaire absent n'est interprété comme zéro et aucune validation n'est reportée au premier pas.

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
produit des espaces distincts de taille arbitraire. La famille publique de rôles est exactement
`ComponentRole`, `Density`, `Momentum`, `Energy`, `Velocity`, `Pressure`, `Temperature` et `Scalar`.
`Momentum(axis=...)` et `Velocity(axis=...)` exigent un axe cartésien typé `x`, `y` ou `z`. Une string
de rôle est refusée ; un token natif inconnu ou réservé et deux rôles qui entrent en collision sur le
même token ABI sont également refusés. Les rôles décrivent la physique d'une composante : ils ne sont
ni inférés par position, ni confondus avec des unités, ni consultés par un `Program` générique.

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
- `FieldDiscretizationProtocol` : stencil, BC, solver, nullspace et gauge ;
- appel dans `Program` : instant logique et politique d'échec.

L'unique autorité callable d'un solve est le `FieldHandle` retourné par
`field = case.field(operator, discretization)`, puis `field(stage_state)`. Les handles de providers
du modèle décrivent seulement les contributions physiques au second membre ; ils ne sont jamais une
route concurrente de solve. Le `FieldContext` reprend exactement les composantes du `FieldSpace`
enregistré. À `resolve`, chaque nœud de solve doit correspondre à exactement un plan de champ du
`Case`, avec la même identité et les mêmes outputs ; zéro correspondance, une ambiguïté ou une
divergence est refusée avant `compile`.

`FieldDiscretization` est l'implémentation builtin de ce protocole, pas une classe centrale à laquelle
les extensions doivent être ajoutées. Tout provider porte un `provider_id` non vide et projette un
schéma canonique v2 exact par `to_data()` ; le `provider_id` de l'objet et celui de cette projection
doivent coïncider. Enregistrement, gel, résolution des références, validation, inspection et lowering
consomment le protocole sans dispatch sur la classe concrète du plan.

Un solve périodique singulier exige un contrat de nullspace et, lorsqu'une valeur unique est consommée,
une gauge. Le runtime ne corrige jamais silencieusement un second membre incompatible.

## 6. Domaine, maillage et layouts

Les layouts publics vivent uniquement dans `pops.layouts` :

```python
from pops.layouts import AMR, Uniform
```

Il n'existe pas de surface `pops.mesh.layouts`. `pops.mesh` conserve les grilles, géométries, politiques
AMR et builders de `LayoutPlan`, mais ne réexporte ni `AMR` ni `Uniform`.

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
immuable des blocs, états et champs à assigner. Un `LayoutPlan` associe explicitement ces sujets aux
layouts et porte les mappings/synchronisations nécessaires. Le plan sait représenter plusieurs
affectations, mais le provider d'exécution livré matérialise exactement un layout par artefact : un
`LayoutPlan` hétérogène est refusé avant la création de l'artefact, sans choix d'un représentant.

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
Le provider natif livré matérialise le coeur maillage/stockage en 2D et ses kernels de transfert,
reflux et sous-cyclage AMR exigent un ratio de transition égal à 2. Une autre dimension ou un autre
ratio est refusé pendant la résolution ou le bind avec les capacités observées. Le coeur de
planification ne normalise jamais la demande vers ce sous-ensemble.

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

Écriture SSPRK2 normative, uniquement avec les opérations génériques de `Program` :

```python
from fractions import Fraction
from pops.time import StagePoint, TimePoint

def explicit_ssprk2(state, rate):
    program = pops.Program("SSPRK2")
    q = program.state(state)
    stage_0 = StagePoint(
        "ssprk2_stage_0", {"main": TimePoint(program.clock, 0)}
    )
    stage_1 = StagePoint(
        "ssprk2_stage_1", {"main": TimePoint(program.clock, 1)}
    )
    k0 = program.value("ssprk2_k_0", rate(q.n), at=stage_0)
    q_stage = program.value(
        "ssprk2_U1", q.n + program.dt * k0, at=stage_1
    )
    k1 = program.value("ssprk2_k_1", rate(q_stage), at=stage_1)
    half = Fraction(1, 2)
    q_next = program.value(
        "ssprk2_step",
        q.n + program.dt * half * k0 + program.dt * half * k1,
        at=q.next.point,
    )
    program.commit(q.next, q_next)
    return program
```

`pops.lib.time.SSPRK2(block[U], rate=A)` retourne un `pops.Program` dont le graphe canonique est
identique à cette écriture. Les factories publiques de programme portent toutes un nom capitalisé :
`ForwardEuler`, `RungeKutta`, `RK4`, `SSPRK2`, `SSPRK3`, `IMEX`, `AdamsBashforth`, `BDF`,
`PredictorCorrector`, `Lie` et `Strang`. `ButcherTableau` et les constantes `*_TABLEAU` sont des
données. Il n'existe ni callable public minuscule concurrent, ni namespace `std`, ni seconde classe
de stepper ; un nom de factory ne sélectionne aucune branche runtime.

### 7.2 Explicite, implicite et IMEX

Un appel explicite évalue un opérateur à un `TimePoint` ou `StagePoint` exact. Un solve implicite
sépare le problème mathématique du solveur :

Un handle callable trouve normalement son `Program` dans ses arguments (`A(q.n)`). Un opérateur
réellement nul utilise la même route operator-first avec l'autorité explicite `L(program=T)` ;
`program=` est refusé si des arguments `ProgramValue` rendent cette autorité redondante.

```python
result = T.solve(
    LocalLinear(operator=L, rhs=b, fields=field_context),
    solver=DenseLU(),
    name="local_linear_stage",
).consume(action=RejectAttempt())
```

Pour un résidu non linéaire local, `LocalResidual` est résolu avec `LocalNewton`. Pour un couplage
multi-états à l'étape suivante, `CoupledImplicitEuler` reçoit le taux couplé, les prédicteurs et les
points qualifiés, puis un `LocalNewton`. La route globale livrée construit un opérateur matrix-free,
l'encapsule dans `pops.linalg.LinearProblem`, puis le résout avec `GMRES` ou `BiCGStab`. Pour
`scope=Hierarchy()`, le provider natif explicite est `CompositeTensorFAC()` ; un scope hiérarchique
sans ce provider est refusé. Tolérance, budget, stratégie et préconditionneur appartiennent au solveur,
jamais au résidu. Il n'existe aucune route publique `Schur` ou `CondensedSchur`, ni dans
`pops.solvers`, ni dans `pops.lib.time`.

Un outcome fallible doit être consommé par une action adaptée à sa phase (`RejectAttempt`, `FailRun`,
etc.) avant que sa valeur puisse contribuer à un commit ou un effet.

À la frontière native, tous les solveurs itératifs retournent le même `SolveReport` : nombre
d'itérations, résidu relatif et une unique paire `SolveStatus` / `SolveAction`. Il n'existe pas de
booléen `converged` parallèle. Une valeur n'est résolue que pour `(kSolved, kNone)` ; toute paire
incohérente est traitée comme un échec et un appel de construction d'échec sans statut/action d'échec
est refusé. Le runtime ne publie jamais l'itéré ou le champ muté d'un report en échec et la transaction
restaure l'ensemble des valeurs acceptées précédentes. Un solveur généré doit porter un critère de
convergence scientifique explicite, distinct de son budget ; atteindre seulement la limite
d'itérations produit `kIterationLimit`, jamais un succès fabriqué.

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
du domaine enfant restent deux appels explicites à `synchronize`. La route native livrée abaisse
`SampleAndHold()` ; toute autre relation sans lowering déclaré est refusée avant publication de
l'artefact.

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

### 7.4 Algèbre et extension des schedules

Un schedule est le produit typé `Schedule(trigger, off=...)` de quatre petites interfaces ouvertes :

- `Domain.native_schedule_domain()` retourne un exact `ScheduleDomainIR` ;
- `Trigger.native_schedule_due()` retourne un exact `ScheduleDueIR` ;
- `OffPolicy.native_schedule_off()` retourne un exact `ScheduleOffIR` ;
- `Schedule.native_schedule_ir()` compose les trois en un exact `ScheduleLoweringIR`.

Une extension est un dataclass immuable et slotté, déclare un `manifest_tag`, projette toutes ses
données comportementales, conserve son type exact pendant le rebuild et participe à l'identité du
graphe. Un dictionnaire ressemblant à l'IR, un retour partiel ou un type transformé pendant le mapping
sont refusés ; aucune classe d'extension n'est enregistrée dans une liste centrale.

Le vocabulaire livré comprend les domaines `AcceptedStep`, `Attempt`, `Stage`, `ClockTick`,
`AMRLevel`, `Event` et `WallOutput`, les triggers `Always`, `Every`, `AtStart`, `AtEnd` et `When`, et les
politiques `Hold`, `Skip`, `Zero`, `AccumulateDt` et `Error`. Ces objets restent typés dans les manifests
et le `ConsumerGraph`. Pour un opérateur du `Program`, le lowering natif courant accepte la timeline
`AcceptedStep` et refuse explicitement un domaine ou une combinaison sans primitive backend ; le fait
qu'une valeur soit représentable dans l'IR n'annonce pas son exécution native.

### 7.5 Transaction d'un pas

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

`states=` sur `case.block` sélectionne des `StateHandle` déclarés par le modèle. Pour un bloc
exécutable, il peut être omis uniquement si le modèle déclare exactement un état ; un modèle
multi-états exige une sélection typée, non vide et sans doublon. Le label du bloc reste un nom
d'instance et ne choisit jamais l'espace
d'état : la route du compilateur est le `local_id` du handle sélectionné, conservé séparément dans
`ResolvedBlock.state_spaces`, le `ProgramModelGraph` et les métadonnées de l'artefact.

Le moteur natif livré accepte exactement un `StateSpace` par bloc. Une sélection de plusieurs espaces
est donc refusée à `resolve`; deux espèces évoluées indépendamment s'expriment par deux blocs qualifiés,
par exemple `case.block("electron_fluid", model, states=(electrons,))` et
`case.block("ion_fluid", model, states=(ions,))`. Les labels `electron_fluid` et `ion_fluid` peuvent
différer des noms d'espaces `electrons` et `ions` sans changer cette route. Sorties, diagnostics et AMR
utilisent ensuite `block[U]`, jamais `U` non qualifié.

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

### 8.4 Moteurs d'exécution privés

`pops.bind` retourne l'unique `RuntimeInstance` authentifiée et seul `pops.run(instance, **controls)`
la fait avancer. `RuntimeInstance` n'expose pas de méthode `run`. Les moteurs `System` et `AmrSystem`
existent uniquement derrière les modules privés d'installation ; `pops.runtime` ne les réexporte pas
et les chemins `pops.runtime.system`, `pops.runtime.amr_system` et `pops.runtime.mesh` n'existent pas.
Une application ne construit donc ni moteur, ni config native, ni plan d'installation pour contourner
`validate -> resolve -> compile -> bind -> run`.

## 9. Consommateurs, sorties et restart

`ConsumerGraph` est l'unique autorité des effets acceptés : `ScientificOutput`, `Checkpoint` et
diagnostics planifiés. Chaque consommateur déclare schedule, handles qualifiés, sélection de niveaux,
format, cible déterministe et comportement d'échec.

Le graphe est résolu avec le layout, authentifié dans le plan et l'artefact, puis détenu par
`RuntimeInstance`. Le snapshot de bind ne possède aucun registre parallèle `outputs` ou `diagnostics` :
les recréer à ce niveau constituerait une seconde autorité et est interdit.

Les formats livrés sont des descripteurs (`HDF5`, `NPZ`, `ParaView`) abaissés vers des writers réels.
La gate finale rouvre indépendamment chaque HDF5 et ParaView émis et vérifie leur contenu structurel ;
l'existence du fichier seule n'est pas une preuve. La route NPZ est exercée par l'exemple IMEX-AMR et
ses tests de format, sans être présentée comme une réouverture supplémentaire de la gate groupée.

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
- programmes explicites, solves locaux, IMEX et solve global matrix-free par `LinearProblem` avec
  `GMRES`/`BiCGStab`, plus `CompositeTensorFAC` pour la portée hiérarchique ;
- HDF5, NPZ, ParaView et checkpoint/restart transactionnels ;
- Kokkos comme backend on-node, MPI lorsque compilé et authentifié ;
- packages C++ externes conformes au manifest et à l'ABI courants.

Les dimensions, ratios, nombres de niveaux, géométries, solveurs et combinaisons device réellement
exécutables sont lus dans les manifests/capability reports. Le provider livré matérialise un seul
layout, un seul `StateSpace` par bloc, le coeur de stockage 2D et les transitions AMR de ratio 2 ; toute
demande hors de cette enveloppe est un refus pré-run, pas une normalisation du plan.

Les maillages non structurés, mobiles/déformables ou changeant de topologie, de nouvelles familles de
stockage, la 3D sur ces routes et une algèbre d'unités ne font pas partie de la release. Ils sont refusés,
pas simulés par des placeholders publics.

## 13. Exemples exécutables normatifs

Quatre scripts sont des tests d'acceptation, pas des esquisses :

1. `examples/final/EXEMPLE_SPEC_FINALE_ADVECTION_SCALAIRE_COMPLET.py` : flux conservatif, parité du
   `Program` SSPRK2 explicite avec `pops.lib.time.SSPRK2`, layout AMR, HDF5/ParaView, checkpoint et
   continuation bit-identique ;
2. `examples/final/EXEMPLE_SPEC_FINALE_MULTIPHYSIQUE_CORE.py` : deux `StateSpace` d'un même modèle
   sélectionnés dans deux blocs qualifiés, layout Uniform, champ elliptique, couplage, HDF5/ParaView et
   restart bit-identique ;
3. `examples/final/EXEMPLE_SPEC_FINALE_ADVECTION_IMEX_AMR.py` : parité graphe, identité sémantique et
   état accepté du `Program` IMEX explicite avec `pops.lib.time.IMEX`, coefficients/stages exacts,
   `AMRExecution.subcycled()`, regrid/reflux, HDF5/NPZ/ParaView, restart strict et continuation
   bit-identique ;
4. `examples/final/EXEMPLE_SPEC_FINALE_15_MOMENTS_HYQMOM.py` : état 15 moments, layout Uniform,
   `pops.lib.time.IMEX`, champ de Poisson, HDF5/ParaView et continuation bit-identique, sans branche de
   scénario dans le compilateur.

`scripts/final_release_contract.py` fixe cet ensemble exact : aucun cinquième script `.py` n'est admis
dans `examples/final/`. Chaque script doit :

- utiliser exclusivement le cycle de vie public ;
- sortir avec un code nul depuis le package installé ;
- accepter `--output-dir` et rester directement exécutable ;
- produire des artefacts réels ensuite rouverts par la gate ;
- imprimer les preuves `HDF5:`, `ParaView:`, `checkpoint:` et `bit-identical restart:` ;
- exercer le restart strict lorsqu'il le déclare ;
- échouer sans fallback si une capacité nécessaire manque ;
- ne pas importer une classe interne pour remplacer un trou d'API.

## 14. Gate de conformance finale

Une release ne peut être déclarée conforme que par
`scripts/run_final_gate.py --evidence <chemin-hors-checkout>`. La commande exige un checkout propre,
refuse d'écraser une evidence existante et produit une evidence JSON liée au commit, à la version du
package, au digest du release contract et au SHA-256 de l'extension native installée. L'evidence est
générée depuis les retours de commandes et ne contient pas de booléens fournis à la main.

La séquence groupée couvre exactement les onze lignes authentifiées suivantes :

1. `official_build` : `scripts/setup_env.sh`, `scripts/build_python.sh`, puis configure/build du preset
   CMake `serial` avec les headers `POPS_INCLUDE` du checkout validé ;
2. `doctor` : `pops.runtime.doctor.doctor()` sur le package installé, sans échec ;
3. `codesign` : `scripts/codesign_pops_extensions.py` sur les extensions installées ;
4. `native_conformance` : CTest complet avec JUnit non vide, sans skip, xfail, failure ni error ;
5. `python_conformance` : suite Python complète, puis lane obligatoire
   `not mpi and not hdf5` avec JUnit all-pass et sans skip caché ;
6. `examples` : les quatre scripts exacts depuis le package installé et leurs quatre marqueurs de preuve ;
7. `artifact_reopen` : parsing indépendant de chaque HDF5/NPZ/ParaView, puis réouverture de chaque
   HDF5 par `h5py` et de chaque archive/array NPZ par NumPy avec `allow_pickle=False` ;
8. `strict_restart` : checkpoint réel et digest complet de son arbre pour chaque exemple ;
9. `documentation` : `docs/check_docs.py` ;
10. `generated_products` : release contract et component catalog régénérés avec `--check` ;
11. `diff` : `git diff --check`, `git diff --cached --check` et checkout encore propre.

`scripts/release_preflight.py --release --tag <tag> --installed --evidence <json>` refuse une
evidence incomplète, issue d'un autre commit, d'un autre digest, d'un autre script de gate ou d'une autre
extension installée. Une exigence de la lane obligatoire ne peut pas être couverte par un test ignoré.

## 15. Décisions finales

- L'interface est objet, operator-first, proche des équations et organisée par responsabilités.
- `Handle` et `Expr` restent séparés pour respecter le data model Python.
- Le flux physique explicite et `DiscretizationPlan` organisé par familles sont conservés.
- Le programme temporel explicite est la norme ; `pops.lib.time` apporte des factories capitalisées
  qui retournent les mêmes `Program`.
- Le domaine produit des handles de frontières typés ; les labels ne portent pas la sémantique.
- Les BC appartiennent au plan numérique ; les transferts appartiennent au layout AMR.
- `pops.layouts` est l'unique surface de layout ; `pops.mesh.layouts` n'existe pas.
- Le `Case` instancie et qualifie, tandis que le `LayoutPlan` matérialise séparément.
- La route globale implicite est matrix-free : `LinearProblem`, `GMRES`/`BiCGStab`, et
  `CompositeTensorFAC` pour `Hierarchy`; aucune façade Schur n'est publique.
- L'AMR conserve ses ratios et raffinements sans les compresser dans une constante globale.
- Le multirate est fondé sur des clocks qualifiées et des synchronisations explicites.
- Les schedules étendent séparément domain, trigger, off-policy et IR sans registre de classes.
- Tous les pas et effets sont transactionnels.
- `ConsumerGraph` est l'unique autorité des sorties, diagnostics et checkpoints.
- La généricité vient de petites interfaces et de manifests, pas de branches par classes.
- Les limites natives sont exposées comme capacités et refus propres.
- `pops.units` n'existe pas ; les espaces sont unitless et refusent les unités opaques.
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
- `scripts/final_release_contract.py` : spécification et ensemble exact des quatre exemples ;
- `scripts/run_final_gate.py` : producteur unique de l'evidence groupée ;
- les quatre scripts de `examples/final/` : conformance utilisateur exécutable.

Toute divergence entre ce document et une gate exécutable est un défaut à corriger. Une fonctionnalité
non prouvée est limitée ou refusée ; elle n'est pas documentée comme livrée.
