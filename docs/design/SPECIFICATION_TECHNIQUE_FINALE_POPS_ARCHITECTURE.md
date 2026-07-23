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
pops.RunReport
pops.RunStopReason
pops.ExecutionContext
pops.set_threads
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

`pops.ExecutionContext` et `pops.set_threads` sont les deux contrôles d'exécution racine ; ils ne
constituent pas un second moteur. `ExecutionContext` matérialise les ressources authentifiées de
lancement. La route MPI actuelle se construit par
`pops.ExecutionContext.mpi_world(artifact)` après compilation et se transmet à
`pops.bind(..., resources={"execution_context": context})`. Le contexte obtient exclusivement du
module C++ le `MPI_COMM_WORLD` natif, son rang, sa taille et ses handles ABI ; aucun objet MPI Python
n'entre dans le contrat public ou privé. Elle refuse tout communicateur custom, toute extension
non-MPI et toute divergence de rang/taille ; aucune exécution série de repli n'est autorisée.

`pops.set_threads(n)` fixe, depuis Python, le nombre positif de threads du backend Kokkos OpenMP
déjà installé. Il doit être appelé avant la première initialisation ou allocation Kokkos ; il prépare
les variables standard `OMP_NUM_THREADS` et `KOKKOS_NUM_THREADS`. Un appel tardif ou un module compilé
sans Kokkos produit un avertissement explicite ; un backend qui ne consomme pas ces variables les
ignore. Cette fonction ne sélectionne pas l'espace d'exécution Kokkos : Serial, OpenMP, CUDA ou HIP
reste une propriété de l'installation native. Aucun descripteur Python inerte ne peut transformer un
build CPU en build GPU.

### 1.1 Objectifs

PoPS doit permettre de :

- décrire la physique avec des états, flux, sources, champs et taux typés ;
- choisir séparément discrétisation spatiale, frontières, maillage, temps, solveurs et sorties ;
- écrire un programme temporel explicite, implicite, IMEX ou multirate avec les mêmes primitives ;
- instancier plusieurs fois un modèle sans ambiguïté de handles ;
- refuser une capacité absente avant la première mutation native ;
- inspecter les décisions, identités, capacités et erreurs de chaque phase ;
- ajouter des composants scientifiques par de petites interfaces, sans branche centrale par classe ;
- exécuter les kernels de production en C++20/Kokkos dans l'`ExecutionContext` exact que le
  provider installé sait transporter, sans communicator ou device global implicite.

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

Un état utilise par défaut ses composantes conservatives comme coordonnées primitives identité. Un
modèle qui possède un vrai changement de coordonnées le déclare en une seule transaction : ordre
primitif, rôles et inverse conservatif ont une autorité commune.

```python
rho, rho_u, rho_v, energy = U
u = model.primitive("u", rho_u / rho)
v = model.primitive("v", rho_v / rho)
p = model.primitive("p", pressure_from(rho, rho_u, rho_v, energy))

model.primitive_state(
    rho, u, v, p,
    conservative=(
        rho,
        rho * u,
        rho * v,
        p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v),
    ),
    roles={
        "rho": Density(),
        "u": Velocity(axis=frame.x),
        "v": Velocity(axis=frame.y),
        "p": Pressure(),
    },
)
```

`primitive_state` accepte uniquement les variables exactes émises par ce `Model`, exige la même
arité que l'état conservatif et refuse un inverse qui lit une variable étrangère ou non sélectionnée.
La déclaration est atomique : une erreur ne laisse ni layout ni inverse partiellement installé. Cette
capacité est state-scoped ; le builtin mono-état fournit ce raccourci, tandis qu'un provider
multi-espèces doit fournir explicitement la même petite interface pour chacun de ses espaces d'état.

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

Un handle scientifique (`FluxHandle`, `FieldHandle`, etc.) et l'opérateur exécutable qui réalise
son calcul appartiennent à deux espaces de noms typés distincts. Le `Module` conserve leur
projection dans un registre `operator_bindings` séparé des alias d'opérateurs : la clé est le handle
scientifique exact, la cible est un `OperatorHandle` authentifié. Cette liaison participe au hash et
au manifest versionné. Elle autorise donc, sans collision, un flux et un taux (ou un champ et son
solveur) portant le même nom physique, mais interdit toute sélection par string, tout handle d'un
autre owner et toute substitution silencieuse de route.

Un modèle sans primitive de pression peut déclarer directement la paire signée requise par HLL,
attachée au handle exact du flux et aux axes typés :

```python
model.wave_speeds(
    F,
    frame=frame,
    values={frame.x: (s_min_x, s_max_x), frame.y: (s_min_y, s_max_y)},
)
hll = riemann.HLL(waves=riemann.waves.ExplicitPair())
```

`wave_speeds` refuse un flux étranger, un axe manquant ou supplémentaire, et un `Handle` utilisé
comme une expression. Un paramètre est lu explicitement avec `model.value(handle)`. La variante
générée depuis le Jacobien reste `model.wave_speeds_from_jacobian(eig=..., blocks=...)` et se lie à
`riemann.HLL(waves=riemann.waves.FromJacobian(...))`.

La source signée est une provenance de modèle, pas une heuristique d'installation. Son kind canonique
(`explicit_pair`, `jacobian` ou `pressure_derived`) entre dans le hash du module, son manifest versionné,
les métadonnées et l'évidence binaire de l'artifact. `CompiledModel` le conserve sous forme de donnée
détachée et le bind le confronte au provider demandé par HLL. Un artifact qui annonce des vitesses
d'onde sans cette provenance est invalide ; aucune source inconnue n'est reclassée en paire explicite.

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

La topologie d'un champ n'est pas déduite d'un booléen `connected`. Un provider `FieldTopology`
matérialise un masque, un label entier par degré de liberté, un vocabulaire de composantes connexes,
leur provenance et un digest canonique. Nullspace, compatibilité du second membre, gauge et solver
consomment exactement ce même quintuplet ; une permutation de labels, un masque divergent ou un digest
étranger est refusé avant le kernel. Il existe donc une base et une contrainte de gauge par composante
connexe, sans branche Poisson ni hypothèse d'un domaine globalement connexe.

`ExternalFieldSolver(topology=..., solver=...)` est une autorité indivisible : les deux composants,
leurs manifests, leurs interfaces et leurs paramètres sont appariés à `resolve`, préparés une seule
fois à `bind`, puis possédés jusqu'à la destruction du runtime. Les interfaces `FieldTopology` ABI v2
et `FieldSolver` ABI v2 forment cette paire sans alias ni table v1. `FieldSolver`
transporte une topologie globale répliquée (bornes du domaine, axes périodiques, métadonnées de tous
les patches et owners) et un tableau de vues locales. Sa requête transporte aussi sans substitution
le quintuplet préparé : masque, labels, vocabulaire de labels dont chaque ligne porte son
`struct_size`, provenance et digest. Ces données sont des copies runtime persistantes adossées à
l'autorité topologique immuable ; omission, remplacement ou mutation par le composant est refusé. La requête et
ses buffers d'autorité sont construits une seule fois après la matérialisation topologique, puis
réutilisés sans reconstruire le JSON ni allouer un tableau de patches à chaque solve.
`FieldTopology.prepare_topology` et `FieldSolver.solve` sont appelés
exactement une fois par matérialisation/solve, y compris sur un rang sans patch local ; il n'existe
pas de boucle de solve indépendante par patch. Chaque métadonnée porte les bornes d'indices, la
coordonnée physique de sa face basse, l'espacement, le centrage, l'identité qualifiée du `LayoutPlan`
source et l'identité du patch. L'identité dérivée de la matérialisation (géométrie, boxes, owners,
périodicité et recette topologique) reste distincte de l'identité source.

`PopsSolveReportV2` contient un unique statut scientifique typé, une action, `iterations`,
`reference_residual_norm`, `residual_norm`, `relative_residual` et une `reason` obligatoire. Pour
l'interface externe V2, la référence reste exactement `||R(x0)||`; le contrat interne préparé décrit
plus bas utilise distinctement `||b-A(0)||`. Il ne
contient ni booléen `converged`, ni résidus ambigus `initial`/`final`. Le ratio doit être cohérent avec
les deux normes (dénominateur `1` seulement lorsque la norme de référence est nulle) et un succès doit
vérifier `residual_norm <= max(relative_tolerance * reference_residual_norm, absolute_tolerance)`.
`IncompatibleRhs` est un échec scientifique explicite. L'entier retourné par le callback signale
uniquement un échec de transport ABI et ne fabrique jamais de statut scientifique.

La représentation matière est typée (`full`, couverture binaire, fraction cut-cell, ids matériau ou
leur combinaison), jamais simulée par un tableau de `1`. La route actuellement prouvée de bout en bout
est plus étroite que cette ABI : `Uniform(CartesianGrid)`, cell-centered, plein matériau, float64,
host et communicateur série. AMR, embedded boundary, multimatériau, GPU, MPI sans consensus global,
conditions de bord dépendantes d'un état/champ/temps et outer solve non linéaire sont refusés à
`resolve`; les accepter dans un manifest ne suffit pas à rendre l'adapter capable.

Cette route sélectionne, pour chacun des deux composants, exactement un variant cible
`{dimension: 2, scalar: "float64", device: "cpu"}`. Un variant uniquement 3D, ou plusieurs variants
2D CPU ambigus, est refusé avant compilation ; une vue 2D ne peut jamais être passée à un binaire
authentifié pour une autre dimension.

Le runtime possède un unique protocole de backend de champ pour les implémentations builtin et
externes : `rhs`, `phi`, configuration de frontière, préparation du second membre, `solve`,
finalisation, snapshot/restore et rapport topologique. Le chemin scientifique ne branche pas sur
« externe » après matérialisation. La provenance n'est pas exposée par un getter parallèle :
`RuntimeInstance.inspect()` et le `RunReport` publient le même schéma `field_providers`, avec
l'autorité déclarée, l'identité du layout source, l'état matérialisé, le digest/provenance observés et
les métriques exactes des patches. Avant matérialisation, les faits runtime sont `None` et la liste de
patches est vide ; aucune valeur sentinelle n'est inventée.
La préparation applique le même contrat de compatibilité du nullspace et la même mise à l'échelle
physique aux deux backends ; la finalisation applique la gauge déclarée avant les ghosts. Une réussite
externe n'est synchronisée ni publiée qu'après vérification de la finitude de chaque degré de liberté
matériel actif. Un échec conserve le snapshot publié et ne copie jamais une sortie fournisseur
partielle ou non finie vers le device.

## 6. Domaine, maillage et layouts

Les layouts publics vivent uniquement dans `pops.layouts` :

```python
from pops.layouts import AMR, Uniform
```

Il n'existe pas de surface `pops.mesh.layouts`. `pops.mesh` expose les grilles, géométries et builders
de `LayoutPlan`, mais ne réexporte ni `AMR`, ni `Uniform`. Son implémentation privée vit dans
`pops.mesh._amr` ; l'ancien chemin `pops.mesh.amr` n'existe pas. L'authoring adaptatif public vit
exclusivement dans `pops.amr`.

### 6.1 Domaine et frame

```python
from pops.domain import Rectangle, RectangleBoundaryNames
from pops.frames import Cartesian2D
from pops.mesh import CartesianGrid, PeriodicAxes

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

# Pour identifier les deux paires de faces opposées :
periodic_grid = CartesianGrid(
    frame=frame,
    cells=(128, 128),
    periodic=PeriodicAxes(frame.axes),
)

# Présentation pure : sauvegarde le rectangle et ses frontières.
domain.show(path="unit_square.svg")
```

Les frontières sont des handles topologiques issus du frame (`frame.boundaries.x_min`, etc.). Les
noms personnalisés sont des labels ; orientation, côté, périodicité et connexions restent typés.
`PeriodicAxes` accepte uniquement des axes du frame, sans booléen, string ou indice. La topologie
canonique dérive alors les paires périodiques et les axes physiques complémentaires. Omettre
`periodic` est le défaut conventionnel borné ; ce défaut est visible dans l'identité et l'inspection.

`CartesianGrid` est l'unique descripteur cartésien public : il n'existe ni descripteur carré
concurrent, ni raccourci entier/tuple dans les APIs qui demandent une grille. Le domaine, le frame,
les cellules et la topologie restent donc visibles et authentifiables. `pops.mesh.PolarMesh` demeure
un descripteur avancé supporté pour l'anneau natif ; il ne constitue pas une seconde route
cartésienne et n'est pas réexporté à la racine `pops`.

`Rectangle.preview(geometry=...)` et son raccourci `Rectangle.show(...)` constituent la surface de
présentation du domaine. Sans géométrie, le renderer montre les bornes et les labels des quatre
frontières. Avec une géométrie, il passe exclusivement par le protocole
`Geometry.level_set(frame) -> LevelSet` : `Disc`, `HalfPlane`, un `LevelSet` analytique ou toute
composition CSG utilisent donc le même échantillonneur, sans branche par forme. Le sampling NumPy et
le renderer Matplotlib sont hors du runtime numérique ; ils ne créent aucun layout et n'entrent dans
aucun kernel. Matplotlib est importé seulement par `show()`. Fournir `path="domain.svg"` enregistre
sans ouvrir de fenêtre ; omettre `path` ouvre la vue interactive.

Le `SystemConfig` uniforme livré ne possède encore qu'un scalaire `n`, un scalaire `L` et aucune
origine. Son lowering accepte donc un `CartesianGrid` seulement si `lower == (0, 0)`, si les deux
longueurs sont égales et si les deux nombres de cellules sont égaux. Toute grille rectangulaire,
anisotrope ou translatée est refusée avant construction du moteur ; elle n'est jamais aplatie vers
un carré représentatif. Son unique booléen natif de périodicité représente exactement deux cas :
aucun axe périodique ou tous les axes périodiques. Une topologie partielle reste valide dans la DSL,
mais ce backend la refuse avant bind tant qu'il ne sait pas conserver cette partition par axe.

Un `Case` ne possède pas son layout. Après validation, `case.layout_subjects()` expose l'ensemble
immuable des blocs, états et champs à assigner. Un `LayoutPlan` associe explicitement ces sujets aux
layouts et porte les mappings/synchronisations nécessaires. Aucun artefact, report, writer ou
executor ne choisit le premier layout comme représentant.

Le provider livré exécute plusieurs layouts `Uniform` distincts lorsque :

- chaque bloc et chaque donnée échangée possède une affectation exacte ;
- le `Program` est séparable en un graphe compilé et authentifié par layout ;
- chaque transfert directionnel nomme explicitement ses ports, sa représentation, son point de
  synchronisation et une opération de l'interface native `Transfer` ;
- le composant C++ qui implémente cette opération fait partie des composants authentifiés de
  l'artefact puis du bind.

L'opération livrée `CONSERVATIVE_CELL_AVERAGE_V1` relie des résolutions différentes d'un même domaine
cartésien : origine, étendue et partition périodique/physique doivent coïncider exactement, et le ratio
de cellules doit être entier dans chaque axe. Son ABI ne porte aucun mapping de coordonnées ; elle
refuse donc deux géométries ou topologies distinctes. Un tel transfert est une nouvelle opération de
l'interface ouverte `Transfer`, avec son propre provider et son mapping authentifié, pas un mode caché
de la moyenne conservative.

Une moyenne conservative fine vers grossier n'invente jamais son inverse grossier vers fin. Un
second sens est une seconde exigence avec sa propre opération et son propre provider ; il n'existe
ni `reverse=True`, ni opération de mapping par défaut lorsque le sens numérique n'est pas déductible
de façon unique.

La synchronisation native `before-step@1` possède une sémantique de snapshot : toutes les sources de
tous les transferts sont capturées avant la première écriture. Un graphe `A -> B -> C` ou un cycle
explicite lit donc partout le même état pré-transfert. L'opération
`CONSERVATIVE_CELL_AVERAGE_V1` remplace sa cible : deux transferts vers le même sujet au même point de
synchronisation sont refusés tant qu'une opération et un provider de merge explicites n'existent pas.

Le provider refuse avant création d'un moteur : un mélange `Uniform`/AMR, plusieurs hiérarchies AMR,
un kernel co-localisé traversant deux layouts sans lowering dédié, un `FieldOperator` multi-layout,
un mapping manquant ou un `Program` non séparable. La route native multi-`Uniform` exige en outre une
stratégie temporelle exacte `FixedDt`; elle refuse les stockages `aux` sans affectation/transfert
explicite, les plans de frontières sans autorité d'installation par layout et toute demande de CFL
globale dépourvue d'une réduction qualifiée inter-layout. Ces refus sont des limites de capacité
exactes, pas une normalisation vers un layout représentatif.

### 6.2 Autorité AMR

Un layout AMR agrège six autorités d'authoring : cinq facettes scientifiques, une autorité de
disposition des patches et deux providers d'exécution :

- `AMRHierarchy` : niveaux et ratios par transition ;
- `AMRTagging` : graphe de prédicats, décisions, hystérésis et conflits ;
- `AMRRegrid` : cadence et règle de reconstruction ;
- `AMRTransfer` : politique par espace/état ;
- `AMRExecution` : relation temporelle entre niveaux ;
- `PatchLayout` : distribution du niveau grossier et, seulement lorsqu'elle est imposée, taille
  maximale de ses patches ;
- un provider `Tagger` qui matérialise le graphe de tagging ;
- un provider `Clustering` qui transforme les tags en boîtes parentes.

```python
from pops.amr import PatchLayout

layout = AMR(
    grid=grid,
    hierarchy=AMRHierarchy(max_levels=..., ratios=(...)),
    tagging=tagging,
    tagger=pops.lib.amr.SymbolicTagger(),
    clustering=pops.lib.amr.BergerRigoutsos(),
    regrid=AMRRegrid(schedule=every(5, clock=T.clock)),
    transfer=transfer,
    execution=AMRExecution.subcycled((
        AMRClockRelation(0, 1, temporal_ratio=2),
        AMRClockRelation(1, 2, temporal_ratio=2),
    )),
    patch_layout=PatchLayout(distribute_coarse=True),
)
```

`PatchLayout` est une autorité de configuration, pas un troisième provider de clustering. Le choix
`distribute_coarse` est explicite et participe à l'identité résolue. `coarse_max_grid=None`, sa valeur
par défaut, demande au provider natif sélectionné de dériver la taille des patches grossiers. Aucun
entier sentinelle n'appartient au contrat public ; une éventuelle représentation sentinelle reste un
détail de lowering privé. Une taille positive n'est écrite dans `coarse_max_grid` que lorsque
l'utilisateur veut réellement contraindre ce choix.

Les builtins de `pops.lib.amr` et les composants externes implémentent le même petit protocole de
provider. Un composant externe est sélectionné sans callback Python :

```python
from pops.amr import ClusteringProvider, TaggerProvider

layout = AMR(
    ...,
    tagger=TaggerProvider(component=my_tagger),
    clustering=ClusteringProvider(component=my_clustering),
)
resolved = pops.resolve(
    pops.validate(case),
    layout=layout,
    components=(my_tagger, my_clustering),
)
```

Les deux valeurs doivent référencer un exact `pops.external.ExternalComponent` portant
respectivement l'interface générée `Tagger` ou `Clustering`. Le même objet exact doit être fourni à
`resolve(components=...)`; son identité de manifest, son interface et sa version traversent
`resolve -> compile -> bind`. Le manifest doit déclarer une classification déterministe `bitwise` ou
`reproducible`, car chaque rang doit produire la même hiérarchie. Un `Tagger` déclare en plus une
capacité `amr_tagging_program` exacte : opcodes feuilles/logiques supportés, nombre maximal
d'instructions, routes de stencil d'indicateur, nombre maximal de termes par axe et les quatre
sorties `refine_candidates`, `coarsen_candidates`,
`refine_equalities`, `coarsen_equalities`. La résolution refuse un opcode ou une capacité absente ;
elle ne réduit jamais le graphe à un prédicat privé du composant.
Le `Tagger` déclare aussi exactement `execution_mode="native_backend"` ou `"host"` et ses espaces
mémoire. Le premier sélectionne un unique variant 2D/float64 CPU, CUDA, HIP, SYCL ou OpenMP target
compatible avec l'allocation du runtime. Le second doit déclarer uniquement `host` : c'est le seul
cas où PoPS matérialise une image host, jamais un fallback implicite. Une déclaration 3D ne rend pas
l'adapter 2D compatible.

Le contrat `Tagger` v2 reçoit, pour chaque patch local, toutes les vues d'états qualifiées utilisées
par le graphe ainsi que son programme lié canonique : feuilles, composantes, seuils, programmes
refine/coarsen, stencils discrets, hystérésis, égalité, conflits et identité. Un stencil discret
transporte sa route versionnée, sa dimension, sa norme, son échelle, son mode de frontière et, pour
chaque axe, offsets, coefficients, ordre de dérivée, ordre formel et halos inférieur/supérieur. Le
spatial method le fournit sous forme typée et sérialisable ; `resolve` refuse une méthode absente ou
ambiguë. Il n'existe aucun choix runtime par nom de reconstruction et aucun fallback vers une
différence centrée. Les moments des coefficients jusqu'à l'ordre formel déclaré sont vérifiés, puis
les halos requis sont comparés aux halos réellement alloués avant le premier appel du provider.
Avec `boundary_mode="ghost_extension"`, les halos same-level, coarse/fine et physiques sont produits
par les autorités `AMRTransfer`/`PreparedBoundaryPlan` exactes au clock et au temps logique du
tagging. Une frontière physique non périodique sans producteur complet, ou une face d'interface
omise sans transfert de ghost correspondant, fait échouer le bind ; un stencil ne lit jamais une
valeur de halo résiduelle.

Le composant évalue ce même programme exact et rend
les quatre masques candidats. En mode natif, les vues empruntent directement les allocations de
champs et le composant exécute sur le backend/stream authentifié ; seuls les quatre masques compacts
sont rapatriés avant le clustering. PoPS reste l'unique autorité qui applique la couverture fine courante,
la politique d'égalité et les conflits refine/coarsen. En mode natif, les états ne sont ni packés ni
réduits globalement. Pour un parent distribué, seule une OR collective groupée des quatre bitmaps est
autorisée ; pour un parent répliqué, les bitmaps doivent être identiques rang par rang et toute
divergence est refusée au lieu d'être masquée par une union. `min_cycles > 0` est refusé
à la résolution tant que le runtime ne possède pas le stockage persistant de décision requis : une
hystérésis ne peut jamais être acceptée puis ignorée. Changer un seuil, le graphe, coarsen, l'égalité
ou les conflits change l'identité et le contenu du programme lié.

L'évaluation logique est trivaluée. Une égalité de feuille produit `Unknown`; `not Unknown` reste
`Unknown`; `Any` vaut `True` dès qu'un enfant vaut `True`, sinon `Unknown` s'il en existe un, sinon
`False`; `All` vaut `False` dès qu'un enfant vaut `False`, sinon `Unknown` s'il en existe un, sinon
`True`. Les quatre masques représentent le résultat des racines, pas l'union des égalités internes.
`EqualityPolicy` transforme ensuite tout `Unknown` de racine en aucune action, candidat refine ou
candidat coarsen, avant `ConflictPolicy`, y compris si l'autre racine vaut déjà `True`.
`non_finite_policy="reject"` est fixe dans la capability et l'ABI v2 : une valeur scalaire, un terme
de stencil ou un gradient dérivé `NaN`/infini interrompt le tagging avant toute logique booléenne.
En particulier `Not(NaN)` ne peut pas devenir `True`. Un composant externe signale ce rejet par son
`PopsComponentStatusV1`; l'adapter ne relit pas tout le champ sur CPU avant l'appel et ne convertit
jamais une erreur numérique en masque `False`.

L'installation prépare chaque table une fois avant la création du runtime. Une table absente, une
capacité de sortie insuffisante ou une identité divergente échoue
avant publication de la hiérarchie ; il n'existe ni callback Python par cellule, ni switch sur
`component_id`, ni fallback vers le builtin après une erreur externe.

Le contrat `Clustering` v1 reçoit un masque dense et retourne des lignes
`[lo_0, ..., lo_(d-1), hi_0, ..., hi_(d-1)]`, bornes inclusives relatives à la région de tags. PoPS
valide capacité, bornes, non-recouvrement et couverture de tous les tags avant de convertir ces
boîtes parentes en layout fin. Les boîtes sont triées lexicographiquement avant validation ; PoPS
vérifie ensuite que cette séquence canonique est identique sur tous les rangs avant publication. La
preuve overlap/couverture est linéaire dans le domaine dense et l'aire couverte ; le consensus MPI
batché utilise un nombre constant de collectives, jamais une collective par boîte. Le
provider ne contrôle ni nesting, ni distribution, ni publication.

Le transfert appartient au layout et n'est pas ajouté une seconde fois au `DiscretizationPlan`.
Les seuils de tagging sont des paramètres du `Case` et sont donc résolus/bindés comme toute autre
valeur. Une expression telle que `norm(grad(ValueExpr(block[U]))) > case.value(threshold)` est
résolue dans un `DiscreteIndicatorContext` explicite. Sa discrétisation spatiale y authentifie le
stencil exact décrit ci-dessus ; AMR ne répète ni `order=`, ni profondeur de halo, et n'invente pas
un gradient continu exécutable.

Le plan normalisé conserve chaque ratio de transition et le raffinement cumulé de chaque niveau.
La relation temporelle de chaque paire parent/enfant est une autorité distincte du ratio spatial.
`AMRClockRelation` porte un ratio rationnel exact et une `AMRRemainderPolicy`. La route intégrale
n'invente jamais `time_ratio = space_ratio`; une relation non intégrale exige explicitement
`EXPLICIT_FINAL_SUBSTEP`, sinon elle est refusée. Le nombre de relations et leurs niveaux adjacents
doivent couvrir exactement la hiérarchie.

Le provider natif livré matérialise le coeur maillage/stockage en 2D et ses kernels de transfert,
correction conservative et sous-cyclage AMR exigent un ratio de transition égal à 2. La correction
coarse/fine reste l'unique ledger de flux détenu par PoPS : aucune interface externe `Reflux`
n'existe, car déléguer ce dépôt créerait une seconde autorité conservative. Une autre dimension ou un autre
ratio est refusé pendant la résolution ou le bind avec les capacités observées. Le coeur de
planification ne normalise jamais la demande vers ce sous-ensemble. Défensivement,
`AmrProgramContext` revalide aussi chaque transition à sa construction et refuse un ratio différent
de 2 avant le premier pas : cette limite appartient au provider natif reflux/average-down installé,
pas aux protocoles publics `AMRHierarchy`, `Transfer` et `AMRExecution`, qui restent extensibles par
sélection d'un autre provider déclarant les capacités correspondantes.

Les critères booléens et les politiques de transfert sont des protocoles authentifiés ouverts. Une
nouvelle implémentation fournit données canoniques, requirements/capabilities et lowering ; elle ne
nécessite pas un `isinstance` ajouté à chaque phase centrale.

### 6.3 Frontières et conditions initiales

Une BC de transport est enregistrée une fois dans `DiscretizationPlan.boundaries`. Une BC de champ
appartient à son `FieldDiscretization`. Une interface multibloc et une frontière coarse/fine sont des
ports distincts. Le graphe de producteurs de ghosts prouve la couverture, la profondeur, le temps et
les dépendances de chaque région avant exécution.

Le lowering produit un plan natif immuable exécuté dans l'ordre de dépendance : halo same-level/MPI,
identifications périodiques, interpolation coarse/fine, faces physiques, résolution des coins, puis
closures numériques. Les valeurs de stage, temps, niveau, rate et itération non linéaire accompagnent
chaque appel dépendant de l'état. Une interface multibloc possède une seule évaluation de flux partagé ;
le runtime disperse ce résultat avec orientations opposées vers les deux résidus. Une closure implicite
installe obligatoirement le couple résidu/JVP et leurs tables exactes d'états, directions, champs et
paramètres. Une interface, orientation, projection, corner policy ou closure sans composant natif
qualifié est refusée à `compile` : un callback Python ou un handle sans implémentation n'est jamais une
route d'exécution.

Sous `MPI_COMM_WORLD`, chaque rang compacte uniquement les cellules de face qu'il possède, le runtime
C++ reconstruit collectivement les deux traces complètes, puis exécute le `NumericalFlux` natif avec le
même batch qualifié sur tous les rangs. Le flux partagé doit être fini et bit-identique entre rangs avant
toute écriture ; seuls les fragments locaux des deux résidus sont ensuite modifiés. Une erreur de
préparation, de trace, de composant ou de consensus est décidée collectivement avant la phase suivante,
afin qu'un échec propre à un rang ne puisse ni publier un demi-flux ni bloquer ses pairs.

Pour un `rhs_jacvec`, le `BoundaryEvaluationPoint` exact du RHS de base est capturé dans le corps du
pas puis transporté jusque dans l'`ApplyFn`; la copie de contexte créée avant `begin_step` ne
reconstruit jamais le temps. Le volume sans contribution additive de frontière est différencié, puis
le JVP natif exact de la closure est ajouté une seule fois. Ainsi la contribution de frontière n'est
ni oubliée ni comptée deux fois. Les scratchs sont persistants, réutilisés et alloués seulement si le
bloc possède réellement un couple résidu/JVP.

La route livrée accepte une linéarisation de frontière d'état avec une direction qualifiée égale à
l'état propriétaire et une sortie. Si le résidu de frontière lit un champ résolu et que
`rhs_jacvec(field_coupled=True)` demande la dérivée totale, la résolution refuse : aucun solveur de
tangente `dField/dState` n'est encore disponible, et un champ primal gelé ne peut pas être présenté
comme sa tangente.

Une interface conservatrice entre deux blocs est déclarée avant validation avec les états déjà
qualifiés par leur bloc et les frontières géométriques du frame :

```python
from pops.mesh.boundaries import BlockInterfaceSide, ConservativeInterface

interface = ConservativeInterface(
    "fluid_to_solid",
    left=BlockInterfaceSide(fluid[U], frame.boundaries.x_max),
    right=BlockInterfaceSide(solid[V], frame.boundaries.x_min),
    numerical_flux=compiled_numerical_flux,
    permutation=(0,),
    right_normal_translation=1.0,
)
interface.attach(fluid_numerics, solid_numerics)
```

`attach()` inscrit la même autorité immuable dans exactement deux `DiscretizationPlan`; ce n'est
pas une double autorité. La résolution la consomme une seule fois, remplace exactement une région
physique de chaque plan par les deux endpoints du même `MultiBlockInterface`, puis retire toute
métadonnée d'authoring. Les deux `BoundaryHandle` ont des owners d'instance de bloc distincts tandis
que l'identité de l'interface et son `NumericalFlux` sont uniques. Toute inscription absente,
dupliquée, concurrente ou ne correspondant pas exactement à une face de chaque bloc est refusée.

La route native livrée exige actuellement deux blocs co-localisés sur le même `LayoutHandle`, des
évaluations de RHS explicites, simultanées et contiguës dans le même point de `Program`, et un
composant `NumericalFlux` authentifié. Elle refuse une interface cross-layout sans
`Mapping`/`Transfer`, le JVP implicite partagé, et une hiérarchie AMR raffinée ou regriddée ; sur AMR,
seul un niveau unique avec hiérarchie figée est exécutable tant que le ledger reflux d'interface
n'est pas fourni.

Une condition initiale associe : handle d'état qualifié, donnée, projection et éventuellement preuve
de reprojection AMR. `pops.lib.initial.Constant` et `Gaussian` sont des données analytiques immuables
réévaluées sur chaque niveau. `pops.lib.initial.BindArray()` déclare au contraire qu'un tableau d'état
conservatif complet sera fourni à `pops.bind(initial_values={block[U]: array})` : le tableau ne pollue
ni le snapshot ni la clé de compilation, le niveau zéro en est l'unique consommateur et les niveaux
fins utilisent le provider de transfert résolu. Le handle d'authoring est authentifié puis remplacé
par le sujet canonique exact du plan ; une clé homonyme provenant d'un autre `Case` est refusée.

Pour un bootstrap AMR, reprojection analytique et prolongation sont donc des choix explicites portés
par la brique de donnée. `initial_values` doit couvrir exactement les sources non analytiques et chaque
`BindArray` doit avoir la forme conservatrice complète `(n_components, ny, nx)` et la précision de
l'artefact. Pour tout artefact AMR, le bind refuse `initial_state` sans condition : le plan IC AMR et
ses `initial_values` typées sont l'unique autorité. Il refuse aussi un tableau de densité qui prétend
satisfaire `BindArray`, une source analytique surchargée et toute valeur manquante.

## 7. Programme de temps

### 7.1 Langage générique

`pops.Program` est un builder de graphe SSA, pas une boucle Python. Les opérateurs principaux sont :

- `state`, `value`, appel d'opérateur et appel de champ ;
- `solve(problem, solver=...)` puis consommation explicite de l'outcome ;
- `keep_history`, `history` et `store_history` pour les historiques qualifiés ;
- `subcycle` et `synchronize` pour les domaines d'horloge ;
- contrôle structuré typé ;
- `commit` et `commit_many` ;
- `step_strategy` pour le contrôleur et le contrat transactionnel.

Chaque anneau d'historique possède dès l'authoring une politique de persistance typée et compilée.
`store_history(name, value, depth=..., checkpoint_policy=...)` couvre les anneaux génériques ;
`keep_history(..., checkpoint_policy=...)` réutilise exactement la même autorité. Une politique
omise devient explicitement `Dense()` dans le graphe compilé. Une politique sélective exige une
profondeur finale connue, et la validation refuse toute politique manquante, orpheline, mal typée ou
dont la profondeur diverge de celle des lectures. Le checkpoint ne devine jamais de fallback.
Dans ces deux méthodes, `depth` est le lag maximal lisible ; l'anneau natif possède donc exactement
`depth + 1` slots en incluant le slot courant `0`. La politique de persistance est toujours validée
contre ce nombre physique de slots, jamais contre le seul lag maximal.
Le plan de checkpoint sépare les slots demandés par la politique des slots effectivement stockés.
Si une regrille est planifiée dans la fenêtre qu'un replay sélectif devrait reconstruire, le plan
effectif est promu en `dense_regrid_safety` : tous les slots sont persistés et le manifeste authentifié
enregistre la demande, la promotion et l'empreinte de calendrier. PoPS ne prétend donc jamais qu'un
replay sur une hiérarchie déjà remappée est bit-identique ; hors d'une telle fenêtre, le plan reste
`policy` et reconstruit uniquement les slots omis.

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

`LinearProblem` porte aussi un certificat mathématique typé, jamais inféré depuis le nom d'un
stencil ou d'un préconditionneur. La valeur par défaut `LinearOperatorProperties.general()` convient
aux méthodes générales. Ses trois faits booléens exacts sont `symmetric`, `positive_definite` et
`positive_definite_on_nullspace_complement`. Les quatre certificats canoniques sont `general()`,
`symmetric_operator()`, `symmetric_positive_definite()` et
`symmetric_positive_definite_on_nullspace_complement()` ; les deux formes de positivité sont
mutuellement exclusives.

La décision de nullspace est keyword-only et obligatoire sur chaque problème :

```python
nonsingular = LinearProblem(A, b, nullspace=None)

singular = LinearProblem(
    A,
    b,
    properties=LinearOperatorProperties
        .symmetric_positive_definite_on_nullspace_complement(),
    nullspace=ConstantNullspace(),
    gauge=MeanValueGauge(0),
)
```

La route matrix-free ne déduit rien des BC, de la périodicité ou du stencil. Une déclaration
`ConstantNullspace()` exige exactement `MeanValueGauge(value)`, est scalaire seulement, et capture
la valeur canonique immuable de la gauge à la construction. Comme un noyau droit constant ne prouve
pas à lui seul que le complément de moyenne nulle est invariant, cette déclaration exige au minimum
le certificat explicite `LinearOperatorProperties.symmetric_operator()`. Les deux attributs IR
`nullspace_contract` et `gauge_contract` sont toujours présents, portent `schema_version=1`, et sont
validés par ensembles exacts de clés/types avant allocation ou émission. `CG` exige le certificat
SPD global avec `nullspace=None`, ou SPD sur le complément avec un nullspace constant ; aucune autre
méthode n'est substituée. `Identity()` est le seul préconditionneur dont la préservation du
complément est actuellement authentifiée. `GeometricMG()` est refusé pour ce contrat tant qu'il ne
publie pas une capacité de préservation explicite. `CompositeTensorFAC()` refuse également le
nullspace constant tant que la gauge composite multilevel n'est pas câblée de bout en bout.
Les solveurs Krylov portent l'arrêt exact
`||b-A(u)|| <= max(rel_tol * ||b-A(0)||, abs_tol)` : le warm start ne change jamais la référence.
Après le test physique initial, une tentative non convergée normalise uniquement sa récurrence par
`||b-A(u0)||`; `||b-A(0)||` reste exclu de cette échelle interne afin qu'une composante immense déjà
satisfaite par le warm start n'annule pas un petit résidu encore fini. Cette normalisation ne modifie
ni le seuil physique ni le `SolveReport`.
Leur footprint persistant est dérivé de la
méthode, du nombre de composantes, de la largeur de halo, du restart et du préconditionneur.
Le restart GMRES accepte exactement les entiers Python de `1` à `INT_MAX - 1` inclus : le workspace
est dimensionné dynamiquement, son coût exact est visible dans le plan scratch, et la borne supérieure
est celle du `int` natif avec une place réservée au terme supplémentaire de la réduction Arnoldi/MPI,
pas un plafond algorithmique arbitraire. La route Newton-Krylov n'impose aucun plafond hérité
d'un ancien tableau fixe.
La largeur n'est pas un booléen « stencil présent » : chaque opération porte une capacité immuable
`StencilAccess(required_ghost_depth=n)` et `set_apply` compose le sous-graphe par le maximum de ces
capacités, sans table centrale de noms d'opérations. `matrix_free_operator(stencil_depth=n)` reste une
contrainte explicite pour un provider plus profond : elle est refusée sous le minimum composé et
transporte autrement tout entier `n >= 0` jusqu'aux allocations natives.

Le préconditionneur livré `preconditioners.GeometricMG()` est un opérateur scalaire : son `phi`, son
second membre et son V-cycle natifs ont exactement une composante. Il est donc refusé à l'authoring
pour `ncomp != 1`, avec une seconde garde native ; PoPS ne diagonalise pas silencieusement un problème
multicomposant. Un tel problème utilise `Identity()` ou un futur provider réellement block-aware.

À la frontière C++, un solve global ne reçoit jamais un callback brut plus un entier de méthode. Le
code généré construit une fois `PreparedAffineLinearProblem`, `PreparedLinearPreconditioner` et
`KrylovWorkspace`, puis prépare chaque évaluation avec un snapshot exact : identité canonique
256 bits du Program/opérateur, révision, macro-pas, fraction d'étape, bits IEEE de `dt` et du temps,
empreintes 256 bits de la topologie native (boxes, distribution, halo, métrique et BC) et des
ressources figées. Le probe réutilise uniquement la topologie complète et la révision immuables
frappées lors de `prepare`, puis recalcule depuis l'unique `ProgramContext` partagé par le step,
l'ApplyFn et le préconditionneur les identités dynamiques de l'horloge et de la révision topologique ;
il ne renvoie jamais simplement la copie du snapshot attendu. La
préparation copie les coefficients variables, matérialise les plans halo/buffers MPI et prépare le
préconditionneur avant la première itération, puis calcule exactement `c = A(0)`. Un
préconditionneur brut peut lui aussi être affine sous des frontières inhomogènes : son objet préparé
possède donc ses propres buffers persistants et calcule `d = M_raw(0)` ; les itérations appliquent
exclusivement `M_lin(v) = M_raw(v) - d`. Elles appliquent de même
`A_lin(v) = A(v) - c` et résolvent `A_lin(u) = b - c`. Toute mutation du snapshot après préparation
est refusée. Les prototypes du problème et du préconditionneur doivent avoir exactement les mêmes
composantes, boxes, distribution et halo. L'itéré et le second membre ne peuvent pas partager leur
stockage ; les slots du workspace sont privés, de forme immuable et conservent leurs plans halo/MPI
préchauffés. Le plan scratch expose séparément le nombre exact de champs persistants, de scalaires
Hessenberg/rotations et de valeurs du payload collectif.

CG et BiCGStab remplacent leur récurrence complète lorsqu'une convergence récursive candidate échoue
à la confirmation par le vrai résidu. BiCGStab maintient autrement `r = s - omega*t` et ne recalcule
le résidu scientifique que pour confirmer une convergence candidate, publier un échec ou produire le
report final : une itération complète ne paie pas un troisième matvec. GMRES agrège toutes les
projections Arnoldi d'une colonne dans une réduction
vectorielle, calcule ensuite la norme projetée directement et déclenche une seconde passe CGS2 batchée
uniquement sous le critère de perte de norme DGKS. Le chemin normal utilise donc deux collectives par
colonne au lieu d'une collective par vecteur de base, sans remplacer la norme scientifique finale.

Sur AMR, le préambule compilé n'est jamais partagé entre des layouts de niveaux différents. À
l'installation, le module matérialise un bundle complet par niveau (scratch, coefficients gelés,
ApplyFn, problème préparé, préconditionneur et workspace), puis le driver sélectionne ce bundle par le
curseur de niveau natif. Un changement d'epoch topologique ou de génération de matérialisation
native (regrid, rollback ou reconstruction de restart, même avec la même valeur d'epoch) invalide et
rematérialise tous les bundles une fois avant l'advance suivant ; deux solves compatibles ne
réallouent rien. Cette règle vaut aussi lorsqu'un même Program compose un solve `Level()` et un solve
`Hierarchy()` : gather/publish utilisent le bundle de leur niveau, tandis que l'unique solve composite
est déclenché par le bundle du niveau racine. Le gel des coefficients tensoriels copie toute la région
allouée, halos compris, car les moyennes de face et termes croisés lisent les voisins inter-boxes.

Les récurrences utilisent une algèbre de champs pure qui ne touche ni le ledger reflux AMR ni les
effets temporels de `ProgramContext`. Aucune allocation de champ, de plan halo, de buffer MPI ou de
scratch de field solve et aucun calcul cellule par cellule n'ont lieu en Python ou dans la boucle
Krylov ; les capacités sont persistantes et les kernels de champ et réductions collectives restent
Kokkos/C++. L'initialisation des vecteurs persistants, y compris le départ froid de chaque V-cycle de
préconditionnement, remplit les cellules valides par un kernel Kokkos sur l'espace d'exécution courant ;
les fantômes restent la responsabilité du plan halo/BC typé qui les écrase avant toute lecture. Aucun
balayage cellule par cellule n'a lieu sur l'hôte dans le hot path. Un résidu préconditionné ou
l'estimation de Hessenberg de GMRES peut seulement demander
une confirmation : seul le résidu scientifique vrai `b - A(u)` peut publier `kSolved`. Le résidu de
référence est `||b - A(0)||`; un warm start déjà convergé retourne zéro itération.

Un outcome fallible doit être consommé par une action adaptée à sa phase (`RejectAttempt`, `FailRun`,
etc.) avant que sa valeur puisse contribuer à un commit ou un effet.

À la frontière native, tous les solveurs itératifs retournent le même `SolveReport` : nombre
d'itérations, norme de référence, norme finale vraie, rapport déclaré, raison et une unique paire
`SolveStatus` / `SolveAction`. Il n'existe pas de
booléen `converged` parallèle. Une valeur n'est résolue que pour `(kSolved, kNone)` ; toute paire
incohérente est traitée comme un échec et un appel de construction d'échec sans statut/action d'échec
est refusé. Le runtime ne publie jamais l'itéré ou le champ muté d'un report en échec et la transaction
restaure l'ensemble des valeurs acceptées précédentes. Un solveur généré doit porter un critère de
convergence scientifique explicite, distinct de son budget ; atteindre seulement la limite
d'itérations produit `kIterationLimit`, jamais un succès fabriqué.

Pour tout solveur linéaire affine et son résidu discret `R(u) = b - A(u)`, le résidu relatif est
`||R(u)|| / ||R(0)||` dans la norme globale définie par le contrat du solveur, avec une base égale à
`1` lorsque `||R(0)|| == 0`. `R(0)` est évalué par l'opérateur préparé exact : mêmes coefficients,
masques, frontières physiques ou générées et topologie que `R(u)`. Il inclut donc le lifting des
frontières inhomogènes ; pour un opérateur linéaire homogène, il se réduit à `b`. Le provider livré
est l'unique produit `L2` global sur toutes les composantes et tous les rangs ; aucun `Linf`, poids ou
masque composite n'est accepté tant qu'un provider de métrique préparé typé ne le transporte pas de
bout en bout. Cette norme ne dépend jamais du warm start. Le critère mixte est
`||R(u)|| <= max(rel_tol * ||R(0)||, abs_tol)`. Le zero-probe et ses buffers sont persistants, et les
normes initiales peuvent être agrégées dans une même collective. Relancer un système inchangé déjà
sous ce seuil retourne donc un succès à zéro itération, au lieu de demander implicitement une
réduction supplémentaire par `rel_tol`.

Un outer solve non linéaire ne réutilise pas implicitement cette définition linéaire. Sa politique de
normalisation est un élément explicite du solveur ; la politique livrée prend le résidu du snapshot
accepté au début de la tentative comme référence et une base égale à `1` si ce résidu est nul. Le
report et le critère utilisent cette même référence pendant toute la tentative. Une norme de résidu
préconditionnée peut guider une itération interne, mais elle ne peut jamais remplacer la norme
scientifique déclarée dans le `SolveReport` publié.

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

Le ledger de flux est l'unique autorité numérique du reflux. Chaque contribution est qualifiée par
owner, bloc, rate/flux conservatif, niveau, face et orientation, puis porte mesure de face, durée du
sous-pas et poids RK/ARK exact. Savepoints, rollback, checkpoint et report conservent ces mêmes
contributions ; aucun registre shadow ni « flux du dernier stage » ne peut piloter la correction. Le
reflux consomme le ledger accepté, puis l'average-down a lieu dans la même phase de synchronisation
rapportée.

### 7.4 Algèbre et extension des schedules

Un schedule est le produit typé `Schedule(trigger, off=...)` de quatre petites interfaces ouvertes :

- `Domain.native_schedule_domain()` retourne un exact `ScheduleDomainIR` ;
- `Trigger.native_schedule_due()` retourne un exact `ScheduleDueIR` ;
- `OffPolicy.native_schedule_off()` retourne un exact `ScheduleOffIR` ;
- `Schedule.native_schedule_ir()` compose les trois en un exact `ScheduleLoweringIR`.

Une extension est un dataclass immuable et slotté, déclare un `manifest_tag`, ainsi qu'une identité
sémantique possédée par sa classe (`component_uri` absolue et namespacée,
`component_version >= 1`), projette toutes ses données comportementales, conserve son type exact
pendant le rebuild et participe à l'identité du graphe. Le chemin de module Python et le `qualname`
ne sont jamais utilisés comme identité persistante. Un dictionnaire ressemblant à l'IR, un retour
partiel, une identité héritée ou un type transformé pendant le mapping sont refusés ; aucune classe
d'extension n'est enregistrée dans une liste centrale. Tous les types du protocole, y compris
`UnresolvedScheduleCondition`, sont importables depuis `pops.time` : une extension n'importe aucun
module `_schedule` privé.

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
if report.accepted_steps == 0:
    raise RuntimeError("aucun macro-pas accepté")
```

La plateforme n'est pas une autorité déclarative fabriquée par l'utilisateur. Son manifeste exact
est dérivé des composants compilés authentifiés et participe à l'identité de l'artefact. Les
ressources d'exécution concrètes (communicateur, device ou handles externes) sont fournies
explicitement à `pops.bind(..., resources=...)` et validées contre ce manifeste avant installation.

Contrat des phases :

| Phase | Entrée | Sortie | Interdictions |
| --- | --- | --- | --- |
| `validate` | exact `Case` mutable | même `Case` gelé | import natif, mutation scientifique |
| `resolve` | `Case` gelé + autorités typées | plan résolu immuable | tableaux runtime, fallback |
| `compile` | exact plan résolu | artefact authentifié | réinterprétation des choix |
| `bind` | artefact + cinq familles de valeurs | instance runtime | changement de structure/algorithme |
| `run` | instance bindée + contrôles numériques | rapport d'exécution | authoring ou sélection de stratégie |

`pops.run` retourne un `pops.RunReport` immuable et observé, jamais un entier nu. Ses compteurs
`accepted_steps` et `rejected_steps` sont locaux à cet appel ; le second compte exactement les
tentatives natives rejetées puis retentées avant acceptation. `final_time` et `final_macro_step`
rapportent l'horloge cumulative réellement publiée. `stop_reason` est un `pops.RunStopReason` ; le
seul arrêt réussi actuellement implémenté est `TARGET_TIME_REACHED`. Un épuisement de `max_steps`,
une garde terminale ou un effet non publiable lève une exception et ne fabrique pas de rapport de
succès. Le rapport transporte les identités authentifiées `run_identity`, `bind_identity`,
`execution_identity` et `artifact_identity`, sans recalcul ni valeur par défaut. Sa section
`field_providers` est la projection immuable du même rapport de provenance que
`RuntimeInstance.inspect()` après l'appel : builtin et externe utilisent le même schéma et seuls les
faits réellement matérialisés y apparaissent.
Le rapport n'a pas de vérité booléenne implicite : le code utilisateur choisit explicitement le
champ observé (`accepted_steps`, `stop_reason`, etc.).

Au début de chaque `pops.run`, le rang zéro affiche un court bandeau PoPS puis la configuration
effectivement installée : cas, target, backend C++/Kokkos, concurrence native active, précision,
communicateur et nombre de rangs, blocs, layouts, stratégie temporelle, intervalle, consommateurs et
identités du run et de l'artefact. Le bilan final rapporte les pas acceptés/rejetés, l'horloge et le
temps écoulé. Ce renderer est une projection Python de faits déjà authentifiés ; il ne choisit aucun
paramètre numérique, ne lit aucun champ et ne prétend pas être un kernel natif. Les rangs MPI non
racine restent silencieux pour éviter un bandeau dupliqué. `console=False` désactive uniquement cette
présentation : cette valeur ne rejoint ni le manifeste ni l'identité numérique du run. Une erreur de
terminal est signalée sur `stderr`, mais ne peut ni masquer une exception numérique, ni empêcher un
rollback, ni convertir un run réussi en échec.

Le suivi pendant la boucle n'est pas une option implicite de `pops.run`. Il est un consumer typé du
`ConsumerGraph`, avec la même autorité de cadence que les sorties scientifiques :

```python
ConsoleMonitor(
    schedule=every(10, clock=program.clock),
    diagnostics=(
        StepChangeNorm(L2(), block=tracer),
        Integral(block=tracer),
    ),
    template=(
        "step={step} t={time:.4e} dt={dt:.3e} "
        "dU_L2={tracer.step_change_l2:.3e} mass={tracer.integral:.6e}"
    ),
)
```

`every(10)` compte les pas macro acceptés et `every_dt(0.1)` impose une grille de temps physique.
Les réductions sont calculées nativement, sur Kokkos puis MPI, uniquement quand le consumer est dû ;
seuls quelques scalaires immuables atteignent Python et seul le rang zéro effectue le formatage. Le
consumer est absent lorsque `enabled=False` : aucun test, calcul, formatage ou I/O n'est alors ajouté
à la boucle d'exécution. `StepChangeNorm(L2())` réutilise l'image transactionnelle `U^n` sans recopier
le champ. Une étape qui change la topologie AMR rend cette seule valeur indisponible et le template
affiche `n/a (AMR regrid)` ; les autres diagnostics dus, comme l'intégrale, restent calculés.

Pour une présentation avancée, `handler=display` remplace `template=`. Le handler est une fonction
Python nommée, sans closure, appelée sur le rang zéro avec un `ConsoleSample` immuable contenant
`time`, `step`, `dt` et les scalaires réduits accessibles par nom (`sample["dU_L2"]` ou
`sample["tracer.step_change_l2"]`). Il ne reçoit ni tableau natif, ni communicateur MPI.

Les seules options de compilation sont celles acceptées par `pops.resolve(..., compile_options=...)` :
`so_path`, `force`, `cxx`, `include`, `std` et `debug`. Le backend est une autorité séparée. Il n'existe
pas de `CompileConfig` public, de `strict=True`, de `sim.run`, ni de `RejectOldManifest`.

`pops.bind` accepte exactement cinq familles : `initial_state`, `params`, `aux`, `resources` et
`initial_values`. L'enregistrement interne qui les authentifie n'est pas importé par l'utilisateur.
`initial_state` est exclusivement la table de blocs d'un layout uniforme ; `initial_values` est la
table typée par `Handle` du plan `InitialCondition` AMR. Tout artefact AMR exige ce plan résolu : il
n'existe ni table de blocs AMR parallèle ni route de compatibilité sans autorité d'initialisation.
Elles ne constituent jamais deux autorités pour le même artefact.
Dans cette release, `resources` est vide ou contient uniquement `execution_context`, valeur typée qui
porte toute l'autorité de lancement. Les clés libres `communicator`, `device`, `stream` ou `allocator`
sont refusées ; elles ne constituent pas un second chemin de configuration.

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

`RuntimeInstance` ne publie ni moteur natif, ni sélecteur de moteur par layout/bloc, ni `InstallPlan` ou
`RuntimePlan`, et n'effectue aucune délégation générique d'attribut. Sa surface explicite se limite aux
identités et rapports, aux lectures d'état, clock, layout, champs et histories, à la réduction native
`integral(block, component=0, levels=...)`, à la vue de rapports AMR, au rapport du programme, ainsi
qu'à `checkpoint` et `restart`. `integral` applique la mesure cartésienne résolue en Uniform et délègue
le masque composite pondéré au provider AMR ; elle ne recopie jamais l'état pour sommer en Python.
Elle ne retourne jamais le `System` ou
l'`AmrSystem` privé et n'expose aucune route `step`, `profile` ou d'assemblage.
`RuntimeInstance.bound_snapshot` est la preuve immuable et authentifiée de l'artefact, des layouts et
des entrées effectivement liés ; cette lecture explicite ne donne aucun accès au moteur privé.

## 9. Consommateurs, sorties et restart

`pops.output.ConsumerGraph` est l'unique autorité publique des effets acceptés :
`ScientificOutput`, `Checkpoint` et
diagnostics planifiés. Chaque consommateur déclare schedule, handles qualifiés, sélection de niveaux,
format, cible déterministe et comportement d'échec.

Un diagnostic embarqué est abaissé exactement une fois vers une `DiagnosticQuantity` : handle propre,
état conservatif unique, layout/niveaux et instruction fermée de réduction native. Une sélection
ambiguë entre plusieurs états est refusée. Parcours des cellules, masque composite AMR et collectifs
MPI restent en C++/Kokkos ; Python n'applique que la transformation scalaire déclarée au résultat
réduit. La pondération métrique uniforme vient de la géométrie normalisée, tandis qu'une réduction
composite AMR déjà pondérée ne l'est jamais une seconde fois. Un `ConservationCheck` ne vaut que pour
une quantité réellement fermée : son baseline accepté est transactionnel et restauré par checkpoint.
Un domaine ouvert doit exposer séparément stockage, flux sortant, sources, reflux et projection, pas
être présenté comme un invariant.

Le graphe est résolu avec le layout, authentifié dans le plan et l'artefact, puis détenu par
`RuntimeInstance`. Le snapshot de bind ne possède aucun registre parallèle `outputs` ou `diagnostics` :
les recréer à ce niveau constituerait une seconde autorité et est interdit.
L'autorité de restart manuel est elle aussi matérialisée pendant `resolve` puis conservée dans le
plan compilé : soit le provider unique d'un nœud `Checkpoint`, soit le builtin v3 identifié quand le
graphe n'en déclare aucun. `RuntimeInstance` ne construit jamais un provider de repli tardif. Tout
provider déclare aussi `validate_snapshot()` et doit produire une préparation compensatable portant
`discard()` et `rollback()` ; ce protocole est vérifié avant qu'un effet accepté puisse être publié.

Les formats livrés sont des descripteurs (`HDF5`, `NPZ`, `ParaView`) abaissés vers des writers réels.
La gate finale rouvre indépendamment chaque HDF5 et ParaView émis et vérifie leur contenu structurel ;
l'existence du fichier seule n'est pas une preuve. La route NPZ est exercée par l'exemple IMEX-AMR et
ses tests de format, sans être présentée comme une réouverture supplémentaire de la gate groupée.
La cible d'un `ScientificOutput` est toujours un chemin logique sans suffixe ; le provider possède
seul l'extension. `schedule=every(100, clock=program.clock)` publie donc un artefact distinct après
chaque centième pas accepté, visible pendant la poursuite du run. Une petite capability de catalogue,
indépendante du writer concret, maintient un fichier
`series__f<identité-de-famille><extension>.series` remplacé atomiquement après le commit ; un pas
rejeté n'y entre jamais. Le même objet typé expose `reopen(path)` et `reopen_series(path)` ; la série
reste paresseuse pour ne pas matérialiser tous les champs historiques, tandis que `latest` et
`verify()` déclenchent les authentifications exactes nécessaires. La famille inclut le provider, la
sélection complète et l'identité du run : des sorties différentes ne sont jamais agrégées par leur
seule extension. Cela permet de remplacer
`ParaView()` par `HDF5()` ou `NPZ()` sans branche sur l'extension. Une sortie
`PER_RANK` ne fabrique pas une fausse timeline à partir des morceaux de rang : une collection
parallèle explicite reste nécessaire.

Un writer externe se sélectionne sur le consommateur, jamais par unicité globale :

```python
ScientificOutput(
    format=ExternalWriter(component=my_writer, extension=".pops"),
    schedule=...,
    fields=(block[U],),
    target="fields/tracer",
)
```

Le format authentifie `component_id`, identité du manifest et interface `Writer`; le même composant
doit traverser `resolve -> compile -> bind` puis être chargé dans le `RuntimeInstance`. Le snapshot POD
remis au writer contient toutes les géométries, champs, niveaux, pièces, noms de composantes,
diagnostics et métadonnées sélectionnés. Une capacité v1 ne peut donc ni prendre le premier champ ou
niveau, ni ignorer une pièce. Deux writers peuvent coexister parce que chaque sortie nomme le sien ;
une cible publiée en collision reste interdite.

Un checkpoint strict conserve au minimum : identités du plan/programme/composants/consumer graph,
états, champs matériels requis, histories, clocks/schedules, contrôleur, hiérarchie AMR, cursors des
consommateurs et contrat de plateforme. Un restart refuse toute divergence non autorisée. La garantie
bit-identique est prouvée par continuation indépendante, pas par comparaison du manifest seul.

Le restart v3 MPI est un protocole collectif du `RuntimeInstance`, jamais une lecture concurrente du
fichier par les moteurs. Tous les rangs authentifient d'abord la même cible ; le rang 0 lit une seule
fois l'artefact, authentifie son enveloppe et diffuse ses bytes exacts ainsi que les cursors via le
communicator porté par `ExecutionContext`. Chaque rang décode alors le payload en mémoire et termine
le préflight complet Uniform, AMR ou multi-layout. Un consensus sans erreur est obligatoire avant la
première mutation native. L'application conserve un snapshot accepté sur chaque rang jusqu'aux
consensus `apply` et `commit` ; toute erreur ou divergence déclenche le rollback de tous les moteurs.
Le multi-layout encapsule les payloads enfants dans le container v3 et les rejoue directement en
mémoire, sans fichiers enfants temporaires ni `np.load` concurrent sur un filesystem partagé.

La capture suit le contrat symétrique avant tout `*_global` natif : chaque rang construit sans
collective le plan complet et ordonné (blocs, niveaux, fields, histories, caches et provenance), puis
un consensus compare son identité. Les accessors collectifs ne démarrent qu'après cet accord. Les
payloads scellés atteignent un second consensus d'identité avant toute écriture rang 0. La publication
finale crée atomiquement un hard-link staging-vers-cible avec sémantique no-clobber, authentifie ce
lien puis retire le staging ; une cible créée concurremment n'est jamais écrasée. Discard et rollback
ne suppriment un chemin qu'après vérification de l'identité `(st_dev, st_ino)` enregistrée : un chemin
remplacé par un tiers est laissé intact et l'échec de compensation est signalé.

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

Le même catalogue génère les IDs et tables C/POD versionnées des interfaces natives (flux numérique,
ghost boundary, closure de champ, tagging, clustering, transfert, solveur de champ, writer et
topologie de champ). Le reflux conservatif reste une autorité interne pilotée par le flux ledger ;
aucune table externe `Reflux` n'est annoncée. Chaque famille possède sa propre version d'interface, indépendante de la version
du protocole enveloppe. Le loader authentifie identité sémantique, manifest, digest du catalogue,
taille/header de table et opérations requises avant de conserver le handle de bibliothèque. Les tables
sont résolues une fois à l'installation ; aucun `dlsym`, nom de classe ou dispatch Python n'entre dans
une boucle de cellules.

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
- `LinearProblem` sans décision `nullspace` explicite, nullspace constant non scalaire, sans
  certificat symétrique, sans `MeanValueGauge`, ou avec un provider ne certifiant pas le complément ;
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
- C++20/Kokkos sur les routes host/`float64` avec communicator série ou
  `ExecutionContext.mpi_world()` explicitement authentifié ;
- packages C++ externes conformes au manifest et à l'ABI courants.

Les dimensions, ratios, nombres de niveaux, géométries, solveurs et combinaisons device réellement
exécutables sont lus dans les manifests/capability reports. Le provider livré matérialise soit une
hiérarchie AMR unique, soit un ou plusieurs layouts `Uniform` reliés par les mappings natifs prouvés.
Il exécute un seul `StateSpace` par bloc, le coeur de stockage 2D et les transitions AMR de ratio 2 ;
toute demande hors de cette enveloppe est un refus avant construction du moteur, pas une
normalisation du plan.

L'ABI et les manifests décrivent un communicator, un datatype, un stream et un device explicites.
La route finale transporte host/`float64` avec le communicator série ou exactement
`MPI_COMM_WORLD`, acquis et authentifié par le runtime C++ lorsque
`ExecutionContext.mpi_world()` est appelé ; un communicator dupliqué, splitté ou personnalisé reste
refusé parce que les moteurs natifs ne disposent pas encore d'une ABI d'injection de communicator.
Le module appelle `MPI_Init_thread(MPI_THREAD_MULTIPLE)` avant la création de threads de travail, ou
se rattache à un monde externe uniquement si `MPI_Query_thread` prouve ce même niveau. Il ne finalise
que le monde qu'il a lui-même initialisé, après la fin du travail natif ; une application hôte conserve
la propriété de son lifecycle MPI.
Python ne possède, n'initialise et n'exécute aucune ressource ou collective MPI. Les sorties
scientifiques choisissent obligatoirement un `ParallelMode` typé :
`SERIAL` pour le contexte série, `ROOT` pour un rassemblement auquel tous les rangs participent suivi
d'un unique writer rang 0, `COLLECTIVE` pour les hyperslabs HDF5 MPIO exacts, ou `PER_RANK` pour des
artefacts locaux qualifiés par rang et un reçu agrégé. Le mode, le format, la sélection, la cible et
l'identité de chaque pièce native (`global_box_index`, `owner_rank`, `replicated`) sont authentifiés
entre rangs avant toute écriture. La route `COLLECTIVE` appelle le backend C++ HDF5 parallèle sur
`MPI_COMM_WORLD`; `h5py` reste uniquement un lecteur/écrivain série optionnel et n'est jamais un
transport MPI. Une dépendance HDF5 parallèle native absente, un mode incompatible ou un backend
Kokkos GPU/device handle non supporté est refusé avant le
constructeur de `System`/`AmrSystem`; aucune route série implicite ne remplace une demande MPI.

Les maillages non structurés, mobiles/déformables ou changeant de topologie, de nouvelles familles de
stockage, la 3D sur ces routes et une algèbre d'unités ne font pas partie de la release. Ils sont refusés,
pas simulés par des placeholders publics.

## 13. Exemples exécutables normatifs

Quatre scripts sont des tests d'acceptation, pas des esquisses :

1. `examples/final/EXEMPLE_SPEC_FINALE_ADVECTION_SCALAIRE_COMPLET.py` : flux conservatif, parité du
   `Program` SSPRK2 explicite avec `pops.lib.time.SSPRK2`, layout AMR avec au moins un niveau raffiné
   réellement exécuté, HDF5/ParaView, checkpoint et continuation bit-identique ;
2. `examples/final/EXEMPLE_SPEC_FINALE_MULTIPHYSIQUE_CORE.py` : deux `StateSpace` d'un même modèle
   sélectionnés dans deux blocs qualifiés, layout Uniform, champ elliptique, couplage, HDF5/ParaView et
   restart bit-identique ;
3. `examples/final/EXEMPLE_SPEC_FINALE_ADVECTION_IMEX_AMR.py` : parité graphe, identité sémantique et
   état accepté du `Program` IMEX explicite avec `pops.lib.time.IMEX`, coefficients/stages exacts,
   `AMRExecution.subcycled()`, regrid/reflux, HDF5/NPZ/ParaView, restart strict et continuation
   bit-identique ;
4. `examples/final/EXEMPLE_SPEC_FINALE_15_MOMENTS_HYQMOM.py` : état 15 moments, layout Uniform,
   `Program` IMEX explicite avec garde de réalisabilité dans sa transaction, champ de Poisson,
   HDF5/ParaView et continuation bit-identique, sans branche de scénario dans le compilateur. Le
   preset `pops.lib.time.IMEX` reste un constructeur d'un `Program` ordinaire ; il ne remplace pas
   cette écriture explicite lorsqu'une garde scientifique spécifique doit être composée.

`scripts/final_release_contract.py` fixe cet ensemble exact : aucun cinquième script `.py` n'est admis
dans `examples/final/`. Chaque script doit :

- utiliser exclusivement le cycle de vie public ;
- construire ses expressions avec `pops.math` et ne jamais importer l'IR interne `pops._ir` ;
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
