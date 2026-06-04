# Conception de l'API modele DSL Python (`dsl.Model`)

Conception + STATUT. Ce document a ete ecrit comme une SPEC cible (API decrite et mappee
ligne a ligne sur le code existant). La Phase A et le backend natif `production` ont
depuis ete livres : les sections ci-dessous gardent le raisonnement de conception, et la
section 0bis (`Statut d'implementation`) marque ce qui est SHIPPE et ce qui reste un GAP.
Chaque affirmation est ancree dans un fichier lu (cite `chemin:symbole`). Les references de
ligne sont indicatives (elles datent de la redaction ; le code a depuis bouge).

## 0bis. Statut d'implementation (a jour `origin/master`)

Recapitulatif rapide ; le detail par section suit (les balises SHIPPE/GAP y sont reprises).

SHIPPE (ne plus lire comme "cible") :
- **Phase A** (#89/#90) : `dsl.Model` facade (`dsl.py:Model`), `Param` nomme (`dsl.py:Param`,
  mode `const` ; `runtime` leve `NotImplementedError`), `CompiledModel` (`dsl.py:CompiledModel`,
  porte `abi_key`/`model_hash`/`cxx`/`std`), `System.add_equation` (`__init__.py:add_equation`,
  dispatch `ModelSpec`->`add_block` vs `CompiledModel`->adder du backend), `adc.FiniteVolume(limiter=,
  riemann=, variables=)` (`__init__.py:FiniteVolume`), `System.run(t_end, cfl)` (`__init__.py:run`).
  `m.flux`/`m.eval_flux` (declarateur vs evaluateur, noms distincts), `m.primitive_vars(**kwargs)`.
- **Backend `production` natif** (#85) : `production` n'est PLUS un alias d'`aot`. `_BACKENDS`
  (`dsl.py`) mappe `production -> ("native", "add_native_block")` ; `compile_native` emet un LOADER
  natif (`emit_cpp_native_loader`) qui inline `add_compiled_model<ProdModel>` sur le `grid_context()`
  REEL du `System` (zero-copie, parite `add_block`), avec garde-fou de cle d'ABI
  (`add_native_block`, `system.cpp:902`).
- **WENO5-Z + SSPRK3 accessibles depuis Python** (#88) : `adc.Spatial/FiniteVolume(limiter="weno5")`
  et `adc.Explicit(ssprk3=True)` (`__init__.py:Spatial`/`Explicit`). MAIS uniquement via le chemin
  natif `add_block` (modele compose `adc.Model`) ; PAS via les chemins `.so`/`CompiledModel` (voir GAP).

GAP (encore cible / differe) :
- **WENO5 sur `CompiledModel`** : les `.so` (`aot`/`production`) allouent 2 ghosts ; WENO5 en exige 3.
  Rejet explicite cote C++ (`system.cpp:793` AOT, `system.cpp:921` natif loader) ET cote Python
  (`add_equation`). Unification des ghosts (le `.so` alloue selon le limiteur) = Phase A residuelle.
- **Ergonomie `compile()`/cache** : `m.compile(so_path=, include=, ...)` exige encore le chemin de
  sortie et le dossier d'en-tetes ; cache automatique (cle = `model_hash`) a faire.
- **`AmrSystem` en production** : `m.compile(target="amr_system")` leve `NotImplementedError`
  (`dsl.py:Model.compile`) ; pendant natif `add_compiled_model(AmrSystem&)` sans binding Python (Phase D).
- **Params runtime** : `m.param(kind="runtime")` leve `NotImplementedError` (changement d'ABI/codegen ;
  section 2b, Phase E).
- **Validation Python device/MPI** du chemin `production` natif : valide bit-identique en C++ sur
  GH200, mais la validation end-to-end DEPUIS Python (`add_native_block` GPU / multi-rang) est en cours.

NOTE HISTORIQUE. Le texte ci-dessous emploie encore le futur ("CIBLE", "a ajouter") la ou la
chose est desormais livree ; se fier a 0bis et aux balises SHIPPE/GAP. Le contenu de conception
est conserve volontairement (justifications, ancres, taxonomie d'erreurs).

Sources lues pour cette conception :
- `python/adc/dsl.py` : `HyperbolicModel` (toutes les methodes), le codegen
  (`emit_cpp_brick`/`emit_cpp_source`/`emit_cpp_elliptic`/`emit_cpp_so_source`/
  `emit_cpp_aot_source`), la facade `compile(backend=)` + `_BACKENDS` + `adder_for`.
- `python/adc/__init__.py` : sucre runtime `Model`/`Spatial`/`Explicit`/`IMEX`/
  `Implicit`/`DivEpsGrad`/`System`/`AmrSystem`/`PythonFlux`.
- `python/system.cpp` + `python/bindings.cpp` : adders `add_block`,
  `add_dynamic_block`, `add_compiled_block`, et metadonnees (`read_block_meta`,
  `variable_names`/`variable_roles`/`block_gamma`), `set_poisson`, `step_cfl`/
  `step_adaptive`.
- `include/adc/core/variables.hpp` : `VariableRole`/`VariableSet`/`role_from_name`/
  `ADC_EXPORT_BLOCK_METADATA`/`ADC_EXPORT_BLOCK_GAMMA`.
- `include/adc/core/physical_model.hpp` : contrat `PhysicalModel`/
  `HyperbolicPhysicalModel`/`aux_comps`.
- `include/adc/runtime/compiled_block_abi.hpp` : ABI AOT `ADC_DEFINE_COMPILED_BLOCK`.
- `include/adc/runtime/dynamic_model.hpp` : `IModel`/`ModelAdapter` (JIT).
- `include/adc/runtime/dsl_block.hpp` : `add_compiled_model` (natif, template).
- `include/adc/runtime/amr_dsl_block.hpp` : `add_compiled_model` cote `AmrSystem`.
- `docs/PAPER_ROADMAP.md` (panier 2), `docs/ARCHITECTURE.md` (section runtime/DSL).

NOTE D'ENVIRONNEMENT. Un agent frere ajoute en parallele un vrai chemin natif
`production` (loader natif, adder `System`, garde-fou de cle ABI). Cette spec est
ecrite AUTOUR de cette direction (production = natif zero-copie `add_compiled_model`,
PAS le `add_compiled_block` hote a marshaling) sans presumer de ses noms de symboles
exacts : on decrit le CONTRAT, pas l'implementation du frere.


## Decisions tranchees (revue)

Points d'API fixes apres revue, pour eviter l'ambiguite a l'implementation :
1. `m.flux(x=, y=)` = DECLARATEUR symbolique ; l'evaluateur numpy est `m.eval_flux(...)` (noms distincts). (section 1)
2. `m.primitive_vars(rho=expr, ...)` accepte les KWARGS (ordre = layout `Prim`) ; forme positionnelle aussi. (section 1)
3. Le flux NUMERIQUE est `adc.FiniteVolume(limiter=, riemann=, variables=)` -- `riemann`, PAS `flux` (qui est le flux physique). (section 6)
4. `m.param(name, value)` retourne un objet `Param` NOMME (`name`/`value`/`kind`), pas un `Const` anonyme. (section 2)
5. `CompiledModel` porte `abi_key` + `model_hash` + flags de build, pas seulement `so_path`/backend/noms. (section 3)
6. Statut GPU natif CLARIFIE : le chemin device-clean a foncteurs nommes est VALIDE GH200 ; il manque seulement son binding Python. Production GPU n'est PAS casse. (section 5)
7. `m.compile(backend, target)` SANS `device=` : capacites GPU/MPI/AMR verifiees au branchement/execution, pas figees a la compilation. (sections 1, 5, 7)


## 0. Etat actuel (factuel, point de depart)

`HyperbolicModel` (`dsl.py:266`) est le SEUL objet modele. Il porte les formules
(arbre `Expr`), trois groupes de generateurs (`emit_cpp_brick`, `emit_cpp_source`,
`emit_cpp_elliptic`), trois enrobages (`emit_cpp_so_source` JIT, `emit_cpp_aot_source`
AOT, et la fabrique `_emit_bricks`/`_emit_metadata` partagee), et la facade
`compile(backend=)`. Trois backends sont declares (`dsl.py:_BACKENDS`) :
`prototype` -> (`jit`, `add_dynamic_block`), `aot` -> (`compile`, `add_compiled_block`),
`production` -> (`native`, `add_native_block`).

> SHIPPE (#85). Au moment de la redaction, `production` etait un ALIAS d'`aot`. Ce n'est PLUS le
> cas : `_BACKENDS["production"] = ("native", "add_native_block")`, et `compile_native`
> (`emit_cpp_native_loader`) emet le LOADER natif zero-copie decrit au point 3 ci-dessous. La suite
> de cette section 0 conserve l'analyse des 3 chemins C++, toujours exacte.

Trois chemins d'execution existent cote C++ :
1. JIT : `.so` expose `adc_make_model`/`adc_model_nvars`/`adc_destroy_model`
   (`dsl.py:emit_cpp_so_source`), charge par `System::add_dynamic_block`
   (`system.cpp:706`) en `IModel<NV>` a dispatch VIRTUEL, residu HOTE Rusanov ordre 1
   (`system.cpp:host_residual`). Hors hot path GPU/MPI (`dynamic_model.hpp:18`).
2. AOT : `.so` expose l'ABI `ADC_DEFINE_COMPILED_BLOCK` (`compiled_block_abi.hpp:156`),
   charge par `System::add_compiled_block` (`system.cpp:738`). Tourne le chemin de
   production (`make_block`<Limiter,Flux>, SSPRK2/IMEX) MAIS sur une grille locale
   reconstruite dans le `.so` avec marshaling de tableaux plats (`copy_state`/
   `write_state`), donc PAS zero-copie, mono-rang (`compiled_block_abi.hpp:56
   DistributionMapping dm(ba.size(), 1)`, commentaire `:24-26` : "sans AMR/MPI").
3. NATIF : `add_compiled_model(System&, ...)` (`dsl_block.hpp:36`), template C++
   qui fabrique les fermetures sur le `grid_context()` REEL du `System` via
   `make_block` (parite bit-identique a `add_block`, validee CPU et GH200,
   `dsl_block.hpp:18-29`). Pendant AMR : `add_compiled_model(AmrSystem&, ...)`
   (`amr_dsl_block.hpp`). C'est la cible du backend `production`, desormais ATTEINTE pour
   `System` (voir SHIPPE ci-dessous).

> SHIPPE (#85), pour `System`. Le template `add_compiled_model(System&, ...)` a desormais un
> chemin Python : `compile_native` (`dsl.py:emit_cpp_native_loader`) emet un LOADER `.so` (deux
> symboles `extern "C"` : `adc_native_abi_key` + `adc_install_native`) qui inline
> `add_compiled_model<ProdModel>` sur le `grid_context()` REEL du `System`. `System.add_native_block`
> (`system.cpp:902`) le `dlopen`, compare la cle d'ABI (rejet explicite si en-tetes/compilateur/std
> divergent) puis appelle l'installateur : bloc natif zero-copie, parite `add_block`. GAP : le pendant
> `add_compiled_model(AmrSystem&)` (`amr_dsl_block.hpp`) n'a PAS de binding Python (`target="amr_system"`
> leve `NotImplementedError` ; Phase D).

> SHIPPE (#89). `dsl.Model` EXISTE maintenant (`dsl.py:Model`, module `adc.dsl`). C'est la facade
> qui COMPOSE un `HyperbolicModel` prive (`_m`) et delegue chaque appel. Le nom `adc.Model`
> (`__init__.py:Model`) reste la fonction distincte qui compose un `ModelSpec` de briques NATIVES
> (chemin (a)) ; les deux coexistent comme prevu (l'un dans `adc.dsl`, l'autre dans `adc`).


## 1. Facade stable `dsl.Model`

> SHIPPE (#89). Cette section est LIVREE : `dsl.Model`, `m.flux`/`m.eval_flux`,
> `m.primitive_vars(**kwargs)`, `m.param`, `m.compile(backend, target)` existent
> (`dsl.py:Model`). Le tableau de mapping ci-dessous decrit l'implementation effective.

CIBLE : `dsl.Model` est la SURFACE stable ; `HyperbolicModel` reste le BACKEND
interne inchange. `dsl.Model` delegue chaque appel a une methode existante de
`HyperbolicModel` (composition, pas heritage : `dsl.Model` detient un
`HyperbolicModel` prive `_m`). Aucune numerique n'est touchee.

Construction : `m = dsl.Model("euler")` cree le `HyperbolicModel("euler")` interne.

Mapping methode cible -> backing `HyperbolicModel` :

| `dsl.Model` (cible) | backe par `HyperbolicModel` (`dsl.py`) | note |
|---|---|---|
| `m.conservative_vars(*names, roles=)` | `conservative_vars(*names, roles=)` `:291` | identique (transfere `roles=`) |
| `m.primitive_vars(rho=expr, u=expr, ...)` (kwargs) ou `(*vars, roles=)` | `set_primitive_state(*vars_or_names, roles=)` `:323` + `primitive(name, expr)` `:301` | STYLE CIBLE = KWARGS `name=expr` : chaque kwarg definit une primitive (`_m.primitive(name, expr)`) ET fixe le layout ORDONNE de `Prim` dans l'ordre des kwargs (Python 3.7+ : ordre d'insertion garanti). La forme positionnelle `(*vars, roles=)` reste acceptee. Les `roles=` (kwarg ou liste) restent supportes pour le mapping role->indice |
| `m.primitive(name, expr)` | `primitive(name, expr)` `:301` | definition d'une primitive par formule |
| `m.aux(name)` | `aux(name)` `:306` | champ auxiliaire (doit etre clef de `AUX_CANONICAL` `:35`) |
| `m.conservative_from(exprs)` | `set_conservative_from(exprs)` `:334` | inverse prim->cons (le DSL ne sait pas inverser) |
| `m.flux(x=, y=)` | `set_flux(x, y)` `:311` | DECLARATEUR symbolique du flux physique (decision tranchee, voir ci-dessous) |
| `m.eval_flux(U, aux, dir)` | `flux(U, aux, dir)` `:354` | EVALUATEUR numpy (debug / proto hote) ; nom DISTINCT du declarateur `m.flux` |
| `m.source(s)` | `set_source(s)` `:313` | optionnel |
| `m.eigenvalues(x=, y=)` | `set_eigenvalues(x, y)` `:312` | |
| `m.elliptic_rhs(e)` | `set_elliptic_rhs(e)` `:314` | optionnel (couplage Poisson) |
| `m.gamma(value)` ou `m.set_gamma(value)` | `set_gamma(gamma)` `:316` | EOS, porte par `ADC_EXPORT_BLOCK_GAMMA` |
| `m.param(name, value)` | `Param` + `Model.param` (#89) | SHIPPE en mode `const` (constante NOMMEE inlinee au codegen, stockee dans `m.params`) ; `kind="runtime"` leve `NotImplementedError` (section 2b, GAP). Cas `name=="gamma"` appelle aussi `set_gamma` |
| `m.check()` | `check()` `:382` | verifie dependances |
| `m.compile(backend, target)` | `Model.compile(...)` (#89, delegue a `HyperbolicModel.compile`) | SHIPPE : renvoie un `CompiledModel` (section 3). `target="system"` cable ; `target="amr_system"` leve `NotImplementedError` (GAP, Phase D). PAS d'argument `device` : les capacites (GPU/MPI/AMR) sont verifiees AU BRANCHEMENT/EXECUTION (`add_equation`/`run`), pas figees comme un drapeau de compilation (eviterait une fausse garantie si le module n'est pas bati avec Kokkos/CUDA). Voir sections 5 et 7 |

COLLISION DE NOMS : TRANCHEE. Dans `HyperbolicModel`, `flux(U, aux, dir)` `:354` est
l'EVALUATEUR numpy (interprete CPU) et `set_flux` `:311` est le DECLARATEUR. Le plan
cible nomme le declarateur `m.flux`. DECISION : sur `dsl.Model`, `m.flux(x=, y=)` est le
DECLARATEUR symbolique (delegue a `set_flux`) ; l'evaluateur numpy est expose sous le
nom DISTINCT `m.eval_flux(U, aux, dir)` (delegue a `_m.flux`). La surface declarative
prime ; aucun nom ne porte les deux sens. (Pas d'alias `m.set_flux` sur la facade : un
seul nom par intention.)

Etat de ces points cible (initialement des GAP ; voir 0bis) :
- `m.param(name, value)` : SHIPPE en mode `const` (#89, classe `Param` + `Model.param`,
  `dsl.py`). Mode `runtime` reste GAP (section 2b, `NotImplementedError`).
- `m.compile(target=)` : SHIPPE (#89). `Model.compile` prend `backend`, `target`, `name`, `cxx`,
  `std`, `require_metadata`. `target="system"` cable ; `target="amr_system"` leve
  `NotImplementedError` (GAP, Phase D). PAS de `device=` (point 7) : capacites verifiees au
  branchement (`add_equation`) / a l'execution (`run`), pas figees comme drapeau de compilation
  (un `device=True` a la compilation donnerait une fausse garantie : un `.so` peut compiler sans
  que le module hote soit device-capable).
- `m.compile()` renvoie desormais un `CompiledModel` (#89, section 3), pas un `str so_path`.


## 2. `m.param(name, value)` : deux modes

> STATUT. Mode (a) SHIPPE (#89, classe `Param`, `dsl.py`). Mode (b) GAP (leve
> `NotImplementedError` ; changement d'ABI/codegen, Phase E). Le raisonnement ci-dessous
> documente pourquoi (b) n'est pas livrable sans changement de moteur.

### Mode (a) : constante figee a la compilation (FAISABLE aujourd'hui)

Le codegen INLINE deja toute constante. Un scalaire Python passe dans une formule est
promu en `Const(float(o))` (`dsl.py:_wrap :110`), et `Const.to_cpp()` renvoie
`repr(self.value)` (`dsl.py:117`) : la valeur est ecrite EN DUR dans le `.so`. Exemple
reel : `dsl_euler/run.py` ecrit `GAMMA = 1.4` puis `(GAMMA - 1.0) * (...)` ; le `1.4`
est inline. Donc `m.param("gamma", 1.4)` en mode constante n'a besoin d'AUCUN moteur
nouveau : c'est du sucre Python qui retourne un `Const` (ou un scalaire) reutilisable
dans plusieurs formules, recompile a chaque changement de valeur.

Forme proposee : `g = m.param("gamma", 1.4)` retourne un objet `Param` NOMME (pas un
`Const` anonyme), porteur de son IDENTITE : `name`, `value`, `kind` (`"const"` au depart,
`"runtime"` reserve, voir mode b). `Param` se comporte comme un `Expr` dans les formules
(il s'INLINE en `Const(value)` au codegen, donc zero-risque cote brique generee), mais
GARDE name/value/kind pour : l'introspection (`m.params`), les logs/diagnostics, la
reproductibilite (un run trace ses parametres), et la transition future vers les params
runtime (mode b) SANS changer la surface utilisateur. Changer la valeur exige
`m.compile(...)` a nouveau (tout est recompile aujourd'hui).

CAS PARTICULIER deja cable : `gamma` a un canal dedie hors formule via `set_gamma`
`:316` -> `ADC_EXPORT_BLOCK_GAMMA` (`variables.hpp:153`), lu par `read_block_meta`
(`system.cpp:179`) pour les couplages inter-especes. `m.param("gamma", ...)` devrait
donc, en plus de l'inliner dans les formules, appeler `set_gamma` pour que la
metadonnee ABI soit coherente (sinon le `System` retombe sur 1.4, `system.cpp:629`/`:844`).

### Mode (b) : parametre runtime (modifiable SANS recompiler) -> PHASE ULTERIEURE

INFAISABLE avec le codegen actuel sans changement de moteur. Justification ancree :
- Le codegen n'a AUCUN concept d'uniforme/membre. Les briques generees lisent
  uniquement les variables conservatives (`U[i]`, `cons_locals` `:468`), les
  primitives derivees (`prim_locals` `:471`), et les champs `Aux` (`a.<nom>`,
  `aux_locals` `:474`). Toute autre valeur est une `Const` inline.
- L'ABI AOT (`compiled_block_abi.hpp:156`) ne transporte aucun parametre : les
  signatures `adc_compiled_residual`/`advance`/`max_speed` (`compiled_block_abi.hpp:159-176`)
  prennent `U`, `aux`, geometrie, schema, pas de bloc de parametres. `Model model{}`
  est default-construit (`compiled_block_abi.hpp:103`,`:118`,`:130`,`:143`), sans etat.
- Cote JIT, `IModel`/`ModelAdapter` (`dynamic_model.hpp`) est aussi sans etat
  parametrique : `M model{}` default-construit (`dynamic_model.hpp:51`).

Deux voies possibles pour un VRAI parametre runtime, toutes deux PHASE 2 :
1. Passer les parametres par le canal `Aux` : declarer un champ aux constant par
   cellule et le peupler depuis Python (comme `set_magnetic_field` peuple `B_z`,
   `system.cpp:911`). Faisable SANS changer le codegen (le param devient un `m.aux`),
   mais limite aux 5 composantes canoniques (`AUX_CANONICAL` `:35`) et au cout d'un
   champ n*n pour un scalaire (abus du canal aux).
2. Etendre l'ABI : ajouter un parametre/bloc de doubles aux signatures
   `adc_compiled_*`, generer un `Model` construit avec ces parametres (membres), et
   propager via `set_density`-like. C'est un CHANGEMENT D'ABI (header
   `compiled_block_abi.hpp` + lecture `system.cpp:add_compiled_block`) ET de codegen
   (struct a membres + constructeur). A marquer explicitement PHASE 2.

VERDICT. `m.param(name, value)` est livrable en mode (a) (constante figee) immediatement
et sans risque. Le mode (b) (runtime, sans recompil) demande un changement de moteur
(ABI + codegen) et reste une phase ulterieure ; `m.param` doit donc soit ne supporter
que (a) au depart, soit accepter un `kind="const"|"runtime"` ou `kind="runtime"` LEVE
explicitement `NotImplementedError` (voir taxonomie section 4).


## 3. Objet Python `CompiledModel`

> SHIPPE (#89). `CompiledModel` existe (`dsl.py:CompiledModel`) et est produit par
> `Model.compile`. Il porte tous les champs ci-dessous (`abi_key`, `model_hash`, `cxx`,
> `std`, `caps`, `params`...). `System.add_equation` (#89/#90) le consomme et aiguille
> sur l'adder du backend. Le tableau et la sous-section consommation decrivent l'implementation
> effective ; voir les notes en ligne pour le cas `production` (desormais natif, plus un alias d'`aot`).

A la redaction, `compile()` renvoyait un `str` (`so_path`) et exposait separement
`adder_for(backend)` pour savoir quel adder `System` employer. Le `CompiledModel` remplace
ce couple `(str, classmethod)` par un objet qui PORTE le chemin et tout ce qu'il faut pour
le brancher correctement.

### Champs

| champ | source (ancre) | role |
|---|---|---|
| `so_path` | retour de `compile_so`/`compile_aot` (`dsl.py:705`/`:743`) | chemin du `.so` |
| `backend` | `backend=` passe a `compile` | `prototype`/`aot`/`production` |
| `adder` | `_BACKENDS[backend][1]` (`dsl.py:792`) | nom de methode `System` a employer |
| `cons_names` | `HyperbolicModel.cons_names` | noms conservatifs (override `names=`) |
| `cons_roles` | `roles_for(cons_names, cons_roles)` (`dsl.py:77`) | roles physiques |
| `prim_names` | `HyperbolicModel.prim_state` | layout primitif |
| `n_vars` | `HyperbolicModel.n_vars` (`dsl.py:340`) | nb composantes |
| `gamma` | `HyperbolicModel.gamma` (`dsl.py:316`) | EOS (None = defaut 1.4) |
| `n_aux` | `aux_n_aux(aux_names)` (`dsl.py:39`) | largeur canal aux requise |
| `params` | dict `{name: Param}` (objets `Param` nommes, section 2) | introspection / reproductibilite |
| `caps` | derive du backend (section 5) | drapeaux CPU/MPI/AMR/GPU |
| `abi_key` | cle ABI baked (compilateur + std + signature d'en-tetes, cf. `abi_key.hpp` / `adc_cases/common/native.py`) | refuser un `.so` incompatible AU CHARGEMENT plutot qu'un UB silencieux |
| `model_hash` | hash stable du modele (formules + roles + n_aux + params) | identifier/reutiliser un `.so` deja compile ; tracer le run |
| `cxx`, `std`, `cxx_flags` | compilateur, standard, flags passes a la compilation | reproductibilite + diagnostic d'incompatibilite ABI |

`CompiledModel` est PRODUIT par `m.compile(...)` : la facade compile le `.so` (via
`compile_so`/`compile_aot` inchangees), puis empaquette le chemin avec les
metadonnees deja connues du `HyperbolicModel` (pas de relecture du `.so` : Python
detient deja noms/roles/gamma/n_aux). Les memes metadonnees sont DEJA emises dans le
`.so` par `_emit_metadata` (`dsl.py:675`) et relues cote C++ par `read_block_meta`
(`system.cpp:179`) ; `CompiledModel` les expose juste cote Python pour le dispatch et
les diagnostics, sans nouvelle source de verite.

Un `CompiledModel` n'est donc PAS un simple `str so_path` : il porte aussi la CLE ABI
et les flags de build (`abi_key`, `model_hash`, `cxx`/`std`/`cxx_flags`). C'est ce qui
permet de refuser au CHARGEMENT un `.so` compile avec un etat incompatible (compilateur,
standard, en-tetes divergents) au lieu d'un comportement indefini silencieux : le chemin
natif `production` compare deja cette cle cote C++ (`abi_key.hpp`, garde-fou du loader),
et `CompiledModel.abi_key` rend la verification + le diagnostic disponibles cote Python.

### Consommation : `System.add_equation(model=compiled, ...)`

CIBLE : `sim.add_equation(name, model=compiled, spatial=, time=, substeps=, names=)`
dispatche sur le bon adder selon `compiled.backend`/`compiled.adder` :

- `backend="prototype"` (`adder="add_dynamic_block"`) ->
  `self._s.add_dynamic_block(name, compiled.so_path, substeps, names, recon)`
  (`bindings.cpp:78`, `system.cpp:706`). NB : `add_dynamic_block` ne prend PAS
  `limiter`/`riemann`/`time` (chemin hote Rusanov ordre 1) ; il prend `recon`
  (`none`/`minmod`/`vanleer`) et `substeps`. La facade doit donc IGNORER (ou refuser,
  section 4) un `spatial.riemann != "rusanov"` pour ce backend (`riemann` = flux
  NUMERIQUE de `FiniteVolume`, cf. section 6 ; `flux` reste le flux PHYSIQUE du modele).
- `backend="aot"` (`adder="add_compiled_block"`) ->
  `self._s.add_compiled_block(name, compiled.so_path, spatial.limiter, spatial.riemann,
  spatial.variables, time.kind, substeps, names)` (`system.cpp:776` ; l'arg C++ du flux
  numerique s'appelle deja `riemann`, `variables` mappe `recon`).
- `backend="production"` (`adder="add_native_block"`) SHIPPE (#85) -> `self._s.add_native_block(name,
  compiled.so_path, spatial.limiter, spatial.riemann, spatial.variables, time.kind, gamma, substeps,
  evolve)` (`system.cpp:902`). Ce n'est PLUS l'alias d'`aot` : l'adder natif zero-copie EXISTE
  (loader `.so` -> `add_compiled_model<ProdModel>` sur le contexte reel ; cle d'ABI verifiee).
  NB : ce chemin ne prend PAS `names=` (les noms/roles viennent des metadonnees du `.so`) ; `add_equation`
  leve si `names=` est fourni (`__init__.py:add_equation`).

`add_equation` est SHIPPE (#89/#90, `__init__.py:add_equation`) : c'est la methode de la facade
Python `System` qui aiguille selon le type de `model` : un `ModelSpec` (`adc.Model(...)`) ->
`add_block` ; un `CompiledModel` -> l'adder de bloc compile/dynamique/natif fixe par le backend.
`System.add_block` (`__init__.py:add_block`) prend un `ModelSpec` de briques natives, PAS un `.so`.
`add_equation` centralise le couplage backend<->adder que `dsl.py` documente comme une regle de
surete ("ne pas brancher un `.so` AOT sur `add_dynamic_block`").


## 4. Taxonomie d'erreurs

Toutes les erreurs sont des `ValueError` (ou `NotImplementedError` pour le differe),
levees AU PLUS TOT (a la declaration ou a `compile`/`add_equation`, pas a l'execution),
message FACTUEL nommant la cause et l'action corrective. Modele des messages existants
de `dsl.py` (p.ex. `:296`, `:822`, `:845`).

| erreur | quand | forme du message (gabarit) | ancre du garde existant |
|---|---|---|---|
| role manquant requis | `compile(require_metadata=True)` sans role ni nom canonique | "le modele '<nom>' ne fournit pas roles physiques (...) ; le .so retomberait sur le fallback (roles 'custom')" | DEJA implemente `dsl.py:837-847` |
| gamma manquant requis | idem, `gamma is None` | "(...) ne fournit pas gamma (set_gamma(...)) (...)" | DEJA implemente `dsl.py:842` |
| param non defini | formule reference un `param` jamais pose | "param '<name>' reference mais non defini (m.param('<name>', valeur))" | etend `check()` `:382` (qui leve deja sur variable non declaree, `:397`) |
| param mode runtime | `m.param(name, value, kind="runtime")` | "param runtime non supporte (changement d'ABI/codegen requis, phase ulterieure) ; utiliser un param constant ou un champ aux" | NOUVEAU, `NotImplementedError` (section 2 mode b) |
| backend inconnu | `compile(backend=x)` hors `_BACKENDS` | "compile : backend <x> inconnu (attendus ['aot','production','prototype'])" | DEJA `dsl.py:821-823` |
| flux incompatible variables | `spatial.riemann in {hllc,roe}` sans primitive `p` declaree | "riemann 'hllc'/'roe' exige une pression : declarer m.primitive('p', ...)" | ANCRE physique : la brique n'emet `pressure`/`wave_speeds` que si `'p' in prim_defs` (`dsl.py:558`) ; `make_block` exige `m.pressure`/`m.wave_speeds` pour HLLC/Roe (cf. `amr_dsl_block.hpp:135`) |
| GPU/MPI/AMR incompatible backend | `add_equation` avec `device="gpu"`/MPI/AMR sur backend non capable | "backend '<b>' n'est pas device-clean / multi-rang : utiliser backend='production' (chemin natif)" | section 5 ; `compiled_block_abi.hpp:24-26` ("sans AMR/MPI"), `dynamic_model.hpp:18-21` (hors hot path GPU) |
| production demande mais seul aot dispo | `compile(backend="production")` tant que le natif n'est pas branche | soit un WARNING explicite ("production -> aot : chemin natif zero-copie non encore branche"), soit erreur si `device="gpu"` exige | `dsl.py:786-791` documente deja l'alias ; aujourd'hui SILENCIEUX (a rendre explicite) |
| require_metadata sur prototype | `compile(backend="prototype", require_metadata=True)` | "backend 'prototype' (...) incompatible avec require_metadata=True ; utiliser 'aot' ou 'production'" | DEJA `dsl.py:830-834` |
| names= mauvaise longueur | `add_equation(names=)` de longueur != n_vars | "names= a K noms mais le bloc '<n>' a NV variables" | DEJA cote C++ `system.cpp:618`/`:832` |

PRIORITE actuelle de resolution des noms/roles (a documenter pour l'utilisateur, NON a
changer) : `names=` explicite > metadonnees ABI du `.so` (`read_block_meta`) > fallback
`u0..` (`system.cpp:826-844` et `:614-628`). Les ROLES et le PRIMITIF ne viennent QUE
de l'ABI (l'API `names=` ne les fournit pas, `system.cpp:828`).


## 5. Matrice de capacites des backends

Etat HONNETE. Lignes = backend `compile()`, colonnes = capacite. La matrice est aussi
materialisee cote code dans `_BACKEND_CAPS` (`dsl.py`), lu par `CompiledModel.caps`.

| backend | CPU serie | MPI | AMR | GPU/Kokkos | zero-copie device | callbacks Python en hot path |
|---|---|---|---|---|---|---|
| `prototype` (JIT, `add_dynamic_block`) | oui (residu hote Rusanov o1) | non | non | non | non | non (C++ dispatch virtuel, sans GIL) |
| `aot` (`add_compiled_block`) | oui (production o2, HLLC/Roe) | non | non | non | non | non |
| `production` (natif `add_native_block`, #85) | oui | oui | non (System mono-niveau) | en cours (validation Python) | oui | non |

> SHIPPE (#85). `production` n'est PLUS l'alias d'`aot`. `_BACKEND_CAPS["production"]` (`dsl.py`)
> declare `{cpu, mpi}=True`, `amr=False` (System mono-niveau ; AMR = Phase D), `gpu=False` par
> PRUDENCE (le chemin natif est device-clean en C++/GH200, mais la validation end-to-end DEPUIS
> Python n'est pas encore faite). Ces capacites sont des drapeaux de diagnostic verifies au
> branchement/execution, pas figes a la compilation (point 7).

Ancres :
- `prototype`/JIT : `dynamic_model.hpp:18-21` ("CHEMIN HOTE / PROTOTYPAGE uniquement ;
  les appels virtuels ne passent PAS dans un kernel GPU (...) ne pas utiliser dans la
  boucle chaude"). Residu hote ordre 1 Rusanov : `system.cpp:host_residual`, recon
  MUSCL hote 0/minmod/vanleer (`system.cpp:41`). Pas de flux HLLC/Roe (Rusanov fige).
- `aot` : tourne le chemin de production `make_block`<Limiter,Flux> / SSPRK2 / IMEX
  (`compiled_block_abi.hpp:21-26`), donc HLLC/Roe et ordre 2 dispos. MAIS sur grille
  LOCALE mono-rang reconstruite dans le `.so` (`compiled_block_abi.hpp:56`
  `DistributionMapping dm(ba.size(), 1)`) avec marshaling `copy_state`/`write_state`
  (`system.cpp:794-824`) : pas zero-copie, pas de halos MPI, pas d'AMR
  (`ARCHITECTURE.md:499` "sans AMR/MPI").
- `production` SHIPPE (#85) : `add_compiled_model` (`dsl_block.hpp`) fabrique les
  fermetures sur le `grid_context()` REEL, residu via `make_block` avec `fill_boundary`
  (halos MPI) + `assemble_rhs` Kokkos sur les vrais `MultiFab`, SANS recopie ; parite
  bit-identique `add_block` validee CPU/Serial ET GH200 (foncteurs nommes). Le backend
  `production` Y EST DESORMAIS RE-ROUTE depuis Python : `compile_native` emet un loader
  `.so`, `System.add_native_block` (`system.cpp:902`) le `dlopen`, verifie la cle d'ABI,
  puis appelle `add_compiled_model<ProdModel>`. AMR : `add_compiled_model(AmrSystem&)`
  (`amr_dsl_block.hpp`) reste SANS binding Python (GAP, Phase D). STATUT GPU (a ne PAS
  lire comme "production GPU casse") : l'ANCIEN chemin a LAMBDAS ETENDUES `__host__
  __device__` segfautait sur Cuda (limite nvcc) ; il a ete remplace par les FONCTEURS
  NOMMES device-clean de `block_builder.hpp`, et c'est CE chemin que `add_compiled_model`
  emprunte, VALIDE bit-identique sur GH200 (parite A==B `dres=0`, multi-box + MPI ;
  #64/#48). GAP restant : la validation device/MPI end-to-end DEPUIS Python
  (`add_native_block` sur GPU / multi-rang) est en cours (cf. section 7 Phase C).
- Aucun backend n'execute de callback Python par cellule : meme `prototype` est un
  modele C++ (issu du codegen), pas un `adc.PythonFlux`. `adc.PythonFlux`
  (`__init__.py:420`) est un chemin numpy hote SEPARE, hors DSL compile.


## 6. Sucre runtime du plan -> existant

Mapping des objets/methodes runtime cibles du plan sur ce qui EXISTE
(`__init__.py`, `bindings.cpp`, `system.cpp`).

| cible (plan) | existant | statut |
|---|---|---|
| `adc.FiniteVolume(limiter=, riemann=, variables=)` | `FiniteVolume` (`__init__.py:FiniteVolume`), remappe sur `adc.Spatial(limiter=, flux=, recon=)` | SHIPPE (#89). DECISION (point 3) : le flux NUMERIQUE (Rusanov/HLL/HLLC/Roe) s'appelle `riemann=`, PAS `flux=`, pour ne pas collisionner avec le flux PHYSIQUE `m.flux` du modele. `limiter=` -> `Spatial.limiter` (none/minmod/vanleer/weno5), `riemann=` -> `Spatial.flux`, `variables=` (`"conservative"`/`"primitive"`) -> `Spatial.recon`. |
| `adc.IMEX(substeps=)` | `adc.IMEX(substeps=)` (`__init__.py:IMEX`) | EXISTE a l'identique. `Explicit(substeps=, method=)` (`ssprk2`/`ssprk3`, #88) et `Implicit(dt_ratio=, substeps=)` (alias d'IMEX) aussi. |
| `adc.DivEpsGrad` | `adc.DivEpsGrad(epsilon=)` (`__init__.py:DivEpsGrad`) | EXISTE. Operateur elliptique `div(eps grad)`. eps(x) variable via `set_epsilon_field`. |
| `adc.DirichletWall.circle(radius=)` | `set_poisson(wall="circle", wall_radius=)` (`system.cpp`) | GAP de sucre : pas d'objet `DirichletWall` ; aujourd'hui c'est un argument chaine de `set_poisson`/`add_elliptic_model`. `DirichletWall.circle(r)` serait un constructeur retournant `(wall="circle", wall_radius=r)`. |
| `sim.add_equation(model=, ...)` | `System.add_equation` (`__init__.py:add_equation`, #89/#90) | SHIPPE : dispatcheur de la section 3 (aiguille `ModelSpec` -> `add_block` vs `CompiledModel` -> adder du backend). |
| `sim.run(...)` | `System.run(t_end, cfl, max_steps)` (`__init__.py:run`, #89) | SHIPPE : boucle `while time()<t_end: step_cfl(cfl)`. `step`/`advance`/`step_cfl`/`step_adaptive` restent exposes. |
| `adc.System(n=, periodic=)` | EXISTE (`__init__.py:System`) | identique. |

Tous ces items sont du SUCRE Python pur-facade : aucun ne touche la numerique C++.
`add_equation` et `run` sont les seuls a porter une vraie logique (dispatch + boucle),
le reste est renommage/remap d'arguments. GAP de sucre restant : `adc.DirichletWall`.


## 7. Sequencement d'implementation

> STATUT global. Phase A SHIPPE (#89/#90). Phase B SHIPPE (#85, pour `System`). Phases C
> (validation Python device/MPI), D (`AmrSystem` en production) et E (params runtime) restent
> des GAP. Le sequencement ci-dessous est conserve pour la tracabilite ; les balises par phase
> indiquent l'etat reel.

Regroupe par dependance. Chaque etape note son WRITE-SET de fichiers.

### Phase A : facade pure-Python, AUCUN changement de moteur -- SHIPPE (#89/#90)

Livree. N'edite que du Python ; ne touche ni `include/**` ni `python/system.cpp`/`bindings.cpp`.
Items 1-6 ci-dessous : tous SHIPPES.

1. `dsl.Model` (section 1) : delegation vers `HyperbolicModel`. WRITE-SET :
   `python/adc/dsl.py` (ajout d'une classe ; ne pas modifier `HyperbolicModel`).
2. `m.param` mode (a) constante (section 2a) + cas `gamma` -> `set_gamma`. WRITE-SET :
   `python/adc/dsl.py`. S'appuie sur `_wrap`/`Const` inchanges.
3. `CompiledModel` (section 3) produit par `m.compile`, empaquetant `so_path` +
   metadonnees deja connues. WRITE-SET : `python/adc/dsl.py`.
4. Sucre runtime non-dispatch (section 6) : `adc.FiniteVolume` (remap de `Spatial`),
   `adc.DirichletWall.circle`, `sim.run`. WRITE-SET : `python/adc/__init__.py`.
5. `sim.add_equation` (section 3) : dispatch `ModelSpec` -> `add_block` ;
   `CompiledModel(prototype)` -> `add_dynamic_block` ; `CompiledModel(aot/production)`
   -> `add_compiled_block`. WRITE-SET : `python/adc/__init__.py` (methode de la facade
   `System`). N'exige AUCUN nouveau binding (les adders existent, `bindings.cpp:78`/`:81`).
6. Taxonomie d'erreurs (section 4) cote Python : etendre `check()` pour les params
   non poses, garde-fou flux/`p`, garde-fou backend/device. WRITE-SET : `python/adc/dsl.py`
   (+ messages dans `add_equation`, `__init__.py`).

La phase A est testable contre les backends `prototype` et `aot` (CPU serie), sans GPU/MPI.

### Phase B : backend natif `production` -- SHIPPE (#85, pour `System`)

Livree pour `System`. C'est du cablage de dispatch (pas de numerique nouvelle).

7. SHIPPE. `_BACKENDS["production"]` (`dsl.py`) vaut `("native", "add_native_block")` ; `compile`
   route sur `compile_native` (`emit_cpp_native_loader`) ; `add_equation` branche sur
   `System.add_native_block`. Le pendant `AmrSystem` reste un GAP (Phase D).
8. SHIPPE (caduc). `production` n'etant plus l'alias d'`aot`, l'ecart a rendre explicite a disparu.
   Reste : `compile(backend="prototype", require_metadata=True)` leve (`dsl.py:Model.compile`), et
   `add_equation` rejette `weno5` / HLLC-Roe-sans-`p` / `names=` sur le chemin natif (section 4).

### Phase C : validation Python device/MPI -- GAP

9. Validation device/MPI du chemin `production` natif DEPUIS Python : parite A==B (`dres=0`) sur
   device, bit-identite multi-rang via `add_native_block`. Deja faite pour `add_compiled_model`
   en C++ (`dsl_block.hpp`) sur GH200 ; reste a valider de bout en bout depuis Python. AUCUN
   write-set Python nouveau (tests).

### Phase D : `AmrSystem` en production -- GAP (phase 2)

10. GAP. `add_equation` cote `adc.AmrSystem` (`__init__.py:AmrSystem`) router vers le pendant
    AMR natif `add_compiled_model(AmrSystem&)` (`amr_dsl_block.hpp`). Aujourd'hui
    `m.compile(target="amr_system")` leve `NotImplementedError` (`dsl.py:Model.compile`).
    LIMITE : `AmrSystem` n'est PAS a parite avec `System` (mono-bloc, explicite) : la facade
    devra refuser un `spatial.riemann="roe"`/`variables="primitive"` sur AMR. WRITE-SET :
    `python/adc/__init__.py` (facade `AmrSystem`).

### Phase E : `m.param` runtime -- GAP (phase 2, changement de moteur)

11. Mode (b) (section 2b) : ABI a parametres + codegen a membres, OU canal aux dedie.
    WRITE-SET LOURD : `include/adc/runtime/compiled_block_abi.hpp`,
    `python/system.cpp` (`add_compiled_block`), `python/adc/dsl.py` (codegen). Hors
    chemin critique de reproduction Hoffart (`PAPER_ROADMAP.md:147-150`, panier 2
    transverse, optionnel).

### Resume de dependance

- A (1-6) : SHIPPE (#89/#90). Le gros de la valeur (`dsl.Model` stable, `CompiledModel`,
  `add_equation`, sucre runtime, erreurs).
- B (7-8) : SHIPPE (#85) pour `System` (backend natif `production` -> `add_native_block`).
- C (9) : GAP, validation device/MPI DEPUIS Python (deja valide en C++ sur GH200).
- D (10) : GAP, phase 2 AMR (bornee par la non-parite `AmrSystem` ; `target="amr_system"` leve).
- E (11) : GAP, phase 2, seul item exigeant un changement d'ABI/codegen (param runtime).
