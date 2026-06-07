# Modèles

Un *modèle* dans `adc` décrit **une équation** : ses formules ponctuelles (flux, source,
vitesses d'onde, second membre elliptique). Il existe **trois façons** d'écrire un modèle, qui
produisent toutes le MÊME objet calculatoire côté coeur C++ et se branchent de la même manière sur
un `adc.System` :

1. **Modèle avec briques (natif)** : on COMPOSE des briques génériques déjà compilées
   (`adc.Model(state, transport, source, elliptic)`). C'est la voie la plus directe pour assembler
   un modèle existant : aucune compilation à la volée, parité production totale (MPI/AMR/GPU).
2. **Modèle DSL** : on ÉCRIT le modèle en FORMULES symboliques (`adc.dsl.Model`), puis on le
   compile en un `.so`. C'est la voie quand le modèle voulu n'existe pas comme brique native.
3. **Modèle hybride** : on MÉLANGE, dans un seul modèle, des briques natives et des briques DSL
   partielles (`adc.CompositeModel`). C'est l'entre-deux : réutiliser une brique native pour un
   slot et écrire l'autre en formules.

Ces trois objets sont des compositions de **briques génériques**. Le coeur reste agnostique au
scénario : il ne nomme aucun cas physique (diocotron, Euler-Poisson, deux-fluides...) ; ce sont des
compositions côté application. Pour le détail des méthodes numériques (reconstruction MUSCL/WENO,
flux de Riemann, intégrateurs SSPRK/IMEX, Poisson multigrille), voir
[ALGORITHMS.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md). Pour l'architecture en couches (modèle / maillage / dispatch /
intégrateur), voir [ARCHITECTURE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ARCHITECTURE.md).

## PhysicalModel : le concept

Toutes les briques satisfont le même CONTRAT C++, le concept `adc::PhysicalModel`
([include/adc/core/physical_model.hpp](https://github.com/wolf75222/adc_cpp/blob/master/include/adc/core/physical_model.hpp)). Un
`PhysicalModel` décrit une équation comme un jeu de **fonctions pures d'états ponctuels** — rien de
plus. C'est le seul axe « quoi calculer » de l'architecture, séparé de l'axe « où / comment itérer »
(maillage + dispatch) et de l'axe « dans quel ordre » (intégrateur + coupleur, cf.
[ARCHITECTURE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ARCHITECTURE.md)).

Le contrat minimal exige quatre fonctions :

- `flux(U, aux, dir)` : le flux physique dans la direction `dir` (0 = x, 1 = y) ;
- `max_wave_speed(U, aux, dir)` : la plus grande vitesse d'onde (pour le CFL et le solveur de
  Riemann) ;
- `source(U, aux)` : le terme source ponctuel ;
- `elliptic_rhs(U)` : le second membre de l'équation elliptique (densité de charge / de masse selon
  le modèle).

Point d'unification : `flux` ET `source` reçoivent `aux` (le canal `adc::Aux` : potentiel `phi`,
gradient `grad_x`/`grad_y`, et champs étendus optionnels `B_z`, `T_e`). C'est ce qui place sous un
même opérateur spatial le transport à dérive (l'`aux` est lu dans le flux) et le fluide compressible
auto-gravitant (l'`aux` est lu dans la source).

Une brique **hyperbolique** complète satisfait en plus `adc::HyperbolicPhysicalModel` : elle porte
les VARIABLES (conservatives et primitives) et les conversions `to_primitive` / `to_conservative`,
parce que variables, conversions et flux sont physiquement liés (un flux est écrit pour une
disposition de variables donnée). C'est cette brique-là que l'on écrit, native ou DSL.

## Modèle avec briques (composition native)

`adc.Model(state, transport, source, elliptic)` compose un modèle à partir de quatre briques
génériques déjà compilées et renvoie une `ModelSpec` (des tags lus côté C++ par la fabrique de
modèles). Python compose les objets ; le calcul cellule par cellule reste C++ compilé (pas de numpy,
GPU/MPI conservés). Les briques réelles, telles qu'exposées par `adc.*` (et leurs structs C++ dans
[include/adc/physics/](https://github.com/wolf75222/adc_cpp/blob/master/include/adc/physics/)) :

**État** (`state=`)
- `adc.Scalar()` : état scalaire (1 variable, p.ex. une densité transportée).
- `adc.FluidState(kind="compressible", gamma=1.4)` : Euler compressible (l'indice `gamma`).
- `adc.FluidState(kind="isothermal", cs2=0.5)` : Euler isotherme (la vitesse du son `cs2`).

**Transport** (`transport=`)
- `adc.ExB(B0=1.0)` : advection scalaire par la dérive E×B (champ magnétique `B0`) —
  `adc::ExBVelocity` dans `physics/hyperbolic.hpp`.
- `adc.CompressibleFlux()` : flux d'Euler compressible (`gamma` vient de l'état) —
  `adc::CompressibleFlux` (alias d'`adc::Euler`).
- `adc.IsothermalFlux()` : flux d'Euler isotherme (`cs2` vient de l'état) — `adc::IsothermalFlux`.

**Source** (`source=`)
- `adc.NoSource()` : pas de source — `adc::NoSource` dans `physics/source.hpp`.
- `adc.PotentialForce(charge=1.0)` : force du potentiel `(q/m) rho E` sur la quantité de mouvement
  (plus travail si 4 variables) — `adc::PotentialForce`.
- `adc.GravityForce()` : force gravitationnelle `rho g` — `adc::GravityForce`.

**Second membre elliptique** (`elliptic=`)
- `adc.ChargeDensity(charge=1.0)` : densité de charge `f = q n` — `adc::ChargeDensity` dans
  `physics/elliptic.hpp`.
- `adc.BackgroundDensity(alpha=1.0, n0=0.0)` : fond neutralisant `f = alpha (n - n0)` —
  `adc::BackgroundDensity`.
- `adc.GravityCoupling(sign=1.0, four_pi_G=1.0, rho0=1.0)` : couplage self-consistant
  `f = sign · 4πG (rho - rho0)` (`sign = +1` gravité, `-1` plasma) — `adc::GravityCoupling`.

`adc.Model(...)` valide la cohérence état ↔ transport (Scalar avec ExB ; FluidState compressible
avec CompressibleFlux ; isotherme avec IsothermalFlux) : un appariement incohérent lève une
`ValueError` immédiate.

Exemple — le modèle diocotron réduit (densité scalaire advectée par E×B, fond neutralisant), tel
qu'utilisé dans le tutoriel pour la comparaison uniforme/AMR :

```python
import adc

model = adc.Model(
    state=adc.Scalar(),
    transport=adc.ExB(B0=1.0),
    source=adc.NoSource(),
    elliptic=adc.BackgroundDensity(alpha=1.0, n0=0.0),
)

sim = adc.System(n=96, L=1.0, periodic=True)
sim.add_block("ne", model=model, spatial=adc.Spatial(minmod=True), time=adc.Explicit())
sim.set_poisson(rhs="charge_density", solver="geometric_mg")
sim.set_density("ne", ne0)          # ne0 : tableau 2D (densité initiale)
sim.step_cfl(0.4)
```

La même `ModelSpec` se branche aussi sur `adc.AmrSystem` (raffinement adaptatif) sans changer le
modèle : `sa.add_block("ne", model=model, ...)`.

## Modèle DSL (écrit en formules)

`adc.dsl.Model` permet d'ÉCRIRE un modèle en formules symboliques : Python compose un arbre
d'expressions (les opérateurs `+`, `-`, `*`, `/`, `**`, `adc.dsl.sqrt` construisent l'arbre, pas une
fonction appelée par cellule), que le DSL traduit en C++ compilable. On déclare les variables
conservatives, les primitives (par des formules), le flux, les valeurs propres, la source et la
contribution elliptique, puis on compile.

Voici le modèle diocotron réduit du tutoriel canonique
([docs/sphinx/tutorials/diocotron_tutorial.py](https://github.com/wolf75222/adc_cpp/blob/master/docs/sphinx/tutorials/diocotron_tutorial.py)), écrit en
formules — il reproduit exactement les briques natives `ExBVelocity` (transport) et
`BackgroundDensity` (elliptique) :

```python
import adc
from adc import dsl

B0 = 1.0      # champ magnétique de fond (porte la dérive E x B)
ALPHA = 1.0   # facteur du second membre elliptique alpha (n - n_i0)

def diocotron_model(n_i0):
    m = dsl.Model("diocotron_tutorial")

    (n,) = m.conservative_vars("n")     # unique variable conservative : la densité (rôle Density)
    m.aux("phi")                        # champs auxiliaires fournis par le solveur (canal adc::Aux)
    grad_x = m.aux("grad_x")
    grad_y = m.aux("grad_y")

    vx = (-grad_y) / B0                  # dérive E x B : v = (-d_y phi / B0, d_x phi / B0)
    vy = grad_x / B0
    m.flux(x=[n * vx], y=[n * vy])       # flux d'advection f = n v(dir)
    m.eigenvalues(x=[vx], y=[vy])        # spectre : une onde, la vitesse de dérive

    m.primitive_vars(n=n)                # scalaire transporté : primitif = conservatif
    m.conservative_from([n])
    m.elliptic_rhs(ALPHA * (n - n_i0))   # couple le bloc au Poisson : rhs = alpha (n - n_i0)

    m.check()                            # toute variable référencée doit être déclarée
    return m

compiled = diocotron_model(n_i0).compile(backend="production")   # -> CompiledModel

sim = adc.System(n=96, L=1.0, periodic=True)
sim.add_equation("ne", model=compiled,
                 spatial=adc.FiniteVolume(limiter="minmod", riemann="rusanov"),
                 time=adc.Explicit())
sim.set_poisson(rhs="charge_density", solver="geometric_mg")
sim.set_density("ne", ne0)
sim.step_cfl(0.4)
```

Détails et points de vigilance du DSL (paramètres nommés `m.param`, rôles physiques,
`require_metadata`, cache du `.so`) : voir la référence courte [DSL_API.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/DSL_API.md) et la
conception [DSL_MODEL_DESIGN.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/DSL_MODEL_DESIGN.md).

## Modèle hybride (briques native + DSL dans un seul modèle)

`adc.Model(...)` compose des briques 100 % natives ; `adc.dsl.Model(...)` génère un modèle 100 %
DSL. `adc.CompositeModel(transport, source, elliptic)` comble l'entre-deux : MÉLANGER, dans UN SEUL
modèle, des briques natives (`adc.ExB`, `adc.PotentialForce`, `adc.ChargeDensity`...) et des briques
DSL PARTIELLES compilées (`adc.dsl.HyperbolicBrick`, `adc.dsl.SourceBrick`, `adc.dsl.EllipticBrick`
suivies de `.compile()`).

Chaque slot accepte SOIT une brique native, SOIT une brique DSL partielle compilée. **Au moins un
slot doit être DSL** : une composition tout-native s'écrit avec `adc.Model(...)`, sinon
`CompositeModel` lève une `ValueError`. Le mélange est compilé en UN `.so` composite (prototype :
backend `aot`), sur le même chemin de production qu'un modèle DSL complet — la numérique native est
réutilisée à l'identique (un struct dérivé cuit les paramètres natifs `qom`, `q`, `cs2`... dans le
type ; aucune re-dérivation). Le slot transport fixe le layout (`n_vars`, noms conservatifs,
primitives, gamma) ; une brique DSL de source / elliptique doit déclarer le MÊME `n_vars`.

Exemple — transport DSL isotherme + source native + elliptique native (extrait de
`python/tests/test_dsl_hybrid.py`) :

```python
import adc
from adc import dsl

CS2, QOM, Q = 0.7, -1.0, -1.0

# Brique hyperbolique DSL répliquant adc::IsothermalFlux{cs2} (3 variables).
def build_iso_transport(cs2):
    b = dsl.HyperbolicBrick("iso")
    rho, rho_u, rho_v = b.conservative_vars("rho", "rho_u", "rho_v")
    u = b.primitive("u", rho_u / rho)
    v = b.primitive("v", rho_v / rho)
    c = dsl.sqrt(cs2)
    b.flux(x=[rho_u, rho_u * u + cs2 * rho, rho_v * u],
           y=[rho_v, rho_u * v, rho_v * v + cs2 * rho])
    b.eigenvalues(x=[u - c, u, u + c], y=[v - c, v, v + c])
    b.primitive_vars(rho, u, v)
    b.conservative_from([rho, rho * u, rho * v])
    return b

m = adc.CompositeModel(
    transport=build_iso_transport(CS2).compile(),  # transport DSL
    source=adc.PotentialForce(charge=QOM),         # source native
    elliptic=adc.ChargeDensity(charge=Q),          # elliptique native
)
compiled = m.compile(backend="aot")                # -> CompiledModel (adder add_compiled_block)

sim = adc.System(n=48, L=1.0, periodic=True)
sim.add_equation("gas", compiled,
                 spatial=adc.FiniteVolume(limiter="minmod", riemann="rusanov"),
                 names=["rho", "rho_u", "rho_v"])
```

Le mélange fonctionne dans les deux sens (transport natif + source/elliptique DSL aussi). Source :
[DSL_MODEL_DESIGN.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/DSL_MODEL_DESIGN.md) section « composition HYBRIDE » et
`python/tests/test_dsl_hybrid.py`.

## Variables conservatives / primitives

Un modèle DSL distingue deux jeux de variables, avec des rôles PHYSIQUES qui permettent au système
de retrouver une grandeur par son sens (et non par un indice littéral) — indispensable aux couplages
inter-espèces.

- `m.conservative_vars("rho", "mx", "my", roles=["Density", "MomentumX", "MomentumY"])` déclare les
  variables conservatives (l'état évolué `U`) et renvoie un tuple de `Var` à dépacker. Le `roles=`
  est optionnel ; sans lui, un mapping canonique nom → rôle s'applique (`rho`/`n` → `Density`,
  `rho_u` → `MomentumX`, `E` → `Energy`...). Un nom non reconnu reste `Custom`.
- `m.primitive(name, expr)` définit UNE primitive par sa formule (en fonction des conservatives ou
  des primitives précédentes), p.ex. `u = m.primitive("u", mx / rho)`.
- `m.primitive_vars(rho=rho, ux=mx/rho, ...)` (forme kwargs) DÉFINIT chaque primitive ET fixe le
  layout ordonné de `Prim` (l'ordre des kwargs). La forme positionnelle
  `m.primitive_vars(rho, u, v, p)` fixe juste le layout à partir de noms déjà définis.
- `m.conservative_from([rho, rho*u, rho*v])` donne l'inverse `Prim → U` (le DSL ne sait pas inverser
  symboliquement les primitives ; on fournit l'inverse explicitement). Il génère `to_conservative`.

L'opérateur spatial peut alors reconstruire en variables primitives (`rho`, `u`, `p`) plutôt que
conservatives — plus robuste pour Euler (positivité de `rho` et `p`) ; voir le choix
`variables="primitive"` de `adc.FiniteVolume` et les détails dans [ALGORITHMS.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md).

## Flux, sources, valeurs propres, RHS elliptique

Ces quatre déclarateurs sont le coeur du modèle DSL ; ils correspondent un à un aux fonctions du
concept `adc::PhysicalModel` lues par le coeur.

- `m.flux(x=[...], y=[...])` : le **flux physique** `F(U, aux, dir)`, une expression par composante
  conservative et par direction. L'opérateur spatial l'évalue aux interfaces puis le passe au
  solveur de Riemann (Rusanov / HLLC / Roe selon `riemann=`). NE PAS confondre avec
  `m.eval_flux(U, aux, dir)`, qui est l'ÉVALUATEUR numpy (debug / proto hôte), ni avec le flux
  NUMÉRIQUE `riemann=` de `adc.FiniteVolume`.
- `m.eigenvalues(x=[...], y=[...])` : les **valeurs propres** (vitesses caractéristiques) par
  direction. Le coeur en tire `max_wave_speed` (borne de Rusanov et pas de temps CFL) ; si une
  primitive `p` (pression) est déclarée, la brique générée expose aussi `pressure` / `wave_speeds`,
  ce qui la rend compatible avec les flux HLLC / Roe (qui exigent une pression).
- `m.source([...])` : le **terme source** `S(U, aux)`, une expression par composante (optionnel). Il
  lit l'état extérieur par le canal `adc::Aux` (p.ex. `grad_x` / `grad_y` pour une force de
  potentiel `-rho grad phi`).
- `m.elliptic_rhs(expr)` : la **contribution au second membre elliptique**, qui couple le bloc au
  Poisson de système (densité de charge `q n`, fond `alpha (n - n0)`, gravité...). Le Poisson de
  système somme les contributions de tous les blocs.

`m.check()` vérifie que toute variable référencée (dans les primitives, le flux, les valeurs
propres, la source, l'elliptique) est bien déclarée (conservative / primitive / aux), et lève une
`ValueError` factuelle sinon. Pour la signification physique et la discrétisation de chaque opérateur
(reconstruction, Riemann, multigrille), voir [ALGORITHMS.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md).

## Compilation : production / AOT / prototype

`m.compile(backend=..., target=...)` traduit le modèle symbolique en un `.so` et renvoie un
`CompiledModel` (qui porte `so_path`, `backend`, l'`adder` à employer, les noms/rôles/gamma/n_aux,
la `abi_key` et le `model_hash`). Le `.so` est mis en cache par `model_hash` : un modèle inchangé
n'est pas recompilé. Le **défaut est `backend="aot"`** — il faut donc demander explicitement
`"production"` pour le chemin natif zéro-copie.

Trois backends, matérialisés côté code dans `_BACKEND_CAPS` (`python/adc/dsl.py`) :

| backend | CPU | MPI | AMR | GPU | rôle |
|---|---|---|---|---|---|
| `production` | oui | oui (np=1/2/4) | via `AmrSystem` | rapporté `False` côté Python | **recommandé** en MPI/AMR ; loader natif zéro-copie (`add_native_block`) |
| `aot` | oui | non | non | non | **DÉFAUT** ; `.so` à marshaling, mono-rang, debug/bench CPU. Porte les params runtime (`set_block_params`) |
| `prototype` | oui (Rusanov o1) | non | non | non | JIT prototype, dispatch virtuel hôte ; ne pas utiliser en production |

`_BACKEND_CAPS["production"]` déclare `{cpu, mpi, amr} = True`. Le chemin natif `production` partage
le moteur de `add_block` (halos `fill_boundary`, donc MPI-capable par construction) et a un pendant
AMR (`m.compile(backend="production", target="amr_system")` → `AmrSystem.add_native_block`). `gpu`
est rapporté `False` PAR PRUDENCE : le chemin natif est device-clean en C++ (validé GH200), mais la
validation end-to-end depuis Python sur un module bâti Kokkos/CUDA reste une étape dédiée ; le module
hôte testé en CI n'est pas bâti GPU.

Ces capacités sont des drapeaux de DIAGNOSTIC, vérifiés au branchement (`add_equation`) ou à
l'exécution — PAS figés comme un argument `device=` de compilation (un `.so` peut compiler sans que
le module hôte soit device-capable). Les **garde-fous** lèvent une `ValueError` au plus tôt :

- backend inconnu (hors `prototype`/`aot`/`production`) ;
- `target="amr_system"` avec un backend autre que `production` (pas de chemin `.so` AMR hors natif) ;
- `compile(backend="prototype", require_metadata=True)` (le JIT ne transporte pas les métadonnées
  utiles) ;
- côté branchement : `riemann` HLLC/Roe sans pression `p` déclarée, `names=` sur le chemin natif
  `production` (les noms viennent des métadonnées du `.so`).

Pour brancher le `CompiledModel`, `System.add_equation` aiguille selon le type : une `ModelSpec`
(`adc.Model(...)`) → `add_block` (natif) ; un `CompiledModel` → l'adder du backend
(`add_dynamic_block` pour `prototype`, `add_compiled_block` pour `aot`, `add_native_block` pour
`production`). Détail complet : [DSL_API.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/DSL_API.md) et
[DSL_MODEL_DESIGN.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/DSL_MODEL_DESIGN.md). Couverture des backends sur GPU/MPI/AMR :
[BACKEND_COVERAGE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/BACKEND_COVERAGE.md).
