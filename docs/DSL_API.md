# DSL_API -- Reference courte du DSL Python (adc.dsl)

Document de reference UTILISATEUR. Pour la conception, le raisonnement et l'historique,
voir [docs/DSL_MODEL_DESIGN.md](DSL_MODEL_DESIGN.md).

---

## 1. Ecrire un modele symbolique

```python
import adc
from adc import dsl

m = dsl.Model("mon_modele")

# Variables conservatives (noms + roles physiques optionnels)
m.conservative_vars("rho", "mx", "my",
                    roles=["Density", "MomentumX", "MomentumY"])

# Variables primitives (kwargs : nom=expression symbolique)
rho, mx, my = m.u[0], m.u[1], m.u[2]
m.primitive_vars(rho=rho, ux=mx/rho, uy=my/rho)

# Flux physique (declarateur symbolique ; m.eval_flux(...) = evaluateur numpy)
m.flux(x=[mx, mx*mx/rho, mx*my/rho],
       y=[my, mx*my/rho, my*my/rho])

# Source (optionnel -- force du potentiel)
phi_x, phi_y = m.a.grad_x, m.a.grad_y
m.source([-rho*phi_x, -rho*phi_y])  # exemple ExB / force

# Second membre elliptique (optionnel -- couplage Poisson)
m.elliptic_rhs(rho)

# Parametre nomme (constante inlinee a la compilation)
g = m.param("gamma", 1.4)
```

---

## 2. Compiler

```python
# Backend recommande (zero-copie, GPU/MPI valides)
compiled = m.compile(backend="production", target="system")
# Pour AMR :
compiled_amr = m.compile(backend="production", target="amr_system")
```

Backends disponibles :

| backend | CPU | MPI | AMR | GPU | Remarque |
|---|---|---|---|---|---|
| `production` | oui | oui (np=1/2/4) | via `AmrSystem` | oui (GH200) | **recommande** ; natif zero-copie |
| `aot` | oui | non | non | non | `.so` a marshaling ; debug/bench CPU |
| `prototype` | oui (Rusanov o1) | non | non | non | JIT proto ; ne pas utiliser en production |

Le `.so` est mis en cache par `model_hash` : un modele inchange n'est pas recompile.

---

## 3. Brancher sur System / AmrSystem

```python
sim = adc.System(n=256, periodic=True)
sim.add_equation("fluide",
                 model=compiled,
                 spatial=adc.FiniteVolume(limiter="vanleer", riemann="rusanov"),
                 time=adc.Explicit(substeps=1))
sim.set_poisson("geometric_mg")
sim.run(t_end=10.0, cfl=0.4)
```

```python
# AMR
amr = adc.AmrSystem(n=128, max_level=2, periodic=True)
amr.add_equation("fluide",
                 model=compiled_amr,
                 spatial=adc.FiniteVolume(limiter="vanleer", riemann="rusanov"),
                 time=adc.Explicit(substeps=1))
```

Points importants :
- `riemann=` nomme le flux NUMERIQUE (Rusanov/HLL/HLLC/Roe) ; `m.flux(...)` est le flux PHYSIQUE.
- `fft` n'est pas supporte sous `System` en MPI `np>1` : employer `geometric_mg`.
- `backend="production"` avec `target="amr_system"` : `AmrSystem` est mono-bloc, explicite ;
  HLLC/Roe/`primitive` sont rejetes cote facade Python AMR (le moteur C++ les supporte, mais le
  binding Python ne les expose pas encore sur ce chemin).

---

## 4. Cache et reproductibilite

`m.compile()` retourne un objet `CompiledModel` qui porte :
- `so_path` : chemin du `.so` compile.
- `model_hash` : hash stable (formules + roles + params) -- cle de cache.
- `abi_key` : cle compilateur/std/en-tetes -- refus explicite si incompatible au chargement.
- `params` : dict des parametres nommes declares via `m.param(...)`.

---

## 5. Points de vigilance

- `m.param(name, value)` : constante INLINEE a la compilation (mode `const`). Changer la valeur
  exige un nouvel appel a `m.compile()`. Le mode `runtime` (sans recompilation) n'est pas encore
  disponible (`NotImplementedError`).
- `adc.PythonFlux` : outil de TEST numpy hote, hors hot path GPU/MPI. Ne jamais utiliser en
  production.
- Roles physiques (`Density`, `MomentumX`, `MomentumY`, ...) : requis pour les couplages
  inter-especes et pour que le `System` retrouve les grandeurs par role. A fournir a
  `conservative_vars(roles=...)` ou a `m.compile(require_metadata=True)`.

---

## 6. Demonstrateurs de reference (adc_cases, ci=true)

| Cas | Fichier |
|---|---|
| ExB mono-espece DSL | `diocotron_dsl/run.py` |
| Deux especes DSL | `two_species_dsl/run.py` |
| Isotherme magnetique DSL | `magnetic_isothermal_dsl/run.py` |
