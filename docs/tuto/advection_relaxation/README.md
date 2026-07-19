# Advection, relaxation et implicite

Ce parcours ajoute les termes implicites sans cacher leur nature numerique. Les deux scripts sont
autonomes, top-level et se lisent dans l'ordre physique du probleme.

## Choisir la bonne version

| Probleme | Script | Solve implicite |
|---|---|---|
| Advection-relaxation locale | [`01_openmp_imex_local.py`](01_openmp_imex_local.py) | `DenseLU` natif par cellule |
| Advection-diffusion-relaxation | [`02_openmp_imex_cg.py`](02_openmp_imex_cg.py) | CG matrix-free global |

Les deux fichiers sont separes volontairement. La relaxation locale ne couple pas les cellules :
lui imposer un Krylov global ajouterait un cout et donnerait une image fausse de la physique. La
diffusion, elle, couple les cellules voisines et justifie une resolution lineaire globale.

## Version 1 : IMEX local

Le premier probleme est

```math
\frac{\partial u}{\partial t}+\nabla\cdot(a u)=-\lambda u.
```

Le transport est explicite et la relaxation est implicite :

```python
implicit_relaxation = model.operator(
    "implicit_relaxation",
    returns=model.local_linear_operator(
        "relaxation_matrix",
        on=U,
        matrix=((-RELAXATION_RATE,),),
    ),
)

program = IMEX(
    tracer_U,
    explicit_operator=advection_rate,
    implicit_operator=implicit_relaxation,
)
```

`pops.lib.time.IMEX` construit IMEX Euler. Son solve local est compile et execute par le `DenseLU`
natif, sans boucle Python sur les cellules.

```bash
python docs/tuto/advection_relaxation/01_openmp_imex_local.py
```

## Version 2 : IMEX global et Krylov

Le second probleme ajoute la diffusion :

```math
\frac{\partial u}{\partial t}+\nabla\cdot(a u)
=\kappa\Delta u-\lambda u.
```

Apres la prediction explicite, le programme resout

```math
\left[(1+\lambda\Delta t)I-\kappa\Delta t\,\Delta\right]u^{n+1}=u^*.
```

L'operateur est symetrique defini positif sur le domaine periodique. Le certificat SPD et le
solveur sont donc explicites :

```python
next_state = program.solve(
    LinearProblem(
        implicit_operator,
        explicit_predictor,
        properties=LinearOperatorProperties.symmetric_positive_definite(),
        nullspace=None,
    ),
    solver=CG(max_iter=80, rel_tol=1.0e-10),
).consume(action=FailRun())
```

L'unique `lambda` Python du script decrit le corps symbolique de l'operateur matrix-free pendant
l'authoring. Les iterations, produits scalaires, applications du Laplacien et reductions sont
ensuite executes dans le runtime C++/Kokkos.

```bash
python docs/tuto/advection_relaxation/02_openmp_imex_cg.py
```

Cette version est un solve de niveau uniforme. Un solve composite AMR ne doit pas reutiliser
silencieusement ce contrat : il requiert une autorite de hierarchie et un provider FAC dedie. Le
parcours avance correspondant est
[`condensed_fac/01_openmp_amr_composite_fac.py`](../condensed_fac/01_openmp_amr_composite_fac.py).
