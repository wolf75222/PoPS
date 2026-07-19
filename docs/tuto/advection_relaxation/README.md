# Advection et relaxation

Les quatre fichiers resolvent une advection avec relaxation. Les deux premiers traitent la source
implicitement. Les deux suivants composent explicitement le transport et la source, une fois avec
Lie et une fois avec Strang.

## Quel script lancer

| Methode | Script | Ordre des operations |
|---|---|---|
| IMEX local | [`01_openmp_imex_local.py`](01_openmp_imex_local.py) | transport explicite, relaxation implicite locale |
| IMEX global | [`02_openmp_imex_cg.py`](02_openmp_imex_cg.py) | transport explicite, diffusion-relaxation par CG |
| Lie explicite | [`03_openmp_lie_splitting.py`](03_openmp_lie_splitting.py) | $T(\Delta t)$ puis $S(\Delta t)$ |
| Strang explicite | [`04_openmp_strang_splitting.py`](04_openmp_strang_splitting.py) | $S(\Delta t/2)$, $T(\Delta t)$, $S(\Delta t/2)$ |

La relaxation locale ne couple pas les cellules. Elle se resout donc cellule par cellule avec
`DenseLU`. La diffusion fait intervenir les cellules voisines et justifie l'emploi d'un Krylov
global.

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

`pops.lib.time.IMEX` construit IMEX Euler. Le `DenseLU` natif compile et execute le solve local.
Python ne parcourt pas les cellules.

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

L'unique `lambda` Python du script decrit l'operateur matrix-free au moment de construire le graphe.
Le runtime C++/Kokkos execute ensuite les iterations, les produits scalaires, le Laplacien et les
reductions.

```bash
python docs/tuto/advection_relaxation/02_openmp_imex_cg.py
```

Ce solve travaille sur un niveau uniforme. Un solve composite AMR utilise une autorite de
hierarchie et un provider FAC. Le cas correspondant se trouve dans
[`condensed_fac/01_openmp_amr_composite_fac.py`](../condensed_fac/01_openmp_amr_composite_fac.py).

## Version 3 : splitting de Lie explicite

[`03_openmp_lie_splitting.py`](03_openmp_lie_splitting.py) avance d'abord l'advection sur un pas
complet, puis applique la relaxation sur ce nouvel etat :

```math
u^{n+1}=\Phi^S_{\Delta t}\!\left(\Phi^T_{\Delta t}(u^n)\right).
```

Le fichier ecrit les deux sous-pas directement dans `pops.Program`. Le transport et la relaxation
utilisent ici Euler explicite, ce qui donne une methode d'ordre un.

```bash
python docs/tuto/advection_relaxation/03_openmp_lie_splitting.py
```

## Version 4 : splitting de Strang explicite

[`04_openmp_strang_splitting.py`](04_openmp_strang_splitting.py) encadre le transport par deux
demi-pas de relaxation :

```math
u^{n+1}=\Phi^S_{\Delta t/2}\!\left(
\Phi^T_{\Delta t}\!\left(\Phi^S_{\Delta t/2}(u^n)\right)
\right).
```

Les deux sous-flots sont d'ordre deux. La source utilise RK2 sur chaque demi-pas et le transport
utilise SSPRK2 sur le pas complet. Cette precision compte : l'ordre symetrique de Strang ne suffit
pas si un sous-flot reste d'ordre un.

```bash
python docs/tuto/advection_relaxation/04_openmp_strang_splitting.py
```
