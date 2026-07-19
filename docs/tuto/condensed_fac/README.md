# Source implicite condensée et FAC composite

Ce tutoriel résout un problème elliptique composite sur une hiérarchie AMR. Le problème physique
contient deux moments couplés par une rotation locale et un potentiel qui réagit à leur
déplacement.

La condensation élimine d’abord le bloc local de moments. Elle produit un problème scalaire
tensoriel sur le potentiel :

$$
-\nabla\!\cdot\!\left(A\nabla\phi\right)=b.
$$

`Hierarchy()` définit un seul problème mathématique sur tous les niveaux.
`CompositeTensorFAC()` choisit le provider natif. Celui-ci assemble les coefficients et le second
membre sur chaque niveau, effectue le solve FAC composite, puis publie les solutions avant de
reconstruire les moments.

## Script

- [`01_openmp_amr_composite_fac.py`](01_openmp_amr_composite_fac.py) : deux niveaux AMR synchrones,
  opérateur condensé explicite et solve FAC en C++/Kokkos OpenMP.

Le fichier commence par le domaine, puis définit les modèles, les plans numériques, le programme,
l’initialisation et l’AMR. Le cycle `validate -> resolve -> compile -> bind -> run` vient ensuite.
La `lambda` passée à `set_apply` définit symboliquement l’opérateur matrix-free. Le runtime ne
l’appelle pas sur chaque cellule.

```bash
python docs/tuto/condensed_fac/01_openmp_amr_composite_fac.py
```

## Contrat de `CompositeTensorFAC`

`CompositeTensorFAC` ne remplace pas n’importe quel Laplacien. Son contrat public demande les trois
briques `condensed_coeffs`, `condensed_rhs` et `condensed_reconstruct`, avec un opérateur scalaire
de portée `Hierarchy()`. Le CG du tutoriel de diffusion ne peut donc pas être remplacé directement
par FAC.

Le marqueur scalaire sert uniquement au tagging AMR. L’état physique condensé comporte trois
composantes, tandis que le provider de tagging direct travaille sur un état scalaire. Les deux
rôles sont donc déclarés séparément.

Le domaine est non périodique, ce qui rend le problème elliptique non singulier. Le script déclare
donc `nullspace=None`.
