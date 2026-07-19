# Source implicite condensée et FAC composite

Ce parcours avancé montre un véritable solve elliptique composite sur une hiérarchie AMR. Il ne
transpose pas artificiellement l’advection scalaire vers FAC : le problème physique contient deux
moments couplés par une rotation locale et un potentiel qui réagit à leur déplacement.

La condensation élimine d’abord le bloc local de moments. Elle produit un problème scalaire
tensoriel sur le potentiel :

$$
-\nabla\!\cdot\!\left(A\nabla\phi\right)=b.
$$

`Hierarchy()` indique qu’il existe un seul problème mathématique sur tous les niveaux.
`CompositeTensorFAC()` choisit le provider natif qui assemble les coefficients et le second membre
sur chaque niveau, effectue un solve FAC composite, puis publie les solutions avant la
reconstruction des moments.

## Script

- [`01_openmp_amr_composite_fac.py`](01_openmp_amr_composite_fac.py) : deux niveaux AMR synchrones,
  opérateur condensé explicite et solve FAC en C++/Kokkos OpenMP.

Le fichier reste volontairement top-level : domaine, modèles, plans numériques, programme,
initialisation, AMR et cycle `validate -> resolve -> compile -> bind -> run` apparaissent dans leur
ordre d’exécution. La petite `lambda` passée à `set_apply` est la définition symbolique de
l’opérateur matrix-free ; ce n’est ni une boucle numérique Python ni un callback exécuté par
cellule.

```bash
python docs/tuto/condensed_fac/01_openmp_amr_composite_fac.py
```

## Pourquoi ce parcours est séparé

`CompositeTensorFAC` n’est pas un alias universel pour n’importe quel Laplacien. Son contrat public
actuel demande les trois briques cohérentes `condensed_coeffs`, `condensed_rhs` et
`condensed_reconstruct`, avec un opérateur scalaire de portée `Hierarchy()`. Un simple remplacement
du CG du tutoriel diffusion par FAC serait donc mathématiquement et techniquement faux.

Le marqueur scalaire sert uniquement au tagging AMR. L’état physique condensé comporte trois
composantes et le provider de tagging direct travaille sur un état scalaire ; garder les deux rôles
séparés rend cette limitation visible au lieu de choisir silencieusement une composante.

Le domaine est non périodique afin que le problème elliptique soit non singulier. Le script déclare
donc honnêtement `nullspace=None` ; la gauge composite périodique n’est pas prétendue disponible.
