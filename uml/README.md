# Diagrammes UML de adc_cpp

Diagrammes d'architecture pour la présentation. Source PlantUML (`.puml`), rendus en
`.png` et `.svg`. Le code n'est pas le sujet : ces diagrammes montrent **l'architecture**
(quoi, comment, niveau d'abstraction, interactions).

| # | Diagramme | Type UML | Ce qu'il répond |
|---|---|---|---|
| 01 | `01_layers` | package / composant | **Vue d'ensemble** : les 5 couches orthogonales + les seams. Quel est le périmètre, comment c'est organisé, ce qu'on écrit vraiment (couche 1). |
| 02 | `02_classes` | classes | **Le cœur** : les 2 concepts (`PhysicalModel`, `EllipticSolver`), les modèles/backends qui les réalisent, les coupleurs paramétrés par template, la composition des données. Montre le niveau d'abstraction (concepts = interfaces statiques, zéro héritage virtuel). |
| 03 | `03_sequence_step` | séquence | **Interaction dynamique** : un pas couplé hyperbolique-elliptique. Qui appelle qui (Coupler → MG → modèle → opérateur → seam). |
| 04 | `04_execution_seam` | composant | **Parallélisation** : un seul `for_each_cell` se résout en série / OpenMP / Kokkos à la compilation. MPI est un axe séparé. |
| 05 | `05_amr_activity` | activité | **Méthode AMR** : le pas adaptatif (sous-cyclage Berger-Oliger r=2 + reflux conservatif). |

## Lecture conseillée pour la présentation

1. **01** pour poser le décor (périmètre + organisation).
2. **02** pour l'abstraction (le point fort à montrer au tuteur : concepts C++20, deux axes de variation indépendants).
3. **03** + **04** pour le « comment ça interagit » et la parallélisation.
4. **05** si on creuse l'AMR.

## Recompiler

```bash
plantuml -tpng *.puml    # PNG
plantuml -tsvg *.puml    # SVG (vectoriel, pour les slides)
```

Les diagrammes reflètent l'architecture réelle (relations extraites de `include/adc/`,
cf. `docs/ARCHITECTURE.md`).
