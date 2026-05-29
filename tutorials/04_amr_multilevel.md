# 04, AMR multi-niveaux (Berger-Oliger + reflux)

Raffiner seulement là où c'est utile. Un niveau fin recouvre une sous-région à `dx/2` et
fait `r=2` sous-pas de `dt/2` pendant que le grossier fait 1 pas de `dt`. Le **reflux**
répare la conservation à l'interface fin-grossier.

![diocotron AMR 3 niveaux](../docs/anim_diocotron_amr3.gif)

## Le mécanisme

1. **Sous-cyclage** (Berger-Oliger 1984) : le fin avance 2 fois plus souvent pour
   respecter sa propre CFL.
2. **Ghosts fins** : interpolés en espace ET en temps depuis le grossier
   (`mf_fill_fine_ghosts_t`) entre deux pas grossiers.
3. **Moyenne descendante** : les cellules grossières couvertes prennent la moyenne des 4
   cellules fines (`mf_average_down`).
4. **Reflux** (Berger-Colella 1989) : à l'interface, la maille grossière est corrigée par
   (flux fin intégré sur les 2 sous-pas) moins (flux grossier). Sans ça, la conservation
   est cassée à l'interface.

Détail mathématique : [ALGORITHMS.md §8](../docs/ALGORITHMS.md).

## En C++

```bash
./build/bin/diocotron_amr  out 128 500     # 2 niveaux
./build/bin/diocotron_amr3 out 128 500     # 3 niveaux emboîtés
python3 scripts/make_diocotron_amr3_gif.py out docs/anim_diocotron_amr3.gif
```

Le coupleur de production `AmrCoupler<Model, Elliptic>`
(`include/adc/coupling/amr_coupler.hpp`) enchaîne, par pas :

```cpp
sim.sync_down();     // fin -> grossier sur toute la hiérarchie
sim.compute_aux();   // Poisson grossier -> aux = grad phi -> injection vers les fins
sim.step(dt);        // amr_step_multilevel_mf : sous-cyclage + reflux, N niveaux
```

L'intégrateur `amr_step_multilevel_mf<Limiter, NumericalFlux>` est générique : il marche en
MUSCL / HLL / HLLC / N composantes, et tout passe par `for_each_cell` (GPU-ready).

## Validation

`test_amr_multilevel_mf` prouve l'équivalence **bit-identique** (`0` exact, 40 pas, 3
niveaux) à la pile Fab2D de référence (`amr_multilevel.hpp`), et la conservation de la
masse à `~1e-12`. `test_amr_coupler` fige la conservation du coupleur de production
(`5.55e-16`).

## Régime de fonctionnement

| Cas pratique | Recommandation |
|---|---|
| Une zone d'intérêt, télescopage | N niveaux mono-box (`AmrCoupler`) |
| Plusieurs zones disjointes mobiles | multi-patch ([05](05_amr_multipatch.md)) |

## Pièges

- Le flux grossier à l'interface doit être échantillonné **avant** d'avancer le grossier
  (centrage temporel correct du registre).
- La moyenne descendante doit précéder la mesure de masse initiale, sinon un blob raide
  fait sauter la masse au premier `average_down` (faux positif de non-conservation).
- Le nesting doit rester propre après un regrid (chaque niveau fin strictement intérieur à
  son parent).
