# Journal des runs ROMEO

Calculs lancés sur ROMEO (URCA) pour mesurer le taux de croissance diocotron
(arXiv:2510.11808). Compte `r250127`, user `rmdraux`. Sorties brutes sur
`/scratch_p/$USER/<jobid>/out` (non sauvegardé, à recopier). Synthèse :
[docs/ROMEO.md](../docs/ROMEO.md).

## Notes de build

- Partition `x64cpu` (AMD EPYC 9654) pour le CPU, `armgpu` (GH200) pour le GPU.
- Dépôt privé : `git clone` échoue (pas d'auth). Source synchronisé par
  `rsync -az --exclude .git --exclude 'build*' ./ romeo:~/adc_cpp/`.
- Le cmake de Spack est cassé sur les nœuds de calcul (`libmd.so.0` absente).
  Les exemples étant header-only, on compile en direct :
  `g++ -std=c++23 -O3 -fopenmp -I include examples/X.cpp -o bin`.

## Job 613961 : convergence WENO5-Z + SSPRK3 (x64cpu, OMP=96)

Modes l = 3/4/5 × résolution effective 256/512/1024. WENO5-Z, SSPRK3, Poisson re-résolu
à chaque étage RK, CFL 0.4. Tous stables. Taux `gamma_norm` (fenêtre du papier, fit
exponentiel R² = 1.00, normalisation `omega_D = rho_bar/(2 pi)`) :

| mode l | analytique | eff 256 | eff 512 | eff 1024 |
|---|---|---|---|---|
| 3 | 0.772 | 0.838 | 0.850 | 0.853 |
| 4 | 0.911 | 0.985 | 0.988 | 0.987 |
| 5 | 0.683 | 0.730 | 0.731 | 0.729 |

Le sur-tir (~+8 %) est plat en résolution : eff 512 et 1024 donnent le même taux.

## Job 613945 : reconstruction NoSlope / VanLeer / AMR (x64cpu, OMP=96)

eff 512 et 1024 (eff 2048 a dépassé le mur horaire `short`). Masse conservée `~1e-13`.
`lin` = fenêtre linéaire fixe `--window 5,14` ; `sat` = pic historique.

| cas | eff | cellules | gamma_norm (lin / sat) |
|---|---|---|---|
| uniforme NoSlope | 512  | 262 144   | 0.650 / 0.583 |
| uniforme VanLeer | 512  | 262 144   | 0.753 / 0.575 |
| AMR ml VanLeer   | 512  | 104 632   | 0.762 / 0.574 |
| uniforme NoSlope | 1024 | 1 048 576 | 0.706 / 0.578 |
| uniforme VanLeer | 1024 | 1 048 576 | 0.748 / 0.582 |
| AMR ml VanLeer   | 1024 | 409 008   | 0.747 / 0.579 |

L'AMR multi-niveau suit l'uniforme à résolution effective égale (~40 % des cellules).

## Job 614089 : sanitizer GPU (armgpu, 1 GH200)

`romeo/sanitizer.sbatch`. compute-sanitizer sur les exemples GPU :
`coupled_kokkos` (memcheck/initcheck/synccheck) et `diocotron_amr_kokkos` (memcheck) =
0 erreur. Checksum `diocotron_amr_kokkos = 4394594.404318` exactement égal au CPU
(bit-identique), dérive de masse `2.2e-16`. Sortie : `romeo/runs/adc_sanitizer.614089.out`.

## Reproduction

```
sbatch romeo/diocotron_highorder_hero.sbatch   # WENO5 + SSPRK3, modes 3/4/5
sbatch romeo/diocotron_recon_hero.sbatch        # NoSlope / VanLeer / AMR
sbatch romeo/sanitizer.sbatch                   # sanitizer GPU
```
Extraction du taux (recopier les CSV depuis le scratch) :
```
python3 scripts/validate_diocotron_growth.py out/<cas>/ring_amp.csv --rhobar 0.9 --target 0.911 --window 4.2,5.2
```
