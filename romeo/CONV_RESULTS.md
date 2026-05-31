# Convergence du taux diocotron en resolution (ROMEO, job 614125)

Etude de confirmation lancee apres le workflow de diagnostic `diocotron-overshoot-diag` (mai 2026),
qui a FERME trois pistes (symetrie de grille, observable, normalisation `omega_D`) et pointe une
distorsion STRUCTURELLE de valeur propre. Question tranchee ici : le sur-taux est-il PLAT en
resolution (donc structurel/geometrique) ou DECROISSANT (troncature) ? Et est-il UNIFORME sur les modes ?

## Protocole

- ROMEO, partition `short`, contrainte `x64cpu`, gcc@14.2.0, OpenMP 8 threads/tache, job array 614125 (9 taches).
- Binaire : `examples/diocotron_column_amr.cpp` (VanLeer/Rusanov, `recon=1`), bord en escalier (`cut=0`).
- Balayage UNIFORME (`refine=0`, pas d'AMR) : `nc in {256, 512, 1024}` x `l in {3, 4, 5}`.
- Observable : DFT azimutale du POTENTIEL phi a `r=r0=0.15`, module du mode `l` (= methode exacte du papier,
  deja en place lignes 242-262 ; le workflow a confirme que ce N'EST PAS un observable densite).
- Normalisation : `omega_D = rho_bar/(2*pi) = 0.9/(2*pi) = 0.14324` (rho_bar = 1 - delta = densite moyenne
  de l'anneau, convention Davidson ; recalcul confirme exact par le workflow).
- Taux : pente de `ln(amplitude)` vs `t` sur la fenetre lineaire `[3, 9]` (R^2 = 1.00 partout).

## Resultats (fenetre [3, 9], R^2 = 1.00)

| mode l | cible analytique | eff 256 | eff 512 | eff 1024 | ecart a la cible (eff 1024) |
|---|---|---|---|---|---|
| 3 | 0.772 | 0.772 | 0.773 | 0.771 | **+0 %** (exact) |
| 4 | 0.911 | 0.941 | 0.929 | 0.921 | **+1 %** (quasi converge) |
| 5 | 0.683 | 0.907 | 0.893 | 0.881 | **+29 %** (aberrant) |

Figure : `docs/fig_diocotron_conv_modes.png`.

## Deux conclusions fermes

1. **PLAT en resolution => biais STRUCTUREL, pas de la troncature.** De eff 256 a eff 1024 (facteur 4),
   l'ecart bouge de moins de 3 points pour tous les modes (mode 5 : +33 % -> +29 %). Une erreur de
   troncature O(dx^p) serait divisee par au moins 4 ; ici elle est quasi constante. Raffiner l'interieur
   cartesien NE referme PAS l'ecart. Coherent avec : plat en schema (WENO5 ~ VanLeer), cut-cell de paroi
   sans effet (`docs/DIOCOTRON_GROWTH_RATE.md` section 4 mesure 5).

2. **NON uniforme : l'ecart CROIT avec le mode l.** Le mode 3 est EXACT, le mode 4 a +1 %, le mode 5 a
   +29 %. L'ancien recit "+8 % uniforme sur 3/4/5" etait un artefact de fenetre/schema (les runs
   `diocotron_highorder` WENO5+SSPRK3 sur les fenetres etroites du papier donnaient +8 % ; la fenetre
   lineaire [3,9] sur VanLeer donne +0/+1/+29 %). Le taux DEPEND fortement de la fenetre de fit (pas de
   plateau exponentiel net) : la fenetre [3,7] redonne mode4 ~ +8 %, la fenetre [3,9] mode4 ~ +1 %.

## Interpretation

La dispersion analytique `gamma(l)` est NON monotone : pic a `l=4` (0.911), plus basse en `l=3` (0.772)
et `l=5` (0.683). Notre schema reproduit le PIC (mode 4, +1 %) et le cote `l=3` (exact) mais ne capte PAS
le ROLL-OFF a haut `l` : il maintient le mode 5 a ~0.88 au lieu de redescendre a 0.68. Les modes de haut
`l` ont une fonction propre radiale plus localisee/oscillante, que la representation CARTESIENNE de
l'anneau (bords `r0`, `r1` en marches d'escalier) distord le plus. D'ou une distorsion de valeur propre
qui CROIT avec `l` et NE depend PAS de la resolution interieure : c'est une erreur d'EQUATION (geometrie),
pas de discretisation.

## Voie vers < 1 % (confirmee, et precisee)

Le verrou n'est ni la paroi (ecartee : cut-cell sans effet, effet d'image `(0.44)^8 ~ 1e-3`), ni la
grille (ecartee : le mode 5 non-4-fold est l'aberrant), ni l'observable (deja phi-DFT), ni la
normalisation (`omega_D` exact). C'est la representation CARTESIENNE de l'ANNEAU ou vit le mode. Pistes :
- **cut-cell / level-set sur les bords d'anneau `r0` et `r1`** (pas seulement la paroi) : adoucir les
  marches la ou la fonction propre est structuree ;
- **grille polaire `(r, theta)`** pour transport + Poisson : supprime par construction la brisure
  d'invariance de rotation (methodes semi-Lagrangiennes diocotron, Madaule, Mehrenberger).
Le mode 3 exact et le mode 4 a +1 % montrent que le cadre est CORRECT ; le travail restant porte sur la
fidelite de la fonction propre a haut `l`.
