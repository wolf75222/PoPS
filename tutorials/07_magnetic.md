# 07, Champ magnétique : rotation cyclotron

Première brique vers le modèle deux-fluides magnétisé complet (Hoffart). Un champ
magnétique uniforme hors-plan `B = B_z ẑ` ajoute la force de Lorentz magnétique.

## La physique

La force magnétique `z (m x B)` sur la quantité de mouvement `m = n u` ne fait que **faire
tourner** `(m_x, m_y)` à la fréquence cyclotron `w_c = |q B / m|`, sans changer `|m|` ni la
densité `n`. La rotation exacte d'angle `theta = z w_c dt` est inconditionnellement stable
(aucune limite `w_c dt`, contrairement à un push explicite) :

$$\begin{pmatrix} m_x' \\ m_y' \end{pmatrix} =
  \begin{pmatrix}\cos\theta & \sin\theta\\ -\sin\theta & \cos\theta\end{pmatrix}
  \begin{pmatrix} m_x \\ m_y \end{pmatrix}$$

Elle est composée au pas électrostatique par **splitting de Strang** (ordre 2) :
`R(theta/2)` avant, le pas électrostatique, `R(theta/2)` après.

## Python

```python
cfg = adc.TwoFluidAPConfig()
cfg.n = 64
cfg.omega_ce = 4.0     # fréquence cyclotron électronique (0 = pas de champ)
cfg.omega_ci = 0.2     # ionique
ts = adc.TwoFluidAPSolver(cfg)
m0 = ts.mass_e()
ts.advance(0.01, 100)
print("masse conservée :", abs(ts.mass_e() - m0))
```

`omega_ce = omega_ci = 0` redonne exactement le comportement non magnétisé (la rotation est
l'identité).

## Validation

`test_two_fluid_cyclotron` isole l'opérateur magnétique : plasma **uniforme** (charge
nulle, donc `E = 0`, transport inerte), avec une quantité de mouvement électronique
uniforme. La seule dynamique est la rotation, donc `(m_x, m_y)` doit tourner à `w_c`. Mesure
par le premier passage à zéro de `m_x ~ m0 cos(w_c t)` :

- fréquence : **0.00%** d'écart théorie/mesure ;
- `|m|` conservée à `8.9e-16` (la rotation préserve la norme) ;
- densité inerte (`E = 0`), masse conservée, à l'arrondi machine.

## Ce qui reste

La rotation pure est la brique validée. Le modèle Hoffart complet demande ensuite :

1. un **push de Boris** combiné E+B (demi-coup E, rotation B, demi-coup E) au lieu du
   splitting de Strang externe, plus précis en régime fortement magnétisé ;
2. le couplage inhomogène (dérive `E x B` et diamagnétique dans le transport) ;
3. la reformulation AP tensorielle sous champ fort.

Voir [ALGORITHMS.md §12](../docs/ALGORITHMS.md).

## Pièges

- En régime fortement magnétisé ET fort champ, le splitting de Strang de E et B perd en
  précision face au push de Boris (E+B au même centrage temporel).
