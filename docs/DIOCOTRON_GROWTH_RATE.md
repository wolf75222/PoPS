# Reproduction du taux de croissance diocotron (vs Hoffart arXiv:2510.11808)

Document consolide : reproduction quantitative du taux de croissance de l'instabilite diocotron
avec adc_cpp, comparaison au papier de reference (Hoffart, Maier, Shadid, Tomas, *Structure-preserving
finite-element approximations of the magnetic Euler-Poisson equations*, arXiv:2510.11808, Section 5.3).
Resultats ROMEO bruts : [romeo/HERO_RESULTS.md](../romeo/HERO_RESULTS.md).

Cible analytique (Petri / Davidson-Felice, geometrie de l'anneau `r0:r1:Rwall = 6:8:16`, reproduite
par `analysis/diocotron_growth.hpp`) : `gamma_3 = 0.772`, `gamma_4 = 0.911`, `gamma_5 = 0.683`.

## 1. Verrou de STABILITE leve (prerequis a tout balayage)

Au-dela d'une resolution effective ~448, la simu partait en `nan` des les premiers pas. Diagnostic :
le **multigrille geometrique DIVERGEAIT** au bord conducteur embedded sur grille fine (coarsening
non-Galerkin + masque du cercle re-evalue par niveau -> correction grossiere incoherente, rayon
spectral du V-cycle > 1, erratique selon l'alignement du cercle). Le warm start propageait la
divergence -> `phi` puis le champ en `nan`. Ce n'etait NI le pas de temps (deja plafonne), NI le
plancher de densite (la densite reste bornee ; seul `phi` explose, au RAYON DE LA PAROI r=0.398).

Correctif : `GeometricMG::solve_robust` (`include/adc/elliptic/geometric_mg.hpp`). Phase 1 = le
V-cycle standard (BIT-IDENTIQUE quand il converge ou stagne) ; SEULEMENT en cas de vraie divergence
(residu final > residu initial) : durcissement STICKY du lissage GS + restart a froid jusqu'a
redevenir contractant. Resultat : stable jusqu'a eff 1024 (uniforme ET AMR `ml`), masse `~1e-14`,
les 8 runs enregistres (eff <= 448) restent BIT A BIT identiques. Details : `docs/HERO_RUN_AMR.md`.

## 2. Methodes (montee en ordre vers le taux analytique)

Le plafond de M1 (`gamma_norm ~ 0.58`) venait de la DIFFUSION du schema (ordre 1 en espace ET en
temps), pas de la physique. Deux leviers classiques, confirmes par la litterature (Jiang-Shu,
Borges WENO-Z, Gottlieb-Shu-Tadmor SSPRK, Ern-Guermond RK ordre 3) :

- **Reconstruction d'ordre eleve** : `NoSlope` (ordre 1) -> `VanLeer`/`Minmod` (MUSCL ordre 2) ->
  **`Weno5`** (WENO5-Z, ordre 5, `operator/reconstruction.hpp`, ordre 5.00 verifie par
  `test_weno_convergence`). Option `recon` de `examples/diocotron_column_amr.cpp` ; `recon=0`
  bit-identique a l'historique.
- **Integration en temps d'ordre eleve** : forward Euler BIAISE positivement un mode en croissance
  exponentielle (instable sur l'axe imaginaire, terme `+ 1/2 omega_r^2 dt`). **SSPRK3** (Shu-Osher)
  enleve ce biais a l'ordre 3. `examples/diocotron_highorder.cpp` : WENO5-Z + SSPRK3, Poisson
  RE-RESOLU a chaque etage RK (couplage stade par stade, `solve_robust`).

## 3. Resultats

### 3a. Convergence colonne : l'AMR suit l'uniforme (ROMEO 613945)

A resolution effective egale, l'AMR `ml` (Poisson multi-niveau) COINCIDE avec l'uniforme pour ~40 %
des cellules (la promesse M2b, a l'echelle) ; VanLeer depasse largement NoSlope :

| cas | eff 512 (lin) | eff 1024 (lin) | cellules vs unif |
|---|---|---|---|
| uniforme NoSlope | 0.650 | 0.706 | 100 % |
| uniforme VanLeer | 0.753 | 0.748 | 100 % |
| AMR `ml` VanLeer | 0.762 | 0.747 | ~40 % |

### 3b. Taux haute precision, modes 3/4/5 (ROMEO 613961, WENO5+SSPRK3)

Fenetre du papier, R^2 = 1.00. L'ordre eleve fait passer le mode 4 de 0.56 (NoSlope+Euler,
sous-evalue, trop diffusif) a ~0.99, du BON cote de 0.911 :

| mode l | analytique | eff 256 | eff 512 | eff 1024 |
|---|---|---|---|---|
| 3 | 0.772 | +8 % | +10 % | +11 % |
| 4 | 0.911 | +8 % | +8 % | +8 % |
| 5 | 0.683 | +7 % | +7 % | +7 % |

## 4. Diagnostic : un sur-tir ~+8 % UNIFORME et PLAT en resolution

Quatre mesures independantes ecartent toutes les causes "faciles" et pointent la GEOMETRIE :

1. **Plat en resolution** : eff 256 ~ 512 ~ 1024 (meme +8 %). Plus de cellules ne referme PAS l'ecart.
2. **Plat en ordre de reconstruction** : `WENO5 ~ VanLeer`. Ce n'est pas l'ordre spatial.
3. **Balayage en delta** : la LIMITE LINEAIRE (delta -> 0) MONTE a +27 % au lieu de baisser. L'accord
   apparent a delta=0.1 etait une compensation fortuite par la saturation. Ce n'est donc PAS une
   contamination nonlineaire ni un effet de fenetre.
4. **Rapport SANS DIMENSION** `gamma / |Re(omega)|` (independant de la normalisation, via la valeur
   propre COMPLEXE `diocotron_eigenvalue` : analytique Re_norm = -2.08 / -2.75 / -3.44 pour l=3/4/5) :
   mesure 0.31 vs analytique 0.331 -> ~5 % de DISTORSION STRUCTURELLE de la valeur propre + ~3 % de
   decalage de normalisation `omega_D`.

Cause : la **geometrie cartesienne** (paroi conductrice en escalier + symetrie 4 du carre brisant
l'invariance de rotation du probleme circulaire), un biais O(1) qui ne converge PAS sur grille
cartesienne. Le transport lui-meme est fidele (invariants verts, section 6).

## 5. Comparaison directe au papier (lecture de l'arXiv)

Methode et physique **identiques** (verifie dans le texte du papier, Section 5.3) :
- vitesse initiale `v0 = -(grad phi0 x Omega)/|Omega|^2` (derive E x B) = notre modele `Diocotron` ;
- mesure : *"DFT du potentiel phi a rayon FIXE r=r0, module du coefficient du mode l"* = notre
  `mode_amplitude`, normalise a l'initial, ajustement exponentiel sur une fenetre etroite ;
- memes cibles analytiques 0.772 / 0.911 / 0.683, memes fenetres de fit ;
- temps : RK explicite ordre 3 (le notre : SSPRK3) ; espace : ordre 2 graph-viscosity dG (le notre :
  WENO5, ordre 5, DONC notre schema n'est PAS la limite).

La DIFFERENCE decisive est la GEOMETRIE, prouvee par la table de convergence du papier (Fig 5.4d) :

| mode 4 | papier (dofs) | gamma_h | ecart | | nous (eff) | gamma_h |
|---|---|---|---|---|---|---|
| | 196 608 | 0.935 | +2.6 % | | 256 | 0.985 (+8 %) |
| | 786 432 | 0.919 | +0.9 % | | 512 | 0.988 (+8 %) |
| | 3 145 728 | **0.913** | **+0.2 %** | | 1024 | 0.987 (+8 %) |

Le papier **CONVERGE** (0.935 -> 0.919 -> 0.913) parce que son domaine est un **DISQUE** dont la
paroi conductrice EST le bord du maillage (epousant, erreur O(h^2) qui s'annule au raffinement).
Notre domaine est une **boite carree avec un cercle conducteur embedded en escalier** : erreur O(1)
qui NE converge PAS (exactement nos donnees plates a +8 %). Meme a leur maillage le plus GROSSIER
(~eff 222) le papier est a +2.6 %, deja meilleur que nos +8 %, et il raffine jusqu'a +0.2 %.

## 6. Indicateurs physiques verifies (fidelite du transport)

`analysis/diocotron_invariants.hpp` + `test_diocotron_invariants` (mode 4, WENO5+SSPRK3, eff 256) :

| invariant | resultat | role |
|---|---|---|
| masse `int rho` | exacte (derive 0) | conservativite (forme flux) |
| energie `1/2 int \|grad phi\|^2` | < 1 % | invariant du systeme ideal |
| moment angulaire `int rho r^2` | < 1 % | invariant diocotron (Davidson) |
| enstrophie `int rho^2` | -5.5 % | Casimir : MESURE la diffusion numerique |
| principe du maximum | `rho in [floor, rho_max]` | pas de valeurs parasites |
| `Re(omega)` (rotation) | reproduit (~+8 %) | 2e moitie de la dispersion |

Figures : `docs/fig_diocotron_highorder.png` (taux vs ordre), `docs/fig_diocotron_invariants.png`
(invariants vs temps).

## 7. Conclusion et prochaine etape

Les leviers de RECONSTRUCTION et d'INTEGRATION d'ordre eleve sont en place et verifies ; ils amenent
le taux mode 4 a ~+8 % de l'analytique (depuis -39 % en ordre 1). Le verrou restant vers < 1 % (et
< 0.5 % comme le papier) est **purement geometrique** : il faut un bord conducteur EPOUSANT, pas plus
de resolution. Deux voies classiques :
- **grille polaire / disque** (comme les methodes semi-Lagrangiennes diocotron, Madaule, Mehrenberger)
  ou la paroi est une surface de coordonnees ;
- **cellules coupees (cut-cell) Shortley-Weller** : Dirichlet `phi=0` impose au vrai cercle dans les
  cellules du bord, au lieu de l'escalier (ordre 2 au bord embedded dans `GeometricMG`).

## 8. Reproduction

```
# stabilite + AMR (recon : 0 NoSlope, 1 VanLeer, 3 WENO5)
g++ -std=c++23 -O2 -I include examples/diocotron_column_amr.cpp -o dca
./dca out 640 3000 0 4 0 1        # uniforme VanLeer eff 640 (stable)

# haute precision WENO5 + SSPRK3, mode l, comparaison analytique
g++ -std=c++23 -O3 -fopenmp -I include examples/diocotron_highorder.cpp -o dho
./dho out 512 800 4 3 0.4         # eff 512, mode 4, WENO5
python3 scripts/validate_diocotron_growth.py out/ring_amp.csv --target 0.911 --window 4.2,5.2

# ROMEO : sbatch romeo/diocotron_highorder_hero.sbatch  (cf. romeo/HERO_RESULTS.md)
```
