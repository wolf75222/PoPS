# Verdict experience geometrie (juin 2026) -- le cut-cell ne recupere PAS le taux

Experience discriminante sur ROMEO GH200 (job 647507), modele COMPLET system-schur,
n=256, t_end=2.0, fenetres papier, l=3,4,5, trois geometries de transport :
square (boite carree, defaut), staircase (masque disque 0/1 a R=16), cutcell
(embedded-boundary aperture+kappa a R=16).

## Resultat brut (taux gamma_numeric, brut, fenetres papier)

| mode | square | staircase | cutcell | papier | erreur |
|------|--------|-----------|---------|--------|--------|
| 3 | 0.037182 | 0.037182 | 0.037182 | 0.772 | -95.2 % |
| 4 | 0.048897 | 0.048897 | 0.048897 | 0.911 | -94.6 % |
| 5 | 0.121080 | 0.121080 | 0.121080 | 0.683 | -82.3 % |

Les trois geometries donnent le MEME taux (differences ~1e-11 = arrondi machine ;
le masque/EB est bien ACTIF mais physiquement sans effet). CONCLUSION DIRECTE :
le masque/cut-cell de domaine au bord externe R=16 NE CHANGE PAS le taux diocotron.

## Pourquoi le cut-cell-a-R etait le mauvais outil

L'instabilite diocotron vit sur l'anneau r0=6 / r1=8, PROFONDEMENT a l'interieur du
domaine R=16. Le masque de disque agit au bord externe R : il ne touche que les
coins (rayon > 16), qui portent rho_min et sont dynamiquement inertes. Le mur de
Poisson (Dirichlet sur le cercle R=16) impose deja le disque pour phi. Donc
confiner le transport a ||x|| < 16 ne change rien a la dynamique de l'anneau.

## Le point qui REORIENTE le diagnostic : deficit RESOLUTION-INDEPENDANT

Le deficit du modele complet est ~ -95 % a n=256 ET a n=384 (quasi identique). Une
DIFFUSION de bord d'anneau (lissage de l'anneau net par la grille cartesienne)
DIMINUERAIT avec la resolution (cellule plus petite => anneau moins lisse). Le
deficit ne bouge PAS avec n. Donc :

- ce n'est PAS (seulement) une diffusion de bord d'anneau resolvable par n plus
  grand (la voie "cartesien haute-res n=768/1024" ne suffira probablement pas) ;
- ce n'est PAS la geometrie du bord externe (cut-cell sans effet) ;
- c'est donc tres probablement STRUCTUREL : normalisation / echelle de temps /
  couplage du chemin system-schur complet. Le taux brut ~0.037 est un PLATEAU.

## Contraste avec le modele REDUIT (qui, lui, recupere)

Le modele REDUIT (derive ExB scalaire) :
- sur grille POLAIRE : l=4 EXACT (0.913 vs 0.911), l=3/5 proches (diag_polar_omega) ;
- sur grille CARTESIENNE : -5 a -27 % a n=192 minmod, et AMELIORE avec ordre/resolution
  (sweep WENO5) -> comportement de diffusion classique, resolution-DEPENDANT.

Le modele complet (resolution-independant a -95 %) se comporte DIFFEREMMENT du reduit.
Le facteur brut entre reduit-polaire (0.155) et complet-cartesien (0.037) est ~4x,
proche du facteur de diffusion d'anneau invoque en M1, MAIS l'independance en
resolution du complet contredit une explication purement diffusive.

## Verdict honnete

- Le cut-cell (#218/#222/#224) est une vraie capacite testee (MMS ordre ~2, masse
  conservee) mais NE corrige PAS le taux diocotron : mauvais outil pour ce verrou.
- Le bug ABI natif (#225) est corrige (runs natifs GH200 possibles) ; le cas est
  executable en natif (DISC #14). Ces gains d'ingenierie restent valides.
- Le deficit du modele complet est RESOLUTION-INDEPENDANT et GEOMETRIE(bord)-
  INDEPENDANT => suspect = STRUCTUREL (normalisation / echelle / couplage du
  system-schur complet), PAS la diffusion de maille ni le bord externe.
- AUCUNE reproduction du modele complet revendiquee.

## Prochain pas recommande (diagnostic, pas gros GPU)

Isoler le facteur structurel : comparer, sur un MEME setup minimal, le taux brut du
chemin system-schur COMPLET vs le chemin reduit ExB, pour localiser d'ou vient le
plateau ~0.037 (normalisation 2pi/echelle de temps du complet ? force de couplage
Schur ? vitesse de derive initiale ?). C'est une etude de normalisation/structure,
pas une montee en resolution ni une nouvelle geometrie.

## MAJ : tentative VOIE 1 (modele complet sur grille polaire) -- mur de WELL-BALANCING

Le chemin polaire (anneau r0/r1 resolu par un axe de grille) a ete assemble (PR adc_cases
#18 : fluide isotherme polaire #209 + Lorentz + Schur polaire #215 + Poisson polaire ;
observable phi sur r=r0). Il S'ASSEMBLE et DEMARRE mais diverge avant la fenetre de fit.
Caracterisation a 3 niveaux :
1. NaN a t~0.02 ; dt plus petit ne fait que RETARDER (t=0.02 -> 0.101 a dt=1e-4) -> PAS le CFL.
2. IC d'equilibre rotatif derivee (bilan radial : centrifuge rho v_theta^2/r - d_r p
   - rho d_r phi + rho B_z v_theta = 0 ; racine ExB-continuee ; signes verifies vs le moteur,
   PR adc_cases #20). Correcte dans le CONTINU.
3. MAIS l'equilibre continu n'est PAS discretement stationnaire : un run delta=0 (sans
   perturbation, nr=256) fait croitre TOUS les modes azimutaux de 0 a ~1e9 en 200 pas.
   Les operateurs discrets (source centrifuge polar_geom_source vs divergence de flux ;
   et/ou l'etage Schur) ne preservent pas le bilan continu.

VERDICT : le fluide polaire complet aux parametres raides exige un SCHEMA WELL-BALANCED
(qui preserve discretement l'equilibre rotatif source-equilibre) -- un vrai chantier CFD,
pas un knob ni une IC continue. Le modele REDUIT ExB scalaire l'evite (pas d'equation de
moment) et c'est pourquoi LUI recupere l=4 exact en polaire : la preuve que la resolution
d'anneau est la clef existe, mais le fluide COMPLET ne tourne pas stable sans well-balancing.
Chantier en cours : workflow polar-wellbalanced (diagnostic du residu discret + fix
well-balanced + test de stationnarite delta=0). AUCUNE reproduction du modele complet
revendiquee (ni cartesien -82/-95%, ni polaire bloque).

## Acquis d'ingenierie de la campagne (independants du verdict scientifique)
- Schur polaire MULTI-RANG MPI (#227 merge, plus mono-rang ; parite np=1/2/4 ~1e-13) ;
  extension MULTI-BOX (#229, fix Kokkos en cours).
- cut-cell EB (#218/#222/#224), Strang generique (#217), fix ABI natif GH200 (#225,
  + test CI de non-regression), cas hoffart executable en natif (DISC #14).
