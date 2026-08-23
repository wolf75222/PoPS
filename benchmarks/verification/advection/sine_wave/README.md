# Advection périodique d’une onde sinusoïdale

Ce premier benchmark de vérification est volontairement écrit comme un tutoriel :
on **génère d'abord une donnée authentifiée**, puis on la trace dans un second
temps. Il ne mélange donc jamais un solveur PoPS avec une bibliothèque de figures.

`generate_data.py` est le fichier de cas pédagogique : constantes en tête, CLI,
imports PoPS, puis neuf sections numérotées qui se lisent linéairement du domaine
jusqu'à l'appel de publication. La configuration AMR et le cycle public
`validate -> resolve -> compile -> bind -> run` y restent visibles. Le support
privé `_case_support.py` contient uniquement les extractions, métriques,
empreintes, provenance et écritures atomiques nécessaires autour du cas.

> État transitoire au 20 août 2026 : les données et l'analyse v3 présentes sont
> une archive historique, non une qualification du source courant. La matrice
> versionnée [matrix.v1.json](matrix.v1.json) doit être exécutée intégralement
> avant d'actualiser [ANALYSIS.md](ANALYSIS.md) et son manifeste. Les anciens
> résultats v1/v2 ne sont pas une preuve non plus.

## 1. Le problème vérifié

On transporte une onde sur le tore unité :

\[
  \partial_t q + \mathbf a\cdot\nabla q = 0,
  \qquad \mathbf x\in[0,1]^d,
  \qquad q(\mathbf x,0)=1+0.10\sin\left(2\pi(x+2y+3z)\right).
\]

Les termes transverses sont tronqués suivant la dimension active : (1,), (1, 2)
ou (1, 2, 3). **x**, **y**, **z** et **diagonal** désignent la direction de
transport; **xy** est le cas 3D dédié à la traversée d'une arête de bloc. Ils
ne changent pas le mode spatial. Après une période T=1, la
translation périodique entière revient à l'état initial. La sonde supplémentaire
à t=0.37 évite qu'un champ immobile semble correct au seul temps final.

La référence est une **moyenne de volume finie exacte**; les normes pondérées
par le volume L1, L2 et Linf viennent de helpers/verification/. L'initialisation
PoPS suit sa projection conservative publique : l'erreur rapportée inclut donc
cette projection, sans la confondre avec une valeur au centre de cellule.

La route calculée est **SSPRK2 + MUSCL(VanLeer) + ScalarUpwind**. Le CFL demandé
est 0.40, mais le CFL effectivement envoyé au runtime est 0.40 / d afin de
rester conservatif pour l'advection non scindée en dimension d.

## 2. Générer une donnée, sans faire de figure

Construire d'abord la dimension ciblée, puis lancer le générateur. Il écrit une
archive NPZ compressée et le JSON de provenance adjacent dans results/.

~~~sh
bash scripts/setup_env.sh --dim 2
bash scripts/build_python.sh --dim 2
env -u PYTHONPATH python benchmarks/verification/advection/sine_wave/generate_data.py \
  --dimension 2 --resolution 32 --mode diagonal \
  --layout uniform --subcycling synchronous
~~~

Par défaut, la chronologie contient **17 snapshots**, de t=0 à t=T.
--time-snapshots accepte une valeur plus grande mais refuse toute valeur
inférieure à 9. Chaque résultat est immuable : son nom contient notamment
_ts<N>_rid<16-hex>, où rid est le préfixe de l'identité SHA-256 construite à
partir du cas, du source fingerprint, de l'artefact/bind et de la provenance
d'exécution. Si le nom existe déjà, le programme refuse de l'écraser.

Le JSON enregistre aussi backend Kokkos, concurrence, MPI/rangs, environnement,
hôte et contexte SLURM. Sa provenance `pops.sine-wave.source.v2` authentifie
**cinq autorités Python** : le générateur linéaire, son support privé et les
trois modules helpers importés. Elle authentifie aussi le diff Git suivi et les
octets de l'extension native effectivement chargée face à `variants.json`. Le
NPZ et le JSON doivent être conservés ensemble : le traceur vérifie leur schéma
v3, leurs empreintes et leurs identités croisées avant toute figure.

En layout uniforme, `--block-size` construit explicitement
`RegularBlocks(max_cells=...)` : la grille est réellement pavée et les
différentes tailles exercent donc les faces internes de blocs. Pour l'AMR, la
grille de base reste sans ce pavage uniforme car le lowering AMR possède sa
hiérarchie; la même option borne alors les patches via `PatchLayout` et
`BergerRigoutsos`. La valeur est enregistrée dans l'identité du résultat dans
les deux cas.

Pour changer de dimension, reconstruire la même dimension avant d'appeler le
générateur :

~~~sh
bash scripts/build_python.sh --dim 3
env -u PYTHONPATH python benchmarks/verification/advection/sine_wave/generate_data.py \
  --dimension 3 --resolution 24 --mode diagonal \
  --layout uniform --subcycling synchronous
~~~

Les directions transverses impossibles sont refusées avant compilation (par
exemple z en 2D).

## 3. Exercer l'AMR de manière explicite

**amr-frozen** utilise une fenêtre géométrique stationnaire et interdit les
regrids après l'initialisation. **amr-mobile** utilise `PrescribedWindow`, une
fenêtre géométrique de centre `PATCH_CENTER + PATCH_VELOCITY * t`, évaluée avec
l'horloge acceptée du `Program`, puis regrille périodiquement. Ce n'est ni un
marqueur transporté ni un callback Python : `Tag(window)` raffine la fenêtre et
`Coarsen(~window)` demande explicitement le coarsening hors de celle-ci.

La périodicité de cette trajectoire n'est pas une propriété implicite du frame.
Le runtime reçoit le masque d'axes périodiques de la topologie exacte et ne
replie centre/distance que sur ces axes; un axe non périodique reste non replié.
La fenêtre doit employer le même frame canonique et la même horloge que le
layout/programme. Les demi-largeurs strictement locales sont contrôlées par le
runtime.

~~~sh
env -u PYTHONPATH python benchmarks/verification/advection/sine_wave/generate_data.py \
  --dimension 2 --resolution 16 --mode diagonal --layout amr-mobile \
  --subcycling subcycled --block-size 8 --time-snapshots 17
~~~

--subcycling synchronous et --subcycling subcycled sont des campagnes
distinctes. Le cas gelé `d2-cf-subcycled` demande `--cycles 3` et 49 snapshots :
la caractéristique matérielle périodique issue du point de référence doit entrer
et sortir trois fois de l'union des boîtes fines natives inchangées. Le patch
mobile reste une obligation séparée de regridding prescrit, sans servir de
substitut à cette traversée statique répétée.
Les tailles de patches AMR se règlent avec --block-size.

Le générateur refuse la qualification AMR si aucune topologie mixte coarse/fine
n'a réellement été observée. Les notes v3 qui attribuent un ancien défaut
d'all-fine à la séparation halo FV/interpolation coarse→fine, ou qui rapportent
deux niveaux actifs et un patch mobile, sont des **archives non recevables** :
elles ne qualifient pas le source courant. Seul le prochain `COMPLETE.json`
authentifié peut établir ces observations; ses métriques et limites seront dans
son `ANALYSIS.md` publié, pas dans ce guide d'exécution.

Les lignes de regrid sur les graphiques sont la **borne droite observée** des
intervalles entre snapshots; elles ne prétendent pas être l'horodatage interne
exact du regrid. Les cas de matrice demandant face, arête 3D ou coin 3D publient
des témoins fail-closed : les `local_boxes` natifs doivent contenir les plans
internes réellement matérialisés, une boîte doit être incidente aux 1/2/3 plans
choisis, et deux snapshots doivent encadrer strictement leur temps exact de
franchissement. Les témoins coarse–fine et patch mobile proviennent des masques
composites, compteurs de regrid et epochs de topologie observés.
Le témoin de patch gelé répété publie en plus les boîtes fines natives et les
intervalles de timeline où la caractéristique matérielle entre ou sort de leur
union; il échoue si cette topologie change ou si les trois entrées et sorties ne
sont pas observées.

## 4. Matrice complète et refus d'écrasement

`run_matrix.py` valide par défaut le schéma, les obligations et toutes les
commandes sans exécuter de solveur. La matrice v1 refuse notamment une obligation
sans cas témoin. Elle couvre Dim 1/2/3, modes d'axe et diagonales valides,
uniforme/AMR gelé/AMR prescrit mobile, subcycling activé/désactivé, trois tailles
de blocs contrôlées (8/16/32), MPI np=1/2/4 avec les topologies natives
1×1/1×2/2×2 en 2D, et np=8 avec une topologie 2×2×2 en 3D. L'exécution
complète est explicite et refuse tout
répertoire de cas déjà présent avant de commencer; chaque sortie est vérifiée
contre les témoins déclarés avant de continuer. Elle reconstruit elle-même les
cinq variantes nécessaires (Dim1, Dim2, Dim3 non-MPI, puis Dim2+MPI et Dim3+MPI), car une
seule extension native ne peut qualifier simultanément ces dimensions et ABI.
Le SHA-256 exact de `matrix.v1.json` est scellé dans le driver : modifier une
résolution, un nombre de cycles/snapshots, une taille de bloc, un rang ou une
topologie MPI est refusé avant toute réservation de sortie ou exécution.
La campagne impose `OMP_NUM_THREADS=2`, `KOKKOS_NUM_THREADS=2` et
`OMP_PROC_BIND=false`; backend et concurrence réellement chargés restent dans
la provenance de chaque paire.

La fin de campagne n'est réussie qu'après publication no-clobber de
`COMPLETE.json`: hashes de toutes les paires, du manifeste, du générateur, du
driver, du support et des helpers authentifiés, ainsi que les reçus natifs. Le
manifeste scelle séparément une autorité source commune aux 37 cas — révision
Git, état sale, SHA-256 du diff suivi, hashes des cinq autorités Python et
inventaire SHA-256 du véritable arbre de build (`CMakeLists.txt`, `cmake/`,
`include/`, `src/`, `python/`, `schemas/`, `scripts/`) — et la vérifie de
nouveau lors du tracé. Cet inventaire inclut aussi un fichier non suivi qui
influe sur le build, mais exclut explicitement sorties, caches, résultats,
rapports et liens symboliques. Les artefacts natifs restent au contraire
attachés individuellement à chaque paire, car ils diffèrent légitimement entre
Dim1, Dim2, Dim3 et la variante MPI. Ainsi deux phases de build ne peuvent pas
être assemblées si elles ont été exécutées avec des révisions ou diffs suivis
différents. Les
séries uniformes x en 1D/2D/3D comportent chacune trois résolutions; l'ordre
L1 doit être au moins 1.75 (tolérance asymptotique conservatrice pour MUSCL
TVD). Les ordres L2/Linf sont rapportés mais ne sont pas artificiellement
qualifiés à deux près des extrema.

~~~sh
env -u PYTHONPATH python benchmarks/verification/advection/sine_wave/run_matrix.py
env -u PYTHONPATH python benchmarks/verification/advection/sine_wave/run_matrix.py --execute
~~~

Le driver MPI n'utilise jamais un `mpiexec` choisi implicitement dans le
`PATH`. Par défaut il prend le lanceur situé à côté de l'interpréteur Python
qui charge l'extension native. Lorsque Python et MPI proviennent de préfixes
distincts, notamment sur ROMEO, fournir le chemin absolu du lanceur associé à
la bibliothèque MPI utilisée au build :

~~~sh
export POPS_MPIEXEC=/chemin/absolu/vers/mpiexec
env -u PYTHONPATH python benchmarks/verification/advection/sine_wave/run_matrix.py --execute
~~~

Après chaque build MPI, une sonde native collective à deux rangs vérifie le
lanceur, les rangs, la taille du monde, l'activation MPI et un `allgather`
d'octets. Elle n'avance aucune simulation. Une incohérence de runtime est donc
refusée avant le premier cas scientifique MPI; en particulier, le cas `np=1`
ne peut plus masquer deux mondes singleton issus de distributions MPI
incompatibles.

## 5. Tracer ensuite, uniquement à partir de la campagne scellée

Le traceur n'importe ni pops ni le générateur, et son CLI public exige
**exactement** `--complete COMPLETE.json`. Il n'accepte ni NPZ positionnel ni
métadonnée isolée. Avant toute création de staging ou de figure, il authentifie
le manifeste et les 37 paires NPZ/JSON qu'il désigne. Une paire seule, y
compris pour examiner une convergence, n'est donc pas une entrée de
publication.

Les échelles de couleurs restent stables dans le temps :

- 1D : profils numérique/exact, erreur et animation GIF;
- 2D : état numérique, exact, erreur, contours et overlay des patches, coupe
  centrale et GIF;
- 3D : coupes centrales xy/xz/yz avec projections de patches AMR, storyboard
  temporel, coupe oblique, carte d'erreur au snapshot L2 maximal, animation GIF
  et isosurfaces rendues dans la figure (VTK est optionnel, aucun fichier VTK
  externe n'est exporté);
- matrice complète : trajectoire du patch mobile prescrite, comparant les
  centres attendus et les centres de boîtes fines aux snapshots de regrid
  observés, et diagramme 3D du reçu MPI `2×2×2`/coin inter-rang à huit rangs;
- tous les cas : conservation de masse, amplitude, phase et normes au cours du
  temps.

Une convergence n'est produite qu'à partir de résultats compatibles de
résolutions distinctes. Les comparaisons AMR (uniforme/AMR,
subcycling/synchrone, interface/bulk et événements de regrid) sont
**fail-closed** : elles exigent même source/révision, backend, concurrence, MPI,
méthode, vitesse, résolution et chronologie. Une incompatibilité arrête le tracé
plutôt que d'assembler une figure trompeuse.

Pour publier le rapport complet d'une matrice réellement terminée (et jamais
d'un essai partiel), passer son unique manifeste scellé :

~~~sh
env -u PYTHONPATH python benchmarks/verification/advection/sine_wave/plot_results.py \
  --complete benchmarks/verification/advection/sine_wave/results/<campagne>/matrix-v1-prescribed-window/COMPLETE.json \
  --figures benchmarks/verification/advection/sine_wave/figures/<publication-nouvelle>
~~~

Le consommateur vérifie les SHA-256 de la matrice, des scripts et des 37 paires
NPZ/JSON avant tout rendu, puis crée le staging et publie une seule fois un
sous-dossier immuable sous `figures/`. Son nom par défaut contient le préfixe de
l'identité de publication SHA-256. Chaque publication contient
`plot_manifest.json`, avec les hashes du manifeste scellé, des 37 paires et de
chaque média. Un dossier cible déjà présent est refusé sans suppression ni
réécriture, même s'il apparaît au dernier instant de publication. Le rapport
publié contient son propre `ANALYSIS.md`; la convergence finale à `T=1` y est
séparée des figures diagnostiques à `t=0.37`, avec chaque média, les
obligations/témoins, la conservation et les environnements réellement reçus.
Sans `COMPLETE.json` authentique, aucun média n'est publié : un répertoire de
retry, même rempli de paires NPZ/JSON, ne remplace jamais ce sceau.

La matrice comprend aussi une étude dédiée de taille de bloc : trois cas 2D en
propagation y, strictement identiques sauf `block_size=8/16/32`. Elle produit
une figure de comparaison seulement si les trois résultats scellés restent
compatibles.

## 6. Interpréter les sorties

Séparer toujours deux familles de métriques :

| Famille | Ce qu'elle répond |
| --- | --- |
| Intégrité | conservation de masse, progression temporelle, mouvement à t=0.37, topologie AMR mixte, cohérence NPZ/JSON/provenance |
| Précision | L1, L2, Linf, amplitude et phase face à l'oracle de moyenne de volume |

L'attente asymptotique est E_h ∝ h² pour le couple espace-temps d'ordre deux.
C'est un guide de lecture, pas un seuil artificiel : Van Leer peut réduire
l'ordre observé, en particulier près des extrema et dans Linf.

## 7. MPI et performance : périmètres séparés

La vérification MPI scientifique appartient à la matrice complète ci-dessus :
le driver construit les variantes Dim2/Dim3 MPI, authentifie le lanceur puis
exécute sans réduction les topologies 1×1, 1×2, 2×2 et 2×2×2. Seul le
`COMPLETE.json` des 37 cas permet de conclure sur l'invariance et la correction.
Le strong/weak
scaling Serial, Kokkos OpenMP, CUDA et Kokkos+MPI relève du workload uniforme
séparé [performance/advection_sine](../../../performance/advection_sine/README.md).
Ses campagnes ROMEO sont soumises, collectées et publiées indépendamment; les
résultats ROMEO finaux ne sont pas encore publiés ici.

## Arbre utile

~~~text
benchmarks/verification/advection/sine_wave/
  generate_data.py   # cas PoPS linéaire en neuf sections, sans tracé
  matrix.v1.json     # matrice complète, cas et obligations versionnés
  run_matrix.py      # validation fail-closed et exécution explicite no-clobber
  _case_support.py   # diagnostics, provenance et publication privés
  plot_results.py    # NumPy/Matplotlib : validation et figures à partir des données
  ANALYSIS.md        # conclusions, limites et figures publiables
  report/            # sélection immuable de figures + manifeste SHA-256
  results/           # données locales ignorées par Git
  figures/           # figures locales ignorées par Git
helpers/verification/ # normes, conservation, convergence et comparaisons réutilisables
~~~

Les futurs cas ne sont ajoutés que lorsqu'ils possèdent un oracle, un schéma de
données et une preuve d'exécution correspondante.
