# Halo accepté évolutif : réception Native bornée

Les vrais runs Serial/MPI2 passent avec ABI8, Kokkos/OpenMP Dim2, MPICH, CP12 et accepted9. Le [reçu hashé](sdk3d8481_evolving_halo_reception.json) sépare les physiques, versions et limites ; le SDK conservé est construit depuis `3d8481c9`, DSO `7fd8c7fe…f846a44`.

| Physique / réalisation | Serial | MPI2 |
|---|---:|---:|
| dc/dt=0, dm/dt=1, synchronisation | 1 PASS, 46.229 s | 1 des 2 cas collectifs PASS |
| Même physique, permutation des composants, ratio 5/2 | 1 PASS, 45.504 s | 1 des 2 cas collectifs PASS ; driver total 91.211 s |
| dc/dt=0, dm/dt=m, Euler et ratio 5/2 | 1 PASS, 45.394 s | 1 cas collectif PASS, 47.386 s |

Les scripts publics définissent leurs équations, discrétisation et méthode temporelle. Aucun traitement central propre à ces modèles n'est ajouté. L'oracle non-auteur affine vérifie tous les carriers, y compris les ghosts : constante 1 exacte et marqueur initial+t sous une borne d'opérations dérivée du code. Les cinq phases, les lignes Halo directes, les membres CP12/accepted9, les masques actifs 48/64, les histoires, diagnostics, géométrie et horloges sont conservés. Reprise/replay sont bitexact. Une copie scellée correctement puis tronquée atteint réellement le préflight carrier ; son refus préserve l'image acceptée. Le premier run avec ZIP_STORED reste FAILED et ses preuves restent disponibles.

L'oracle multiplicatif discrimine la réalisation temporelle : facteur coarse `1+DT` et fine `(1+2DT/5)^2*(1+DT/5)`, pour `DT=1/64`, deux macro-pas, trois sous-pas fine dont le reste. Les calendriers un pas, ratio spatial 2 et reste omis sont refusés. La contre-revue indépendante recalcule la partition avec Fraction ; l'erreur Native maximale 4.44e-16 reste dans la borne `gamma(24*n+1)` indépendante du résultat observé. Cette formule est vérifiée sur les cellules valides actives ; les ghosts complets sont capturés et comparés pour la reprise, sans oracle général de leur formule dans ce cas.

Chaque route authentifie le vrai paquet et son DSO, avec les 1 136 fichiers installés et identités exactes avant/après. Le nombre de rangs MPI ne multiplie pas le nombre de cas. Audit affine (`/Users/romaindespoulain/dev/tmp/pops-evolving-halo-final-independent-20261002/REVIEW.md`) ; audit temporel (`/Users/romaindespoulain/dev/tmp/pops-api040-native-reception-evidence-20261001/sol61-halo-temporal-discriminator-independent.json`).

Reproduction : reprendre les variables SDK et PATH des [commandes N8/N16](sdk3d8481_spatial_v4_scientific_reception.md), puis sélectionner le fichier `tests/python/integration/amr/test_public_evolving_accepted_halo.py` (deux cas) ou le node `tests/python/integration/amr/test_public_accepted_halo_substep_growth.py::test_public_accepted_halo_five_halves_substep_growth`, avec une nouvelle sortie. Les drivers `run_installed_checks.py` et `run_installed_mpi_checks.py` et leurs commandes littérales sont enregistrés dans les result.json liés. Faire la capture Serial `--identity-only --output .../after` après la fin du test ; le driver MPI la fait lui-même.

Le refus de checkpoint prouve le préflight de reprise. Le rollback après préparation d'un Halo pendant une tentative reste à recevoir via le relais privé versionné en préparation. Field/GhostBC, autorité bootstrap, regrid, AMR générale, GPU, ROMEO et CI restent ouverts. Le texte brut de l'exception carrier n'est pas conservé séparément : la preuve est l'assertion Source précise et le JUnit Native passant. Aucune réception globale de C23/C34 ni des 94 obligations.
