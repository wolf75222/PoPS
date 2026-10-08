# PR brouillon - état réel de l’intégration PoPS 0.4.0

Snapshot du 8 octobre 2026. **Mission encore ouverte ; cette PR est un brouillon.**
La révision reconstruite et exécutée pour CPU/MPI est `6d898599a3f66fa9aa3c049ba0f3114f1960b6dc`.
Cette PR ajoute ensuite le correctif privé de portabilité AMR testé dans la sonde
CUDA737830. Les résultats CPU/MPI précédents restent liés à6d ; un nouveau
rebuild/relink et une réception CPU/GPU du code intégré final restent nécessaires.

## Ce que contient la PR

La PR expose toute la branche cumulative `codex/api040-integrated-main-20261004`
contre `master`, pas uniquement le dernier correctif. Avant cet ajout de statut,
le diff comporte 3 512 fichiers et 1 155 commits : notamment 337 fichiers Python,
146 headers C++, 14 sources C++, 46 exemples, 1 106 fichiers de tests et 1 787
documents/preuves. Ces nombres décrivent le diff Git ; ils ne sont pas des nombres
de fonctionnalités reçues ou de tests exécutés. Les travaux antérieurs inclus
restent identifiables par leur historique ; aucune attribution globale d’auteur
ni validation universelle n’en découle.

Le code couvre le frontend/IR et son abaissement vers les composants réels
C++/Kokkos/MPI/AMR, les contrats de valeurs SSA, les plans/providers Field et Aux,
les mécanismes temporels, les checkpoints/restarts et leurs tests. Les documents
historiques et [le corpus](corpus.json) détaillent chaque réception limitée.
Le mini-runtime du handoff n’est pas substitué au paquet de production.

Les derniers changements intégrés ont corrigé l’incidence des invalidations Aux
par niveau, les bindings et plans retenus lors d’un restore AMR, ainsi que le
callable Poisson RHS V2 pour le compilateur CUDA. Les tests exercent Phi→Q→Psi,
les restores répétés, le refus collectif d’une image incompatible et la source
STATE qui consomme Q après restore/rollback. Le contrat privé de bindings et le
contrat callable sont versionnés à 1 ; ABI Native13, package8 et read/input2 sont
conservés. La signature des 383 headers est devenue
`af8d3a678a42f9d28abedc386bad5af8d987eb694a9042439414de1ac9c278e1`.

## Ce qui a réellement été exécuté et reçu

Environnement CPU local : conda `pops`, Python3.12, Dim2/double, Kokkos5.2
OpenMP+Serial, MPI et HDF5 parallèle. Les tests Python importent le paquet
installé, sans PYTHONPATH, et authentifient le vrai module Native :
`a9367a4af3c9989ea03ebcaf66b76b8e20f5dd5bcd5bdf16b6b1172e942b5ca2`.
La wheel est `414add1e2b1289c31c37f51ca3bbf8476c2409839b069094ab5cd06071767622`.

| Run | Résultat réel | Portée/preuve |
|---|---|---|
| Build officiel Python | PASS ; 24 actions C++, sept bindings et Core AMR reconstruits ; wheel/install/prove/doctor PASS | [Build reçu](evidence/draft_status_20261008/native-build-install.json). Signature af8 réelle, fullmanifest Source/wheel/install synchronisé. Ccache4hits/20misses agrégés ; pas de mesure du compilateur interne par source. |
| C++ af8, représentant STATE | 1 PASS, zéro fail/error/skip | Binaire et deux DSOs frais, vrais guards du host ; distinct de la cohorte suivante. |
| C++ Field17 | 17 avec un rang, 17 sur chacun des deux rangs MPI : 51 exécutions | [Réception C++](evidence/draft_status_20261008/cpp-first-and-cohort.json), census uniquement trois cas legacy. |
| C++ Registry | 21 PASS avec un rang | Total cohorte C++72 ; templates ND1/2/3 ne qualifient pas les Native1/3. |
| PDE IMEX nonautonome original | 1 PASS, 35,54 s ; 64 cellules sauvegardées | [PDE reçue](evidence/draft_status_20261008/first-original-pde.json). U=84/25 et F=144/25 : erreurs4,818e−13 et1,926e−12 sous les bornes originales1,210e−11 et4,114e−11. Y=48/25 est une référence, pas une mesure. |
| OwnCheckpointRetry original | 1 PASS, 93,80 s | [OwnRetry indépendant](evidence/draft_status_20261008/own-retry-independent.json). Deux publications du même seal178806octets ; reference/accepted/restored et continuation/restart tableaux/JSON/Program exacts. Deux niveaux AMR ; compteurs regrid/topologie2→3. |
| Huit non-régressions Python originales | 8 PASS, zéro fail/error/skip, 482,66 s | [Native8 indépendant](evidence/draft_status_20261008/native8-independent.json) : manual/preset/rejection/publication rollback et cinq guards temporal restart ; 17 checkpoints/809 charges vérifiées. Déclaration producteur `-x` non étayée : commande réelle sans `-x`. |
| Contrats/packaging statiques | PASS | release_preflight en mode développement ; 172 API +7 ABI +19 sdk-root +185 sdk-support =383 headers signés, 9 test-only. Ce n’est pas GitHub CI. |

Les équations, gardes et budgets scientifiques originaux n’ont pas été assouplis.
Un premier nouvel oracle Q=12 partout était incorrect car la SSA doublait le
niveau actif seulement. Son échec est conservé ; le nouveau témoin utilise des
copies high-Q réellement acceptées et des contrôles exacts de source/Q, Q=2Phi,
finitude et changement de forcing. Il ne qualifie pas la convergence du PDE
composite non uniforme. Voir [la critique](STATE17_SOURCE_CRITIQUE.md).

Les scopes d’inventaire restent explicites : certains reçus comptent fichiers
et symlinks, d’autres incluent les répertoires. Une observation OpenMP2 n’est
attribuée qu’au processus réellement instrumenté. Aucun de ces runs CPU n’est
une réception scientifique MPI2/GPU/3D générale.

## GPU : erreurs, correction et tâche actuelle

Le job ROMEO736186 a échoué lors de la compilation du callable Poisson V2 ; le
callable nommé corrigé est intégré. Le job737206, même code6d/SDKaf8, a ensuite
échoué dans trois range-for bracés de amr_layout_transfer.cpp, avec NVCC12.6.85,
GNU13.4 et Kokkos5.2.1. [La réception négative complète](evidence/draft_status_20261008/cuda737206-negative-independent.json)
vérifie 104 éléments/469901230octets et les anciennes ressources préservées.
Aucun Native GPU installé ni sept bindings CUDA terminés n’en découle.

Le [patch candidat et la sonde](wip/cuda_braced_range/README.md) sont exposés
séparément, **désormais intégrés au runtime après contre-revue Source et preuve NVCC ciblée**. La sonde737830
compilant une seule TU originale puis candidate a terminé0:0 en49s, selon le
retour actuel du moniteur : baselineexit1 avec les trois erreurs reproduites,
candidateexit0, objet ELF64 LE AArch64 ET_REL903032octets, SHA
`5b2ed0915d30176d2c0f8f333eebbe36e3446623b5b89e52cbea89842f5703b3`.
[La réception complète de la sonde](evidence/draft_status_20261008/cuda-private-probe-independent.json)
a vérifié551éléments/96441149octets et les emprunts avant/après. L’émission
originale cudafe montre des plages `({…})` ; le candidat utilise les tableaux
nommés. Le patch est intégré, avec contrat privé de portabilité version1,
sans changement des corps, ordre, budgets, gardes, FP ou signature des headers.
Les argv internes sont dérivés de xtrace, pas d’une instrumentation syscall.
Ce résultat ciblé ne vaut pas build complet du Native ni calcul PDE GPU.

Conformément à la préférence de stockage corrigée, les nouveaux travaux sont
placés dans `/gpfs/scratch/rmdraux/` personnel, avec répertoire de tâche0700.
Home ou autre espace personnel reste préféré au projet commun ; un sous-dossier
projet/rmdraux n’est qu’un dernier recours. Les anciennes preuves/emprunts
projet sont conservés et lus seulement. Leurs chemins actuels avec `/rmdraux/`
sont distingués des anciens chemins historiques, sans supposer une cause de
relocation. La sonde n’a modifié ni anciennes sources, ni environnements, ni jobs.

## Première CI GitHub de la PR681

Sur le commit initial de PR `a82e626896a7ab79883d7a14ad95013efd95fbb5`,
les premiers checks sont **en échec**, malgré les validations locales listées.
Le [run CI37799501091](https://github.com/wolf75222/PoPS/actions/runs/37799501091)
rapporte un internal compiler error GCC13 dans apply_identity_attributes lors du
prewarm de amr_system.cpp sur Serial/OpenMP/MPI, et des écarts de classement des
includes/inventaires dans les tests Python architecture. Les jobs dépendants
sont donc partiellement sautés et le gate d’agrégation échoue. Les correctifs
correspondants et une CI au SHA final restent à réaliser.

Le [run Docs37799501092](https://github.com/wolf75222/PoPS/actions/runs/37799501092)
avait320violations de forme/liens : tirets longs et liens absolus vers des preuves
locales indisponibles sur Linux, ainsi que3références au handoff non suivi.
La correction de portabilité documentaire conserve les labels et chemins de
provenance en texte littéral, ou lie les petites copies suivies ; aucune règle
CI ni garde n’est assouplie. `bash scripts/build_docs.sh` passe localement après
ces corrections. Cela ne déclare pas les prochains checks GitHub verts.

## Ce qui reste

Les **86 obligations T1–T6/C01–C40/M01–M28/W01–W12 +8 principes =94** ne sont
pas toutes fermées. Les limites détaillées du corpus restent la référence.

1. Reconstruire/relinker les consommateurs CPU après le patch privé intégré,
   reconstruire le vrai Native CUDA/sept bindings, puis exécuter les cas
   scientifiques GPU et leur non-régression. Tester les autres backends et
   dimensions à la révision correspondante ; ne rien hériter de builds anciens.
2. Compléter les producteurs Field V2 encore refusés : diffusion, RHS de chemin,
   solve spatial retenu, transport principal, résultats couplés multibloc et
   boucles remappées. Le premier défaut concret est `diffusive_rhs` non admis et
   sa frontière allocation/écriture non exposée au ticket SSA. Un simple ajout
   d’allowlist serait incorrect. Les réalisations natives existent : il faut
   raccorder leur publication réelle et collective.
3. Compléter la lecture d’un Field résolu caché sous DerivedAux avec un vrai
   FieldContext, ses droits/versions/représentations/point, pas en supprimant le
   refus. Exercer les nouvelles physiques **et méthodes** via Python public,
   écrites par un non-auteur, sans recettes centrales spécifiques.
4. Ajouter les réceptions manquantes : faute de préparation des bindings,
   ownership/rangs vides des nouveaux cas AMR, rollback de l’AcceptedSnapshot
   englobant, regrid/history/cache pertinents réellement peuplés, convergence
   spatiale/temporelle, coûts à calcul comparable et campagnes distribuées.
5. Rafraîchir la matrice par exigence, les exemples et les preuves de chaque
   backend ; recevoir la CI GitHub sur le SHA final. Cette PR n’est pas déclarée
   green, merge-ready ou scientifiquement qualifiée sur toute la mission.

## Reproduction

Les commandes exécutées exactes, leurs entrées et sorties sont dans les reçus
et les scripts associés. Workflow du dépôt : setup_env une fois par worktree,
puis build incrémental. Depuis le checkout de la révision voulue :

```sh
source /Users/romaindespoulain/miniforge3/etc/profile.d/conda.sh
conda activate pops
rtk proxy env -u PYTHONPATH -u PYTHONOPTIMIZE PYTHONDONTWRITEBYTECODE=1 \
  bash scripts/build_python.sh --dim 2 --mpi
rtk proxy env -u PYTHONPATH -u PYTHONOPTIMIZE PYTHONDONTWRITEBYTECODE=1 \
  python docs/development/api_040/run_installed_checks.py \
  --output /tmp/pops-original-imex-reception \
  --test tests/python/integration/runtime/test_imex_nonautonomous_field.py
```

Le build officiel installe le paquet ; valider identité/doctor/headerSignature
avant d’interpréter le test. Le dossier output doit être neuf. Les caches,
prefixes Kokkos/HDF5/MPI et configurations réellement utilisés sont consignés
dans les reçus ; les backends non exécutés ne deviennent pas reçus par cette
recette. Les preuves volumineuses originales restent externes ; seules les
petites réceptions vérifiées sont copiées [ici](evidence/draft_status_20261008/manifest.json).
