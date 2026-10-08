# PR brouillon - état réel de l’intégration PoPS 0.4.0

Snapshot du 8 octobre 2026, actualisé à 19:52 UTC pour la
[PR brouillon 681](https://github.com/wolf75222/PoPS/pull/681).
**Mission encore ouverte ; cette PR est un brouillon.**
Actualisation suivante du 8 octobre : le code `db206100` a été reconstruit et
installé avec Native `ed2f610f22c5a50b673049ef7fb38854d37f074517d536ad39da1dbe73ed821b`.
Les dix tests Python originaux passent à nouveau et leurs données sont reçues
indépendamment : PDE1, OwnRetry1 et Native8. Les
[reçus db/ed2](evidence/current_db_ed2_20261008/manifest.json) distinguent cette
réception des modifications Source suivantes. Sept objets bindings af8 sont
réutilisés après authentification ; les deux compilations runtime appartiennent
au premier essai, terminé en erreur pendant MPI_Finalize. Le second essai avec
FI_PROVIDER=tcp réussit le lien, la wheel, l’installation et doctor. Le wrapper
du premier essai sort1 et sa commande Ninja échoue143 ; le libellé conflant ces
sorties dans le reçu producteur est explicitement corrigé, sans réécrire le reçu.

La campagne complète GPU737948 a ensuite échoué après la construction et
l’installation de Kokkos5.2.1/HOPPER90 : NVCC refuse une lambda device sous une
méthode privée de CompositeFAC. Treize TUs runtime sur17 ont terminé ; zéro
binding sur7, aucune wheel/install Native GPU ni PDE reçu. L’allocation Slurm
h100 expose ici un GH200120GB, CC9.0. Les nouvelles ressources restent dans le
scratch personnel. Les4298 fichiers réguliers empruntés sont inchangés ; l’unique
ligne d’enregistrement Conda du nouvel environnement a été retirée après
conservation des preuves, retrouvant exactement les deux lignes précédentes.

Le raccordement diffusive_rhs→FieldV2 et la preuve algébrique de la mise à jour
acceptée sont intégrés au commit `53fd6ec25fd67351aefe0a5d2ac1101be7041a78`.
La preuve utilise les dépendances typées et coefficients rationnels des
expressions ; aucun nom de méthode ou tableau connu ne sélectionne le calcul. Le
[contrat de traduction](core_expression_translation_contract_v1.md) reprend
le critère utilisateur : cœur figé, expressions traduites exactement, et
variations de noms/coefficients reçues séparément sur ce même cœur.

La réception suivante est désormais réelle : le cas public nonconstant Euler
compile, bind et exécute le vrai paquet installé C4/Nativeed2. Les192 valeurs
de chaque tableau sont reçues indépendamment, avec les72 charges du checkpoint.
Erreurs état/prédicteur6,661e−16 sous2,274e−12 ; potentiel5,863e−12 sous1,465e−10.
Sur le même cœur `fd27f604adde381bad38e6f04986805ed05f33b549a8dc8c36b622770a9b6aa5`,
les résultats du modèle/méthode/SSA renommés sont bit à bit identiques au cas
initial. κ0,1→0,125 change réellement le coefficient C++ et les tableaux selon
l’oracle : gain0,9847759065→0,9809698831, différence maximale de l’état0,0018544761
et du potentiel0,0001393585. Les [reçus du cœur figé](evidence/fixed_core_diffusion_20261008/manifest.json)
conservent ces faits ; ils ne prouvent pas la généricité de tout le langage.

Le même cas a aussi terminé avec deux vrais rangs MPI : un PASS par rang, zéro
failure/error/skip. Le lanceur utilise l’identité et pytest dans le même processus
pour conserver les descripteurs PMI ; le checkpoint a un chemin partagé et les
tableaux sont propres à chaque rang. Deux erreurs antérieures de lanceur et de
chemin du fixture sont conservées. La [réception MPI2 indépendante](evidence/fixed_core_diffusion_20261008/mpi2-independent.json)
est maintenant PASS : identités des deux rangs authentifiées, quatre tableaux
globaux bit à bit égaux entre rangs et au cas série, même oracle indépendant,
checkpoint partagé et ses72 charges reçus. Le census local des cellules/owners
n’a pas été capturé et n’est pas inféré. Il s’agit d’un cas distribué physique,
pas de deux cas. Ces corrections ne modifient aucun fichier du cœur.

Le nouveau cœur contient 1198 fichiers figés, d’empreinte
`bd29159d6ea92bae81cd96b827eeb3712aa3455942b928acd0f0e0ede728dda9`.
La wheel `64174ea5855a9978c3ae7b61ed116be96217911a68b1a2fab434b059fca0bc90`
installe 1168 charges Python identiques aux sources ; Native reste `ed2…` et
SDK `af8…`. Les deux écritures publiques SSPRK2 et les trois constructions Euler
passent et sont reçues indépendamment sur ce nouveau cœur : cinq cas série,
zéro failure/error/skip, 360 charges de checkpoint vérifiées. Les écritures
SSPRK2 diffèrent de `4,440892098500626e−16` sur l’état accepté ; les trois autres
tableaux sont bit à bit égaux. L’ordre flottant des expressions est conservé.
Les [reçus SSP et Euler actuels](evidence/fixed_core_methods_53fd_20261008/manifest.json)
ne remplacent pas les reçus historiques du cœur `fd27…` et ne prouvent pas
la généricité universelle. SSPRK2 a aussi terminé un PASS sur chacun de deux
vrais rangs MPI. La réception indépendante vérifie l’oracle, les quatre
tableaux globaux bit à bit égaux entre rangs et au cas série, ainsi que le
checkpoint commun de72 charges. Le census local des cellules/owners reste
absent. Les candidates CUDA/Moving/Aux restent externes. La CI et les86
obligations+8 principes sont ouvertes.

L’injection publique d’échec tardif est intégrée dans les tests au commit
`4fbd42ec`, suivie de la correction de provenance `29e9143c`. Le premier
essai a compilé le modèle et le programme, puis échoué avant bind sur une
sérialisation de chemin ; cet échec est conservé séparément. Le second essai
Euler termine PASS1, zéro failure/error/skip : garde réellement refusée,
images avant/après de l’état accepté bit à bit identiques et contrôle accepté
avec le même artefact, sur deux bindings distincts. La contre-revue indépendante
a reçu ces images et l’oracle des192 cellules. Les historiques du binding
refusé étaient initialement non initialisés, fill0 ; ce résultat ne ferme pas
le rollback d’anneaux déjà peuplés. SSPRK2 refus/contrôle est également reçu.
`store_history(depth=1)` déclare le lag maximal et produit deux slots physiques ;
l’assertion adaptative a été corrigée à fill2 avant son premier run. Ce run
effectue un rejet puis deux acceptations et vérifie l’état final, mais échoue
d’abord dans l’oracle du prédicteur : le fixture extrait le slot0 recyclé après
rotation, au lieu du slot1 du dernier intervalle accepté. La correction
d’observation `e01eb5b8` passe au run suivant avec les mêmes équations et bornes ;
les deux slots physiques sont conservés. La réception indépendante vérifie les
huit tableaux de192 cellules, le gain de deux Euler, le potentiel à2DT et le
rapport de deux acceptations/un rejet. Elle n’infère pas un checkpoint ni un
rollback englobant à partir de la seule assertion fill2.
Le fixture de continuation après un premier pas accepté est intégré en
`a6f104f6`, après quatre tests Source et revue mathématique indépendante ; son
run Native reste à faire. Il demande un refus via la composition publique
`limit−2560*tau`, puis compare les deux slots déjà peuplés et le contrôle de
deux pas. Les snapshots Aux/historiques utilisent
des accesseurs privés du journal Native ; ils ne qualifient pas une API publique
de snapshot, un changement de paramètres sur la même instance ou la révocation
d’un ticket retenu. Le retry adaptatif corrigé et la réception MPI2 restent à
terminer. Les [preuves d’échec tardif](evidence/late_refusal_20261008/manifest.json)
conservent les déclarations antérieures au run et les négatifs.

La réception historique détaillée ci-dessous concerne
`6d898599a3f66fa9aa3c049ba0f3114f1960b6dc`.
Cette PR ajoute ensuite les correctifs privés de portabilité NVCC et GCC13,
la portabilité documentaire et la correction des inventaires/sélecteurs CI.
Les résultats CPU/MPI précédents restent liés à6d ; un nouveau
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

Le dernier commit poussé lors du relevé courant est
`53fd6ec25fd67351aefe0a5d2ac1101be7041a78`. La CI de ce SHA comporte
des échecs d’architecture Python, MPI/C++ et de plusieurs shards Python ;
son exécution n’est pas reçue comme verte.
La branche contient tout le code suivi terminé à ce checkpoint. Le dossier
de référence local `PoPS_Codex_handoff_0.4.0/` demeure non suivi ; les archives,
environnements et sorties binaires volumineuses restent aux chemins des reçus.

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
| Réparation architecture/catalogue CI | 158 +53 =211 PASS, zéro fail/error/skip | [Reçu du lot](evidence/draft_status_20261008/architecture-ci-repair.json). Contrôles de sources, inventaires, includes et sharding ; import explicite du vrai frontend de ce checkout pour cette portée statique. Aucun import Native ni calcul scientifique dans ce lot. |

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

## CI GitHub et correctifs intégrés

Sur le commit initial de PR `a82e626896a7ab79883d7a14ad95013efd95fbb5`,
les premiers checks sont **en échec**, malgré les validations locales listées.
Le [run CI37799501091](https://github.com/wolf75222/PoPS/actions/runs/37799501091)
rapporte un internal compiler error GCC13 dans apply_identity_attributes lors du
prewarm de amr_system.cpp sur Serial/OpenMP/MPI, et des écarts de classement des
includes/inventaires dans les tests Python architecture. Les jobs dépendants
sont donc partiellement sautés et le gate d’agrégation échoue. Les correctifs
correspondants ont ensuite été intégrés ; leur réception au SHA final reste
nécessaire.

Le [run Docs37799501092](https://github.com/wolf75222/PoPS/actions/runs/37799501092)
avait320violations de forme/liens : tirets longs et liens absolus vers des preuves
locales indisponibles sur Linux, ainsi que3références au handoff non suivi.
La correction de portabilité documentaire conserve les labels et chemins de
provenance en texte littéral, ou lie les petites copies suivies ; aucune règle
CI ni garde n’est assouplie. `bash scripts/build_docs.sh` passe localement après
ces corrections. Le check Docs est ensuite **PASS** sur `d227c8b5`, puis sur
`65861f43` dans le [run Docs37804221414](https://github.com/wolf75222/PoPS/actions/runs/37804221414).

Le correctif GCC13 de `65861f43` déplace uniquement le type privé local Carrier
vers Impl ; son contrat est [versionné à 1](gcc13_private_carrier_portability_v1.md).
Le [run CI37804221197](https://github.com/wolf75222/PoPS/actions/runs/37804221197)
compile désormais amr_system.cpp avec succès sur les prewarms Serial, OpenMP
et MPI. C’est une preuve de compilation de la TU avec le compilateur CI,
distincte des calculs CPU/Python encore liés à6d.

Au [relevé conservé](evidence/draft_status_20261008/ci-65861f43-snapshot.json)
du 8 octobre à16:29:59UTC, les checks comptent54SUCCESS,43FAILURE et un shard
Python encore en cours ; la CI globale n’est donc pas verte. Les écarts
architecture/catalogue C++ sont corrigés dans le lot211PASS : quatre fragments
et six contrats amont classifiés, anciennes limites conservées avec régions
supplémentaires disjointes et bornées, cible external-field backend réellement
enregistrée, découverte runtime pour les déclarations device conditionnelles,
inventaire C++201→202 et miroir MPI121→123. Tous les anciens coûts restent
exacts ; le nouveau poids2s/0,2s est explicitement estimé, sans mesure prétendue.
Le raffinement déterministe du sharding réduit le maximum modélisé910,3325s
à902,62s, sous la borne inchangée906s. Ces résultats locaux attendent la CI.
La [contre-revue indépendante](evidence/draft_status_20261008/architecture-ci-independent-review.json)
admet les dix fichiers exacts du lot, sans relancer ni prétendre les tests Native.
Les [deux logs CI conservés](evidence/draft_status_20261008/manifest.json)
permettent de relire la compilation GNU13.3 et le refus du catalogue Python.

Un échec supplémentaire a été lu directement dans le shard Python5 :
`ci_pytest_timings.py` refuse dix fichiers absents du catalogue de durées
Python, avant leur exécution. Cette correction reste à faire. Les autres
shards en échec demandent également un diagnostic de leurs logs ; leur succès
ne découle ni des211 contrôles locaux ni des prewarms.

## État des travaux en cours à ce checkpoint

| Travail | État réel | Prochaine étape |
|---|---|---|
| Code cumulé et correctifs récents | Publiés dans cette PR brouillon ; les anciens travaux sont conservés | Relire le diff et poursuivre la réception. |
| CI de53fd6ec2 | Échecs réels ; run37833215669 encore actif | Diagnostiquer les logs puis recevoir les contrôles requis au SHA final. |
| Native CPU après les deux changements C++ | db206100 reconstruit/installé, Nativeed2 reçu ; wheel53fd installée | Recompiler après les prochains headers C++ ; ne pas hériter de la réception. |
| Science CPU | PDE1/OwnRetry1/Native8 reçus surdb ; cinq nouveaux cas publics reçus sur53fd | Non-régression de la nouvelle preuve, échec tardif/contrôle et retry. |
| Native CUDA complet et PDE GPU | Pas de campagne complète suivante soumise à ce checkpoint | Construire depuis le SHA figé dans le scratch personnel, puis qualifier les calculs réels. |
| Extension publique diffusion→FieldV2 | Euler/deux SSPRK2 et MPI2SSP reçus ; rollback initial Euler/SSPRK2 reçus | Retry adaptatif et refus MPI2 ; rollback d’anneaux peuplés à compléter. |

La préparation du prochain build CPU est conservée dans
`/Users/romaindespoulain/dev/tmp/PoPS-private-cpp-af8-native-and-science-preparation-v1-20261008/`.
Son [plan](evidence/draft_status_20261008/next-native-preparation-plan.json)
documente le build db réellement exécuté : deux objets runtime et un nouveau
lien, sept bindings réutilisés après authentification de leurs sources,
flags, dépendances et objets af8. Les futures modifications de headers exigent
une nouvelle construction de tous leurs consommateurs.

## Ce qui reste

Les **86 obligations T1–T6/C01–C40/M01–M28/W01–W12 +8 principes =94** ne sont
pas toutes fermées. Les limites détaillées du corpus restent la référence.

1. Reconstruire/relinker les consommateurs CPU après le patch privé intégré,
   reconstruire le vrai Native CUDA/sept bindings, puis exécuter les cas
   scientifiques GPU et leur non-régression. Tester les autres backends et
   dimensions à la révision correspondante ; ne rien hériter de builds anciens.
2. Compléter les autres producteurs Field V2 : RHS de chemin,
   solve spatial retenu, transport principal, résultats couplés multibloc et
   boucles remappées. La publication diffusive réelle est désormais raccordée,
   avec sa frontière allocation/écriture et son ticket SSA ; les autres
   réalisations doivent recevoir leur propre raccordement réel et collectif.
3. Compléter la lecture d’un Field résolu caché sous DerivedAux avec un vrai
   FieldContext, ses droits/versions/représentations/point, pas en supprimant le
   refus. Exercer les nouvelles physiques **et méthodes** via Python public,
   écrites par un non-auteur, sans recettes centrales spécifiques.
4. Ajouter les réceptions manquantes : faute de préparation des bindings,
   ownership/rangs vides des nouveaux cas AMR, rollback de l’AcceptedSnapshot
   englobant, regrid/history/cache pertinents réellement peuplés, convergence
   spatiale/temporelle, coûts à calcul comparable et campagnes distribuées.
5. Rafraîchir la matrice par exigence, les exemples et les preuves de chaque
   backend ; compléter le catalogue de durées Python, diagnostiquer les autres
   shards et recevoir la CI GitHub sur le SHA final. Cette PR n’est pas déclarée
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

Les nouveaux cas publics se reproduisent avec le même pilote du vrai paquet
installé. Sur ce poste, `FI_PROVIDER=tcp` est nécessaire au MPICH configuré ;
les préfixes des dépendances du build doivent être conservés dans l’environnement.
Les sorties doivent avoir un chemin neuf, hors du dossier du prototype :

```sh
rtk proxy env -u PYTHONPATH -u PYTHONOPTIMIZE PYTHONDONTWRITEBYTECODE=1 \
  FI_PROVIDER=tcp POPS_NATIVE_DIM=2 OMP_NUM_THREADS=2 OMP_PROC_BIND=false \
  python docs/development/api_040/run_installed_checks.py \
  --output /tmp/pops-public-diffusion-new \
  --test tests/python/integration/runtime/test_public_diffusion_field_predictor.py
rtk proxy env -u PYTHONPATH -u PYTHONOPTIMIZE PYTHONDONTWRITEBYTECODE=1 \
  FI_PROVIDER=tcp POPS_NATIVE_DIM=2 OMP_NUM_THREADS=2 OMP_PROC_BIND=false \
  python docs/development/api_040/run_installed_checks.py \
  --output /tmp/pops-late-refusal-new \
  --test 'tests/python/integration/runtime/test_public_diffusion_field_late_refusal.py::test_public_late_refusal_same_artifact_control[ssprk2]'
```

Pour MPI2, le pilote d’identité et pytest doivent s’exécuter dans le même
processus sur chaque rang. Le [reçu MPI2](evidence/fixed_core_methods_53fd_20261008/mpi2-independent.json)
pointe le `run_rank.py` réellement exécuté, ses deux identités et ses sorties.
Le pilote MPI du dépôt possède aussi ce mode worker en processus ; la commande
suivante est la recette du dépôt, distincte du lanceur exact déjà reçu :

```sh
rtk proxy env -u PYTHONPATH -u PYTHONOPTIMIZE PYTHONDONTWRITEBYTECODE=1 \
  FI_PROVIDER=tcp python docs/development/api_040/run_installed_mpi_checks.py \
  --output /tmp/pops-late-refusal-mpi-new --ranks 2 --dimension 2 --threads 2 \
  --test 'tests/python/integration/runtime/test_public_diffusion_field_late_refusal.py::test_public_late_refusal_same_artifact_control[ssprk2]'
```

Le pilote série lance pytest dans un sous-processus ; le lancer tel quel sous
MPICH ne reproduit pas le worker MPI en processus.

Pour reproduire le lot211 contrôles de sources après activation de conda pops,
depuis ce checkout, choisir des noms XML neufs. Le bootstrap sélectionne
volontairement le frontend Source ; ces commandes ne qualifient pas Native :

```sh
rtk proxy env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 python -c '
import sys, pathlib, importlib.util
sys.path.insert(0, str(pathlib.Path.cwd() / "python"))
print("Source pops origin:", importlib.util.find_spec("pops").origin, flush=True)
import pytest
raise SystemExit(pytest.main([
    "-q",
    "tests/python/architecture/test_amr_program_support_parity.py",
    "tests/python/architecture/test_final_nd_amr_consumers.py",
    "tests/python/architecture/test_ci_cpp_include_impact.py",
    "tests/python/architecture/test_ci_impacted_selection.py",
    "--junitxml=/tmp/pops-architecture-owned-new.xml",
]))'
rtk proxy env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 python -c '
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd() / "python"))
import pytest
raise SystemExit(pytest.main([
    "-q",
    "tests/python/architecture/test_automatic_reflux_balance_fence.py",
    "tests/python/architecture/test_automatic_projection_balance_fence.py",
    "tests/python/architecture/test_final_nd_state_consumers.py",
    "tests/python/architecture/test_prepared_reflux_runtime_execution_fence.py",
    "tests/python/architecture/test_program_only_temporal_facades.py",
    "tests/python/architecture/test_cpp_suite_registration.py",
    "--junitxml=/tmp/pops-architecture-dependent-new.xml",
]))'
```
