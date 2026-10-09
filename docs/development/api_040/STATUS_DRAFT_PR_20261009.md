# PoPS 0.4.0 : état reçu du 9 octobre 2026

[PR681](https://github.com/wolf75222/PoPS/pull/681) reste un brouillon. Les 86 obligations T/C/M/W et les huit principes restent ouverts. Ce document rapporte les exécutions du gel **4017/939/acf0** et distingue les corrections CI suivantes. Les anciennes preuves gardent leurs identités et leurs limites.

Le critère de généricité est opérationnel : la physique **et la méthode** d'un nouvel article sont écrites en équations et compositions Python du langage typé ; leur traduction devient le C++ exécuté. Le cœur figé ne choisit, remplace ou refuse aucune expression en fonction d'une identité d'article, de modèle ou de formule. Chaque opération a une définition mathématique indépendante des modèles. Les identités servent à authentifier la provenance, les propriétaires et les caches. Les dépendances et l'autorité des champs doivent découler des expressions résolues, sans second assemblage technique demandé au scientifique.

## Dernier lot réellement construit et installé

| Élément | Identité reçue |
|---|---|
| Source effectivement construit | `4017d29d9adc066e2ab64d2eed3b99f1caa8f63c` |
| Native installé Dim2 | `939d331c0f28150429266516e0e6b57fe84986f48d336290ec613a7c4dd85941` |
| SDK, 384 headers signés | `acf0eeca77739e7d4dd8a7da161af75667a3f898959e7464961ea243db9107fa` |
| Cœur conservateur, schéma2, 1220 fichiers dont bindings | `743bc354455285b4828e6d80039949317c31f63ac6fc3c5432ae2897487594a8` |
| Wheel | `0d3c66b61ce95291dcd59c23d183a06b058d8fdcf473bf8ec58a69f16e32f696` |

Le build officiel incrémental réussit en **96,04 s**. Il mesure 24 objets et 24 actions de sortie C++ Ninja, sans réemploi d'objet qualifié. Le nombre réel d'exécutions du frontend C++ reste indéterminé : ni les octets identiques ni les statistiques agrégées du cache ne suffisent à le compter. La réception physique indépendante vérifie 40 entrées du lien, 1174 fichiers Source/install/wheel, les 1179 payloads immuables de la wheel et les 384 headers. Le lien et l'installation ont les mêmes 17 sections MachO de code, constantes et données ainsi que le même UUID ; les 111 octets différents sont limités aux métadonnées de chargement et à la signature.

Backend exécuté : AppleClang21, Kokkos5.2 OpenMP/Serial, MPICH4.1.2, Dim2 double, HDF5 parallèle1.14.3. OMP2 est configuré ; aucun débit, occupation des threads ou facteur d'accélération n'en est déduit. Les tests chargent le vrai paquet installé dans l'environnement `pops`, sans `PYTHONPATH`, avec les identités enregistrées avant et après. Les lecteurs indépendants ne chargent pas Native.

## Exécutions et réceptions scientifiques actuelles

| Campagne sur 4017/939/acf0 | Résultat réel | Réception indépendante et limites |
|---|---|---|
| Publication Field gardée puis diffusion | 3/3 PASS world1 | Trois variantes, dont renommage et coefficients modifiés. Les 192 valeurs restent bit identiques après renommage ; les coefficients modifiés changent le C++ et le calcul sauvegardé. Références Fraction/FFT et 42 pins de sorties contrôlés ; pas de preuve complète graph→binaire déduite. |
| PDE IMEX nonautonome original | 1/1 PASS, 30,90 s | `U=84/25`, `F=144/25`, tableaux et checkpoint reçus sous les bornes originales. `Y=48/25` reste une référence sans tableau intermédiaire sauvegardé. |
| Méthodes publiques, refus tardif/retry et capacité Uniform | 10/10 PASS, 349,54 s | Euler/SSPRK2, mutation, renommage, trois journaux de rollback exact et historique peuplé reçus. Un carrier UniformCP9 dans chacun des trois états ; 315 fichiers, 25 DSO et neuf closures CP authentifiés. Aucun transfert de couverture multi-layout. |
| Retry original et huit cas natifs de reprise | 9/9 PASS, 615,76 s | 23 checkpoints et 1121 payloads reçus, 22 NPZ scientifiques, 28 NPY et 20 DSO. Restart, presets, refus, histories et réemploi du même sceau vérifiés. Trois digest corrompus et six archives rescellées mais sémantiquement invalides caractérisés. |
| MPI2 représentatif, PDE original et Euler | 2/2 PASS par rang, 76,10 s | Identités des deux rangs, références IMEX et égalités des sorties reçues. |
| MPI2 méthodes/refus/capacité/publication gardée | **13/13 PASS par rang**, zéro erreur/skip, 586,53 s | Les deux gardes de norme qui échouaient sur 988 passent après reconstruction, avec les mêmes seuils. 97 comparaisons rank/world1 bit identiques dans le périmètre enregistré ; commandes et données actuelles reçues. |
| Publication couplée, selector world1 identique à MPI2 | 1/1 PASS, 38,82 s | Addendum indépendant : 24 paires d'états, 48 paires de membres numériques CP bit identiques et six paires clock/ring égales, sur trois phases. Les archives CP complètes et le framing des diagnostics ne sont pas déclarés identiques entre un et deux rangs. |
| Inputs statiques : variable, diagonal linéaire, diagonal lisse × Euler/SSPRK2 | **6/6 PASS** | Six états et 9216 lignes d'échanges reçus avec une référence scalaire FV indépendante ; défaut maximal `1,78e-15`. Slots/stades et absence de publication Field vérifiés. CPU world1 uniquement ; C++ Program et IR complet non conservés comme preuve portable. |

Les runs world1 de publication gardée, du PDE original, des méthodes publiques et des reprises totalisent **23 exécutions PASS**. Le cas couplé supplémentaire, les six Inputs statiques et le test du cache compilé ci-dessous ont des reçus distincts ; ce ne sont pas une unique suite monolithique. Les durées sont celles des invocations enregistrées, sans comparaison de performance entre campagnes différentes.

La norme AMR répliquée compte chaque sample fini une seule fois. Le retry donne maintenant `0,074104… < 0,1`, contre `0,104799…` avec duplication à deux rangs ; la première acceptation de l'historique peuplé donne `28,124317… < 30`, contre `39,773791…`. La seconde tentative reste refusée sous sa garde20. Le contrat `program-norm2-ownership@1` conserve la composante0, les samples actifs EB de chaque niveau, le coarse couvert par le fine et la garde nonfinie de toutes les copies. `dot_all` et une norme composite pondérée sont d'autres opérations.

Les [reçus actuels et leur manifeste](evidence/current_4017_939_20261009/manifest.json) conservent commandes, XML, identités, contre-réceptions, inventaires et chemins des données brutes. Le [lot988 historique](evidence/current_988_f8d_20261009/manifest.json) reste conservé, y compris ses deux échecs MPI ; aucun ancien PASS n'est réétiqueté. La première baseline4017 a refusé le lancement du modèle parce que `Kokkos_ROOT` était absent. Le premier collecteur Input a échoué après calcul en appelant une propriété comme une fonction. Ces deux négatifs de préparation restent séparés des exécutions corrigées et reçues.

Les gaps sont explicites : commandes Program absentes dans le lot retry/Native8 malgré C++ et DSO conservés ; pas d'états finaux macro/adaptive/directguard sauvegardés dans certaines fixtures ; pas de reload frais des checkpoints couplés, de preuve complète graph→binaire, de résidu elliptique composite indépendant, ni de convergence multilevel/EB par ces seuls témoins. Un PASS Native/JUnit et une réception mathématique des tableaux sauvegardés sont deux faits distincts.

## Code intégré et corrections suivantes

La tranche intégrée modifie les véritables modules Python/C++ de PoPS. Les propriétaires bas niveau remplacent les imports circulaires ; 72 corps numériques restent identiques à l'AST. `model.expression_language@1` réexporte les mêmes objets mathématiques. La preuve SSP V2 utilise les Inputs résolus réellement consommés et leur contenu actuel, sans sélectionner une méthode par son nom. Les contrats `accepted-static-provider-read@1`, `public-library-alias@1` et `program-norm2-ownership@1` sont versionnés. ABI13 demeure inchangée ; les consommateurs du SDK modifié ont été reconstruits pour la cohorte reçue.

La cohorte Source du commit4017 reçoit **904 PASS**, zéro échec/erreur/skip : 873 d'architecture, 25 SSP et six de propriété temporelle. Native y est effectivement bloqué : quatre tentatives d'import refusées, aucune extension ni module chargé. Cette validation d'intégration par l'auteur ne remplace pas les contre-revues indépendantes. Le négatif903/1 du commit7da et sa correction d'inventaire strict sont conservés dans les [preuves de l'intégration précédente](evidence/reviewed_integration_20261009/source-7da-inventory-correction-manifest.json).

| Correction après les runs4017 | Code et preuve actuels | Réception restante |
|---|---|---|
| Route CI du test de cache | La wheel téléchargée est installée dans un environnement privé, authentifiée et vérifiée avant pytest, via le même outil que les shards. Contrôles négatifs indépendants reçus ; **un test réel PASS** sur la wheel actuelle, backend MPI world1, 75,79 s. | Exécution GitHub LinuxSerial sur le SHA poussé suivant ; pas de succès CI déduit du Mac MPI. |
| Routage des deux fichiers 1D | `native_dimensions.json` affecte dimension1 aux cas de cisaillement périodique et au cas auto-cohérent, d'après leurs limites/cells/états réellement résolus. Les six autres fichiers du shard gardent dimension2. | Vrai Native Dim1 reconstruit et science reçue ; aucune géométrie modifiée. |
| Champs dépendant de l'état du stade et preuve SSP | Contre-exemple Source réel : le Field résolu dépend du stade courant et ne satisfait pas la prémisse des Inputs statiques. Sol prépare l'autorité générique par publication/RHS dans les vrais plans ProgramField et la factory du compilateur. | Patch versionné, cas nouveau multiFields/multistades et négatifs, revue non-auteur, intégration, reconstruction et réception native. Aucune recette dédiée au cas d'origine ni théorème d'ordre/positivité générale revendiqué. |
| C++ M2, AMR1/8 et PendingProof | Correctifs intégrés et contre-revues ciblées CPU1/MPI2 conservées ; la fixture PendingProof vérifie la vraie identité canonique, le motif exact de refus, huit fautes et rollback. | Non-régression C++ complète pertinente sur le build final O3, avec contributions distribuées, rangs vides et copies nonfinies. Les précédentes TU ciblées O0/objets988 ne valent pas réception O3 de tout le build4017. |

La validation ciblée des deux corrections CI reçoit **60/60 PASS**, zéro erreur/skip, 17,56 s, avec Native bloqué. Elle teste le diff de travail exact au-dessus de4017 ; les preuves de route, installation et backend y sont des fixtures Source et subprocess simulés. Les deux builders scientifiques réels se résolvent en dimension1 sans compilation. Cette validation ne revendique ni Native ni succès GitHub ([reçu](evidence/current_4017_939_20261009/ci-focused-source/current-ci-route-source-receipt.json)).

Les commits documentaires et de routage CI suivants ne changent pas les identités des runs du tableau. Aucun inventaire, corps de test, shard ou seuil scientifique n'est supprimé pour rendre la CI positive. La CI distante4017 contient des échecs réels ; la route de cache et les deux erreurs de dimension ont été diagnostiquées sur ses logs. Aucun état merge-ready ni succès d'agrégation requis n'est revendiqué.

## ROMEO et GPU

Tous les nouveaux fichiers sont dans `/gpfs/scratch/rmdraux/PoPS-final-full-native-cuda-dim2-gh200-um-98804c68-20261009`, privé mode700. Le home en dépassement n'est pas utilisé pour les installations ; aucun nouveau travail n'est écrit dans l'espace projet commun. Le compilateur déjà présent dans le sous-dossier personnel projet est seulement lu. Quota réel, grâce et croissance sont contrôlés avant chaque phase. Les prévisions de croissance ne sont ni des consommations mesurées ni un plafond personnel demandé par l'utilisateur.

Le GPU reste un gel distinct : **Source988 / SDKde2 / coreb756**. Les échecs de préparation738094,738116 et738155 sont conservés ; leurs jobs dépendants ont été effectivement annulés sans exécution. La reprise738173 reçoit une vraie sonde CUDA : pointeur `cudaMemoryTypeManaged`, accès Host/Cuda, lancement et synchronisation réussis, GH200 compute9.0, runtime/header12060. Elle échoue ensuite avant l'entrée du build officiel PoPS parce que `Kokkos_ROOT` n'est pas exporté ;738174 est annulé, jamais lancé. Ce PASS est une allocation et un microkernel CUDA, sans qualification de PDE ni de Native PoPS.

La reprise additive R5 exporte `Kokkos_ROOT` et `POPS_KOKKOS_ROOT` vers le propre préfixe Kokkos avant les contrôles et les processus enfants. Sa revue Source indépendante préserve le loader CUDA12.6 en premier, CMake, RPATH, finally, quota, sélecteurs et équations. Après admission physique fraîche, Root a réellement soumis **738183** pour le build et **738184** avec dépendance afterok. Le build738183 échoue réellement après29 s : le script officiel cherche par défaut un environnement nommé `pops`, alors que le préfixe privé actif est `pops_final_cuda_dim2`. Le journal de commande et le marker d'entrée sont conservés ; aucune action C++/Ninja ni admission Native GPU n'est reçue. Le dépendant738184 a été effectivement annulé, jamais lancé. Une préparation additive R6 doit sélectionner explicitement l'environnement privé, vérifier le vrai CLI dans Slurm et authentifier ce refus antérieur, sans effacer le marker ni refaire le setup. Environnement, setup officiel effectué une fois, Kokkos et sonde existants sont conservés. **Aucun Native GPU, kernel PDE GPU ou résultat scientifique GPU n'est encore reçu**, et le gel988 ne qualifie pas4017.

## Reproduire les périmètres reçus

Dans le worktree de la révision construite, utiliser le véritable environnement `pops`. Le setup officiel a déjà été fait une fois pour le worktree actuel ; dans un autre worktree, effectuer `scripts/setup_env.sh` une fois. L'invocation de construction exacte, ses variables, commandes et wheel sont dans [le reçu du build4017](evidence/current_4017_939_20261009/build/invocation.json). Le parcours officiel est `scripts/build_python.sh --dim 2 --mpi` ; reconstruire, installer et authentifier les identités avant d'interpréter un run d'une nouvelle révision.

Les commandes suivantes supposent l'environnement `pops` activé et sa vraie installation Kokkos. `POPS_INCLUDE` désigne les headers de ce checkout ; la preuve d'installation doit confirmer qu'ils correspondent au SDK du Native. Les pins ci-dessous sont ceux du build4017 reçu, pas des valeurs à conserver après changement de code. Changer de révision exige un nouveau reçu et les pins du nouveau build.

```sh
env -u PYTHONPATH -u PYTHONOPTIMIZE -u PYTEST_ADDOPTS \
  Kokkos_ROOT="$CONDA_PREFIX" POPS_KOKKOS_ROOT="$CONDA_PREFIX" \
  POPS_INCLUDE="$PWD/include" \
  FI_PROVIDER=tcp OMP_NUM_THREADS=2 OMP_PROC_BIND=false \
  POPS_NATIVE_DIM=2 POPS_REQUIRE_NATIVE_TESTS=1 \
  POPS_GUARDED_FIELD_SOURCE_SHA=4017d29d9adc066e2ab64d2eed3b99f1caa8f63c \
  POPS_GUARDED_FIELD_NATIVE_SHA=939d331c0f28150429266516e0e6b57fe84986f48d336290ec613a7c4dd85941 \
  python docs/development/api_040/run_installed_checks.py \
  --output /chemin/prive/nouveau-recu \
  --test tests/python/integration/runtime/test_imex_nonautonomous_field.py::test_public_nonautonomous_imex_field_reads_explicit_time
```

```sh
env -u PYTHONPATH -u PYTHONOPTIMIZE -u PYTEST_ADDOPTS \
  Kokkos_ROOT="$CONDA_PREFIX" POPS_KOKKOS_ROOT="$CONDA_PREFIX" \
  POPS_INCLUDE="$PWD/include" FI_PROVIDER=tcp POPS_REQUIRE_NATIVE_TESTS=1 \
  POPS_GUARDED_FIELD_SOURCE_SHA=4017d29d9adc066e2ab64d2eed3b99f1caa8f63c \
  POPS_GUARDED_FIELD_NATIVE_SHA=939d331c0f28150429266516e0e6b57fe84986f48d336290ec613a7c4dd85941 \
  python docs/development/api_040/run_installed_mpi_checks.py \
  --output /chemin/prive/nouveau-recu-mpi2 --ranks 2 --dimension 2 --threads 2 \
  --test tests/python/integration/runtime/test_public_diffusion_field_predictor.py \
  --test tests/python/integration/runtime/test_public_diffusion_field_late_refusal.py \
  --test tests/python/integration/runtime/test_guarded_field_diffusion_publication.py \
  --test 'tests/python/integration/runtime/test_field_publication_instances_runtime.py::test_installed_three_instance_solved_provider_reads[cells0-False-False]'
```

Les six Inputs statiques ont leurs [invocations et références de réception](evidence/current_4017_939_20261009/static-inputs/remaining5/invocation.json), leur [runner reçu](evidence/current_4017_939_20261009/static-inputs/reviewed-runner/run_native_reception.py) et leur [contre-réception indépendante](evidence/current_4017_939_20261009/independent/static-inputs/independent-reception-ready.json). Ils passent uniquement après authentification du build officiel terminé. Les scripts et l'admission ROMEO R5 sont conservés dans [le même manifeste](evidence/current_4017_939_20261009/manifest.json) ; ils conservent les chemins privés et le gel988, sans lancer de CUDA sur le nœud de connexion.

La réception complète Dim1/3, GPU, multilevel AMR/EB, regrid/restart/convergence et les coûts à calcul comparable restent à établir pour chaque ligne concernée. Les tableaux T1–T6, C01–C40, M01–M28, W01–W12 et principes1.1–1.8, leurs équations et mappings sont préservés dans `mission_reception_compact_20261001.md` et `corpus.json`. Ces résultats bornés ne démontrent pas à eux seuls la généricité de toutes les opérations du langage.
