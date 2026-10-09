# PoPS 0.4.0 : avancement reçu du 9 octobre 2026

[PR681](https://github.com/wolf75222/PoPS/pull/681) reste un brouillon. Le nouveau code est poussé jusqu'à `e27d065a5eb7c086ced60b751a2468f729361ff0`. Les **94 obligations**, 86 lignes T/C/M/W et huit principes, restent ouvertes. Le [corpus](corpus.json) conserve toutes les équations et les mappings antérieurs.

La physique et la méthode d'un nouvel article doivent s'écrire en équations et compositions Python du langage typé, puis devenir elles-mêmes le C++ exécuté. Le cœur figé ne choisit, remplace ou refuse une expression d'après une identité de modèle, d'article ou de formule. Les identités authentifient les propriétaires, la provenance et les caches. Le témoin ci-dessous apporte une réception bornée ; la preuve complète frontend→binaire reste ouverte.

## Code intégré

Le contrat [StageProviderRead v1](stageproviderread_v1.md) dérive l'autorité des champs du vrai Program et des plans de résolution enregistrés. La fabrique de production transmet ces plans automatiquement ; le scientifique ne recompose pas une seconde chaîne technique. La preuve SSP reste conditionnelle à la prémisse Forward Euler et aux coefficients convexes, sans theorem ajouté de positivité, entropie ou convergence.

Un non-auteur a écrit une [bibliothèque Python publique](../../../examples/migration/scientific/api040_stage_fields_library.py) et un [script scientifique linéaire](../../../examples/migration/scientific/api040_three_stage_screened_fields.py). Il assemble deux champs écrantés, trois lectures/résolutions de stade, SSPRK3 et un donneur figé, sans recette de physique dans le cœur. Les tests publics couvrent le cas, le renommage et une modification de coefficient Python.

Trois composants C++/Kokkos/AMR utilisent désormais des foncteurs dont le type peut être compilé par NVCC : transfert partitionné, coefficients du solveur tensoriel et copie/mise à zéro AMR. La contre-revue conserve exactement les corps mathématiques, les plages, les réductions, l'ordre flottant et les gardes. La CI conserve son budget de 35 minutes et sa couverture ; une partition supplémentaire accueille les deux nouveaux fichiers, avec renommage valide de la fixture.

## Identités réellement construites

| Élément | Identité |
|---|---|
| Source du build CPU Dim2 | `8c1f55701cab7c60b3519bd343e3b4c1f193570f` |
| Source des corrections CI et du dernier test public | `e27d065a5eb7c086ced60b751a2468f729361ff0` |
| Native CPU installé | `cee947c25c771f8c5a155e5f4cb990de308de14fe77e9f55c654e3b3837dee6f` |
| SDK | `087fead68988afda3c730e232f15b58ded9a07a1e4387b212c33248ac7661626` |
| Cœur conservateur, schéma2, 1221 fichiers dont bindings | `95d0ccaa1e7aa39f9cbb01d539c4e31a5a7e6e143207b088fd9c2730482963f5` |
| Wheel du build Dim2 | `f9dca3938c50a7f8541f4a4a7bd3868c05904da235ada053d0e8e757af368afb` |

Le build officiel a clos en **100,26 s**, avec 24 actions C++ Ninja observées ; le nombre de processus frontend n'est pas déterminé. La réception physique et scientifique indépendante réauthentifie 1175 fichiers Source/install/wheel et 384 headers. AppleClang21, Kokkos5.2 OpenMP/Serial, MPICH4.1.2, double et Dim2 sont les backends reçus. Les cinq changements après8c1 concernent CI et fixture ; le cœur et le SDK sont byte exacts. Le Native n'est jamais réétiqueté comme un nouveau build e27. Un défaut de métadonnées du collecteur initial est conservé : cinq lignes `physical_before` contiennent le hash d'après build, alors que les vrais hashes du gel `pre-build.json` prouvent les cinq changements.

## Exécutions reçues

| Exécution | Résultat réel | Limite |
|---|---|---|
| Architecture + SSP + StageProviderRead + propriété temporelle | 941 tests : 939 PASS et deux échecs CI de catalogue/capacité ; Native bloqué | Négatif conservé |
| Correction ciblée CI intégrée | 56 PASS, zéro échec/erreur/skip, 18,39 s | Local Source, pas CI GitHub |
| Cas représentatif CPU world1 | Capture réelle close0, 35,31 s ; oracle et réception non-auteur acceptés | Un pas, Uniform Dim2 |
| Renommage valide | Quatorze grandeurs canoniques identiques bit à bit | Deux offsets sérialisés CP de chaînes diffèrent légitimement |
| Décroissance Python 11/10 | Oracle inchangé dans sa méthode ; différence d'état final `6,3484e-5`, coefficient présent dans le C++ exécuté | Pas de convergence ou théorème global |
| Trois tests publics corrigés | **3 PASS**, zéro échec/erreur/skip, 100,91 s ; 42 grandeurs, 27 historiques et 432 tableaux CP reçus indépendamment | Le premier cohort2PASS/1refus de grammaire reste conservé |
| C++ Release/O3 world1 | **26 tests CTest PASS**, zéro échec/erreur/skip | Trois cibles AMR/provider/publication/nullspace |
| C++ MPI2 | **Trois groupes PASS**, dix cas GTest par rang, vingt exécutions cas-rang | Transfert composite, nullspace et interface conservative ; pas MPI2 du nouveau cas scientifique |

Le cas résout `(-Delta+3)phi=q+d` et `(-Delta+5)psi=q-(3/4)d`, puis `q_t=(3/20)Delta(q)-(4/5)q+(1/5)phi+(2/5)psi`, avec `d_t=0`. Il exécute les trois stades SSPRK3 depuis leurs propres états. Le donneur reste identique bit à bit ; les deux champs résidents sont ceux du dernier stade réellement exécuté, de temps1/2, et non ceux d'un maximum de temps. L'erreur maximale de l'état est `6,67e-16`, celle des champs `2,70e-14`, et le résidu physique indépendant `5,87e-12`, sous les budgets gelés avant Native. Aucune tolérance n'a été ajustée après mesure. Sept captures volontairement fausses sont refusées par le lecteur indépendant.

Les [reçus retenus](evidence/current_8c1_cee_20261009/manifest.json) et leur [publication](evidence/current_8c1_cee_20261009/publication.json) sont des copies byte exactes. Les fichiers binaires, tableaux et checkpoints complets restent aux chemins originaux authentifiés dans les reçus. Les anciennes [preuves4017](evidence/current_4017_939_20261009/manifest.json), leurs négatifs et le [précédent document](evidence/current_8c1_cee_20261009/history/STATUS-before-stage-fields.md) sont préservés ; aucun PASS ancien ne ferme une campagne du nouveau cœur.

## ROMEO et CI

Tous les travaux récents ROMEO sont dans des namespaces privés mode700 du scratch ; aucun nouveau fichier n'est placé dans le projet partagé. Le build GPU R6 **738213** a réellement exécuté NVCC puis échoué sur 16 diagnostics d'accès privé dans deux unités de traduction. Le dépendant **738214** a été annulé sans exécution. Ces logs ont motivé les trois correctifs C++ désormais intégrés et reçus CPU. Le gel GPU échoué reste Source988/SDKde2 ; il ne qualifie ni8c1 ni e27. Une ancienne sonde CUDA Managed GH200 est positive ; aucun Native PoPS ni PDE GPU n'est encore reçu.

R7 est préparé pour construire Sourcee27 en réutilisant les dépendances privées R6. Son admission doit authentifier une nouvelle source, le quota réel, les dépendances et le loader CUDA ; son build sera séparé de l'admission runtime après build. Aucun job futur, résultat ou hash Native GPU n'est inventé. Home et travaux antérieurs sont conservés.

La CI distante Source397 contient de vrais échecs. Deux logs de shards sont retenus : 52 reçoit8FAIL/50PASS, 53 reçoit5FAIL/167PASS. Ils montrent six familles de défauts : profils GPU sur Serial, M27 Dim1 mal routé, découverte des headers Kokkos, compilation du composant test M19 sans flags MPI, identité du communicateur W11, et prémisse SSP d'un consommateur State. Ils restent distincts des 56 contrôles locaux de catalogue. Aucun succès de la porte d'agrégation ni état merge-ready n'est revendiqué ; la CI requise doit être examinée sur le SHA final poussé.

## Reproduire

Dans un worktree neuf, utiliser `scripts/setup_env.sh` une fois. Le worktree actuel a déjà été préparé. Activer l'environnement `pops`, conserver sa vraie installation Kokkos et utiliser les headers installés du wheel authentifié. Le [journal d'invocation](evidence/current_8c1_cee_20261009/build/actual-official-invocation.sh) donne les commandes exactes du build reçu. Le parcours officiel reste :

```sh
scripts/build_python.sh --dim 2 --mpi --wheel-dir /chemin/prive/wheel-neuf -- \
  -C build.verbose=true -C cmake.define.CMAKE_EXPORT_COMPILE_COMMANDS=ON
env -u PYTHONPATH -u POPS_INCLUDE -u PYTHONOPTIMIZE -u PYTEST_ADDOPTS \
  PYTHONDONTWRITEBYTECODE=1 FI_PROVIDER=tcp OMP_NUM_THREADS=2 OMP_PROC_BIND=false \
  POPS_NATIVE_DIM=2 POPS_REQUIRE_NATIVE_TESTS=1 \
  python -m pytest -q tests/python/integration/runtime/test_public_three_stage_screened_fields.py
```

La fixture authentifie le vrai `pops` installé, le Native et les headers, et conserve Program, C++, DSO, états, historiques et CP. Les [commandes C++](evidence/current_8c1_cee_20261009/cpp/actual-commands.sh) utilisent le preset du dépôt `mpi`, Release/O3, six vraies cibles et un build séparé du cache Python. Le premier filtre CTest n'a exécuté que les trois groupes MPI ; le [second](evidence/current_8c1_cee_20261009/cpp/world1/actual-commands.sh) sélectionne les labels des trois cibles restantes et reçoit26tests. Pour un autre chemin, adapter seulement les chemins de sortie et de Kokkos/Conda ; recalculer les pins, sans réutiliser les hashes ci-dessus comme preuves d'un rebuild.

## Travail restant et prochaine tranche

L'injection d'échec après la troisième résolution et acceptation de champ est préparée et contre-revue en Source. Sa version2 exige le hash du vrai C++ normal reçu ; son adaptation au compilateur réel reste séparée et doit être reçue avant exécution. Elle doit encore démontrer rollback, retry et restart sur tous les états, historiques, auxiliaires, clocks et checkpoints, sans whitelist masquant des différences. Aucun run de cette injection n'est déclaré ici.

Il reste à construire et recevoir Dim1/3, rejouer les anciennes suites affectées sur ce cœur, qualifier le nouveau cas sous MPI2/GPU et AMR multilevel/EB/regrid/convergence, corriger et recevoir toute la CI requise, persister les rapports CG et les clocks de publication par stade, établir la preuve complète graph→binaire et mesurer les coûts à calcul comparable. Les tableaux T1–T6/C01–C40/M01–M28/W01–W12 et principes1.1–1.8 du [suivi compact](mission_reception_compact_20261001.md) gardent leurs obligations complètes. La mission demeure en cours.
