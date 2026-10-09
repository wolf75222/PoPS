# PoPS 0.4.0 : état reçu du 9 octobre 2026

[PR681](https://github.com/wolf75222/PoPS/pull/681) reste un brouillon. Les 86 obligations T/C/M/W et les huit principes restent ouverts. Ce document sépare le dernier lot exécuté des corrections suivantes. Une empreinte ou une compilation ne ferme pas une réception scientifique.

Le critère de généricité est opérationnel : les équations **et la méthode** d'un nouvel article doivent être des compositions Python du langage typé ; leur traduction devient le C++ exécuté. Le cœur ne choisit aucun calcul par identité d'article, nom, formule ou recette reconnue. Les identités authentifient la provenance et les caches. Les opérations ont une définition mathématique indépendante des modèles.

## Dernier lot réellement construit

| Élément | Identité reçue |
|---|---|
| Source | `98804c6849c9966e05b5397126bab983a5fa0227` |
| Native installé Dim2 | `f8d9281232039f3ed3f92b7a14ae6530aa18baf41257a3c99a9629a50af61f64` |
| SDK, 384 headers signés | `de2b5ef24a9fae48ab143d4200aab2885c9400dcd43712fa74928f5f2c4e954a` |
| Cœur conservateur, schéma2, 1213 fichiers dont bindings | `b756f21226ad09e2108026bc9c3bfce9a251d1480e8a81e0e733db0c0845871a` |
| Wheel | `8ba01454f883750be892a8bbeaef9c2b82d16491f919a5bfd7420116c61e1272` |

Le build officiel incrémental réussit en 231,88 s : 24 actions C++ observées, 17 runtime et sept bindings. Quatre objets changent d'octets ; 20 restent identiques **après compilation**. La contre-réception rehash 1167 fichiers Source/installés/wheel, les objets et le lien natif ; la wheel contient 1169 membres `pops` physiquement authentifiés. Backend réellement exécuté : AppleClang21, Kokkos5.2 OpenMP/Serial, MPICH4.1.2, Dim2 double, HDF5 parallèle1.14.3. OMP2 est configuré ; aucun débit ou facteur d'accélération n'en est déduit.

## Exécutions scientifiques et injections d'échec

| Campagne sur ce lot | Résultat réel | Réception et limites |
|---|---|---|
| Publication Field gardée puis diffusion, trois variantes | 3/3 PASS world1, zéro erreur/skip | Deux modèles publics, deux propriétaires, un Program. Valeurs Field et état final vérifiées indépendamment, checkpoint exact. Renommer tous les noms préserve les 192 valeurs bit à bit ; changer les coefficients change le C++ et les 192 résultats concernés. |
| PDE IMEX nonautonome original | 1/1 PASS, 29,91 s | `U=84/25`, `F=144/25`, bornes originales ; 64 valeurs et checkpoint reçus. `Y=48/25` reste une référence sans mesure directe. |
| Retry original et huit cas natifs de reprise | 9/9 PASS, 551,37 s | États/Fields/horloges/historiques/sauvegardes reçus ; retry republie le même seal. Les corruptions rejetées sont conservées séparément. |
| Méthodes publiques, refus tardif/retry, capacité Uniform | 10/10 PASS, 348,64 s | Euler/SSPRK2, mutation et renommage, restaurations byte exact, historique peuplé. Capacité : **un** carrier UniformCP9 direct dans chacun des trois états, 66 payloads par checkpoint ; pas de transfert de la précédente couverture multi-layout. |
| MPI2 représentatif, PDE original et Euler | 2/2 PASS par rang, 72,52 s | Rangs0/1 et identité du vrai paquet reçus ; U/F et méthode Euler reçus indépendamment. |
| MPI2 méthodes/refus/capacité/publication gardée | 11/13 PASS par rang, 2 FAIL, zéro erreur/skip, 523,84 s | Retry automatique et première acceptation de l'historique peuplé échouent sous les gardes originales. Les résultats positifs et les deux échecs restent distincts : 13 checkpoints physiques uniques, 850 payloads rehashés et égalité des tableaux entre rangs reçus indépendamment. Aucun succès de continuation du cas peuplé n’est déduit. |

Les deux échecs MPI ont un contre-exemple natif : un maillage coarse répliqué donne `norm2=8*sqrt(2)` à deux rangs, contre8 pour une unique copie physique. Les mesures sur les vrais cas donnent0,104799… après demi-pas contre0,074104… attendu pour la garde0,1, et39,77379… contre28,12432… pour la garde30. La correction de propriété MPI est intégrée et reçue en contre-revue Source ; le rebuild et la réception scientifique sur le nouveau SDK restent à effectuer ; aucune tolérance n'est changée. Le contrat historique de `norm2` porte sur la composante0 et la somme des samples actifs EB de chaque niveau, sans exclure le coarse couvert par le fine ; une norme composite pondérée représente un autre calcul. `dot_all` constitue une opération distincte.

Les [reçus copiés et leur manifeste](evidence/current_988_f8d_20261009/manifest.json) conservent les identités, commandes, XML, contre-réceptions et emplacements des données brutes. Le lecteur scientifique ne charge pas Native. Les premières hypothèses erronées de lecteurs (encodage d'identités binaires opaques, chemins pytest-current, nombre de carriers) restent archivées dans leurs emplacements d'origine ; elles ne sont pas attribuées au producteur.

## Code intégré, réception native suivante

| Lot | Avancement | Vérification restante |
|---|---|---|
| Architecture et autorité TemporalTau | Intégrés dans les vrais modules ; revue indépendante et composition Source900 PASS | Paquet reconstruit, cohorte Source du commit final et CI de ce nouveau SHA. |
| Preuve SSP des Inputs statiques | V2 intégrée et reçue sur Source, authentifie le contenu résolu actuel et les remplacements réellement exécutés ; contrat `accepted-static-provider-read@1` | Réception native des six constructions après installation. Le défaut V1 d'identité mise en cache reste un négatif archivé. |
| Quatre preuves C++M2 | Corps réels ajoutés, 22/22 C++ locaux, quatre sélectionnés passent sur chaque rangMPI2 ; contre-revue indépendante | Non-régression du nouveau cœur ; aucune substitution des corps par une liste textuelle. |
| Tests C++AMR1/8 | Deux corrections intégrées ; contre-revue indépendante : quatre commandes CPU1/MPI2, six cas-rangs PASS | Réexécution sur le build final ; ratio2 avant bootstrap, budget avant premier advance. Préparation lazy du moteur/ledger dans la transaction initiale conservée. |
| CI depuis un paquet réellement installé | Route intégrée : wheel à partir du vrai Native, installation privée par dimension et suppression de PYTHONPATH ; trois sciences et26 unités sur installation passent dans la candidate | LinuxSerial/ELF et contrôles GitHub sur le SHA final. Les inventaires/shards ne sont pas réduits. |
| Norme AMR répliquée | Correction générique intégrée et reçue par un non-auteur ; composante0, chaque niveau et non-finitude conservés | Rebuild SDK/Native/DSO et relance des deux cas en échec, des contributions distribuées/rangs vides/nonfinies. |


Cette tranche modifie 57 chemins de code, tests et contrats. Les propriétaires bas niveau remplacent les imports circulaires ; 72 corps numériques sont inchangés à l'AST. Le langage public `model.expression_language@1` réexporte les mêmes objets mathématiques. La preuve SSP utilise les plans résolus réellement consommés, sans sélectionner une méthode par son nom. Les contrats `accepted-static-provider-read@1`, `public-library-alias@1` et `program-norm2-ownership@1` décrivent les obligations et limites. ABI13 demeure inchangée ; le SDK modifié impose la reconstruction des consommateurs.

La fixture C++ `PendingProofSurvivesPublicationAndRejectsRankLocalDamage` partage maintenant l'identité canonique réellement installée entre route d'état, frontière et ledger. L'ancienne identité étrangère cachait le ledger après suppression des preuves : le refus série n'était pas testé. Les huit fautes et le rollback sont conservés, le motif exact de refus est ajouté ; trois cas-rangs CPU1/MPI2 passent dans la candidate, reçus par un non-auteur. Ce correctif de fixture ne change ni production ni SDK ([preuves](evidence/reviewed_integration_20261009/pending-proof-manifest.json)).

Les [reçus de cette intégration](evidence/reviewed_integration_20261009/manifest.json) distinguent validation de l'auteur, contre-revues et exécutions C++ ciblées. La composition Source externe reçoit 900 PASS, zéro échec/erreur/skip ; la revue indépendante du routage CI ajoute huit contrôles Source avec Native effectivement bloqué. Les cinq secondes de deux nouvelles lignes d'inventaire sont des estimations non mesurées en CI ; les anciennes durées et l'ordre JSON sont préservés. Les binaires C++ de preuve utilisent les objets Source988 gelés et des TU candidates ; ils ne qualifient pas le futur Native complet. En particulier, la TU norm2 est compilée à O0, pas un nouveau build O3 complet.

Le Native installé reste celui du tableau Source988 jusqu'au prochain build officiel. La présente intégration ne lui attribue ni nouveaux résultats scientifiques ni nouveau succès de CI. Le gel Source du commit, son cœur mesuré, le nouveau SDK et le Native installé seront enregistrés après cette reconstruction.

## ROMEO et backends

Le job738094 échoue **avant compilation** sur une hypothèse incorrecte d'absence de Mamba dans Miniforge ; son dépendant738095 est annulé, jamais exécuté. Les journaux et l'environnement créé sont préservés. La reprise additive738116 passe le vrai Conda CLI, le setup officiel effectué une fois et compile Kokkos ainsi que la sonde CUDA, puis échoue127 en120s au chargement de `libcudart.so.12`. Le dépendant738117 est effectivement annulé, sans lancement. Le fichier CUDA existe et son SHA est authentifié ; son répertoire manque au chemin de bibliothèques de la sonde. La prochaine reprise additive doit réutiliser environnement/setup/Kokkos existants, exécuter la sonde dans une allocation Slurm puis construire le vrai Native PoPS. La sélection du vrai Conda CLI a été vérifiée avant le clonage réel ; Mamba installé est enregistré sans être employé par cette reprise.

Tous les nouveaux fichiers ROMEO sont dans `/gpfs/scratch/rmdraux/PoPS-final-full-native-cuda-dim2-gh200-um-98804c68-20261009`, namespace privé mode700. Le home en dépassement n'est pas utilisé pour l'installation ; aucun nouveau travail n'est écrit dans l'espace projet commun. Quota réel, période de grâce et croissance sont contrôlés à chaque phase. Profil demandé : GH200/HOPPER90, CUDA12.6, Kokkos5.2.1 avec mémoire unifiée. Ce gel Source988 n'authentifie pas la future correction de norme. **Aucun Native GPU, kernel PDE GPU ou résultat scientifique GPU n'est encore reçu.**

La CI Source988 présente des échecs réels ; aucun succès d'agrégation requis ni état merge-ready n'est revendiqué. Dim1/3, GPU, refinement/restart/convergence de chaque nouveau cas et coûts à calcul comparable restent à recevoir selon la ligne concernée. Ces acquis finis ne démontrent pas à eux seuls la généricité pour toutes les opérations ou articles.

## Reproduire

Utiliser le véritable environnement `pops`, sans `PYTHONPATH`, dans le checkout de la révision souhaitée. Le setup officiel a déjà été fait une fois pour ce worktree ; dans un autre worktree, effectuer son setup officiel une fois. Construire avec `scripts/build_python.sh --dim 2 --mpi`, puis authentifier la wheel installée et Native avant d'interpréter les résultats. Les reçus cités ci-dessus contiennent les commandes exactes du gel988 ; ils ne doivent pas être réétiquetés pour une autre révision.

```sh
env -u PYTHONPATH -u PYTHONOPTIMIZE -u PYTEST_ADDOPTS \
  FI_PROVIDER=tcp OMP_NUM_THREADS=2 OMP_PROC_BIND=false \
  POPS_NATIVE_DIM=2 POPS_REQUIRE_NATIVE_TESTS=1 \
  python docs/development/api_040/run_installed_checks.py \
  --output /chemin/prive/nouveau-recu \
  --test tests/python/integration/runtime/test_imex_nonautonomous_field.py::test_public_nonautonomous_imex_field_reads_explicit_time
```

Pour la campagne MPI, après avoir enregistré les deux empreintes attendues `POPS_GUARDED_FIELD_SOURCE_SHA` et `POPS_GUARDED_FIELD_NATIVE_SHA` correspondant au **nouveau build réellement installé** :

```sh
env -u PYTHONPATH -u PYTHONOPTIMIZE -u PYTEST_ADDOPTS \
  FI_PROVIDER=tcp POPS_REQUIRE_NATIVE_TESTS=1 \
  python docs/development/api_040/run_installed_mpi_checks.py \
  --output /chemin/prive/nouveau-recu-mpi2 --ranks 2 --dimension 2 --threads 2 \
  --test tests/python/integration/runtime/test_public_diffusion_field_predictor.py \
  --test tests/python/integration/runtime/test_public_diffusion_field_late_refusal.py \
  --test tests/python/integration/runtime/test_guarded_field_diffusion_publication.py \
  --test 'tests/python/integration/runtime/test_field_publication_instances_runtime.py::test_installed_three_instance_solved_provider_reads[cells0-False-False]'
```

Les familles T1–T6, C01–C40, M01–M28, W01–W12 et principes1.1–1.8 gardent leur tableau et leurs équations dans `mission_reception_compact_20261001.md` et `corpus.json`. La prochaine tranche doit fermer ses défauts prouvés, reconstruire le vrai paquet, vérifier les références indépendantes, puis étendre la réception selon ce tableau.
