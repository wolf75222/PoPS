# Diffusive IntegralState AMR : projection initiale aux faces physiques

Base : `b468a55f814f13bb5a6813c15d99adbb4b67326d`. Checkout exclusif `PoPS-sol61-diffusive-amr-diagnosis`, branche `codex/api040-sol61-diffusive-amr-diagnosis`. Aucun environnement partagé, build, installation ou JIT n'a été exécuté ici. Le correctif C++ nécessite une nouvelle compilation native et un SDK cohérent avant réception.

## Échec natif reçu et diagnostic sauvegardé

La réception installée Dim2 ABI5 rapporte trois succès et un échec parmi les quatre tests publics de diffusion : le cas AMR donne `right = 0.7000195904` au lieu de `0.70001`. Les critères restent inchangés (`2e-12`). Les preuves sont dans `outputs/installed-integral-transport-diffusion-public-authority-dim2-abi5-20260930/pytest-tmp/test_public_diffusive_integral1`, au même niveau workspace que `work`.

Le script indépendant `tests/review/sol61_diffusive_amr_initial_projection.py` lit ces vrais fichiers, vérifie le SHA du checkpoint et des images natives, décode les bytes `POPSEX02` et compare le décodage au reçu JSON. Il n'exécute pas de code natif. Les identités sont :

- artefact : `pops.artifact.v1:sha256:09c4b174ea59745a2f5f690e0b20175916a06556ae40395a811500b94d20958d` ;
- SDK/ABI Dim2 : `307f61570e511eae829151680b4dd2e24425907803f4ec78b2dbb32a19b82540|/usr/bin/clang++|c++20|dim=2` ;
- checkpoint accepté SHA256 : `a75c0a2199c26abe42f18409a158b6227acb5c75d8761dfacb0753c3695a105e` ;
- ledger natif SHA256 : `ca09f9d1d17aa6b46179f1a439ece299bf00f8224c7316e202a480f2707e1a24`.

La seule boîte fine est `[26,0]..[63,63]`, soit 2432 cellules fines valides et 416 cellules coarse actives. La face droite appartient entièrement au niveau fin ; la gauche reste coarse. Le niveau coarse initial reproduit exactement les moyennes du champ `1+x`. La restriction des valeurs fines vers les parents couverts conserve leurs moyennes à `2e-14`. L'intérieur fin reproduit aussi le champ affine. L'erreur de projection maximale `0.0078125` apparaît uniquement dans les deux enfants du dernier parent : l'état réellement lié contient `[1.984375, 1.984375]`, au lieu des moyennes fines `[1.9765625, 1.9921875]`.

| Face / niveau / sous-pas | Nombre | Flux natif | Durée | Montant |
| --- | ---: | ---: | ---: | ---: |
| gauche / coarse / 0 | 32 | 0.1 | 1e-4 | -1e-5 |
| droite / fine / 0 | 64 | 0.2 | 5e-5 | 1e-5 |
| droite / fine / 1 | 64 | 0.191808 | 5e-5 | 9.5904e-6 |

Les aires, durées, orientations, multiplicités et identités d'incidence sont correctes. Aucune face droite coarse couverte n'est présente. Une récurrence Forward Euler indépendante, partant de la dernière cellule fine sauvegardée et de la valeur physique du bord, retrouve exactement les flux `0.2` puis `0.191808`, donc le montant droit `1.95904e-5`. La masse composite vaut `1.5` initialement et `1.5000095904` après acceptation ; sa variation égale le montant total du ledger à `2e-14`. Le ledger et la conservation correspondent ainsi à l'état réellement initialisé. La projection affine perdue précède l'évaluation des échanges.

## Correctif générique

Le bootstrap `BindArray` utilise `prepare_regridded_state_transfer` avec la politique publique `StateTransfer()`, dont la prolongation est `ConservativeLinear`. Cette préparation utilisait un stencil centré sans qualifier les faces physiques. Le limiteur monotonized-central rencontrait le ghost physique copié au bord : une différence nulle annulait la pente et les deux enfants recevaient la moyenne parent.

La préparation de prolongation linéaire accepte maintenant une autorité `PhysicalParentBoundary` explicite, comme la route existante de fantômes coarse/fine. Au premier ou dernier parent physique, le même kernel prend la pente de son voisin intérieur. Le stencil requis et sa preuve collective utilisent le même calcul de boîte ; seuls les voisins extérieurs aux faces physiques déclarées sont retirés de cette boîte. Les voisins périodiques et intérieurs restent obligatoires. Une face physique à parent unique refuse la reconstruction linéaire faute de deux valeurs intérieures ; la route d'injection explicitement choisie reste disponible et inchangée.

Le runtime transmet la périodicité réelle de la configuration aux transferts de bootstrap, de remaillage des états et des historiques. Les sources auxiliaires explicitement injectées conservent leur politique. Le recouvrement antérieur, la restriction, la publication collective et le rollback sont conservés. Aucune équation, diffusivité, durée, identité de Rate, sélection d'échange, règle de couverture ou tolérance du bilan n'est modifiée. Aucun traitement spécifique au modèle `1+x` n'est ajouté au runtime.

La fixture native ajoute un contrôle de l'état initial réellement lié sur toutes les cellules fines valides, après sauvegarde du checkpoint initial. Cela localise immédiatement une régression de projection avant le calcul du bilan. Deux tests C++ supplémentaires couvrent reproduction affine et moyennes parentes en 1D/2D/3D, origines décalées, rapports anisotropes et deux composantes, puis refus des stencils périodiques/intérieurs absents et du parent physique unique. Ces tests C++ sont préparés pour l'intégrateur, sans claim d'exécution ici.

## Réception effectuée et limites

- Audit des vrais états/ledger sauvegardés : succès, nombres ci-dessus ; ce succès qualifie le diagnostic de l'ancien échec, pas la nouvelle correction native.
- Source/émission Python du contrat public : quatre succès, un test host C++ volontairement non exécuté conformément à la séparation des builds.
- Ruff et contrôle des whitespace : succès.
- Aucun test C++ compilé, aucun nouveau résultat natif AMR, MPI, EB, GPU ou Dim3 obtenu ici.

Commande d'audit, depuis ce checkout :

```sh
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python tests/review/sol61_diffusive_amr_initial_projection.py /Users/romaindespoulain/Documents/Codex/2026-09-28/dans-le-d-p-t-pops/outputs/installed-integral-transport-diffusion-public-authority-dim2-abi5-20260930/pytest-tmp/test_public_diffusive_integral1
```

Après rebuild et réception dans un nouveau répertoire, la même commande avec `--require-affine` exige les moyennes fines exactes. Sur les anciennes preuves, cette option échoue volontairement avec l'erreur de projection mesurée. L'intégrateur doit exécuter `test_nd_transfer`, les tests de fantômes préparés, puis les quatre tests publics de diffusion avec `POPS_REQUIRE_NATIVE_TESTS=1` sur le nouvel artefact installé. Les six tests transport/restart/refus déjà reçus par l'intégrateur restent des résultats antérieurs distincts ; ils ne qualifient pas ce nouveau correctif.
