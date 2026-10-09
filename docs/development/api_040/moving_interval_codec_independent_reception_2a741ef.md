# Réception indépendante de source du codec ALE corrigé

Cible exacte : `0116827` + `2df5090126c8ae3d8b812eef1647db89ee431338` + `2a741effb06a11b0b351bef2629a9d09e8ce7bb8`. Le codec de cette cible a le SHA256 `8a7687baae58f9e6480733c3f04b5026f605c386a7ea0751c148950dbc47d703`. Checkout exclusif `PoPS-sol61-ale-codec-fixed-review`, branche `codex/api040-sol61-ale-codec-fixed-review`. Aucune source de production, installation ou environnement partagé n'est modifié par cette réception. L'ancien rapport et les probes contre `011` restent conservés dans leur checkout ; leurs résultats ne sont pas réutilisés comme preuve du correctif.

## Résultat et portée

Les contre-modèles démontrés sur `011` sont fermés par les nouveaux prédicats de source : mesures anciennes et nouvelles exactement dérivées des coordonnées, produits/flux relatifs/résidu/échelle finis avant Reynolds, support et identité source des occurrences comparés, clock/tolérance explicitement déclarés et vérifiés. Le codec force un ledger imbriqué POPSEX02 qui conserve ces métadonnées même sans IntegralState.

Le reçu est aussi confronté aux autorités temporelles extérieures : `physical_time + dt == accepted_time` exactement, sans tolérance, et `tick == macro_step - 1`. Le probe teste le cas valide, le décalage d'un ULP, deux mauvais macro-steps et le temps infini. La façade réelle refuse treize représentations mal typées/nonfinies/nonscalaires de `t` ou `macro_step` avant l'appel de validation temporelle native. Pour ce test de façade, ce validateur est explicitement remplacé par un spy : aucun payload simulé n'est présenté comme une vraie image POPSEX03 reçue nativement.

La lecture des anciens formats reste compatible : après normalisation des deux seuls ajouts (`force_extended=false` et son OR), le serializer de ledger est exactement identique au blob `011`. Les bytes par défaut suivent donc le même parcours ancien ; seul le codec03 demande explicitement l'extension. Le reader AMR legacy est byte-identique. Neuf appels sans géométrie mobile retournent les mêmes budgets Python que `011`.

Le supplément de capacité ajoute maintenant `cells*(2+3*ncomp)*clock_ticks*record_bytes`. Cinq probes de la vraie fonction de budget, à 8 cellules, couvrent les largeurs 1/2/16/256/1024. Pour 256 composantes la capacité passe à 4728544 bytes, au-dessus de la borne minimale de ledger 468176 ; à 1024, 18822880 contre 1869008. Cela ferme le contre-exemple de budget précédent. Comparer à cette borne inférieure ne constitue pas une réception d'un checkpoint réellement émis à toutes ces largeurs ni une preuve globale d'allocation maximale.

## Injections C++ préparées pour réception centrale

Le target existant `test_program_runtime` inclut `moving_interval_codec_independent_review.inc`. Les quatre injections précédentes ont été adaptées : la fixture déclare `clock_authority="review-clock"` et `geometry_tolerance_authority` avant de copier la déclaration installée. Chaque test exige un **roundtrip valide par ASSERT_NO_THROW avant son injection**. Un échec dû à une autorité absente ne peut donc pas faire passer le test.

- Mesure falsifiée : autorité, déclaration et reçu ont tous la tolérance 2 dès le cas valide ; seuls volumes et densités changent ensuite, avec Reynolds et ledger nuls exacts. Le refus doit venir de l'identité des mesures, pas d'une tolérance discordante.
- Overflow : le roundtrip valide précède le passage à des champs encore finis dont le produit de Reynolds déborde ; endpoints et mesures restent cohérents.
- Topologie/allocation : roundtrip valide, puis prototype à nombre de composantes étranger et longueur UINT64_MAX du ledger imbriqué.
- Support : déclaration d'IntegralState et roundtrip valide, puis support extérieur inventé sur les records internes, avec clés et montants inchangés.

Deux tests supplémentaires préparent la réception des points end-time/macro-step et du staging. Le premier teste le cas temporel valide puis mauvais temps/cursors et une clock installée étrangère. Le second utilise la vraie façade C++ System en Dim1 : capture acceptée à profondeur zéro, refus sans reçu terminal à profondeur un même avec staging explicite, refus à profondeur deux et sous restart, puis égalité de l'image après rollback. La réussite d'un vrai staging terminal à profondeur un reste à recevoir via le test natif existant de l'auteur, qui l'exerce après une mise à jour mobile réelle. Ce test System est explicitement indisponible en Dim2/3 ; les probes directs du codec utilisent des champs rangés Dim1, sans prétendre qualifier un provider mobile Dim2/3.

**Aucun de ces tests C++ n'a été compilé ni exécuté par cette revue.** Ils sont prêts pour le build/réception centrale, pas marqués verts. Aucun résultat MPI, GPU, AMR mobile ou nouvel end-to-end natif n'est revendiqué.

## Staging, restore et rollback

La table des 32 combinaisons profondeur 0/1/2/3, staging explicite, restart et reçu terminal correspond au garde réel de `2a` : profondeur zéro hors restart autorisée ; profondeur un seulement avec staging explicite et reçu terminal ; profondeurs supérieures et restart refusés. L'image doit ensuite satisfaire la même autorité temporelle et les mêmes équations du codec. Cette table est un calcul de prédicats sur source, pas une exécution des transactions natives.

La façade Uniform restaure les champs, puis le clock avant le mailbox, ce qui permet au restore natif de confronter le reçu au clock réellement restauré. Le préflight Python prend au contraire les autorités depuis le payload, avant publication. Les champs physiques doivent toujours correspondre exactement à ceux encodés dans le codec03. Les snapshots de pas et d'installation conservent la moving map ; leur copie passe par les copies profondes des Fab. Les garanties MPI de métadonnées, propriétaires, faces partagées et refus collectif restent celles examinées dans le rapport `011` ; leur réception réelle n'a pas été répétée ici.

## Commandes et preuves effectuées

```sh
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python tests/review/sol61_ale_codec_2a741ef.py
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -I - <<'PY'
import sys, pathlib
root = pathlib.Path.cwd()
sys.path[:0] = [str(root / 'python'), str(root)]
import pops, pytest
assert pathlib.Path(pops.__file__).resolve() == root / 'python/pops/__init__.py'
raise SystemExit(pytest.main(['-q', 'tests/python/unit/runtime/test_continuation_transitions.py', '-k', 'mailbox or provisional or allocation_failure or exchange_capacity']))
PY
```

Probe de la cible source/host : succès, aucune dimension native sélectionnée. Façade ciblée : **8 succès, 15 désélectionnés**. Ruff et whitespace : succès. Aucune nouvelle vulnérabilité concrète restante n'a été démontrée dans ce lot borné ; les tests natifs C++ et les reprises réelles restent nécessaires sur le SHA central intégrant ces freezes. Le raccordement public Python SSA demeure distinct de cette revue de codec.
