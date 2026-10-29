# ROM-16 : récupération et registre des preuves, 29 octobre 2026

Ce lot conserve les sources disponibles et rend les manques vérifiables. La fiche
[ROM-16](https://linear.app/romain7522/issue/ROM-16) reste ouverte : les patches
post-b6 ne sont pas récupérés et la CI native historique de b6 reste rouge.

Autorité : bloc « Intention et autorité » de la fiche,
[plan des sessions](https://linear.app/romain7522/document/plan-des-sessions-planmd-spec-03-ee15cf7fcf6e),
[intentions de Romain](https://linear.app/romain7522/document/intentions-de-romain-citations-exactes-4c506ef09c3f)
et [spec 0.3](https://linear.app/romain7522/document/spec-03-api-python-de-pops-cible-normative-2a2f962f9c42),
K6, MG5 à MG7, CF14, CF15 et §18, phase 0. Le manifeste conserve l'empreinte
SHA256 du texte normatif téléchargé, 4 346 lignes.

## Sources et branches

| Lot | Branche / SHA | Base | Fichiers | CI et classement |
| --- | --- | --- | --- | --- |
| b6 conservé | `rom16/recovered-b6-20261029`, `b6e64c1d23f5c0565c41674d262ca26b401e8339` | Parent `d17c0535d0d0baf0b5c7f6f06d7e842b9760276b` ; série depuis `77acb67747c18666e31c24f6b51ffbe290496bb8` | [36 fichiers contre le parent](b6_files_vs_parent.txt), [1 143 contre la base de série](b6_files_vs_integration_base.txt) | Récupéré, poussé, non intégré ; CI native historique rouge |
| Lots post-b6 | Aucune branche source authentifiée | Inconnue | 12 chemins de patches et 2 livraisons, détaillés ci-dessous | Originaux introuvables dans le périmètre audité ; CI non établie |
| v15d2 | Archive documentaire, sans nouvelle branche source | Livraison documentaire du 14 octobre, distincte de b6 du 16 octobre | [Inventaire et empreintes](receipts.json) | Manifeste, handoff et reçu lisibles ; reçus individuels en partie non authentifiés |
| Présent manifeste | `rom16/recovery-manifest-20261029`, tête exacte publiée dans le commentaire Linear | `bb9aea0cc56fabe6b003bfbee5daee28d38ac7e7` | Ce dossier uniquement | PR documentaire séparée ; résultats exacts publiés dans Linear |

La branche originale `codex/pops-api-v02-local` et
[PR #687](https://github.com/wolf75222/PoPS/pull/687) pointaient déjà sur b6.
La nouvelle branche de conservation garde exactement ce commit, son arbre et
ses parents. Les 16 commits de la série, leurs parents, identités et fichiers
sont conservés dans [b6_commits.json](b6_commits.json). Le `merge-base` de b6 et
du master distant courant est `bb9aea0cc56fabe6b003bfbee5daee28d38ac7e7`.

Les auteurs et committers des 16 commits sont `desp0042` ; le contrôle des
identités et trailers ne trouve aucun nom interdit dans cette série ni dans les
20 derniers commits de b6. Aucun commit n'a été réécrit. Le nouveau commit utilise
l'identité Git existante de Romain et les push utilisent son compte gh existant.
Voir [identity_audit.json](identity_audit.json).

## CI existante, conservée dans son périmètre

La [CI de b6](https://github.com/wolf75222/PoPS/actions/runs/38840872542)
contient 199 jobs : 154 succès, 43 échecs, un annulé et un ignoré.
Le gate agrégé échoue. Les workflows
[Docs](https://github.com/wolf75222/PoPS/actions/runs/38840872541) et
[no-ai-authors](https://github.com/wolf75222/PoPS/actions/runs/38840872390)
ont réussi sur ce SHA. [b6_ci.json](b6_ci.json) conserve les 45 jobs sans succès,
leurs liens et les observations tirées de trois logs ciblés.

Constats bornés : erreur Pyright dans `traversal.py:42`, échecs MPI de propriété
du runtime et de double initialisation Kokkos, timeout existant de 100 minutes
du shard Python 0. Les causes des 39 autres shards échoués n'ont pas été relues.
La correction de ces familles relève de B03. Ce lot ne relance pas la campagne.

Une CI verte de cette PR documentaire ne constitue aucun nouveau niveau K6
E, G, C, R ou S pour b6. Un reçu natif reste lié à sa source, son artefact, sa
configuration et sa cible d'origine. L'archive v15d2 déclare elle-même
`new_scientific_or_native_execution=false`, `gpu_qualified=false` et
`api_02_universal_certification=false`. Les succès, échecs et remplacements
historiques restent distingués dans [receipts.json](receipts.json).

## Manques post-b6 et limite de la recherche

Les refs et reflogs locaux ne donnent aucun descendant de b6. Le contrôle des
objets inaccessibles réussit : 69 commits, aucun descendant de b6. Voir
[object_audit.json](object_audit.json). Le dépôt local est shallow ; ce contrôle
porte sur les objets disponibles, pas sur une sauvegarde externe inconnue.

Le cache `/Users/romaindespoulain/Library/Caches/PoPS-api-v02` est absent ; son
ancien worktree est prunable. L'autre checkout du dossier de projet est à
`77acb677`. Son état dirty reste indéterminé : ses commandes de statut bloquent.
Il a été laissé intact.

Les chemins suivants sont attestés par le journal historique, sous
`completion-v02/` dans le cache absent. Chaque original est absent. Leur base
exacte est inconnue ; aucune application sur b6 n'est supposée correcte.

| Fichier relatif au cache | Ligne du journal |
| --- | --- |
| `models/child-field-increment022/freeze-uniform0001/changes.patch` | 78237 |
| `backend/gc-census022/changes.patch` | 78417 |
| `methods/boundary-context-fixture021b/changes.patch` | 78417 |
| `models/child-field-increment022/test-hygiene0001/changes.patch` | 78417 |
| `backend/ci-child022/changes.patch` | 78546 |
| `backend/ci-child022/delivery.json` | 78546 |
| `root/compiled-eb022/freeze0001/changes.patch` | 79298 |
| `root/compiled-eb022/freeze0001/delivery.json` | 79298 |
| `models/child-field-increment022/amr-cps/native-path/freeze0002/changes.patch` | 79672 |
| `magnetic/path-producer-window022/changes.patch` | 79672 |
| `magnetic/path-producer-window022/changes-r2.patch` | 79773 |
| `models/child-field-increment022/amr-cps/freeze-coherent0001/changes.patch` | 79927 |
| `magnetic/deferred-spatial-flux019/freeze-amr-cont7-0001/changes.patch` | 80572 |
| `root/mapping-reconciliation023/changes.patch` | 80614 |

[receipts.json](receipts.json) conserve le chemin et le SHA256 du journal, les
lignes citées, les empreintes historiques disponibles et les prochaines actions.
La présence de fragments de commandes et de reçus dans un journal ne garantit
pas la restitution complète d'un patch. Aucun patch n'a été reconstruit ici.

Les anciennes archives source et plusieurs reçus v15d2 sont marqués dataless
par macOS. Leur taille et leurs empreintes déclarées sont inventoriées ; leur
contenu n'a pas été authentifié. Leur seule présence ne ferme aucun écart.

## Vérification et reprise

[manifest.json](manifest.json) fixe les sources, chemins, commandes, limites et
prochaines actions. [SHA256SUMS.txt](SHA256SUMS.txt) couvre les inventaires.

Vérification documentaire avec le script du dépôt, dans l'environnement `pops` :

```sh
env -u PYTHONPATH PATH=/Users/romaindespoulain/miniforge3/envs/pops/bin:$PATH bash scripts/build_docs.sh
```

Aucun moteur, test, seuil, skip, timeout ou assertion n'est modifié. CF15 ne se
déclenche pas : ce lot n'ajoute ni cas ni mécanisme. Aucune purge, aucun reset,
aucun écrasement de branche existante, aucun force-push.

Prochain lot borné : récupérer un original manquant depuis une sauvegarde
indépendante ou un enregistrement complet, établir sa base et vérifier ses
empreintes avant sa branche source. Les échecs CI de PR #687 restent visibles
pour B03 ; B01 reste ouverte jusqu'à réception des sources manquantes.
