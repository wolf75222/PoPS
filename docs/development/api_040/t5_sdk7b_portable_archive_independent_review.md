# Réception indépendante de l’archive historique T5 SDK7b

L’archive authentique est reçue depuis ses vrais fichiers sauvegardés, avec le
sceau externe fourni par root. Une copie froide donne exactement le même résultat
en interdisant, par audit Python des ouvertures, les 40 fichiers natifs originaux
et les 49 fichiers de l’archive donneuse. Le processus refuse tout import `pops`.
Cette réception offline ne lance ni simulation ni compilation.

## Autorités et inventaire

- Archive root : `outputs/native-t5-sdk7b-portable-archive-20260930`.
- SHA externe du manifeste :
  `854974b3447561d22f8c767821cf8ce85eb1c1a8232e2ec3f8f0e56141579a4b`.
- SHA des pins originaux fournis par root :
  `7a6d07c39e434b256e08792ff6c89d991c8525068eb0df0c0a37e43c12521bc2`.
- Source native historique : `634cba3511fef957e65a21fcae3ab257f164b360`.
- SHA de l’extension native historique :
  `8a217a0fff5a131729a842dd222e2e08346034eb53b4f461fc17fc8cc291b563`.
- Signature SDK historique :
  `7b503163f41cc33040c296679e3b3530a9716c91ecfe4fdce6f69dca6f07dad0`.
- Oracle scientifique historique inchangé : SHA
  `21041d64af5378a35b5a8717cc0300b6a399d95cc2c9209e66d66e5e2ed54f89`.
- Fixture historique archivée, identique à `git show 634cba35:...` : SHA
  `db1d5153eeb59c8e15e80f22837a9889c30e08f37dc99dab96a1e422fc12962d`.

Le manifeste fixe 48 fichiers ; il constitue lui-même le 49e fichier présent.
Le total réel est **1 084 339 octets**, sans lien symbolique. Les 40 fichiers
originaux, dont l’identité et les 39 fichiers de phases, conservent exactement
leurs SHA. La transformation des pins ne change que leurs 40 chemins absolus en
chemins relatifs. Le checker exige une bijection de ces 40 chemins, un inventaire
exact des fichiers et répertoires, des chemins canoniques sans alias, des fichiers
réguliers, des tailles bornées et les sceaux historiques indépendants. La limite
de 64 MiB est un budget de ce checker de réception, pas une limite de production.

Les chemins absolus qui subsistent à l’intérieur des reçus sauvegardés sont des
provenances historiques. La copie froide prouve qu’ils ne sont pas utilisés pour
lire les états natifs. Les pins relatifs sont la seule route de lecture de ces
40 fichiers. L’archive originale a été revérifiée inchangée après les injections.

## Science historique conservée

Les cas sont restart N8, restart N16 et retry N8, avec `dt=.01`, `gamma=.3`,
`q0=.7`, tolérance `3e-13`. L’oracle indépendant reçoit la capture pré-pas de q,
la réaction originale `U*=U*(1-gamma*q*dt)`, le transport FV et le flux extérieur
droit unique qui alimente q. Il reçoit également les frames, occurrences,
consommations, mesures, états de restart et refus de publication du retry.

Sur les sept pas recalculés, incluant les reprises, les maxima sont :

| Vérification | Erreur maximale |
| --- | ---: |
| Champ physique | 2.220446049250313e-16 |
| Bilan d’inventaire physique | 8.304988641238964e-17 |
| Bilan du ledger natif | 1.951563910473908e-18 |

Les quantités N8 sont `.7118500624999999` puis `.7236549032181168` ; N16,
`.7119124312499999` puis `.7237795053013649`. Retry N8 donne le même premier
pas. Les ledgers ont respectivement 32 et 64 incidences. Le résultat recalculé
est identique au reçu scientifique root, hormis le SHA du document de pins
désormais relatif. Aucun état positif n’a été fabriqué.

## Contre-protocoles et correctif du checker

Les huit copies négatives sont reçues par l’ancien `verify_archive.py` et
refusées par le nouveau checker. Deux conservent le **sceau externe root exact** :
un état remplacé par un lien symbolique interne vers des octets identiques, et
un répertoire vide ajouté. Ils démontrent deux écarts réels à l’inventaire exact
et à l’interdiction de liens.

Les six autres utilisent exclusivement des sceaux de contre-modèles, clairement
marqués comme ne constituant pas une autorité root : doublon de mapping, ligne
de mapping inutilisée, chemin portable absolu, doublon d’inventaire, clé JSON
dupliquée et source historique modifiée. Ils testent les gardes structurels après
rescellement explicite ; ils ne contournent pas le sceau externe root original.

Le checker corrigé est `tests/review/sol61_t5_portable_archive_checker.py`. Il
accepte l’archive existante sans la modifier. Pour l’intégrer comme nouveau
`verify_archive.py` dans une prochaine archive, root devra produire un **nouveau**
sceau externe du manifeste, parce que les octets du checker auront changé ; les
40 pins natifs et l’oracle historique restent inchangés. Le checker suppose un
arbre stable pendant la lecture ; il ne constitue pas un protocole de lecture
concurrente d’un arbre hostile mutable.

## Douze contre-modèles scientifiques rescellés

Les neuf injections antérieures sont rejouées sur les vrais fichiers portables :
q erroné dans la réaction, flux droit altéré, signe de scale omis, axes transposés,
trace intérieure utilisée comme trace extérieure, durée déplacée d’un ULP,
horloge étrangère, état restauré modifié et état du refus modifié. Le cas de
transposition échoue au contrat de forme après vérification des pins externes ;
les autres atteignent les contrôles d’authentification interne ou scientifiques.

Trois injections supplémentaires changent les états, q et les flux de toutes les
faces x de façon cohérente, puis rescellent les tableaux checkpoint, CBOR,
digests de reprise, reçus, wire et les 40 pins de fichiers : source lue sur l’état
après transport, q lu à l’endpoint, source originale évaluée deux fois. Le
snapshot est authentifié avant le refus scientifique dans les trois cas. Leurs
bilans internes ont des résidus de `1.1558e-16`, `1.8757e-16` et `1.8323e-16` :
un bilan seul pourrait les accepter. L’équation indépendante les refuse tous
avec `reaction/candidate q capture or FV field update mismatch`.

Ces copies altérées sont uniquement des négatifs explicitement rescellés par le
harness ; elles ne sont ni des reçus natifs positifs ni de nouveaux pins owner.
Le résultat détaillé est `t5_sdk7b_portable_archive_independent_receipt.json`.

## Réparation de la fixture historique extensible

Le reçu root SDK20d reste historique : 362 PASS et un échec de test, dû à la
comparaison de tout le fichier courant avec une fixture antérieure à l’extension
`physical_global`. Le test réparé authentifie maintenant les trois sources
historiques par des SHA fixes sur `634cba35`. Pour la fixture courante, il exige
les AST exacts de `_case`, `_oracle`, `_receipt_check` et des constantes
`DT/GAMMA/Q0/TOL`, ainsi que N8/N16, le retry `.5` et les pins externes.
L’ajout d’une sélection de réception est permis ; une modification de physique
est explicitement refusée par un contre-test. La fonction réparée passe contre
MAIN `b2959ba2df0b88a2af2a5dfa6e6a5b2042a49da9` en lecture seule, sans réécrire MAIN. Cela ne qualifie pas une
nouvelle exécution native de `physical_global=True`.

## Commandes et portée

Dans le checkout privé `work/PoPS-sol61-t5-archive-reception` :

```sh
T5_PY=/Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python
T5_ARCHIVE=/Users/romaindespoulain/Documents/Codex/2026-09-28/dans-le-d-p-t-pops/outputs/native-t5-sdk7b-portable-archive-20260930
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM "$T5_PY" -I tests/review/sol61_t5_portable_archive_checker.py --archive "$T5_ARCHIVE" --manifest-sha256 854974b3447561d22f8c767821cf8ce85eb1c1a8232e2ec3f8f0e56141579a4b --output outputs/sol61-t5-archive-strict-positive.json
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM "$T5_PY" -I tests/review/sol61_t5_portable_archive_countermodels.py --archive "$T5_ARCHIVE" --countermodels-dir outputs/sol61-t5-portable-negative-countermodels-v3 --output outputs/sol61-t5-portable-countermodel-reception-v3.json
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM "$T5_PY" -I -c 'import sys;from pathlib import Path;sys.path.insert(0,str(Path.cwd()));import pytest;raise SystemExit(pytest.main(["-q","--tb=short","tests/review/test_sol61_t5_archive_protocol.py","tests/review/test_sol61_integral_feedback_offline_contract.py"]))'
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM "$T5_PY" -I -c 'import runpy;from pathlib import Path;ns=runpy.run_path("tests/review/test_sol61_integral_feedback_offline_contract.py");f=ns["test_source_fixture_contract_is_frozen_and_files_need_external_state_pins"];f.__globals__["ROOT"]=Path("/Users/romaindespoulain/Documents/Codex/2026-09-28/dans-le-d-p-t-pops/work/PoPS");f();print("Read-only current MAIN: PASS")'
```

Résultats : positif réel reçu, copie froide reçue avec 89 fichiers originaux
interdits et import PoPS interdit, 8 refus protocole, 12 refus scientifiques,
47 tests source/protocole PASS. Les répertoires négatifs doivent être vides avant
un replay ; un nouveau nom conserve les preuves antérieures. Aucun build,
installation, JIT, modification d’en-tête, SDK ou MAIN n’a été réalisé ici.

La portée reste le témoin historique Program `q*S`, Uniform Dim2, un rang. Une
bibliothèque compilée MPI ne fournit pas de qualification MPI2. Cette revue ne
qualifie ni direct physical GlobalState, ni AMR, ni ALE, ni un SDK ultérieur.
