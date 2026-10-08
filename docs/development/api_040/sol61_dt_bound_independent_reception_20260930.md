# Contre-réception indépendante dt_bound - 30 septembre 2026

Candidate exacte : `9c4209a5e5ecb05ac71a62f4d2c21b6e3acd4300`.
Checkout exclusif `PoPS-sol61-dt-bound-review`, branche
`codex/api040-sol61-dt-bound-review`. Source-only Python identifié par son
`pops.__file__`. Aucun changement de production, MAIN, header, SDK ou installation.

## Captures publiques reçues

Replay byte-identique du probe gelé
`a7ecd3ca47b636aa058514a382786b9d75c58e93` : **exit 0**, deux véritables
émissions publiques `validate → resolve → ProgramModelGraph → emit_cpp_program`
avec `ctx.dot_all`. L'ancien `u.n` principal ne produit plus `KeyError: 0` ;
le bloc query-only reçoit sa route et ne produit plus l'erreur de block-index map.
Le probe n'est pas dupliqué dans le présent commit.

La contre-suite indépendante vérifie quatre Program physiques déclarés via
Model/Case/numerics et le futur helper native : trois composantes distinctes,
permutation `(2,0,1)`, deux types de capture. Le bloc query-only est déclaré
avant fluid dans la Case, mais absent des updates. Le Program conserve fluid=0
et ajoute query_only=1 ; le bound utilise exactement son owner et n'émet aucun
commit. IR et hash ne sont pas modifiés par la résolution/lowering.

Les captures temporelles directes et transitives (`value` de point next, puis
contraction de cette candidate) refusent sans publier de bound et restaurent
l'IR complet et le compteur d'identifiants. Un comparateur indépendant importe
la source du parent exact depuis `git archive` dans un subprocess séparé :
Program legacy et unbounded conservent IR complet, semantic complet, hash et
C++ byte-identiques. Le test ne se contente pas de hashes attendus auteur.

La closure itérative authentifie les noeuds, vérifie la whitelist préexistante
sur les dépendances, et reste une vue de lowering. Aucun nouveau schéma ni
identité DAG n'est introduit par cette correction. La sélection de tests auteur
reçoit également déduplication, captures partagées et longue chaîne de captures.

## Lacune readonly supplémentaire, conservée comme FAIL

Le builder public suivant passe actuellement :

```python
def builder(P, cfl):
    P.commit(passive.next, side_update)  # candidate top-level déjà créée
    return cfl * P.hmin()
```

`commit` ajoute une entrée globale dans le Program sans créer de noeud dans le
sub-DAG. `readonly_dt_bound_nodes` ne voit donc que cfl/hmin/scalar_op, accepte,
et le Program publie un second commit. Le test indépendant
`test_callback_cannot_hide_a_commit_outside_its_scalar_dependencies` exige le
refus et la restauration exacte ; sur 9c4209a5 il **échoue : DID NOT RAISE**.
Le contre-cas est transmis à l'auteur/intégrateur. Cette lacune semble historique,
pas introduite par 9c ; elle interdit cependant de déclarer tous les callbacks
read-only reçus. Le présent lot ne modifie ni le code ni le résultat pour
transformer ce défaut en PASS.

Sélection exécutée : **38 PASS, 1 FAIL** en 12.51 s. Les deux captures réparées
sont donc reçues source/émission ; le refus de cette mutation non-SSA reste ouvert.

## Fixture native publique préparée, pas exécutée

`tests/python/integration/runtime/test_dt_bound_capture_independent_runtime.py`
contient trois tests collectés :

- Deux captures, chacune avec ordre normal et permutation de trois composantes.
  Vrai `pops.compile`, `pops.bind`, puis `pops.run` avec AdaptiveCFL. Le moteur
  évalue le bound native ; aucune requête publique de stabilité standalone
  n'existe actuellement sur RuntimeInstance et aucun executor/.so privé n'est
  utilisé pour la simuler.
- Oracle dense indépendant sur toutes les composantes et cellules 4×4 : fluid
  initial `(2,5,7)` donne dt `.25/1249`, query-only `(11,13,17)` donne `.25/9265`.
  Deux mises à jour acceptées avec facteurs distincts `(2,3,5)` détectent une
  capture ignorée, un owner ou composant zéro implicite, et une lecture stale
  après le premier pas. La query-only doit rester strictement intacte.
- Initials query-only avec NaN actif ou overflow fini `1e308` dans la troisième
  composante : le nouveau contrat dot_all doit lever avant toute publication.
  Les deux états, time=0, macro_step=0 et les cursors doivent rester identiques.
  Le fixture n'attrape pas une erreur de bind pour la déclarer reçue comme un
  refus du bound : la réception réelle devra distinguer ces chemins si bind
  refuse préalablement une initiale non finie.

Collect-only : **3 tests collectés** en 0.45 s ; Ruff PASS. Les quatre cas
normaux de cette fixture ont été authorés, validés, résolus et émis dans la
contre-suite source. Aucun `pops.compile/bind/run` native de cette fixture n'a
été exécuté ici ; l'intégrateur possède cette réception après reconstruction.

Limite historique explicitement conservée : System.cpp654–659 et
amr_system.cpp18034–18040 ignorent un résultat scalaire Program dt_bound <=0
ou non fini. Le refus NaN/overflow demandé ici provient de l'évaluation dot_all,
pas d'un consumer scalaire qui serait prétendu fail-closed. La décision de
changer/versionner ce consumer appartient au lot dédié de l'intégrateur.

## Commandes

```sh
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM PYTHONPATH=python /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python tests/review/sol61_dot_all_dt_bound_gap.py
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM PYTHONPATH=python /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest -q --tb=short tests/review/test_sol61_dt_bound_capture_independent.py tests/python/unit/codegen/test_dt_bound_capture_closure.py tests/python/unit/time/test_program_authoring_atomicity.py tests/python/unit/time/test_typed_provenance_guards.py
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM PYTHONPATH=python /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest --collect-only -q tests/python/integration/runtime/test_dt_bound_capture_independent_runtime.py
```

La deuxième commande conserve l'exit 1 du défaut readonly non-SSA. Ces résultats
ne reçoivent ni native/MPI/GPU, ni PDE, ni codec checkpoint, ni la future version
du consumer scalaire. Les receipts locaux et copies gelées untracked sont préservés.
