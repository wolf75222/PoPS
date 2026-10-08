# Contre-réception indépendante pureté dt_bound - 30 septembre 2026

Candidate production exacte : `7c68cbaad14024868922664461fa804d4067128c`,
parent `9c4209a5e5ecb05ac71a62f4d2c21b6e3acd4300`.
Checkout exclusif `PoPS-sol61-dt-bound-purity-review`, branche
`codex/api040-sol61-dt-bound-purity-review`. Les seuls commits ajoutés avant
le présent gel sont les réceptions de tests 893379b et 8c87ffe ; `python`,
`include` et `src` restent byte-identiques au candidat 7c. Aucun MAIN/env,
header, SDK, installation, build/JIT modifié.

## Défaut et captures réparées reçus source

Le contre-test public de commit caché figé dans 893379b, **inchangé**, passe
désormais : le builder ne peut plus ajouter un commit global portant sur une
candidate préexistante puis retourner une expression scalar read-only.
Le refus restaure le Program et conserve son unique publication déclarée.

Les deux captures publiques réparées par 9c restent reçues sur cette tranche :
ancien état principal et query-only absent des updates, incluant les quatre
cas à trois composantes/permutation de la fixture native préparée. Le véritable
pipeline public de validation/résolution puis émission C++ utilise les bons
owners et n'introduit aucun commit dans le bound. Les captures temporelles
directes/transitives restent refusées. IR complet, semantic complet, hash et
C++ des Program legacy/unbounded restent byte-identiques à l'archive parent
pré-correction, vérifiés dans deux subprocess indépendants.

## Contre-probes nouvelles après allocations partielles

Le test indépendant `test_sol61_dt_bound_purity_review.py` ne se contente pas
du seul setter metadata. Il crée dans le callback un TimeState query-only,
sa lecture SSA et une contraction dot_all, puis tente successivement une
mutation publique de cadence, stratégie ou history. Pour les trois cas :

- Refus exact `read-only callback changed authoring metadata`.
- IR complet, identifiants/régions et registre TimeState restaurés exactement.
- Aucun bound publié ; le commit principal demeure seul.
- Une lecture SSA sortie du callback échoué ne devient pas réutilisable : le
  vrai `Program.dot_all` la refuse avec `was not authored by this Program`.
- Une nouvelle requête pure peut recréer l'état, se valider/résoudre et
  s'émettre réellement, avec `ctx.state(1)` et un seul `ctx.dot_all(1,...)`.

Le fix contrôle l'effet net du callback sur les containers/attributs Program.
Les anciennes entrées sont protégées par identité, les allocations query
append-only restent permises, les attributs non autorisés sont protégés par
défaut. La validation transitive whitelist de la closure demeure distincte.
Ce mécanisme n'ajoute ni schéma IR ni fake run identity, et ne modifie pas
le consumer scalaire historique <=0/nonfinite.

## Vérifications et limites

Sélection cohérente : **51 source PASS en 15.06 s**, incluant les huit contre-
tests indépendants précédents, les trois nouveaux, les tests auteur de captures
et mutation et les régressions d'atomicité/provenance. Ruff PASS. Aucun défaut
production supplémentaire démontré dans cette tranche bornée.

```sh
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM PYTHONPATH=python /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest -q --tb=short tests/review/test_sol61_dt_bound_purity_review.py tests/review/test_sol61_dt_bound_capture_independent.py tests/python/unit/codegen/test_dt_bound_capture_closure.py tests/python/unit/time/test_program_authoring_atomicity.py tests/python/unit/time/test_typed_provenance_guards.py
```

La fixture native publique InitialCondition/BindArray/ConservativeCellAverage,
exact initial_values, états initialisés avant run, deux mises à jour et frontières
MPI2 reste gelée séparément dans 8c87ffe. Elle a été reçue source et collectée,
mais compile/bind/run natif doit encore être exécuté par l'intégrateur sur son
package reconstruit. Les présentes assertions n'authentifient ni MPI/Kokkos
natif, ni GPU, ni PDE, ni codec checkpoint, ni l'admission native d'un NaN au
bind. Le witness public de refus est l'overflow depuis une initiale finie.
