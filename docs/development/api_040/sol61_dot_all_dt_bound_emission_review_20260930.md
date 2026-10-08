# Deux contre-cas publics dt_bound - 30 septembre 2026

Source reçue : `a77e1b1ce09b84171fd3f63ffc64f6995614b8e5` ; checkout exclusif
`PoPS-sol61-dot-all-a77-review`. Le nouveau dot_all se sérialise en IR v7,
mais les deux captures ci-dessous échouent à l'émission C++ de la véritable
route publique résolue. La correction numérique AMR de a77 ne corrige pas ce
seam Python/codegen. Aucun changement de production dans ce lot.

Le script indépendant `tests/review/sol61_dot_all_dt_bound_gap.py` construit
un Model à deux composantes et une Case Uniform périodique 4×4, un Program
avec une transaction identité et FixedDt(.1), puis un bound
`cfl / (1 + P.dot_all(state.n, state.n))`. Il appelle `validate`, `resolve`,
`ProgramModelGraph.from_resolved_blocks` et `emit_cpp_program` sur l'objet
temporel effectivement résolu. Il ne forge pas un nouvel identifiant SSA,
ne monkeypatch pas les maps, et ne réémet pas artificiellement les opérations.

| Capture | Résultat réel a77 | Cause à fermer |
| --- | --- | --- |
| `u.n` lu dans le corps principal avant le callback | `KeyError: 0` | Le SSA de l'état mis en cache est extérieur au DAG dt_bound ; l'émetteur isolé n'a pas la capture. |
| État du bloc readonly `bound_data` lu uniquement dans le callback | `ValueError: block ... is not declared in the Program block-index map` | Le map des blocs n'inclut pas ce bloc déclaré seulement dans le sous-programme bound. |

Les deux Program sont authorés, validés et résolus ; ce sont des erreurs
d'émission, pas des refus scientifiques légitimes démontrés. Le premier cas
préserve la lecture principale utilisée par le commit identité ; le second
laisse `bound_data` readonly. Le même probe devra recevoir les deux émissions
sur le futur correctif, avec une route de capture explicite et sans fallback
vers des opérations temporelles/state-changing.

```sh
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM PYTHONPATH=python /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python tests/review/sol61_dot_all_dt_bound_gap.py
```

Exécution authentique sur a77 : **exit 1**, deux résultats `FAILED` enregistrés
dans `outputs/sol61-dot-all-a77-independent/dt-bound-receipt.json`. Le script
enregistre également le chemin du package Python réellement importé. Un code
retour nul requiert l'émission réelle des deux Program et la présence de
`ctx.dot_all(` dans le C++ ; les exceptions ne sont pas converties en PASS.
Ruff reçoit le script. Aucune compilation de ce C++, exécution native,
qualification MPI/GPU/PDE ou installation n'est revendiquée.

Les tests URI/IR7 et la réception du noyau/provider a77 sont gelés séparément.
Le succès de la seule sérialisation dt_bound n'est donc pas utilisé pour
masquer ces deux contre-exemples d'émission. L'intégrateur a confié la
correction bornée à l'auteur de la route Python/codegen.
