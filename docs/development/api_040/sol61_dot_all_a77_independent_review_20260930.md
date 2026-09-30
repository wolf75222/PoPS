# Réception indépendante bornée de dot_all a77e1b1 — 30 septembre 2026

Candidate exacte : `a77e1b1ce09b84171fd3f63ffc64f6995614b8e5`.
Comparaison legacy : `21a56b910c421eee465afcb6f16123dd309360a9`.
Checkout exclusif : `PoPS-sol61-dot-all-a77-review`, branche
`codex/api040-sol61-dot-all-a77-review`. Aucun changement de production,
d'installation ou de SDK. Cette réception est source/host ; elle ne reçoit
ni MPI/Kokkos natif, ni GPU, ni SSA public exécuté, ni PDE, ni codec checkpoint.

## Résultat reçu et écart encore présent

Le probe indépendant exécute 47 assertions avec le vrai noyau
`FiniteOwnedDotKernel`, le helper local, les méthodes des deux providers, les
deux visiteurs AMR et les templates natifs de classification/résolution
`ProgramOwnerFieldIdentity`. 46 assertions reçoivent les propriétés attendues ;
la dernière assertion supplémentaire reproduit volontairement une lacune de
contrat de stockage : AMR accepte des ghost-counts différents. Le même programme
passe sous AddressSanitizer et UndefinedBehaviorSanitizer.

Les six méthodes legacy `dot`, `norm2`, `norm_inf` Uniform/AMR et le visiteur
`for_each_owner_active_level_` sont byte-identiques à 21a56b9. Le nouvel opérateur
reste une contraction de toutes les composantes actives et possédées ; son test
de rotation euclidienne ne déclare pas une norme physique universelle.

Le probe reçoit les contre-exemples suivants :

- Vecteur `(2,3,5,7,11)` : contraction 208, rotation des deux premières
  composantes 208, corruption des composantes suivantes 20404.
- Coarse couvert + fine actif : 13, et non l'ancien double comptage 26.
  Intersection effective du masque EB et de la couverture AMR ; NaN coarse
  couvert ignoré, NaN fine actif refusé.
- NaN, infinities, overflow du produit, de l'accumulation des composantes,
  des patches et des niveaux refusés avant la somme collective simulée ;
  overflow de la somme globale refusé après celle-ci.
- Masques non finis, largeur incorrecte, layout/distribution/rank-space
  incompatibles refusés avant somme. Identités Scratch et History réellement
  classifiées, owner étranger refusé, couverture sélectionnée par niveau natif
  et non ordinal de callback.
- Répliques : seul lane.rank()==0 contribue ; les autres répliques restent
  validées et refusent le NaN actif. Rangs locaux vides participent à la
  séquence de vote puis somme.
- Uniform refuse une largeur, un layout, une distribution, un rank-space ou
  un ghost-count incompatible.

**Écart concret restant, AMR ghost-count** : champ gauche fine accepté de
ghost-count 0, copie Direct droite de même largeur/layout/distribution/rank-space
avec ghost-count 99 : `dot_all` retourne 13. Le chemin réel appelle le validator
legacy AMR `require_same_field_contract_`, qui ne compare pas `ghosts()`.
Le probe reproduit l'acceptation ; elle n'est pas qualifiée comme un refus reçu.
Le contrat retenu par l'intégrateur exige la même compatibilité de stockage que
le nouveau chemin Uniform. Il faut un guard limité au nouveau dot_all, convergé
avant réduction, puis réception sur son propre SHA ; les méthodes legacy restent
préservées. Source : `amr_program_context_spatial_operations.inc:362` et
`amr_program_context_history_checkpoint_services.inc:218`.

## Limites des substitutions et ordre collectif

Les vrais FieldView, Box, Index et templates d'identité sont inclus. Le stockage
MultiFab est un substitut host possédant ses allocations ; les réductions locales
parcourent les vraies vues en série. Les facades de hiérarchie, la lane et les
collectives sont des substituts explicites. Les séquences `P,V,S` signifient
préflight du masque Uniform, convergence de l'erreur locale, somme globale.
AMR observe `V,S`. Les injections locales refusent avant `S` ; l'overflow global
refuse après `S`. Ceci vérifie les appels et branches extraits, pas le progrès
réel de plusieurs processus MPI.

L'appel Uniform au masque avant le try du provider n'est pas à lui seul une
preuve de deadlock : la route native prepared du masque possède sa convergence
interne des erreurs et son consensus. Les substituts du présent probe n'exécutent
pas cette route native. La réception native/MPI appartient à l'intégrateur.

## Source, identité et reproduction

Le probe authentifie ses headers contre les git objects du SHA exact avant
compilation et écrit les hashes des fonctions extraites dans son receipt.
Archive `include` + `python` :
`72ed0713530bf3f4610be10496ee903ece09f332e8d91d03bd9e9c5bbd937395`.
TU host générée :
`d81df3a378301d667c890578f6efeddc4910f1cd97c99f888b8ac7e94ec66fb7`.

```sh
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM PYTHONPATH=python /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python tests/review/sol61_dot_all_a77_review.py
rtk proxy /usr/bin/clang++ -std=c++20 -O0 -g -fsanitize=address,undefined -Iinclude -I/Users/romaindespoulain/miniforge3/envs/pops-api040/include outputs/sol61-dot-all-a77-independent/probe.cpp -o outputs/sol61-dot-all-a77-independent/probe-sanitized
rtk proxy outputs/sol61-dot-all-a77-independent/probe-sanitized
```

Les deux exécutions ont imprimé `47 independent actual-kernel/provider/identity
checks PASS`. Ruff reçoit le nouveau script.

Sur ce même SHA, 27 tests source/host ont également passé en 4.52 s : les deux
probes gelés indépendants de `f212b9cbe7ace97600b1af3e71c6966f6addfd87`
(`test_sol61_dot_all_21a56b9.py`) et
`7ac0b265ca4a14fe3f22b8e0d99d6818c7f24f0b`
(`test_sol61_vector_reduction_oracles.py`), copiés sans modification, ainsi que
le test auteur `test_dot_all_native_contract_host.py`. Ces copies ne sont pas
dupliquées dans ce commit. Ils couvrent URI/refus exacts, IR v7 conditionnel et
récursif (branches/while), anciens graphes et hashes, déclarations de largeur,
owner/order, sérialisation de dt_bound et témoins vectoriels. La sérialisation
dt_bound ne constitue pas une réception de son émission C++ : les deux gaps
résolus publics sont enregistrés séparément dans le probe dt_bound.

Les receipts complets sont locaux sous `outputs/sol61-dot-all-a77-independent/` ;
les probes permettent de les régénérer. Aucun résultat antérieur n'est utilisé
comme réception native de a77e1b1.
