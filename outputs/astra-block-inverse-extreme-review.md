# Contre-revue indépendante — block_inverse / block_apply_inverse

Agent Astra Semantics, checkout isolé PoPS-resource-lifetime. Aucun cœur modifié par le relecteur. Fichier indépendant autorisé par Astra Protocols : `docs/development/api_040/evidence/block_inverse_extreme_review.cpp`, exécutable host autonome avec main.

## Résultats

- Avant correctif : 343 contrôles exécutés, 148 échecs (`outputs/block-inverse-extreme-review-red.log`). L'alias out==rhs a été ajouté ensuite. Le compteur varie avec les branches de vérification ouvertes après succès de l'inversion.
- Après correctif : **465 contrôles, zéro échec**, compilation clang++ C++20 -O2.
- Même reçu : **465 contrôles, zéro échec**, -O2 avec AddressSanitizer + UndefinedBehaviorSanitizer ; aucun diagnostic.

Commandes reproductibles depuis ce checkout :

```
rtk proxy /usr/bin/clang++ -std=c++20 -O2 -Iinclude docs/development/api_040/evidence/block_inverse_extreme_review.cpp -o /tmp/pops-block-inverse-extreme-review
rtk proxy /tmp/pops-block-inverse-extreme-review
rtk proxy /usr/bin/clang++ -std=c++20 -O2 -fsanitize=address,undefined -fno-omit-frame-pointer -Iinclude docs/development/api_040/evidence/block_inverse_extreme_review.cpp -o /tmp/pops-block-inverse-extreme-review-sanitized
rtk proxy /tmp/pops-block-inverse-extreme-review-sanitized
```

## Discrimination

Le reçu couvre N=1,2,3,5 : diagonales 1e308/1e-308 ; matrices denses aux échelles mixtes (1e308,1e-308,1e150,1e-150,1) avec permutation de colonnes ; oracle analytique indépendant pour D(I+alpha 11ᵀ)P ; inversion et application à un vecteur avec solution exacte ; annulation dont la somme intermédiaire déborde mais la solution est finie ; solve diag(1e-320) avec résultat1 alors que l'inverse est non représentable ; refus de résultat non représentable ; singularité/NaN/Inf et conservation intégrale des sentinelles ; alias rhs/out ; invariance du verdict de tolérance sous changement positif d'échelle de chaque ligne ; parité bit à bit de rotations ordinaires.

Lecture du correctif : admission commune via équilibrage des lignes et pivot relatif max(tol,N epsilon), publication seulement après calcul temporaire fini, inverse remis à l'échelle par colonnes, apply sans inverse original matérialisé par produits en mantisse/exposant et accumulation renormalisée. La voie historique ordinaire garde son ordre arithmétique dans sa plage sûre. Aucun finding bloquant supplémentaire reproduit. Demande documentaire transmise : les commentaires publics doivent remplacer l'ancien seuil absolu sur déterminant par le contrat relatif courant.

Portée : tests host binary64 du header réel, sans runtime PDE ni compilation MPI/Kokkos/GPU. Ils ne qualifient pas la convergence d'un solveur global, la performance ni le comportement de toutes les matrices mal conditionnées. Les tests gtest durables et le reçu natif intégré restent détenus par le propriétaire/parent.

## Réception complémentaire du propriétaire

Astra Protocols a corrigé les commentaires publics et ajouté la décision
`docs/development/api_040/block_inverse_scaling_v1.md`. Le véritable TU gtest,
compilé légèrement contre le header final et la bibliothèque gtest existante,
passe **14/14 en binary64**, y compris les anciennes parités exactes N2/N3.
Les cinq nouveaux tests et le résidu N4 passent aussi **6/6 en binary32**.
L'ancien test N4 exigeait une délégation bit à bit à une élimination non
équilibrée ; il vérifie maintenant le résidu de la matrice originale avec une
borne de roundoff explicite `8*N*epsilon*(1+sum(abs(products)))`.

Les **12 tests publics/source/fragments C++ de condensation repassent en
11,12 s** avec le header final. Aucun build global, MPI, Kokkos ou GPU n'a été
lancé par ce propriétaire. Les logs host complémentaires sont présents dans
`outputs/block-inverse-gtest-{host,float-host}.log` ; le reçu source est
`outputs/t3-condensation-scaled-green.xml`.
