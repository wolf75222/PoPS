# M17 : préparation FP et coût hôte appariée

Ce témoin compile deux en-têtes réellement matérialisés depuis Git : ancien `8d438518`, nouveau `5000328`. Chaque variante reçoit ses propres en-têtes mathématiques `include/pops/numerics/moments` au même SHA, avant les dépendances communes du checkout. Aucun ancien émetteur n'est réintroduit en production. Les commandes, hashes, sorties brutes, sources et exécutables hôtes sont conservés sous :

`/Users/romaindespoulain/dev/tmp/sol61-m17-paired-mixture-host-20261002`

Reproduction depuis le WT :

```sh
rtk proxy python3 tests/review/sol61_m17_paired_host.py /un/nouveau/dossier
```

Le dossier doit être nouveau ; les preuves existantes ne sont pas écrasées. La commande compile C++20, `-O2 -DPOPS_NATIVE_DIM=2 -fno-fast-math -ffp-contract=off`. Le programme n'est pas une DSO PoPS et n'importe pas PoPS. Il utilise 32 mélanges de deux gaussiennes, directions signées, densités égales/proches/rapport2, et moments terminaux non gaussiens. Les deux variantes utilisent les mêmes arrays effectifs sauvegardés et le même ordre canonique. Refus densité négative, moment NaN et direction infinie : statuts identiques `2,1,5`. Tous les 32 états admissibles donnent Success pour flux et intégrale. Inversion des endpoints et direction nulle : égalités exactes vérifiées dans chaque exécutable.

## FP et référence indépendante

Les sorties ne sont **pas bit à bit identiques** ancien/nouveau. Maximum de `abs(new-old)/(1+abs(old))`, flux/intégrale/borne ensemble : `2.0840071082221546e-16`. Aucun critère scientifique ni seuil existant n'a été modifié.

La référence indépendante `tests/python/support/fan_li15_oracle.py` calcule les coefficients par fonction génératrice, les flux par intégration gaussienne par parties, et le produit principal par la double somme multi-indices. Le harness intègre ce dernier sur le segment **brut** avec deux quadratures Gauss-Legendre flottantes indépendantes, 32 puis64 points. Il n'utilise ni le générateur de production, ni les coefficients polynomiaux émis, ni la récurrence analytique des poids de densité.

Maxima mesurés sur les 32 entrées effectives :

| Comparaison normalisée | Maximum |
| --- | ---: |
| Nouveau flux / référence primaire | `4.001166780394643e-16` |
| Nouvelle intégrale / quadrature64 | `2.74493942460234e-16` |
| Quadratures32 /64 | `5.997023818736086e-17` |

La convergence entre ces deux quadratures est une vérification numérique bornée ; elle ne constitue pas une borne rigoureuse universelle d'erreur de quadrature ou de stabilité. Les cas près du bord constitutif, densités extrêmes et toutes architectures FP restent hors ce témoin.

Permutation : la compilation/exécution du noyau réellement émis, canonique et inverse, a été rejouée séparément : **1 PASS en1.86 s**. Elle vérifie tous flux/intégrales et borne exacte sous permutation, compare l'ancien noyau et garde l'orientation. XML `/tmp/sol61-m17-paired-permutation.xml`, SHA256 `efd7bcd526fba88ffeab682d04c72df29d1d3d9de6e5fae1ca4f63370f91c1ec`.

## Coût steady local

Chaque répétition exécute30000 couples flux+intégrale avec checksum sauvegardé ; sept répétitions par variante, mêmes32 états. Médiane ancienne : **1935.06 ns/couple** ; nouvelle : **745.51 ns/couple** ; ratio nouvelle/ancienne **0.3853**. Un ancien échantillon monte à7297.93 ns ; toutes les valeurs brutes sont conservées. Le CPU n'est pas isolé, les variantes sont exécutées séquentiellement et les allocations/registres de la grosse expression peuvent dépendre du compilateur. Ce signal local ne qualifie ni une performance applicative, ni Native PoPS, MPI, GPU ou ROMEO.

Reçu JSON externe SHA256 : `8031f96848d2c9a2bb959e5e62c316243c55d6df594bdf3225fcb8daac968c85`. La première exploration sur gaussiennes simples est conservée séparément sous `sol61-m17-paired-host-20261002` ; le reçu de mélanges ci-dessus est le résultat final de ce lot. Les qualifications et éventuelles mesures backend appartiennent à ROOT.
