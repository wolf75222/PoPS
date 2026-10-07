# Accepted traces in logical child-clock regions, contract 2

Le contre-exemple W11 compose une équation FV `dU/dt = -div(F)` avec deux ticks enfant, puis un commit parent `U_next = (U_n + U_child)/2`. Le courant extérieur droit alimente une quantité persistante `q`, avec `q0 = 0.7`, capacité normalisée `C = 1` et conversion de signe `scale = -1`. Le profil spatial non constant permet de vérifier un véritable changement de volume. Les équations restent dans le [support public](../../../tests/python/support/w11_child_trace_case.py) ; le compilateur ne reconnaît aucun nom de modèle ou de physique.

Le premier `validate -> resolve -> compile` sur le vrai paquet ce024 refusait ce cas avant bind : le détachement cherchait le RHS enfant uniquement dans les valeurs du scope principal (`KeyError`). Après cette frontière, les quadratures refusaient le commit issu de `subcycle`/`synchronize`, et les carriers de faces restaient locaux au body sans livraison acceptée. Ces négatifs sont conservés dans [le lot auteur](/Users/romaindespoulain/dev/tmp/sol61-w11-child-trace-native-20261007/before-refusal.json) et [le témoin Source non-auteur](/Users/romaindespoulain/dev/tmp/sol61-w11-subcycle-source-independent-20261007/report.md).

## Contrat et réalisation

`pops.accepted-trace-regions@2` authentifie la région propriétaire de chaque évaluation FV retenue. Le walker décompose les commits affines globaux, propage leurs coefficients aux régions et traite l'entrée de boucle comme un carrier transporté une seule fois. Le body doit posséder un carry affine unitaire. Chaque tick émet ses nouveaux montants avec son `dt` local exact ; le poids parent et les poids des stages restent distincts. Une nouvelle évaluation RHS arrête l'ascendance, et la validation antérieure des solves spatiaux, coefficients négatifs et transformations opaques est conservée.

La publication des traces et leur consommation vers `q` s'effectuent dans chaque tick, avant la capture candidate et les gardes du parent. Les sous-boucles imbriquées composent les durées et phases natives existantes. Pour Uniform, un scope d'évaluation neutre restaure le stage après la qualification propre à chaque RHS, y compris sur exception. Il n'est pas appliqué à AMR, dont le scope modifie le sous-pas : le chemin antérieur est conservé et cette extension régionale AMR reste refusée. Les selectors de diffusion du scope principal restent consommés dans ce scope.

Une capture globale est un jeton préparé, pas un `Real`. Ses consommateurs scalaires publics lisent maintenant `ctx.integral_candidate_value` avec l'identité et les unités exactes. La vérification native de propriétaire, tentative, point et image de ledger reste active ; les expressions pointwise conservent leur route existante. Aucun cast ou résultat prescrit ne remplace cette lecture.

| Frontière | Version et compatibilité |
| --- | --- |
| Program serialized IR | 25 uniquement lorsqu'un selector de trace possède une région enfant |
| Contrat régional | `pops.accepted-trace-regions@2`, carry `unit-affine`, duration `exact-logical-child` |
| Identité sémantique | La clé `external_trace_regions_v2` est authentifiée puis incluse dans la projection ; champs et versions falsifiés refusés |
| Plans antérieurs | IR et projection antérieurs conservés ; aucune clé régionale ajoutée aux anciens plans |
| Syntaxe publique / Native ABI | Réutilisation des API existantes ; aucun nouveau header ou ABI par ce portage Python |

Le cache d'artefacts distingue également le C++ effectivement émis. Les [tests non-auteur](../../../tests/python/unit/codegen/test_subcycle_integral_independent.py) vérifient Euler et SSPRK2, nested 2x2, poids parent 1/2, absence de double comptage du seed, conservation et refus opaques/scheduled. La [contre-revue Source finale](/Users/romaindespoulain/dev/tmp/sol61-w11-subcycle-source-independent-r2-20261007/report.md), pins `afefe225cdec08b8be6e240328d43fd1e5df54efcc9e528d9986e66d45274188`, reçoit huit PASS, sept lectures scalaires préparées et la compatibilité d'un ancien IR5. Elle n'exécute aucun Native.

## Réception scientifique

La baseline des trois cas originaux [Uniform/AMR1/AMR2](/Users/romaindespoulain/dev/tmp/sol61-w11-integral-existing-baseline-ce024-20261007/report.json), pins `955d96be2359fa7bc5a301a2b1ccb6bd3ad4a9aafd9dff8c61c396754491dd99`, passe réellement sur ce024 et conserve les 2 563 membres installés avant/après. Elle qualifie le chemin antérieur, séparément du nouveau W11.

La [cohorte Source affectée](/Users/romaindespoulain/dev/tmp/root-w11-regional-source-validation-r2-20261007/command.json) reçoit 96 PASS. Six échecs d'une cohorte plus large sont reproduits sur une [archive baseline exacte](/Users/romaindespoulain/dev/tmp/sol61-w11-six-failure-baseline-20261007/report.md) ; le fixture sans primitive des cinq tests detach est corrigé sans modifier les assertions et reçoit six PASS. Le défaut IMEX de contexte Field solve7/read6 demeure distinct. La cohorte scalaire complémentaire reçoit 93 PASS et trois FAIL (deux golden de source dt-bound et un diagnostic de callback retime) ; les trois sont reproduits sur la même archive baseline fcadfe34, sans changer golden ou garde. Ces diagnostics restent conservés et ces résultats Source ne remplacent pas la non-régression native.

Le [test natif public](../../../tests/python/integration/runtime/test_w11_child_trace_rollback.py) exige la consommation préalable de `q`, un historique enfant réellement rempli/rotaté, le refus du parent, la restauration de State/q/history/ledger/clocks, puis une reprise sur la vraie frontière `t_end = 0.08` comparée au FV indépendant. Son estimateur de changement de State, seuil `0.009`, est fixé avant exécution : la tentative `dt = 0.1` dépasse le seuil, la reprise `dt = 0.08` le respecte. Il utilise les lectures et checkpoints publics.

La qualification native de ce nouveau test reste à produire. Uniform/MPI2, CUDA, AMR régional, carry non unitaire, range/schedules opaques et gaine cinétique M14 complète gardent leurs couvertures distinctes. Cette extension ne clôture pas le corpus des 94 exigences.
