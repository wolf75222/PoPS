# IMEX: exact State and Field coordinates

Le correctif de bibliothèque conserve le tableau ARK et les données `Y_i`. Le Field utilisé par `E` et le taux explicite lisent maintenant la même SSA, matérialisée à `c_explicit`. Le taux implicite garde le stage joint à `c_implicit`. Le [manuel de l'exemple final](../../../examples/final/EXEMPLE_SPEC_FINALE_ADVECTION_IMEX_AMR.py) suit la même composition que le preset [IMEX](../../../python/pops/lib/time/imex.py).

Le défaut préexistant créait `field_state = copy(Y_i)` au temps implicite, résolvait le Field sur cette nouvelle SSA, puis le lisait avec l'ancien `Y_i`. La garde FieldContext refusait correctement solve7/read6. Une simple substitution du lecteur par `field_state` aurait déplacé le RHS explicite au temps implicite : nommer ensuite une copie du taux au stage joint ne retime pas son évaluation. Les deux coordonnées doivent donc être conservées dans la composition Python.

Les équations restent :

\[
Y_i=P_i+h a^I_{ii} I(t_n+c_i^I h,Y_i),\qquad
E_i=E(t_n+c_i^E h,Y_i,F_i),\qquad
I_i=I(t_n+c_i^I h,Y_i).
\]

Le Field consommé par `E_i` est résolu sur une copie de coefficient un de `Y_i`, à sa coordonnée explicite. Cette copie conserve les valeurs de State ; elle n'interpole pas une autre solution physique. Aucun coefficient A/b/c, solve implicite ou garde stale n'est changé. Il s'agit d'une correction de composition dans la bibliothèque et l'exemple, sans nouveau mécanisme central, syntaxe publique, champ de contrat ou version ABI.

Le [contre-exemple et les tests Source](../../../tests/python/unit/time/test_imex_field_stage_identity.py) distinguent IMEX Euler, ARS222 et un tableau à abscisses séparées. Les preuves auteur (`/Users/romaindespoulain/dev/tmp/sol61-imex-exact-field-stage-source-20261007/report.json`, pins `078fc7f6b98f2fbc305ee8258da6a2713e4ee5134b26bdf814e7a42c793ef530`) reçoivent 47 PASS pertinents. Après correction du manuel, trois tests supplémentaires (`/Users/romaindespoulain/dev/tmp/sol61-imex-manual-preset-source-20261007/report.json`, pins `764afaee28ff1f07ca6182ba09c321719e3b74740a1e30c69ec826f4b75d2916`) passent, dont l'égalité complète graphe/données/hash/identité sémantique manuel/preset et la résolution du Field AMR. Les négatifs antérieurs sont conservés. Sept défauts de certificats RK d'une cohorte plus large sont reproduits avant ce patch ; ces 47/3 PASS ne signifient pas un green Source global.

La contre-revue indépendante (`/Users/romaindespoulain/dev/tmp/sol61-imex-field-state-time-independent-20261007/report.md`), pins `37d651d3af2aa4eeb8210899afb3bb7240c695435b03a608f44fbd1ac939034b`, reçoit 16 cas de trois cibles pytest en 10.19 secondes : coordonnées distinctes, mêmes SSA solve/read, copie de `Y_i`, garde stale et parité manuel/preset. Son libellé historique « huit groupes » est imprécis ; command.json/XML sont l'autorité des trois cibles et 16 cas, sans réécriture du lot fermé.

Une référence numérique indépendante distingue les temps : avec `U_n=3`, `t_n=1`, `h=1/4`, `I(t,Y)=-(1+t)Y`, `F(t,Y)=(2+t)Y`, `E=F` et le tableau Euler déclaré ici, `Y=48/25`, `F_E=144/25` et `U_next=84/25`. Utiliser le temps implicite pour le Field donnerait `87/25`. Ces critères sont fixés avant toute nouvelle exécution native.

Le build officiel du commit `9d4cdcca19abd7d8acfc2151497d8f2a967cddcc` retourne RC0, sans répéter setup. Le wheel SHA `bbb3e87eefd048b64f92a134a9aed9bf715f0f31911bc941612c782494b9d395` contient exactement le nouveau `imex.py` (`e6158934c813658741655e5e4d55df6680b7802b29161b5953481589b007c7a7`). Le Native `64c8922e58cf484a1b1188b899e25e84569707ae2090ecbd26f515daf2e1e9c6` reste identique au build précédent, car cette correction de bibliothèque ne change pas le C++. Le test importe le vrai paquet `pops` installé, avec `PYTHONPATH` supprimé ; le contexte après tentative mesure Dim2, MPIworld1, Kokkos OpenMP, concurrence effective de2 et mémoire Host.

L'appel original `run_manual_and_restart` atteint compilation, bind et avance, puis échoue pendant la préparation du consommateur checkpoint : `ConsumerPublicationError: state carrier checkpoint requires prepared accepted state`. L'ordre est `begin -> advance -> stage consumers -> commit -> accept consumers -> finalize`. La capture publique `checkpoint_state_carriers()` refuse correctement la profondeur1. Le checkpoint des diagnostics porte aussi une garde depth0. Le XML contient un test, une erreur, zéro failure/skip ; aucun pas accepté, restart, parité preset ou oracle numérique IMEX complet n'est reçu.

La réception indépendante ROOT (`/Users/romaindespoulain/dev/tmp/root-imex-original-negative-reception-9d4-20261007.json`, SHA `11c10d361aa20093c3a782e0c71c8505d3c0ffdb94ccb78d64e5e3b0fa93a58c`) authentifie les 2 845 entrées déclarées du lot fermé : 2 602 fichiers réguliers, 242 dossiers et un lien, 90 023 689 octets réguliers. Le lot auteur conserve le négatif, le wheel, les DSO compilés, C25, contexte, XML et backup du paquet antérieur ; pins `05e9e2fc1e97a2b29985e182d313c3cb96bb1e66bef899d3e55816e6c604a893`. Ce résultat impose une capture candidate transactionnelle distincte : figer les octets avant commit, publier après commit et conserver le rollback ainsi que les gardes publiques. Son code, sa nouvelle identité Native et ses tests doivent être reçus séparément.

Le cas original advection/relaxation IMEX AMR doit encore vérifier états acceptés, solves consommés, coordonnées, restart et critères scientifiques inchangés. CUDA, MPI2, 3D, convergence, coûts et tous les autres critères des 94 IDs gardent leurs preuves distinctes.

## Réception ultérieure de l'original, 7 octobre 2026

La capture candidate ABI12, la projection transactionnelle des diagnostics et le budget
du graphe complet des consommateurs lèvent successivement les refus précédents, conservés
avec leurs artefacts. Sur Source2ded/Native77d/Headerc04a, les deux workflows originaux
passent dans le vrai paquet installé : manuel/restart/continuation, puis preset/parité/rejet.
La contre-lecture reçoit les vrais états, champs, checkpoints et regrid de deux à trois niveaux.
Un test non-auteur publie réellement le checkpoint, déclenche un échec, puis reçoit le rollback
exact des données sauvegardées et la suppression des fichiers publiés.

Le [reçu détaillé](prepared_amr_checkpoint_capture_v1.md#received-original-workflows-and-publication-failure)
porte les identités, les commandes et les limites de ces essais Dim2/Host/MPIworld1.
Il ne reçoit pas encore le cas nonautonome fixé ci-dessus, le Retry propre au checkpoint,
les historiques non vides, MPI2, CUDA,3D, convergence ni coûts comparables.

## Témoin nonautonome intégré et compilation du champ refusée

Source `d5b0cfbacc6a2bfe5778ceba95d6fe3bbc7cce5d` intègre le
[cas public indépendant](../../../tests/python/support/imex_nonautonomous_field_case.py)
et son [test numérique](../../../tests/python/integration/runtime/test_imex_nonautonomous_field.py).
Les corrections de [distinction Aux/Field](imex_imposed_aux_field_classification_20261007.md)
et d'[autorité StateStorage AMR](amr_explicit_local_storage_authority_20261007.md)
lui permettent de franchir validate, resolve et verify. Sa réalisation utilise le vrai
GeometricMG sur une hiérarchie AMR d'un niveau, aux mêmes64 cellules, tolérances et oracles.
Les refus Uniform/MG, Uniform/CG screened et FAC explicite à un niveau sont conservés.
Ce témoin ne reçoit aucune qualification coarse/fine ou Uniform screened.

Le rebuild officiel d5b0 passe avec sept bindings réellement recompilés et le module relinké.
Le Native installé a SHA
`76af57236d6a65630febc1e974452a0010abef1c3e9c269ab32c5c15655daac3`,
ABI12 et HeaderSignature
`773b8630b18aa821b0e8394c460dfd62ec990f6d3866807bf7581b358f8420e8`.
Le wheel retenu a SHA `266e42a0f88c4be08d6d4501b6b34398a8ffbcee7a1640cb5eb47ed59a7f4ea0`.
Ces résultats reçoivent la reconstruction locale, pas une PDE ni CUDA.

Le premier test échoue en5.95 secondes pendant la compilation du composant Field, avant
bind : `model_native.cpp:146:30: use of undeclared identifier 'stage_tau'`.
Le RHS généré contient `return ((pops::Real(3) + stage_tau) * u);` sans déclaration de l'Aux.
La commande du compilateur, stderr, log et XML sont retenus. Le fichier C++ temporaire
complet a été supprimé par le compilateur et n'est pas reçu ; il ne sera pas reconstitué
comme preuve historique. Aucun état accepté, Field numérique, restart ou résultat Y n'existe
pour ce test. La correction doit porter sur l'émission et la préparation génériques des Aux,
avec le State, la frame et le point exact du solve, sans substitution spécifique de `stage_tau`.

Lot fermé : `/Users/romaindespoulain/dev/tmp/sol61-imex-nonautonomous-native-d5b0cfb-20261007`,
pins `67200763a39e83248c66290bd92f9bbf0cb2e074fa58bf83e0ecb7a671e9ded3`.
