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

La réception Native reste à produire après installation du commit intégré. Le paquet 64c antérieur et les anciens résultats IMEX ne qualifient pas cette correction. Le cas original advection/relaxation IMEX AMR doit suivre validate -> resolve -> compile -> bind -> run, puis vérifier états acceptés, solves consommés, coordonnées et critères scientifiques inchangés. CUDA, MPI2, 3D, convergence, coûts et tous les autres critères des 94 IDs gardent leurs preuves distinctes.
