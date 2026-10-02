# Field→Ghost initial — SDK11–14, preuves et échecs conservés

Actualisation ROOT : SDK13 Source002860fe a terminé sa vraie construction, 28 TU,
wheel/install/doctor rc0, 1 140 fichiers installés, DSOdf7e5a87 et SDK180a9615.
Le premier essai atteint Field/Ghost puis échoue sur un diagnostic absent de
la façade Python (1 FAIL80,38 s). Le correctif test-only e659 utilise le vrai
propriétaire natif const. Le second essai exécute bind, Field/Ghost initial et
une étape FE acceptée ; il échoue au checkpoint CP12, dont le budget omet le
membre des carriers AMR (1 FAIL65,60 s). Les 1 140 sources et DSO sont exacts
avant/après les deux essais. Journaux, XML et captures ne sont pas rescellés.

La contre-revue des données sauvegardées trouve OriginalF initial jusqu'à
1,0327e−8 pour une garde inchangée de1e−10. Le contrat FAC composite par défaut
(rel_tol1e−9, forcing16) autorise1,6e−8 ; son diagnostic interne n'a pas été
sauvegardé. La réalisation explicite publique maintenant intégrée demande
rel_tol7,6923e−13, soit cutoff≤1,25e−11 pour le forcing maximal16,25 de cette
tranche. Elle ne modifie ni l'équation, ni le maillage, ni les seuils indépendants.
La fixture scientifique est versionnée @2. Une marge de budget ne garantit pas
à elle seule une qualification Native : le nouvel essai reste nécessaire.

Les ghosts xmin acceptés valent3,046875 à m2,03125 et t1/64. Le getter Field
retient pourtant phi≈2. ForwardEuler résout au stage t_n : ce getter peut
observer le dernier SolveOutcome/warm-start, alors que Ghost consomme une
solution temporaire endpoint. Le point d'observation doit donc être authentifié.
Aucune publication supplémentaire ou solve déclenché par un getter n'est ajouté
pour satisfaire une assertion. La vraie image consommée, son point exact et
le cache checkpoint sont des preuves distinctes encore en réception.

SDK14 est un clone offline de62 paquets, avec SDK13 préservé. FAC et injections
Ghost préparées sont intégrées, 37 contrôles Source/host passent16,62 s sans
skip. Le correctif budget dérive une capacité structurale des formes/ghosts/
composants configurés ; sa contre-revue finale et44 tests Source/host passent.
La première hypothèse d’allocation est retirée : l’ancienne surcharge utilisait
des vues string_view/span sans copie. Le nouveau conteneur est explicitement
préparé sous vote et sa durée de vie vérifiée. Les injections utilisent
un vrai composant natif, sa table ABI et le propriétaire AMR ; elles restent
préparées Source seulement. Reconstruction, observation scientifique exacte,
checkpoint/restart, injection puis Serial/MPI2 restent obligatoires. Aucune
réception complète ou fermeture globale94 n'est affirmée.

Les détails et pins actuels figurent dans le
[reçu de progression @3](initial_field_ghost_sdk11_sdk12_progress_20261002.json).
Le [plan SDK14](../../../tests/review/sol61_sdk14_initial_field_ghost_plan.md)
conserve les commandes installauth, le monde MPI réel et le contrôle après
un échec. Les paragraphes ci-dessous sont l'historique SDK11/12, daté avant
les résultats SDK13 ; leurs reçus immuables restent valides dans leur portée.


La composition publique `Inflow((interior_trace(U,"c"), phi+1+logical_time()))`
est abaissée en un composant authentifié depuis ses expressions et dépendances.
L'[exemple linéaire](../../tutorials/initial_field_ghost/01_public_initial_field_ghost.py)
préserve les équations `dc/dt=0`, `dm/dt=m`, flux zéro et
`phi-alpha*laplacian(phi)=m`, alpha1/8, Neumann0, FE dt1/64 et AMR8×8/deux
niveaux. Les seuils Field/originalF `1e-10`, la capture avant tout getter et les
gardes de checkpoint/restart de la fixture scientifique restent inchangés.

| Gel construit / vrai paquet | Compilation et installation | Essai public réel | Réception |
|---|---|---|---|
| SDK11 Source5cfc1558, DSOd96ee34f, 1 139 sources | rc0, 28 TU, wheel/native identiques | 1 FAIL, 49,42 s : bind, `invalid pops.expr.key.v1 node` | Aucun Field/Ghost exécuté |
| SDK12 Source3843390d, DSOf3b8bd30, 1 139 sources | rc0, 28 TU, wheel/native identiques | 1 FAIL, 52,84 s : vrai Field initial, `AMR exact field solve has an invalid evaluation point` | Aucun Field/Ghost reçu |

Les identités et les 1 139 fichiers installés sont exacts avant/après chacun
des essais ; les anciens ENV et artefacts sont conservés. Le
[reçu de progression](initial_field_ghost_sdk11_sdk12_progress_20261002.json)
référence les journaux, XML, builds et reçus ROOT immuables. Source/host est
distinct de Native : 107 contrôles intégrés avaient passé avant SDK11 ; le
correctif de délégation de SDK12 a reçu39 contrôles indépendants et un cas
legacy séparé. L'échec d'un essai natif n'est pas transformé en réception.

Le premier défaut était une valeur réservée numérique routée vers le validateur
d'expressions builtin. Le protocole `native-boundary-component-values@1`
authentifie désormais un seul binding complet pour le producteur et une seule
région exacte, avant la délégation externe. Le doublon exact trouvé par le
non-auteur est refusé. Le validateur d'AST, les expressions et leurs supports
n'ont pas été assouplis.

Le second défaut est dans le solveur Field natif : son point exact n'admet
`dt=0` que sous l'autorité de rematérialisation topologique existante. Le
bootstrap initial dispose d'une autre autorité. La correction Source958c4b8b authentifie désormais ce point initial par
`PreparedAcceptedInitialFieldPoint@1`, limitée au vrai bootstrap et au point
exact du propriétaire. La portée empruntée est réinitialisée par RAII après
vote collectif ; le même solveur natif construit le vrai témoin de producteur.
La requête initiale passe en version3, avec marqueur explicite ; les requêtes
positives/topologiques gardent leurs bytes version2. Dix contrôles auteur et
un contre-test indépendant8b805b59 passent en Source/host. Le nouvel ENV SDK13
est un clone de62 dépendances ; sa reconstruction réelle depuis002860fe est
en cours. Aucune réception Native initiale n'en est déduite. La tranche reste
obligatoire.

Contrats intégrés : `accepted_initial_ghost@1`, interface11 additive au catalogue,
préparation/destruction partagées avec GhostV1 ; GhostV1 garde `dt>0`.
`PointwiseGhostValues@1` lit la région native des ghosts ; `interior_trace`
exprime séparément le clamp du State primaire. `native_interface_extensions@1`
authentifie les tables additives exactes. Le numéro ABI8, CP12 et accepted9
ne changent pas les layouts antérieurs ; les signatures SDK et le rebuild
restent requis. Les détails et limites sont dans le
[tutoriel](../../tutorials/initial_field_ghost/README.md) et les revues Source.

Reproduction du second essai conservé depuis le checkout
`/Users/romaindespoulain/dev/tmp/pops-api040-initial-ghost-native-20261002`
au gel3843390d :

```sh
env -u PYTHONPATH POPS_NATIVE_DIM=2 PYTHONNOUSERSITE=1 \
  OMP_NUM_THREADS=1 POPS_THREADS=1 OMP_PROC_BIND=false FI_PROVIDER=tcp \
  /Users/romaindespoulain/miniforge3/envs/pops-api040-initial12/bin/python \
  docs/development/api_040/run_installed_checks.py --output /tmp/pops-initial12-new-receipt \
  --test tests/python/integration/amr/test_public_initial_field_ghost.py::test_public_initial_field_fresh_before_ghost_and_positive_point
```

La commande exacte de construction et ses variables sont conservées dans le
build lié par le JSON. L'exemple et les contre-tests de callback sont Source
préparés seulement : aucune fraîcheur Field, rollback initial AMR, MPI ou GPU
n'est reçu à partir d'un test de header, d'un getter ou d'un bind échoué.
L'injection en préparation doit observer le vrai owner AMR avant/après une
faute du composant natif ; une exception Python artificielle n'est pas une
preuve de cette propriété.
