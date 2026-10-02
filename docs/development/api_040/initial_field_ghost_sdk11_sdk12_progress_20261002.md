# Field→Ghost initial — SDK11–14, preuves et échecs conservés

Actualisation ROOT SDK14 — build Source8d438518 réel rc0/28 TU, DSO2738b99f,
SDK8c238f29, 1 140 fichiers installés authentifiés. La
[réception scientifique positive ROOT](/Users/romaindespoulain/dev/tmp/pops-api040-native-reception-evidence-20261001/sdk14-initial-field-ghost-positive-scientific-root-reception.json) reçoit Serial et MPI2 :
vraie Field consommée avant Ghost au point initial et après une FE, OriginalF
initial maximal1,02e−11 et accepted maximal4,61e−12 pour la garde intacte1e−10,
CP12/accepted9 et restart bit exact. Serial possède28 exports, MPI2 en possède34 ;
deux rangs exécutent le même test MPI, ce ne sont pas deux cas indépendants.
Le reçu immuable8e335073 sépare cette réception de l'injection et de la mission94.

L'observation complète candidate est opt-in, générique et qualifiée par le slot,
le consommateur, le point temporel et la topologie. Elle est prise après le Solve
réel et ses votes, avant le Ghost réel. Le getter reste le warm-start du dernier
stage FE ; il ne déclenche aucune résolution. La capture de signature utilise
les données JSON typées de ComponentManifest, et le lecteur indépendant@4
respecte les axes NumPy(y,x) du writer, y compris les rectangles non carrés.
Les exports antérieurs mal typés et les refus de lecteurs restent conservés.

La non-régression SDK14 exécute trois tests Serial Halo/Stage et restauration ;
l'exemple public linéaire termine et reload ses états exacts. Leurs identités
complètes et les 1 140 sources avant/après sont égales et contre-revues.
L'exemple utilise le FAC par défaut : son succès ne qualifie pas la garde SCI1e−10.
Les suites Source99PASS et27PASS sont distinctes des runs et ne sont pas
additionnées aux suites qui les recouvrent.

L'[injection native@6 ROOT](/Users/romaindespoulain/dev/tmp/pops-api040-native-reception-evidence-20261001/sdk14-initial-ghost-capture-fault-root-reception.json) est reçue Serial et MPI2. Le recompute ROOT
rehash les57 pins du gel indépendant, relit les9/14 exports réels, compare ses
résultats au rapport non-auteur et authentifie les1 140 fichiers du paquet et du
checkout ainsi que le DSO. En Serial, le XML compte1PASS49,277 s ; en MPI2,
chaque rang compte le même cas1PASS54,344 s (un seul cas partagé). Les wrappers
mesurent respectivement52,082 s et57,646 s. Le Field réel2 est lu avant l'écriture
candidate, puis le callback ABI natif déclenche l'échec sur le propriétaire xmin
déduit du carrier réel. L'abandon outer-bootstrap original restaure exactement
les State valides+grown, métadonnées accessibles et readiness non matérialisée.
Le point temporel +0 est aussi vérifié par bits au recompute ROOT.

Le diagnostic parent-rejected MPI reste explicitement canonical-refused : aucun
State/registre intermédiaire n'est inventé ou reçu et la cause native d'origine
est conservée. La contre-revue Source@6 et31 contrôles offline par configuration
couvrent les refus et mutations de copies ; ces suites ne sont pas deux runs
Native supplémentaires. Les payloads Field/registre/auxiliaires indisponibles,
caches/history de cette faute, rollback nested parent et constructeur antérieur
ne sont pas certifiés. Les échecs@3/@4 et MPI@5 restent immuables et distincts.
Cette tranche obligatoire bornée est terminée ; la mission94 reste active.

Portée : CPU arm64 Kokkos OpenMP1/MPICH Dim2, Serial et MPI2 sur même machine ;
Field scalaire uniforme FAC, AMR8×8/deux niveaux synchrones, une FE dt1/64,
ghosts physiques xmin à indices tangentiels valides. Pas de réception GPU,
inter-nœuds, autre dimension, modèle complet, ROMEO SDK14 ou GitHub CI actuel.
Le suivi machine est [progression@5](initial_field_ghost_sdk11_sdk12_progress_20261002.json).
La mission complète reste active, avec94 obligations conservées.

Reproduction Native depuis le checkout gelé
`/Users/romaindespoulain/dev/tmp/pops-api040-initial-ghost-native-20261002`
(test Source53d62a5e ; build Source8d438518, aucun diff de production).
Chaque sortie doit être neuve ; les réceptions originales ne sont pas écrasées.
Pour refaire le positif, remplacer le node par celui de
`tests/python/integration/amr/test_public_initial_field_ghost.py::test_public_initial_field_fresh_before_ghost_and_positive_point` indiqué dans les reçus positifs.

```sh
POPS_RECEPTION_ENV=/Users/romaindespoulain/miniforge3/envs/pops-api040-initial14
POPS_RECEPTION_OUTPUT=/Users/romaindespoulain/dev/tmp/pops-initial-ghost-reproduction-serial
cd /Users/romaindespoulain/dev/tmp/pops-api040-initial-ghost-native-20261002
rtk proxy env -u PYTHONPATH CONDA_PREFIX="$POPS_RECEPTION_ENV" \
  Kokkos_ROOT="$POPS_RECEPTION_ENV" POPS_KOKKOS_ROOT="$POPS_RECEPTION_ENV" \
  CMAKE_PREFIX_PATH="$POPS_RECEPTION_ENV" \
  POPS_INCLUDE="$POPS_RECEPTION_ENV/lib/python3.12/site-packages/pops/include" \
  POPS_NATIVE_DIM=2 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 POPS_THREADS=1 OMP_PROC_BIND=false FI_PROVIDER=tcp \
  PATH="$POPS_RECEPTION_ENV/bin:/Users/romaindespoulain/miniforge3/bin:/usr/bin:/bin:/usr/sbin:/sbin" \
  "$POPS_RECEPTION_ENV/bin/python" docs/development/api_040/run_installed_checks.py \
  --output "$POPS_RECEPTION_OUTPUT" \
  --test tests/python/integration/amr/test_public_initial_field_ghost_failure.py::test_public_initial_ghost_rank_fault_keeps_prepublication_owner
```

Avec le même préfixe env, refaire `run_installed_checks.py --identity-only
--output "$POPS_RECEPTION_OUTPUT/after"` même après un échec, puis comparer
l'identité entière et les bytes `source-files.json` avant/après. Pour MPI2,
utiliser une sortie distincte vide et `run_installed_mpi_checks.py --output
"$POPS_RECEPTION_OUTPUT" --ranks 2 --dimension 2 --threads 1 --timeout 1800
--test tests/python/integration/amr/test_public_initial_field_ghost_failure.py::test_public_initial_ghost_rank_fault_keeps_prepublication_owner`.
Ce driver conserve ses authentifications avant/après et XML par rang.
Les commandes exactes et environnements des exécutions reçues sont dans les
`root-commands.json` liés aux sorties du sceau. Le
[lecteur indépendant reproductible](/Users/romaindespoulain/dev/tmp/sol61-sdk14-initial-capture-fault-independent-20261002/README.md)
permet le recompute offline sans importer PoPS ni écrire dans les originaux.
Le [driver ROOT](/Users/romaindespoulain/dev/tmp/pops-api040-native-reception-evidence-20261001/sdk14-capture-fault-root-receiver.py) est figé ;
son reçu existant se lit sans le régénérer. Les hashes ne sont pas un résultat CI.

Historique SDK13 (avant le build SDK14 et sa réception positive) :

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
