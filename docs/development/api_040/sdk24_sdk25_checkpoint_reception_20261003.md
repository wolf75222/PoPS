# SDK24/25 — checkpoint Uniform9, migration et consommateurs reçus

La tranche obligatoire UniformCP9/V@3 est reçue par ROOT dans les profils ci-dessous. Cette réception ne ferme aucune des94 obligations dans sa portée complète. Les preuves Source, build/install, Native, math et engineering sont distinctes ; GPU, inter-node, CI et performance à calcul comparable restent non reçus. L'[index transportable](sdk24_sdk25_receipt_index_20261003.json) épingle14 reçus ROOT et six preuves/commandes complémentaires par SHA exact. Tous les historiques sont conservés.

## Code et identités

[ROOT intégration Source/build](sdk24_sdk25_receipts/root-sdk25-main-source-build-integration.json) authentifie Main8be7294199266d81eeff9a9af05aa4582a0fcf55 et le freeze ROMEO717936347672aeb4bc9146e2b3065bfd42cf1ef5 :1192 blobs production égaux et24 fichiers modifiés réellement comparés Git/filesystem. Source79 est réellement **79PASS,0FAIL/ERROR/SKIP**,17NativeLoader explicitement exclus ; le package Source exact est importé sans prototype. Les17 migrations Native ont été reçues ensuite, séparément. Source552 exact7bb historique n'incluait pas runtime_instance_gate ; SourceRED545P1F et ENOSPC454 ne deviennent pas des réceptions complètes.

[SDK25 build732329](sdk24_sdk25_receipts/root-sdk25-build-reception-732329.json) reçoit24objets CXX effectivement reconstruits,1192 fichiers Source,1157 installés,1148 identity-source,1154 wheel-record et SDK24 préservé1156 fichiers. Native `a661f3b1a7440850985d44f4b3936a94edd7059e570a20633bbe3f96bc64b97e`, header `8ee6d09b428cfe172d08008253c2ea490eb733f9b1458f6f2ed8da1584effbf5`, codegen `778b74730e4b09c0ff9d7c85cfbae5cc1ea242259fe0304929aff26a6f8db327`. ABI8, UniformCP9 et AMRCP12 sont distincts et inchangés ; un build ne qualifie pas une simulation.

Le contrat `checkpoint_restart_epoch@1` authentifie tous les owners, leurs chemins layout canoniques et leurs priorRun/lineage typés avant mutation ; `checkpoint_regrid_epoch@1` est distinct. Snapshot/compensation restaurent les owners et gardent l'autorité antiABA monotone. La nouvelle capture diagnostic candidate@1 est additive : depth1external noncommitted/lifecyclevalide, sans restart/solve pending. Ancienne API noarg accepted-idle, symboles/layouts publics et chemin AMR restent conservés. Aucun modèle/formule/nom de slot n'a été reconnu dans le cœur ; aucune équation, constante ou garde scientifique n'a été relâchée.

## Réceptions SDK24 conservées

Fan–Li CPU732296 et MPI2same-node732307 sont **deux reçus ROOT distincts** : CP9 full-grown State, trois refus, restore et continuation/replay5–8 bit-identiques ; math OriginalFan–Li15=10+5 sur N16×16,dt1e−4,SSPRK2 avec SPD/rho,GL24/48 et DOP853 sur les vraies images valid. MPI2 est un scénario collectif/deuxrangs, avec partition authentifiée et parité global valid. Cela corrige le wording préliminaire « pasMPI2CP9 » ; les réceptions canonical/reverse SDK21 restent séparées. Aucun AMR, autre troncature, Field/cache/history ou formule Ghost n'en découle.

V@3 CPU732320 reçoit width1/N8/world1/rootowner[] homogène OriginalQ/projection/lifting/history et CP12 replay, pas le signedD/fluxCF discriminant. TransportsCPU732321 reçoit13phases State local/complete/valid et CP9 refus/restore/replay : son oracle PDE indépendant n'a pas été recalculé. Ces portées ne se fusionnent pas.

## Réceptions SDK25 fermées par rejeu ROOT

| Reçu et jobs | Portée authentifiée | Limites |
|---|---|---|
| [CP9 profils732338/732339](sdk24_sdk25_receipts/root-sdk25-cp9-profiles-reception-732338-732339.json) | Deux profils parbackendCPU1/MPI2same-node : full-grown local/global valid State, clock, checkpoint9/refus/restore/replay et join par noms de blocs. Fan–Li original : huitpas continuous/replay, neuf images valid parbranche, SSPRK2/GL24/48/DOP853 indépendants. | Les deux transports sont **State-only**, pas oracle PDE reçu. Aucun Uniform→AMR,Field/history/cache/Ghostformula ou autretroncature hérités. |
| [V5 matrix732335/732336](sdk24_sdk25_receipts/root-sdk25-v5-matrix-reception-732335-732336.json) | Quatreprofils width1/2,N8/16 parbackendCPU1/MPI2, **huit profils logiques ; douze nœuds JUnit (quatre CPU et huit MPI), répartis dans trois XML** ; C25 allrank avantbind, initialcomplet, CP12State/history/restart/replay. ROOT a rejoué le lecteur et vérifié la contre-revue indépendante (15 PASS). RatioOriginal maximal rapporté/recalculé9.20720856194327e−11 sous garde1e−10 inchangée. | Rootowner homogèneOriginalQ uniquement ; pas graphe Native childowner, signeddiffusion/CFflux discriminé,Fieldcache ou Ghostformula. |
| [Migration732331 et provisoire732332/732333](sdk24_sdk25_receipts/root-sdk25-migration-provisional-native-reception-732331-732332-732333.json) | MigrationCPU1 **23PASS=17NativeLoader+6Source** : historique2→8 exactvalid-only, restart public `valid_only_legacy8`, refus ciblés/no-clobber. ProvisoireCPU1/MPI2same-node : vraiCP9publiéstep1/t1/256 puis faultPython explicite et compensation ; Stategrown/valid/clockstep0 restaurés exactement. | Grown/diagnostics courants validés puis omis du target8, jamais fabriqués pour2. ALLmigrationC25 nonretenu. Provisoirecursors[] seulement ; compteurantiABA nonmesuré, diagnostics/historybytes nonmesurés, aucune faute stageC++/retry/math nouvelle. |
| [Non-régression732340/732341/732343/732345](sdk24_sdk25_receipts/root-sdk25-nonreg-reception-732340-732341-732343-732345.json) | Diagnostics : septcas logiques par lancementCPU/MPI2,3Source+4NativeUniform/AMR,24pairesNPZ bitexact au total +capacity/refusal. RetryCPU : deuxtests Nativeaccept/reject/exhaustion. HistoryCPU : unitemprocessisolé appelle huitchecks authentifiés,5Source+3Native ; AB2≤1e−12,coldringless>1e−9,variable-stride `np.array_equal`. | Pas de nouvelle comparaison offlinefullState pourretry. AB2 n'est **pas un byteclaim**. stdoutenfantsuccesshistory nonretenu. Les diagnostics nonreg ne prouvent pas les diagnostics de la faute provisoire. |

Chaque reçu rapporte des replays ROOT identiques aux rapports indépendants ; les autorités scientifiques indépendantes restent séparées des sceaux ROOT. Le C25 retainedTU/commande/DSO ne constitue pas une preuve cryptographique transitive compiler→DSO. Aucune duplication de rang ne multiplie les expériences logiques.

## Négatifs et compatibilité préservés

Les builds/runs négatifs SDK22/23 et SDK24 restent FAILED : C++Layout sansbegin/end, JSONclockmethod732229, admissionCP9member732268, closedrun732282, childomissionff4c, consumers732305/732306, migrationdiagnosticpair732309, before-imagejoinordernative/NPY732308 avantinjection et diagnosticaccepted-idle732322(events=[] avantpublication). Les corrections et reruns reçus ne réécrivent pas ces échecs.

L'historique v2→8 reste valid-only ; `state_storage='valid_only_legacy8'` est explicite et le défautstrict9 refuse8. Le carrier9 est décodé par le codec commun, géométrie/components/Real et projectionvalidbits attestés ; la paire diagnostic courante est typée/budgétée/décodée, nonempty permise, puis omise. Aucun ghost ou diagnostic historique absent de2 n'est déduit du courant.

732342 reste **FAILED,zérotest**, mauvais sélecteur de fonction pour `PythonProcessFile`, sans exécutionNativeAB2. La [correction épinglée](sdk24_sdk25_receipts/correction.json) sélectionne le fichier entier ;732345 reçoit ensuite l'item isolé et ses huitchecks. Le nom ancien bit-identical ne remplace pas la vraie gardeAB2≤1e−12.

## Reproduction officielle

Le dossier de préparation ROOT est `/Users/romaindespoulain/dev/tmp/sol61-romeo-api040-composition15-preparation-20261002`. Setup `scripts/setup_env.sh` unefois/worktree, puis build incrémental réel `scripts/build_python.sh --dim 2 --mpi --wheel-dir "$run/wheels"`, en préservant l'installation précédente suivant build25.sbatch. Les scripts et l'executionSource79 exacte sont dans l'index ; aucune nouvelle soumission n'est lancée par cette documentation.

Pour Native, utiliser le **vrai package installé** dans l'env authentifié, `env -u PYTHONPATH`, et vérifier `pops.__file__` sous `sys.prefix`. Ni SourcePYTHONPATH, mini-runtime, callbackmodèle central ni seuil alternatif :

```sh
env -u PYTHONPATH python docs/development/api_040/run_installed_checks.py --output "$out" --test tests/python/unit/codegen/test_checkpoint_migration.py
env -u PYTHONPATH python docs/development/api_040/run_installed_checks.py --output "$out" --test tests/python/integration/io/test_time_history_checkpoint.py
env -u PYTHONPATH python docs/development/api_040/run_installed_mpi_checks.py --output "$out" --ranks 2 --dimension 2 --threads 1 --timeout 6000 --test tests/python/integration/runtime/test_public_evolved_stage_amr_v3.py
```

Serial n'a pasdimension/threads ; MPIlesexposeexplicitement. HarnessROOT : `python3 BASE/submit_native.py --label <label> --world-size <1|2> --source-freeze 717936347672aeb4bc9146e2b3065bfd42cf1ef5 --test <fichier-ou-node-authentifié>`. AB2doitprendrelewholefileci-dessus. LesCPU sontDim2OpenMP,MPICHcompilé ; CPU1 ne signifie pasKokkosSerial niMPI2.

## Travail restant

Les94 obligations restent actives dans leurs scopes complets. La tranche obligatoire CP9/V@3 bornée est terminée ; prochain lot : composition publique M18/W09 (quadrature/near-boundary et distinctionBOUNDARY/INFEASIBLE) ou couplage public M19 BGK/Vlasov–Poisson, encore à implémenter/recevoir. Les anciennes preuves free-streaming/entropie restent historiques, pas une réception de ces nouveaux couplages. AMR/GPU/inter-node/CI/performance comparable et toute physique manquante demeurent explicites.
