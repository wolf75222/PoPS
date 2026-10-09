# SDKbb416 : réception AMR12 Serial et MPI2

La [preuve ROOT avec SHA des fichiers réels](sdkbb416_amr12_scientific_reception.json) reçoit quatre cas Serial et quatre cas MPI2 : N8/N16, une/deux composantes, deux pas, checkpoint, recharge et replay. Le lecteur officiel retourne `received` dans les deux exécutions. Les sceaux d'inventaire et d'approbation sont distincts, calculés par ROOT avant réception ; les fichiers du premier refus restent conservés.

| Vérification réelle | Résultat | Durée |
| --- | --- | --- |
| Variantes natives Serial | 4 passes, zéro failure/error/skip | 454.001168 s |
| Variantes natives MPI2 | 4 passes par rang, zéro failure/error/skip | 930.587080 s |
| Protocole natif Serial | 1 passe, 9 refus nommés, rollback exact | 153.777685 s |
| Protocole natif MPI2 | 1 passe par rang, 10 refus collectifs, rollback exact | 262.359533 s |
| Lecteur intégré et attaques Source | 357 passes, zéro skip | 8.02 s |

Source native `fc670393ce346557583a64dac82908541483fef8`, SDK `bb416020c8971068a97b22b37fa6a2d495b28a8f42c562d9bedceee8a7b6aa5d`, DSO `488123fe4d7c4ecb3ac604f4eec9e97525a8e745034ea01a69624616d8b09e54`. Les 1 132 fichiers Python/header et l'identité du paquet construit sont identiques avant/après chaque lot. Le paquet importé est celui de `pops-api040-ir17/lib/python3.12/site-packages/pops`, authentifié par les runners ; le prototype du handoff n'est pas utilisé. Backend réel : arm64, Apple LLVM21, CPU Kokkos5.2, MPICH, Dim2. Le lot Serial utilise un seul processus du binaire compilé avec MPI ; une construction sans MPI reste à qualifier. ABI native6, payload AMR12, accepted-state8 et TagSelection1 restent des domaines de version séparés.

Le profil reçu est `homogeneous-original-composite-Q-tag-selection@3`. Les bytes grown, historiques, diagnostics et horloges sont exacts après recharge/replay ; le masque coarse est strictement partiel (48/64 ou 192/256 cellules actives), le volume composite vaut1 et l'oracle indépendant vérifie Q non linéaire, les contraintes et les montants conservés. [L'audit Serial indépendant](../../../tests/review/hooke_serial_variants_archive.md), [l'audit MPI2](../../../tests/review/hooke_mpi2_archive_review.md) et [le complément refus MPI2](../../../tests/review/hooke_mpi2_refusals_closed.md) lisent les véritables fichiers sauvegardés. La contradiction du protocole est une augmentation de exactement1 ULP dans64 mots coarse Q0 ; les blobs natifs état/diagnostics, les tableaux sauvegardés et les métadonnées reviennent aux bytes d'origine.

Le lecteur @3 corrige explicitement les hypothèses historiques sur les contrôles entiers typés, le registre de services Original vide, les histoires globales, les boîtes fine-only, le contrôleur temporel et les références d'identité en digest brut32. Les lecteurs @1/@2 et les échecs antérieurs restent intacts. La [contre-revue du lecteur](../../../tests/review/sol61_amr_reader_v3_counterreview.md) couvre l'orchestration complète avec les sceaux ROOT existants ; les 357 tests Source ne remplacent pas les exécutions natives du tableau.

Pour reproduire la lecture officielle des archives exactes depuis le dépôt, utiliser un environnement Python avec NumPy, puis :

```sh
POPS_RECEPTION_EVIDENCE=/Users/romaindespoulain/dev/tmp/pops-api040-native-reception-evidence-20261001
env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops/bin/python \
  tests/review/sol61_evolved_stage_amr_saved_reception_v3.py receive \
  --pins "$POPS_RECEPTION_EVIDENCE/sdkbb416-amr12-root-reception-serial/root-owner3.json" \
  --pins-sha256 a70ded5561a0c224b1c99f2754bf137854a785ac0e49c30190ec665d71cfa174 \
  --approval "$POPS_RECEPTION_EVIDENCE/sdkbb416-amr12-root-reception-serial/root-approval3.json" \
  --approval-sha256 647e75645419d59aafb382684de7e6b52124ea362058e44961a6a2a23f533fd3
env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops/bin/python \
  tests/review/sol61_evolved_stage_amr_saved_reception_v3.py receive \
  --pins "$POPS_RECEPTION_EVIDENCE/sdkbb416-amr12-root-reception-mpi2/root-owner3.json" \
  --pins-sha256 df16494175a298b938e31cc316fa11086b0dceb822b442e827fec262928bea6f \
  --approval "$POPS_RECEPTION_EVIDENCE/sdkbb416-amr12-root-reception-mpi2/root-approval3.json" \
  --approval-sha256 de4a3a5d45b0cdd99b7eaafb1bf75243d44fce893ca8047df28e7ad1f0001247
```

Les commandes natives originales sont conservées dans les `result.json` authentifiés et les commandes de build dans les [reçus de construction](sdkbb416_dim2_native_build.md). Exécuter les mêmes nodes avec `run_installed_checks.py` ou `run_installed_mpi_checks.py --ranks 2 --dimension 2 --threads 1`, `env -u PYTHONPATH` et l'environnement natif ci-dessus, toujours vers un nouveau dossier. Le node physique est `tests/python/integration/runtime/test_public_evolved_stage_amr.py::test_public_evolved_stage_amr_checkpoint_and_composite_Q` ; celui des fautes est `tests/python/integration/runtime/test_native_amr12_state_carrier_failures.py::test_native_amr12_state_carrier_refusals_and_exact_rollback`. Les runners vérifient le paquet réellement installé. Le runner Serial doit être suivi de `--identity-only --output DOSSIER/after` ; le runner MPI effectue ses audits avant/après automatiquement.

Portée exacte : état homogène à flux nul. Cette réception n'établit pas les flux constitutifs non constants, la restriction non linéaire, M06/M13 complets, GPU, ROMEO, OpenMPI ou GitHub CI. Elle ne reconstruit pas cryptographiquement le graphe C++→DSO et ne décode pas entièrement les images opaques Program/auxiliaires. Les leases privés de capture ne sont pas déduits des champs sauvegardés. Les huit tests complémentaires publicTag ont ensuite réellement échoué sur la garde de ghosts constants ([preuve indépendante conservée](../../../tests/review/hooke_public_tag_actual_failure.md)) ; cette obligation demeure ouverte, La capture bound/accepted réelle confirme des cellules valides à1−2ulp, des ghosts fins nuls dès le bind et des ghosts coarse nuls après publication. La reproduction exacte des constantes et un contrat explicite de préparation des halos sont des extensions distinctes en cours ; le stockage brut ne promet pas HaloReady. Le lecteur spatial non constant dispose d'un profil distinct et reste à recevoir sur ses propres archives natives.
