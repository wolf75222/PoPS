# ROMEO Halo et reprise après échec - réception bornée @10

Le vrai paquet construit depuis `5582d98ad49dd5d93d5ededa43448fb9c54e8cd4`
reçoit les quatre cas ci-dessous sur CPU x86 Kokkos OpenMP/MPICH Dim2.
La contre-revue Sol6.1 et son rejeu ROOT vérifient 39 pins bruts, 216 membres
exportés et six XML de rang : aucun échec, erreur ou skip. Le
[reçu](romeo_halo_stage_v10_reception.json) lie les deux sceaux ROOT aux archives
et aux lecteurs exacts. Cette réception ne ferme pas la mission complète.

| Cas réel | Job SLURM | Rangs | Pytest par rang | Acquis |
|---|---:|---:|---:|---|
| Croissance, sous-pas 5/2 | 731349 | 1 | 95,727 s | Cellules actives, carrier complet constant, checkpoint et replay exacts |
| Croissance, sous-pas 5/2 | 731350 | 2 | 94,245 s | Même cas collectif, origines et bibliothèques de chaque rang |
| Échec après préparation/fence | 731351 | 1 | 97,998 s | Deux owners en échec, rollback, retry et contrôles alignés |
| Échec après préparation/fence | 731352 | 2 | 108,383 s | Injection sur le dernier rang, vote collectif et rollback |

Le paquet Python/pytest/NumPy vient du clone ENV7 ; les dépendances du module
natif et du compilateur viennent d'ENV3, vérifiées par les origines chargées.
Les 1 137 sources installées correspondent exactement à Git avant/après chaque
job. DSO `3496ac88cc8ee99b314f5c21f12afe2bc71338f20be3b754d90c29e119388446`,
SDK `678ecad64a90a445dec23c8de219358cac76ad038012429973093c034fa580e8`, ABI8,
CP12/accepted9. ROOT a aussi relu les bytes distants du DSO, des 1 137 sources,
de la wheel et du registre Conda après les jobs : tous les hashes correspondent.
Les inventaires ENV3/7 complets restent des fingerprints capturés par les jobs.
Il n'existe pas de preuve cryptographique générale C++→DSO à l'instant du calcul.

Le calcul indépendant de croissance a une erreur maximale de
`6.661338147750939e-16`. Il distingue un pas coarse unique, un ratio temporel2
et l'omission du reste du rapport5/2. Le composant constant est exactement1
dans tous les ghosts capturés. Après échec, les carriers, états valides,
métadonnées et checkpoints retrouvent les images d'avant tentative. Le retry
est comparé à un contrôle continu et à un autre owner ayant réellement subi
la même tentative rejetée. Les ordinaux causaux3 contre2 restent présents ;
aucune remise à zéro de l'allocateur n'est utilisée. Le résidu de croissance
indépendant du retry vaut au plus `4.440892098500626e-16`.

Les racines effectives de calcul sont les `/tmp/api040-v10-<job>-<suffix>`
attestés sur XFS des nœuds ROMEO. Les exports durables conservent ces chemins
originaux et les bytes, sans réécrire les preuves. Le job731317 antérieur reste
FAILED : GPFS refuse `renameat2(RENAME_NOREPLACE)` par EINVAL22. Le job731329
vérifie séparément cette incapacité GPFS et la capacité XFS, y compris les
refus de destination existante. Aucune relaxation du contrat de publication
ou substitution de résultat n'est introduite. Un SIGKILL/perte de nœud peut
empêcher l'export et interdit une réception sans preuve complète.

Rejeu de la contre-réception enregistrée, sans import de PoPS :

```sh
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 \
  /Users/romaindespoulain/miniforge3/envs/pops/bin/python \
  /Users/romaindespoulain/dev/tmp/sol61-romeo-v10-independent-reception/audit.py
```

Les entrées SLURM exactes et leurs hashes sont dans
`/Users/romaindespoulain/dev/tmp/sol61-romeo-api040-reception-v10/plan-pins.json`.
Les vrais nœuds pytest sont `test_public_accepted_halo_five_halves_substep_growth`
et `test_public_accepted_halo_rank_local_failure_restores_before_publication`
dans `tests/python/integration/amr/`. Les construire depuis le même gel avec
`scripts/setup_env.sh` puis `scripts/build_python.sh --dim 2 --mpi` et utiliser
`run_installed_checks.py --test <nœud>` avec `env -u PYTHONPATH` permet une
nouvelle campagne, dont les résultats devront être reçus séparément.

Limites : MPI sur un même nœud seulement ; pas de GPU, MPI inter-nœuds,
publication directe GPFS, panne du transport MPI lui-même, Field→Ghost initial,
autres modèles ou réception du SDK11/12. Les durées incluent JIT et setup ;
elles ne mesurent pas une performance de solveur à calcul comparable. Le job
étranger696509 et les anciennes campagnes restent préservés. Aucun résultat
GitHub CI n'est déduit de ces exécutions.
