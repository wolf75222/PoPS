# Executer le tutoriel sur plusieurs plateformes

Le modele, les flux et le programme temporel sont les memes dans chaque fichier. Seule la
plateforme change. Chaque fichier la fixe directement et se lance sans argument.

| Maillage et plateforme | SSPRK2 preimplemente | SSPRK2 explicite |
|---|---|---|
| Uniforme, OpenMP 7 threads | [`01_openmp_preset_ssprk2.py`](01_openmp_preset_ssprk2.py) | [`02_openmp_explicit_ssprk2.py`](02_openmp_explicit_ssprk2.py) |
| Uniforme, MPI natif | [`03_mpi_preset_ssprk2.py`](03_mpi_preset_ssprk2.py) | [`04_mpi_explicit_ssprk2.py`](04_mpi_explicit_ssprk2.py) |
| AMR, OpenMP 7 threads | [`05_openmp_amr_preset_ssprk2.py`](05_openmp_amr_preset_ssprk2.py) | [`06_openmp_amr_explicit_ssprk2.py`](06_openmp_amr_explicit_ssprk2.py) |
| AMR distribue, MPI natif | [`07_mpi_amr_preset_ssprk2.py`](07_mpi_amr_preset_ssprk2.py) | [`08_mpi_amr_explicit_ssprk2.py`](08_mpi_amr_explicit_ssprk2.py) |

## OpenMP : sept threads explicites

Le choix d'OpenMP se fait au moment de construire Kokkos. Depuis la racine du depot, preparer
l'environnement, installer Kokkos Serial + OpenMP, puis reconstruire le module :

```bash
bash scripts/setup_env.sh
conda activate pops
bash scripts/kokkos_openmp_conda.sh
bash scripts/build_python.sh --clean
```

Les scripts OpenMP fixent le nombre de threads avant d'initialiser le runtime natif :

```python
import pops

pops.set_threads(7)
```

Le nombre de threads est defini dans le fichier Python. Le bilan final affiche le backend Kokkos
charge. Les scripts se lancent sans argument :

```bash
python docs/tuto/scalar_advection/01_openmp_preset_ssprk2.py
python docs/tuto/scalar_advection/02_openmp_explicit_ssprk2.py
python docs/tuto/scalar_advection/05_openmp_amr_preset_ssprk2.py
python docs/tuto/scalar_advection/06_openmp_amr_explicit_ssprk2.py
python docs/tuto/scalar_advection/09_openmp_amr_gradient_ssprk2.py
python docs/tuto/scalar_advection/10_openmp_amr_synchronous_ssprk2.py
python docs/tuto/scalar_advection/11_openmp_runtime_parameters.py
python docs/tuto/scalar_advection/12_openmp_amr_outputs.py
python docs/tuto/scalar_advection/13_openmp_amr_restart.py
```

Les deux premiers scripts ecrivent les champs utilises par les figures :

- `results/01_openmp_preset_ssprk2.npz` ;
- `results/02_openmp_explicit_ssprk2.npz`.

Le script 12 publie le format choisi par `OUTPUT_FORMAT` a chaque echeance physique
`OUTPUT_EVERY_DT` sous `results/12_openmp_amr_outputs/`. Avec `output.ParaView()`, il produit des
VTU compresses, un PVD temporel standard et la recette portable `.view.json`/`.view.py`; HDF5 et NPZ
restent selectionnables sans changer le graphe du cas. Un vrai PVSM se demande explicitement avec
`MaterializedPVSM` et un `pvpython` reel. `POPS_CATALYST=1` active en plus la pipeline Catalyst 2
strictement pour cette execution OpenMP serie. La visualisation live est refusee avec MPI; les
sorties scientifiques progressives PVTU/HDF5 restent disponibles pour suivre un calcul distribue.
Le script 13 ecrit son checkpoint sous
`results/13_openmp_amr_restart/`.

Les figures comparent ces deux executions OpenMP :

```bash
python docs/tuto/scalar_advection/plot_openmp_results.py
```

## MPI natif : monde explicite

La construction distribuee active MPI et le HDF5 parallele natif. Reconstruire le module avec :

```bash
bash scripts/setup_env.sh
bash scripts/build_python.sh --mpi --clean
conda activate pops
```

Les quatre scripts MPI fixent un thread par rang. Apres `pops.compile`, ils construisent le monde
natif et le transmettent au bind :

```python
pops.set_threads(1)
execution_context = pops.ExecutionContext.mpi_world(artifact)
simulation = pops.bind(
    artifact,
    initial_state={"tracer": initial_state},
    resources={"execution_context": execution_context},
)
```

Le monde MPI vient du runtime natif, sans `mpi4py` ni handle construit en Python. L'identite du
backend et le contrat MPI font partie du cache compile. Chaque fichier se lance avec `mpiexec` :

```bash
mpiexec -n 2 python docs/tuto/scalar_advection/03_mpi_preset_ssprk2.py
mpiexec -n 2 python docs/tuto/scalar_advection/04_mpi_explicit_ssprk2.py
mpiexec -n 2 python docs/tuto/scalar_advection/07_mpi_amr_preset_ssprk2.py
mpiexec -n 2 python docs/tuto/scalar_advection/08_mpi_amr_explicit_ssprk2.py
```

Ces fichiers MPI calculent puis affichent le bilan de chaque rang. Les scripts OpenMP produisent les
figures ; les fichiers MPI n'ont donc pas de branche de publication ni d'ecriture concurrente.

L'API de sortie sait neanmoins executer les writers asynchrones `PER_RANK` et `COLLECTIVE` sur un
worker post-commit MPI. Ces modes exigent `MPI_THREAD_MULTIPLE`. Le runtime duplique une lane par
consumer, ne reprend jamais `MPI_COMM_WORLD` depuis un worker et ordonne toutes les sessions d'un
run dans un FIFO partage. ParaView relaie par defaut des morceaux VTU bornes vers le rang zero avant
de publier le PVTU, sans filesystem partage; `SharedDirectory()` est l'alternative explicite lorsque
tous les rangs voient le meme repertoire.

`LiveVisualization` et le backend Catalyst integre restent strictement serie. Leurs modes `ROOT`,
`PER_RANK` et `COLLECTIVE` sont refuses avant le bind et aucun `catalyst/mpi_comm` n'est transmis.
Une simulation MPI se suit au fil du calcul avec les PVTU ou HDF5 progressifs produits par
`AsyncScientificOutput`.

Le writer VTU generique sait encoder des snapshots cartesiens authentifies 1D, 2D ou 3D, avec les
champs centres cellules dans `CellData` et les champs nodaux dans `PointData`/`PPointData`. La capture
native PoPS et le backend Catalyst integre restent 2D et centres cellules. Catalyst accepte un seul
consumer/pipeline combine et un seul run par `RuntimeInstance`. Sa reservation globale n'est jamais
relachee : un autre run Catalyst demande un nouveau processus. Plusieurs runtimes concurrents dans
un meme processus ne sont pas supportes avec HDF5 asynchrone ou Catalyst. Le worker PoPS fournit
deja l'asynchronisme : l'async interne Catalyst est force a zero et un `CATALYST_ASYNC_ENABLED` actif
est refuse. Le script 12 garde une file en memoire. Une application peut ajouter `DurableJournal`,
mais sa garantie au moins une fois ne commence qu'apres le handoff durable `pending`; elle n'est pas
atomique avec le pas accepte ou un checkpoint et les archives `delivered` non purgees doivent etre
gerees par l'application.

Sur Slurm, les memes scripts restent inchanges :

```bash
srun --ntasks=4 --cpus-per-task=1 \
  python docs/tuto/scalar_advection/03_mpi_preset_ssprk2.py
srun --ntasks=4 --cpus-per-task=1 \
  python docs/tuto/scalar_advection/04_mpi_explicit_ssprk2.py
srun --ntasks=4 --cpus-per-task=1 \
  python docs/tuto/scalar_advection/07_mpi_amr_preset_ssprk2.py
srun --ntasks=4 --cpus-per-task=1 \
  python docs/tuto/scalar_advection/08_mpi_amr_explicit_ssprk2.py
```

## Verifier l'environnement

Apres chaque build :

```bash
python -c "import pops; from pops.runtime.doctor import doctor; print(pops.__version__); doctor()"
```

Le nom exact du backend Kokkos installe est aussi affiche dans les bilans des scripts. Ces petits
cas mesurent surtout les couts de lancement ; leurs temps ne constituent pas un benchmark de
scaling.
