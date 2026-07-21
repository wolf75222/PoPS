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

Le script 12 publie un instantane tous les `OUTPUT_EVERY_STEPS` pas acceptes sous
`results/12_openmp_amr_outputs/`. Son format tient sur une ligne : `output.ParaView()`,
`output.HDF5()` ou `output.NPZ()`. Le script 13 ecrit son checkpoint sous
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
