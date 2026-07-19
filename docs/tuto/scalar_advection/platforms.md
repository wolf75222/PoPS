# Executer le tutoriel sur plusieurs plateformes

Le modele, les flux et le programme temporel restent identiques. Chaque fichier montre une plateforme
precise et le cycle complet sans argument de ligne de
commande, helper partage ou branche de detection du backend.

| Plateforme | SSPRK2 preimplemente | SSPRK2 explicite |
|---|---|---|
| OpenMP, 7 threads | [`01_openmp_preset_ssprk2.py`](01_openmp_preset_ssprk2.py) | [`02_openmp_explicit_ssprk2.py`](02_openmp_explicit_ssprk2.py) |
| MPI natif, 1 thread par rang | [`03_mpi_preset_ssprk2.py`](03_mpi_preset_ssprk2.py) | [`04_mpi_explicit_ssprk2.py`](04_mpi_explicit_ssprk2.py) |

## OpenMP : sept threads explicites

Le backend OpenMP se choisit lors de la construction de Kokkos. Depuis la racine du depot,
preparer l'environnement, installer Kokkos Serial + OpenMP, puis reconstruire le module :

```bash
bash scripts/setup_env.sh
conda activate pops
bash scripts/kokkos_openmp_conda.sh
bash scripts/build_python.sh --clean
```

Les deux scripts OpenMP appellent ensuite cette autorite publique avant tout objet susceptible
d'initialiser le runtime natif :

```python
import pops

pops.set_threads(7)
```

Le nombre de threads ne vient donc ni d'un argument cache, ni d'une branche sur le backend, ni d'une
variable shell. Le bilan final affiche le backend Kokkos reellement charge. Lancer les deux variantes
temporelles sans argument :

```bash
python docs/tuto/scalar_advection/01_openmp_preset_ssprk2.py
python docs/tuto/scalar_advection/02_openmp_explicit_ssprk2.py
```

Elles ecrivent respectivement :

- `results/01_openmp_preset_ssprk2.npz` ;
- `results/02_openmp_explicit_ssprk2.npz`.

Les figures comparent ces deux executions OpenMP :

```bash
python docs/tuto/scalar_advection/plot_openmp_results.py
```

## MPI natif : monde explicite

La route distribuee officielle active ensemble MPI et HDF5 parallele natif. Reconstruire le module
pour ce contrat exact :

```bash
bash scripts/setup_env.sh
bash scripts/build_python.sh --mpi --clean
conda activate pops
```

Les deux scripts MPI fixent un thread par rang et ne possedent pas de chemin serie. Apres
`pops.compile`, ils construisent inconditionnellement le monde natif et le transmettent au bind :

```python
pops.set_threads(1)
execution_context = pops.ExecutionContext.mpi_world(artifact)
simulation = pops.bind(
    artifact,
    initial_state={"tracer": initial_state},
    resources={"execution_context": execution_context},
)
```

Il n'y a ni `mpi4py`, ni handle MPI fabrique en Python, ni moteur parallele alternatif. L'identite du
backend et du contrat MPI fait deja partie du cache compile ; aucun repertoire de cache n'est impose
par une variable shell. Lancer chaque fichier sans argument :

```bash
mpiexec -n 2 python docs/tuto/scalar_advection/03_mpi_preset_ssprk2.py
mpiexec -n 2 python docs/tuto/scalar_advection/04_mpi_explicit_ssprk2.py
```

Ces fichiers MPI se limitent volontairement au calcul et au bilan de chaque rang. Les figures sont
produites par les variantes OpenMP, ce qui evite toute branche de publication ou ecriture concurrente
dans les scripts MPI minimaux.

Sur Slurm, les memes scripts restent inchanges :

```bash
srun --ntasks=4 --cpus-per-task=1 \
  python docs/tuto/scalar_advection/03_mpi_preset_ssprk2.py
srun --ntasks=4 --cpus-per-task=1 \
  python docs/tuto/scalar_advection/04_mpi_explicit_ssprk2.py
```

## GPU : emplacement reserve

Aucun script GPU n'est cree dans ce tutoriel. Une installation Kokkos CUDA ne suffit pas a prouver
le pipeline Python final, et le tutoriel ne fabrique pas de contexte ou de fallback. Le futur fichier
GPU prendra la place suivante dans le parcours seulement lorsque son API publique reelle sera
fournie :

```text
05_gpu_<api-publique-a-definir>.py
```

## Verifier l'environnement

Apres chaque build :

```bash
python -c "import pops; from pops.runtime.doctor import doctor; print(pops.__version__); doctor()"
```

Les scripts restent minimaux et ne dupliquent pas cette verification dans le chemin de simulation.
Le nom exact du backend Kokkos installe est affiche dans les quatre bilans. Les temps de petits cas
$64\times64$ sont domines
par les couts de lancement et ne sont pas publies comme un benchmark de scaling.
