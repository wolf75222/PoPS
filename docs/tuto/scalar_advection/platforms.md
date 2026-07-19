# Executer le tutoriel sur plusieurs plateformes

Le modele, les flux et le programme temporel ne changent pas avec la plateforme. Le backend
est choisi au build, puis le meme script est lance avec les ressources de la machine.

## CPU, un seul thread

Construire le module officiel :

```bash
bash scripts/setup_env.sh
bash scripts/build_python.sh
conda activate pops
```

Sur une installation Kokkos OpenMP, limiter l'execution a un thread donne le chemin CPU
mono-thread :

```bash
OMP_NUM_THREADS=1 KOKKOS_NUM_THREADS=1 OMP_PROC_BIND=false \
  python docs/tuto/scalar_advection/01_pops_library.py
```

Le script affiche le backend Kokkos reel. Il ne faut pas appeler une execution OpenMP
« Kokkos Serial » si le manifest annonce `OpenMP`.

## CPU OpenMP

Le backend OpenMP se choisit lors de la construction de Kokkos, pas avec une variable au
moment du lancement. Dans l'environnement `pops`, preparer une fois l'installation
Serial + OpenMP puis reconstruire le module :

```bash
conda activate pops
bash scripts/kokkos_openmp_conda.sh
bash scripts/build_python.sh --clean
```

Le nombre de threads est ensuite fixe avant l'initialisation de Kokkos :

```bash
OMP_NUM_THREADS=4 KOKKOS_NUM_THREADS=4 \
OMP_PROC_BIND=spread OMP_PLACES=cores \
  python docs/tuto/scalar_advection/01_pops_library.py
```

Sur un cluster, utiliser le nombre de CPU alloue par l'ordonnanceur. Par exemple avec Slurm :

```bash
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export KOKKOS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
srun --cpu-bind=cores python docs/tuto/scalar_advection/01_pops_library.py
```

## MPI

La route distribuee officielle active ensemble MPI et HDF5 parallele natif :

```bash
# --clean est necessaire lorsqu'on bascule depuis un artifact serie deja configure.
bash scripts/build_python.sh --mpi --clean
conda activate pops
POPS_CACHE_DIR="${TMPDIR:-/tmp}/pops-tutorial-mpi" \
OMP_NUM_THREADS=1 KOKKOS_NUM_THREADS=1 OMP_PROC_BIND=false \
  mpiexec -n 2 python docs/tuto/scalar_advection/01_pops_library.py
```

Le cache de code genere est volontairement distinct du cache serie. Un changement de backend
natif ne doit jamais reutiliser un artifact de modele compile pour un autre contrat.

Le script lit le contrat du `CompiledArtifact`. Pour un artifact MPI, il cree explicitement
`ExecutionContext.mpi_world(artifact)` puis le fournit a `pops.bind`. Il n'utilise ni
`mpi4py`, ni collectives Python, ni moteur parallele alternatif. La copie globale finale sert
uniquement a produire la figure apres l'execution native.

Sur Slurm :

```bash
POPS_CACHE_DIR="${TMPDIR:-/tmp}/pops-tutorial-mpi" \
srun --ntasks=4 --cpus-per-task=2 \
  python docs/tuto/scalar_advection/01_pops_library.py
```

Les deux niveaux de parallelisme peuvent etre combines si le module MPI a lui-meme ete
construit contre l'installation Kokkos OpenMP :

```bash
conda activate pops
bash scripts/kokkos_openmp_conda.sh
bash scripts/build_python.sh --mpi --clean
```

On choisit ensuite le nombre de threads par rang :

```bash
export OMP_NUM_THREADS=2
export KOKKOS_NUM_THREADS=2
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
POPS_CACHE_DIR="${TMPDIR:-/tmp}/pops-tutorial-mpi-openmp" \
  mpiexec -n 4 python docs/tuto/scalar_advection/01_pops_library.py
```

## GPU

L'environnement peut preparer une installation Kokkos CUDA avec :

```bash
bash scripts/setup_env.sh --cuda
```

Ce n'est pas encore une promesse d'execution du pipeline Python final. La version 1.0.0
n'annonce pas de provider de contexte GPU executable pour ce tutoriel et doit refuser cette
route avant `run`. Le cas GPU n'est donc ni execute ni presente comme valide ici. Il sera ajoute
quand le `PlatformManifest`, la memoire des composants et le contexte d'execution GPU seront
supportes de bout en bout.

## Verifier l'environnement

Apres chaque build :

```bash
python -c "import pops; from pops.runtime.doctor import doctor; print(pops.__version__); doctor()"
```

Le rapport doit correspondre au backend que l'on veut mesurer. Les temps d'un run de
tutoriel ne sont comparables que si la grille, le nombre de rangs, le nombre de threads et le
backend annonce sont identiques.

## Execution verifiee

Les commandes ci-dessus ont ete executees le 19 juillet 2026 sur un Apple M1 Pro (8 coeurs),
avec PoPS 1.0.0, une grille $64\times64$, 29 pas acceptes et $t_{fin}=0,2$. Le tableau donne
le temps de la boucle native rapporte par le script ; il exclut la compilation initiale du
modele.

| Programme | Backend annonce | Ressources | Temps natif |
|---|---|---:|---:|
| `SSPRK2` preimplemente | Kokkos OpenMP | 1 thread, 1 processus | 0,029741 s |
| `SSPRK2` preimplemente | Kokkos OpenMP | 4 threads, 1 processus | 0,032976 s |
| `pops.Program` explicite | Kokkos OpenMP | 4 threads, 1 processus | 0,033584 s |
| `SSPRK2` preimplemente | Kokkos OpenMP + `MPI_COMM_WORLD` | 1 thread x 2 rangs | 0,034608 s |

Le preset, le programme explicite et le run MPI produisent ici le meme champ final bit a bit
(`max_abs = 0`). Ce petit probleme sert a verifier le parcours, pas a mesurer le scaling : les
couts de lancement et de communication dominent deja le calcul sur $64\times64$ cellules.
