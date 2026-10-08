# Inventaire backend ROMEO - 2026-10-01

Inventaire réel en lecture seule ; aucun sbatch/srun, build, installation, copie de source, import numérique ou backend test exécuté. Réalisé depuis MAIN source `3b2b05098e6bf4f23dccc67b43baf4f1556b0e39`, branche `codex/api-040-native-20260928`, avec fichiers non suivis existants. Gel documentaire ensuite dans le worktree docs-only base `6fafed46`. L'état ordonnanceur est un instantané et doit être revérifié avant une campagne.

L'objectif utilisateur, attachment `0e1884a0-8812-45f3-ac4e-1f68b82a78e3/goal-objective.md`, lignes34–39, autorise explicitement transferts/builds/campagnes CPU/MPI/GPU via ssh romeo sans plafond personnel. Cette autorisation ne supprime pas les limites du compte et ne transforme pas une ressource visible en backend reçu. La prochaine tranche native locale AMR v12 reste obligatoire avant l'élargissement ROMEO.

| Probe / ressource | Fait observé | Limite |
|---|---|---|
| SSH BatchMode | Alias romeo atteint hostname romeo1, utilisateur rmdraux ; connexion rc0 | Aucun secret/credential exporté |
| squeue -u rmdraux | En-tête seul, aucun job actif ou pending du compte | État instantané ; jobs d'autres comptes non inspectés |
| Association Slurm r250127 | Cluster romeo2024, QOS normal ; GrpTRES cpu1536,gres/gpu60 ; MaxSubmit100 | Budget heures restant inconnu ; plafond global partagé, pas promesse d'allocation immédiate |
| Association default | MaxJobs0/MaxSubmit0 | Choisir explicitement r250127 lors de la future soumission |
| instant par défaut | UP, max1h, 102 nœuds, TotalCPUs25152, GPUs232 | Compteurs de partition ; ne décrivent pas l'accès libre instantané |
| short | UP, max24h, 99 nœuds, TotalCPUs24384, GPUs224 | Allocation via SLURM |
| long | UP, max30j, 64 nœuds, TotalCPUs16128, GPUs160 | Allocation via SLURM |
| Nœuds CPU | sinfo :192 CPU, mémoire1160629+ MB, sans GPU | Version/architecture compute à authentifier dans le futur job |
| Nœuds GPU | sinfo :288 CPU, mémoire820802 MB, gres gpu:h100:4 | Pas de nvidia-smi/device execution ; aucune qualification CUDA/GPU |
| Login | x86_64 ; Python3.9.21 ; GCC11.5.0 ; aucun module chargé | Aucun build/compute sur login |
| CUDA/NVHPC | module avail :cuda12.6, NVHPC24.11, variantes HPCX/BYO/noMPI | Disponibilité != compilation d'un paquet PoPS compatible |
| OpenMPI | GNU4.1.7, GNU4.1.6.16.0 ; AOCC4.2/4.1.7 et4.1.6.16.0 ; aarch64/4.1.7(+cuda) | Charger le profil adapté dans job ; ne pas confondre avec MPICH local |
| Stockage | df /home/rmdraux :gpfs2.8P,524T utilisés,2.3P disponibles | Taille filesystem globale, pas quota utilisateur |
| Quota | quota -s rc1, stdout/stderr vides | Quota individuel/expiration/budget portail inconnus |

Les champs QOS normal/instant/short/long consultés pour MaxWall, MaxJobsPU, MaxSubmitPU, MaxTRESPU et GrpTRES sont vides. Les MaxTime des partitions et associations ci-dessus restent les limites explicitement constatées ; les champs vides ne prouvent pas une absence universelle de limite.

Sources et installations existantes, non modifiées :

| Chemin distant | Identité / éléments observés | Utilisation future |
|---|---|---|
| /gpfs/scratch/rmdraux/hoffart-pops-20260914 | environment.sh,build_runtime.sbatch,qualify_runtime.sbatch,qualify_shared_binding.sbatch,qualify_restart.sbatch,shared_segment.sbatch,src,releases,runtime-releases,runs,evidence | Préserver les campagnes/checkpoints ; lire profils/script exacts avant réemploi. Pas d'identité extension affirmée par cette inspection |
| /gpfs/scratch/rmdraux/pops | Git HEAD c15b6836ba7c32bb3ad39ac66e9fd4b86e9959da | Ancien checkout ; ne représente pas MAIN ni SDK5ec |
| /gpfs/scratch/rmdraux/pops-adc-runtime-authority-1cae357aa | Git HEAD49e536027b3e18fbd4b0f823cb6842dcec3021d1 | Ancienne authority ; aucune réception transférée au source courant |
| /gpfs/scratch/rmdraux/kokkos et kokkos-src | Git HEAD15dc143e5f39949eece972a798e175c4b463d4b8 | Compatibilité flags/ABI/install non établie |
| /gpfs/scratch/rmdraux/kokkos-install, kokkos-install-pic, kokkos-x64-pic, kokkos-x64-pic-serial | Répertoires existants | Ne pas écraser ; lire CMake/version/compiler avant réemploi |

Scripts dépôt identifiés : [romeo_run.sh](../../../tests/gpu/romeo/romeo_run.sh) charge cuda12.6, cible nvcc C++20/sm90 et `$HOME/pops_dsl_gpu`; [romeo_kokkos_build.sh](../../../tests/gpu/romeo/romeo_kokkos_build.sh) configure CUDA+Serial/Hopper90 avec nvcc_wrapper dans `$HOME/pops_dsl_kk`. Ces chemins historiques n'ont pas été trouvés parmi les répertoires home PoPS de la probe bornée. Aucun script n'a été exécuté ; aucun n'est une réception NativeAMRRoot.

Reproduction minimale de l'inventaire, sans mutation :

```sh
ssh -o BatchMode=yes -o ConnectTimeout=12 romeo 'hostname'
ssh -o BatchMode=yes romeo 'squeue -u rmdraux -o "%.18i %.12P %.32j %.8T %.10M %.10l %.6D %R"'
ssh -o BatchMode=yes romeo 'sinfo -o "%P %a %l %D %c %m %G"'
ssh -o BatchMode=yes romeo 'sacctmgr -nP show assoc where user=rmdraux format=Cluster,Account,Partition,QOS,DefaultQOS,GrpTRES,MaxTRES,MaxJobs,MaxSubmit'
ssh -o BatchMode=yes romeo 'scontrol show partition'
ssh -o BatchMode=yes romeo 'bash -lc "module -t avail"'
ssh -o BatchMode=yes romeo 'quota -s'
```

Les probes réellement utilisées ont été groupées par Python distant `subprocess.run(...,timeout=5..15,capture_output=True)` à travers ssh BatchMode avec timeout global ; stdout bornée à8–14k caractères par commande. Les répertoires ont été listés à un seul niveau par glob, sans lecture des contenus scientifiques ni recherche globale. Ces commandes sont une reproduction, pas une nouvelle campagne.

Après réception native AMR v12 locale : revérifier squeue/association/quota, créer une installation distincte du checkout final avec les scripts dépôt, authentifier SHA source, SDK/header, compiler/Kokkos/MPI/CUDA, roue et DSO, puis exécuter un cas représentatif CPU sous SLURM avant variantes MPI/GPU. Enregistrer les équations/oracles réels, XML non filtrés, checkpoints/identités et réception indépendante. Ne pas réutiliser le hash DSO local Apple arm64 sur ROMEO, ni appeler GPU reçu à partir de module avail.

La [documentation officielle de connexion](https://romeo.univ-reims.fr/documentation/ressources/romeo_2025/se_connecter/) confirme les quatre serveurs login romeo1–romeo4. La section d'entrée a été consultée ; les rubriques d'exécution/installation doivent être lues au moment du profil réellement retenu.
