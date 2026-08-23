# Analyse des campagnes de performance

## Statut

Aucune valeur ROMEO n’est reportée ici avant la collecte d’une campagne complète
au format `pops.performance.advection-sine.measurement.v3`. Les anciennes
tentatives R1–R8 sont exclues : elles correspondaient au workload C++ autonome
supprimé, ou à une preuve GPU invalide.

## Ce qui sera analysé

Pour chaque point, le collecteur forme le maximum des temps de rang à chaque
répétition. Il publie ensuite la médiane, la MAD et le débit
`2 × cellules_globales × pas / médiane`. Le facteur deux vient des deux étapes
SSPRK2. Les figures générées par `plot_scaling.py` lisent uniquement ce résumé.

Un summary est refusé si la source ou le build Release ne sont pas liés par les
reçus, si `launch.json` ne documente pas les ressources SLURM de chaque point,
si le pas de temps dyadique diverge, si tous les rangs attendus ne sont pas
présents, si les boîtes locales ne pavent pas exactement le domaine, si le
backend/concurrency/MPI ne correspondent pas, ou si l’oracle hors chronométrage
échoue. Le contrat CMake impose `POPS_USE_HDF5=OFF`, même pour MPI : aucune
mesure ne peut donc attribuer à HDF5 une dépendance de cette campagne. Pour CUDA, l’UUID
`cudaGetDeviceProperties.uuid` doit être non vide et distincte pour chaque rang.

La publication finale contient un `COMPLETE.json` qui inventorie et hache les
fichiers bruts et le rapport collecté. Cette preuve de cohérence est nécessaire
pour interpréter un résultat, mais elle ne remplace ni une exécution complète ni
une analyse de l’environnement matériel.

## Gabarit de compte-rendu après collecte

| Campagne | Points qualifiés | Médiane de référence | Observation | Limite |
|---|---:|---:|---|---|
| `serial_reference` | à compléter | à compléter | à compléter | CPU x64 seulement |
| `cuda_reference` | à compléter | à compléter | à compléter | GPU ARM seulement |
| strong/weak OpenMP | à compléter | à compléter | à compléter | réseau et placement |
| strong/weak CUDA MPI | à compléter | à compléter | à compléter | UUID et topologie GPU |

Ne pas interpréter une efficacité faible avant de vérifier le reçu Slurm, les
fréquences, l’affinité, les boîtes locales et l’absence de contention. Ne pas
extrapoler ces résultats uniformes à l’AMR.
