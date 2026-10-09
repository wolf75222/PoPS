# AMR layout-transfer host-range portability @1

CUDA737206 rejetait trois range-for bracés de la TU réelle, alors que le build
CPU les compilait. La sonde737830, même NVCC12.6.85/GNU13.4/Kokkos5.2.1/flags
stricts, reproduit les trois erreurs dans l’original et compile le candidat.
Le host C++ émis par cudafe transforme les anciennes plages en `({…})` ; les
nouveaux tableaux nommés évitent cette émission invalide.

Le contrat privé version1 utilise std::array<const MultiFab<Dim>*,3> pour les
masques, std::array<std::size_t,5> pour les valeurs sérialisées du budget, et
std::array<const MultiFab<Dim>*,4> pour les champs de capacité. Les cardinalités,
éléments, ordre et instant d’observation restent identiques. Chaque corps de
boucle, garde de overflow/shape, budget, calcul flottant et wire contract est
byte-identique après inversion des trois headers et de l’include array.
Aucune allocation heap, nouveau kernel, modèle ou branche par formule ajouté.

C’est un changement du CPP privé, sans modification de header signé : SDKaf8,
NativeABI13, package8/read2 restent inchangés. Les consommateurs liés à la TU
et le Native doivent néanmoins être reconstruits/relinkés à la nouvelle source.
La sonde qualifie la compilation de cette TU CUDA seulement ; elle ne qualifie
ni sept bindings CUDA, ni Native GPU installé, ni physique/convergence GPU.

Preuves : [contre-revue Source](evidence/draft_status_20261008/cuda-array-source-review.json),
[réception originale/candidate](evidence/draft_status_20261008/cuda-private-probe-independent.json),
[patch exact et scripts](wip/cuda_braced_range/README.md).
