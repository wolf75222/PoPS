# Astra contre-revue native Sol6

Revue du diff `reconstruction.hpp`, `test_weno_convergence.cpp` et
`test_program_runtime.cpp`, puis lecture des transactions natives dans
`src/runtime/system/system.cpp`. Aucun changement de ces fichiers par Astra.

Pas de finding bloquant sur le diff inspecté. Minmod évite le produit des
différences pour déterminer le signe. VanLeer emploie le rapport borné des
magnitudes puis multiplie la plus petite : les tests couvrent subnormal minimal,
maximum fini, signes opposés, zéro, NaN et infini. Les valeurs invalides restent
invalides pour les gardes de flux. Ceci ne qualifie pas les anciens MC/Superbee,
dont le traitement des NaN est distinct.

Le test TVD est un témoin numérique limité au scalaire périodique, flux upwind,
Euler explicite, CFL 0.5, 600 données pour chacun des deux limiteurs. Il ne prouve
ni positivité système, ni comportement AMR, ni tous les CFL/flux, et son commentaire
respecte cette frontière.

Le test transactionnel consomme une durée de cache, modifie état/histoire/cache et
stade un échange, provoque un refus natif, réessaie, accepte l'enfant puis annule
le parent. Le code `finalize_step_transaction` restaure correctement le pointeur
du snapshot parent après finalisation de l'enfant ; `rollback_step_transaction`
restaure ce snapshot. Les assertions vérifient durée/mailbox en plus des états,
horloge et histoire. Couverture pertinente ; elle ne démontre pas les collectifs
MPI adversariaux ni AMR. Compilation/exécution de ces cibles sous responsabilité root.
