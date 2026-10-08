# GCC13 private carrier portability @1

Les jobs GNU13.3 Serial/OpenMP/MPI ont déclenché un internal compiler error dans
apply_identity_attributes, pendant l’instanciation de vector<Carrier> dans
rematerialize_fields_after_topology_change. La CI identifie le type local, sa
création via make_shared<vector<Carrier>>, puis le chemin commit_bootstrap_level.
L’avertissement de linkage interne précédent est distinct du crash fatal.

Le contrat privé version1 place le même agrégat dans Impl comme
AcceptedAuxiliaryStateCarrier, et conserve un alias local Carrier. Les quatre
champs, leur ordre, leurs opérations implicites, allocations, shared_ptr/vector,
Kokkos View et les pointeurs empruntés sont inchangés. Les captures, durées de
vie, clocks/session/epochs/paramètres et gardes exactes restent identiques.
L’inverse de cette transformation restitue byte-pour-byte le CPP initial.

Aucun header signé ni ABI public ne change : signatureSDKaf8, ABI13/package8/read2
restent identiques. Le Native et consommateurs de cette TU doivent cependant
être reconstruits/relinkés et authentifiés à la nouvelle source.

La syntaxe Clang enregistrée et la [contre-revue Source](evidence/draft_status_20261008/gcc13-carrier-source-review.json)
admettent le patch ; elles ne prouvent pas la réparation GNU13.3. Les nouveaux
jobs CI GNU13.3 au SHA intégré doivent compiler la TU réelle. Une sonde réduite
GNU13.4 préparée, non exécutée, ne remplace pas cette preuve : un baselinePASS
avec13.4 indiquerait seulement une non-reproduction par cet autre compilateur.

La réception Field/Registry/PDE/OwnRetry/Native8, ainsi que CUDA complet, reste
requise après ce changement de CPP privé. Aucune acceptance historique a936
n’est transférée par l’égalité du headerSignature.
