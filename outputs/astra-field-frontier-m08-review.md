# Revue indépendante d219db7 et réception M08

Base examinée : d219db7, a5d9be7, 4713a9b, appliqués comme dépendances dans le checkout isolé PoPS-principal-group. Aucun fichier de production `python/pops` modifié par cette revue.

## Frontière de publication

Trois contre-tests source indépendants réussissent dans `test_field_publication_frontier_independent.py` :

1. Une vraie source dépendant du champ publié par un premier solve forme un nouvel état ; le second solve publie seulement ses deux entrées actuelles. L'ancien contexte reste refusé sur cet état, même à la même date physique.
2. Une altération adversariale du véritable nœud de coefficients fait lire une autre version du même bloc que la charge : publication refusée pour provenance actuelle ambiguë.
3. Un état obsolète réellement lu par l'équation ne peut être remplacé par un état supplémentaire courant lors de la publication ; le champ reste également refusé par le RHS de cet état courant.

La coupure aux nœuds State élimine la provenance historique ayant servi à construire l'état ; elle ne supprime pas les identités des entrées de charge/coefficient de l'équation actuelle. La coupure aux solves ancêtres distingue une observation de résultat d'une liaison actuelle de l'équation. Aucun défaut bloquant supplémentaire reproduit sur cette frontière. Cela ne constitue pas une validation de toute réécriture IR possible ni une preuve native.

## M08 : finding et correctif

L'oracle emploie bien -Laplacien périodique à cinq points, le gradient centré (x,y), la vitesse (d_y phi,-d_x phi), les faces Rusanov d'ordre1 et la divergence conservative. Un test analytique indépendant avec mode cos(2x)sin(3y) vérifie signe de Poisson, coefficients Fourier discrets, orientations des deux gradients et divergence nulle.

Le programme possède deux Model/Case blocks hétérogènes charge/tracer. Un solve initial à c=0 est partagé par leurs deux RHS, le probe distinct reste à c=0 et hors trajectoire, puis un solve du prédicteur à c=1 est partagé par les deux seconds RHS. Les deux commits suivent .5 U0 + .5(U1+dt R1). Les histoires depth1 décrivent le dernier pas : stage0/probe à t_end-dt et stage1 à t_end ; l'état final accepté est à t_end. Le contrôle natif reste à exécuter.

Finding : l'ancien seuil FV 3e-7 ne discriminait pas le réemploi erroné du champ initial au second RHS. Les écarts max fresh/stale après deux pas dt=.02 valent 4.440542267225567e-9 à N32 et 2.2170445479474665e-9 à N64. Test rouge observé avant correction : `test_native_tolerance_rejects_deliberately_stale_transport[32]` échoue avec ce seuil.

Correctif autorisé, physique inchangée : erreur fresh≤3e-11 ; séparation des oracles≥21 tolérances calculée avant compilation ; distance native à l'oracle stale≥20 tolérances ; références stale enregistrées dans le NPZ, marge et erreurs dans le reçu. L'inégalité triangulaire rend ces critères cohérents. Le seuil est une obligation de précision supplémentaire, pas une conséquence garantie du seul résidu max1e-10 ; la documentation le dit explicitement. Aucune relaxation ne sera appliquée si la réception native échoue.

Validation finale : **14 tests source/math réussis** (trois adversariaux de frontière, régression de frontière existante, authoring/resolve/émission réelle M08, cinq tests math existants, quatre tests indépendants de stencil et discrimination). `git diff --check` net. Aucun build lourd ou run natif exécuté.
