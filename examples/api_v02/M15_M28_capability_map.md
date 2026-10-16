# M15–M28 : profils fermés et limites de réception

Branche `codex/pops-api-v02-local`, base `77acb67747c18666e31c24f6b51ffbe290496bb8`. Chaque résultat natif appartient à son identité enregistrée, jamais à toute version ultérieure des sources.

`models_M15_M28.build_model("Mxx", **parameters)` retourne le CaseSpec commun. L’import ne compile ni ne lance de simulation. Les oracles NumPy contrôlent les états natifs après calcul ; aucun n’est un callback du programme scientifique. Chaque ligne décrit un profil fini, pas toute la famille du corpus ni toutes les garanties du papier.

| ID | Profil effectivement écrit | Limites |
|---|---|---|
| M15 | HyQMOM 1D par récurrence S04, ordres2/4 (respectivement3/5 moments), FV HLL périodique, Euler ; profils lisse et Riemann. Le dispatcher choisit l'ordre4. | Deux ordres reçus par les tests dédiés ; ni tout ordre, ni frontière exacte de réalisabilité/vide, ni étude de convergence de chocs. |
| M16 | Quinze moments B.1 originaux, FV HLL avec spectre calculé, Lorentz non nul, référence indépendante. | Admission cartésienne conditionnelle ; gaussienne corrélée dans direction(1,-1) exige un refus natif. |
| M17 | Fan–Li D2M4 tensoriel complet DF+B, chemin droit en moments bruts, Gauss4, FV Rusanov, Euler. | Référence indépendante Gauss24 ; pas réception de chocs ou de tout chemin. |
| M18 | Entropie discrète MB, M₂/M₃ sur Gauss6 [-1,1] : Newton plein compilé à chaque étape RK4, D0 spatial sur 16 cellules. Défaut : M₂ ; ancien M1 analytique disponible explicitement. | Support/quadrature fixés ; profils lisses intérieurs ; aucune garantie globale de réalisabilité, positivité, stabilité entropique ou identification au flux continu. |
| M19 | Vlasov–Poisson–BGK1x1v :16 cellules physiques,8 volumesv, Green du Poisson discret jauge0, transportx/v, Maxwellienne projetée sur trois invariants. | Vitesse tronquée à flux nul, positivité gardée ; pas théorèmeH ni conservation exacte énergiechamp, pas3x3v/AMR. |
| M20 | Deux espèces1x1v, FP symétrique, moments courants, fluxv centré et projection pondérée des défauts nombre/impulsion/énergie, transport spatial. | Projection choisie, aucun théorème d’entropie ou Chang–Cooper ; domaine positif gardé. |
| M21 | EM slab f_s(z,v∥,μ), quadrature2πB/m, Bconstant, polarisation Boussinesq symbolekperp² ; dA/dt éliminé avec même Dv que taux cinétique, A et résidu original dérivé enregistrés. | Pas opérateurperp multidimensionnel, ExB/courbure/miroir/collision ; ni tokamak/full-f3D, ni DG ou dispersionAlfvén qualifiés. |
| M22 | TransportM1 gris et échange matière/radiation, bilans énergie/impulsion. | Variante thermodynamique déclarée ; aucune garantieAP transférée. |
| M23 | MHD complète, flux Hall/Ohm dépendant de vrais voisins spatiaux, petite onde circulaire. | Profil planaire2.5D, Bxconstant ; réception de fréquence nécessaire, bilan seul insuffisant. |
| M24 | BN convectif sept équations, pI=p1/uI=u2, chemin droit Gauss2. | Pas relaxation phasique ni contacts exacts réceptionnés. |
| M25 | GPR : rho,m,E,A3×3,rhoJ3 évolués, énergie et relaxation explicites. | Profil de cisaillement choisi ; pas limites singulières ou géométrie3D générale. |
| M26 | Agrégation–diffusion non linéaire, vraie grille16×1, interaction native globale cos, mobilité de face arithmétique, Euler ; perturbation distante. | Midpoint direct ; pas positivité inconditionnelle ni AMR. |
| M27 | CH non linéaire : mu=c³−c−kappaLapc puis Lapmu ; deux opérateurs spatiaux successifs,16×1, Euler4pas. | Pas solve implicite ni stabilité inconditionnelle. |
| M28 | Volumes mobiles1D, mesures balayées et flux relatifs du même intervalle, GCL et variante non uniforme. | ArtefactDim1 nécessaire ; pas ALE multi-D ou changement de topologie. |

## Mécanismes génériques

FiberAxis/FiberRepresentation retiennent axes, unités, nœuds, poids, arêtes, voisinages, bords et bijection multi-indice/composante. Moments, pullbacks et divergences deviennent des expressions ordinaires compilées sur chaque vraie cellule physique. Le descripteur immuable est inclus dans Representation/Space, hashes, to_data et reconstruction. Aucune branche native ne lit un nom de modèle.

cell_field(...,quadrature=CellMidpoint()) et cell_mean(...,template=...,quadrature=CellMidpoint()) déclarent reconstruction constante et quadrature. Les champs calculés gardent propriétaire, étage, espace, largeur et représentation. Une interaction globale peut alimenter un opérateur spatial puis des expressions. Les routes AMR non qualifiées restent refusées.

M21 réalise -D0 par un noyau Fourier fini exact sur sa grille périodique. StencilFlux porte un bilan de faces accepté ; son résultat ne peut être caché dans une multiplication non linéaire comme un simple champ intermédiaire. Le profil emploie donc le port spatial générique, sans callback.

## Preuves et tests

La commande de réception et les règles d'identité figurent dans `README.md`.
Chaque ligne du reçu de campagne conserve son profil, son C++ exécuté, ses
métriques, ses limites et son identité native. Les résultats d'installations
antérieures restent des preuves historiques, sans transfert au paquet courant.

`test_api_v02_models_M15_M21.py` reçoit les profils Dim2 et le refus M16 oblique
sans publication d'état ; M15 utilise le test Dim1
`test_api_v02_hyqmom_orders_native.py`. `test_api_v02_spatial_pdes.py` reçoit M26,
sa perturbation distante et M27. Les tests d'entropie reçoivent les deux ordres
M18 ; ceux de l'ALE comparent aussi des maillages non uniformes et mobiles.
Tous suivent validate→resolve→compile→bind→run. Les tests source et les builds
ne sont ni ces exécutions, ni la CI, ni une qualification scientifique générale.
