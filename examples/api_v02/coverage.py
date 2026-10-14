"""Portée scientifique des témoins API v0.2, indépendante de leur réception.

Ces données décrivent les obligations qui restent hors du profil fini choisi.
Elles ne constituent ni un résultat de test, ni un statut de compilation, ni
une qualification des articles entiers. Les reçus de réception font autorité
pour le code, le binaire, les paramètres et les contrôles réellement exécutés.

Les titres Mxx sont transcrits du corpus local ; ses anciens statuts ne sont
pas repris. Les obligations T09 visent le nouveau témoin spatial Duhamel de
``method_interval_transport.py``, et non son ancien témoin homogène.
"""

SOURCE_TITLE_SOURCE = "docs/development/api_040/corpus.json: models[*].title"

SOURCE_TITLES = {
    "M01": "Advection scalaire",
    "M02": "Burgers / flux utilisateur",
    "M03": "Euler / EOS",
    "M04": "Advection–diffusion",
    "M05": "Navier–Stokes avec conduction",
    "M06": "Enthalpie / Stefan",
    "M07": "Saint-Venant / topographie",
    "M08": "Centre-guide / advection–Poisson",
    "M09": "Euler–Poisson magnétique de Hoffart",
    "M10": "Dérive–diffusion–Poisson",
    "M11": "Poisson–Maxwell–Stefan",
    "M12": "Multifluide cinq/dix moments–Maxwell",
    "M13": "Réactions / streamers / photoionisation",
    "M14": "Gaine / paroi / circuit",
    "M15": "HyQMOM 1D / ordres",
    "M16": "HyQMOM15 magnétique",
    "M17": "Fan–Li15",
    "M18": "Fermeture entropique MN",
    "M19": "Vlasov–Poisson / BGK",
    "M20": "Fokker–Planck multiespèces",
    "M21": "Gyrocinétique full-f",
    "M22": "Rayonnement M1–matière",
    "M23": "MHD Hall",
    "M24": "Baer–Nunziato",
    "M25": "GPR thermomécanique",
    "M26": "Agrégation–diffusion",
    "M27": "Cahn–Hilliard",
    "M28": "ALE / volumes mobiles",
}

# Une exécution sur une cible ne qualifie pas les autres cibles. Cette note
# commune évite notamment d'annoncer à tort une absence d'AMR pour T08.
COMMON_SCOPE_NOTES = (
    "La dimension compilée, la variation spatiale effective, le maillage, les "
    "frontières, le nombre de rangs MPI et le backend sont ceux du reçu ; un "
    "profil plan dans un binaire multidimensionnel ne qualifie pas toutes les directions.",
    "La réception d'un profil ne se transfère pas à un autre binaire, à un "
    "autre nombre de rangs, au GPU, à l'AMR ou au remaillage. Les contrôles "
    "AMR effectivement exécutés, notamment T08, restent des preuves distinctes.",
    "Un accord avec un oracle discret vérifie le schéma choisi ; il ne prouve "
    "ni la convergence vers le problème continu, ni les garanties générales "
    "de l'article, ni une propriété pour tous les états admissibles.",
)

OBLIGATIONS = {
    "M01": (
        "Le témoin est une advection scalaire 2D périodique à vitesse constante ; "
        "les frontières entrantes/sortantes et les vitesses variables ne sont pas qualifiées.",
        "L'erreur à temps court et la conservation ne remplacent pas une étude "
        "de convergence spatiale et temporelle par raffinement.",
    ),
    "M02": (
        "Le profil périodique comporte un choc et une détente avec flux de "
        "Godunov, mais reste invariant dans la direction transverse.",
        "La convergence vers la solution entropique et les interactions "
        "réellement multidimensionnelles ne sont pas qualifiées par l'oracle FV fini.",
    ),
    "M03": (
        "Le système Euler complet est testé sur une onde d'entropie périodique "
        "avec une EOS stiffened-gas fixée ; les autres EOS ne sont pas qualifiées.",
        "Les chocs forts, le voisinage du vide, la positivité générale et la "
        "convergence du schéma restent hors de ce profil lisse.",
    ),
    "M04": (
        "La diffusion utilise un tenseur constant anisotrope avec termes "
        "croisés ; les tenseurs variables ou dégénérés et les frontières "
        "non périodiques ne sont pas qualifiés.",
        "La positivité, la dissipation générale et la convergence de la "
        "composition transport–diffusion restent à établir au-delà du témoin fini.",
    ),
    "M05": (
        "Les contraintes visqueuses, leur travail et la conduction sont présents ; "
        "la réception périodique ne qualifie pas les conditions de paroi ou "
        "les frontières ouvertes du système compressible.",
        "L'accord avec le bilan discret ne prouve pas la convergence "
        "Navier–Stokes ni la positivité globale pour des états plus sévères.",
    ),
    "M06": (
        "Le témoin Stefan est plan, à une phase, avec plateau latent et apport "
        "thermique au bord ; les interfaces courbes et les configurations "
        "multidimensionnelles ne sont pas qualifiées.",
        "La convergence de la position du front et de son flux thermique "
        "reste distincte du contrôle fini d'enthalpie.",
    ),
    "M07": (
        "L'équilibre lac au repos inclut des cellules sèches, mais ne "
        "qualifie pas le mouillage/séchage dynamique ni les ruptures de barrage.",
        "La positivité hors du pas choisi et les écoulements réellement 2D "
        "sur topographie restent à qualifier.",
    ),
    "M08": (
        "La variable q est une fluctuation de densité/charge signée de "
        "moyenne nulle ; le témoin ne reçoit pas la positivité d’une "
        "densité totale obtenue en lui ajoutant un fond constant.",
        "Le couplage centre-guide utilise des résolutions Poisson fraîches "
        "aux étapes sur un domaine périodique avec jauge de moyenne nulle ; "
        "les autres frontières et jauges ne sont pas qualifiées.",
        "Les résidus de champ et la conservation à temps court ne qualifient "
        "pas la convergence ni la dynamique à temps long.",
    ),
    "M09": (
        "Le témoin Euler–Poisson magnétique explicite ne réalise pas le "
        "schéma milieu/Schur de Hoffart ; ses propriétés énergétiques "
        "ne peuvent pas lui être transférées.",
        "Les limites quasi neutre et fortement magnétisée ainsi que la "
        "conservation exacte de l'énergie totale ne sont pas qualifiées.",
    ),
    "M10": (
        "Le flux Scharfetter–Gummel et Poisson sont couplés à chaque étape, "
        "avec coefficients fixés et frontières périodiques ; les lois de "
        "transport non linéaires et les autres frontières ne sont pas qualifiées.",
        "La positivité et la dissipation d'entropie générales du schéma "
        "explicite complet, notamment en régime raide, restent à établir.",
    ),
    "M11": (
        "Le témoin est un mélange ternaire périodique à fractions strictement "
        "positives ; la disparition d'espèces et les mélanges dégénérés ne "
        "sont pas qualifiés.",
        "La contrainte de masse et l'élimination des frictions ne transfèrent "
        "pas au schéma FV centré une preuve générale de positivité, "
        "d'entropie ou de convergence issue d'un autre schéma.",
    ),
    "M12": (
        "Le système 2D3V couple cinq moments électroniques, dix moments "
        "ioniques et Maxwell avec flux de chaleur central nul ; les "
        "fermetures de chaleur non locales ne sont pas qualifiées.",
        "Les contrôles de Gauss, de divergence et de travail semi-discret "
        "ne prouvent pas une conservation exacte de l'énergie par RK4, "
        "la positivité globale ni la validité de régimes de reconnexion.",
    ),
    "M13": (
        "Le streamer plan utilise des coefficients locaux fixés et un "
        "groupe photoionisant de Helmholtz ; une énergie électronique "
        "dynamique et une photoionisation multigroupe calibrée ne sont pas qualifiées.",
        "Les bilans de charge/noyaux et l'oracle de convolution ne "
        "qualifient pas la vitesse d'un front physique ni sa convergence.",
    ),
    "M14": (
        "Le témoin gaine–circuit est froid, avec réservoir prescrit et "
        "paroi absorbante dans un régime strictement sortant ; les régimes "
        "thermiques, réfléchissants ou émissifs ne sont pas qualifiés.",
        "Le bilan conjoint des charges de face et du condensateur ne "
        "qualifie pas une conservation d'énergie thermique ni la convergence "
        "des courants et de la structure de gaine.",
    ),
    "M15": (
        "Les ordres HyQMOM 2 et 4 ont des profils lisses et de Riemann ; "
        "les autres ordres et les états au bord exact du cône des moments, "
        "dont le vide, ne sont pas qualifiés.",
        "Les gardes de Hankel et l'oracle FV fini ne prouvent pas la "
        "convergence du système de moments ni une admissibilité générale "
        "pour toute donnée réalisable.",
    ),
    "M16": (
        "La variante B1 à quinze moments possède un contre-exemple "
        "à l'hyperbolicité oblique : son Jacobien directionnel a des valeurs "
        "propres non réelles pour une gaussienne corrélée admissible. "
        "Une hyperbolicité universelle serait une assertion fausse.",
        "Le témoin cartésien admis et le refus du contre-exemple ne "
        "qualifient donc pas une évolution générale dans toutes les "
        "directions ; il faudrait préciser une restriction du domaine "
        "ou une autre fermeture avant cette extension.",
    ),
    "M17": (
        "Le témoin Fan–Li15 utilise un chemin droit dans les moments bruts "
        "et une quadrature Gauss4, avec variation spatiale selon x ; "
        "les autres chemins et interactions multidimensionnelles ne sont pas qualifiés.",
        "La comparaison à une quadrature plus fine et les contrôles "
        "spectraux locaux ne prouvent pas la réalisabilité globale, "
        "le comportement au bord SPD ni les propriétés entropiques des chocs.",
    ),
    "M18": (
        "Les fermetures entropiques M2 et M3 utilisent Newton et une "
        "quadrature Gauss6 sur [-1,1] ; elles sont discrètes en vitesse, "
        "et ne constituent pas la fermeture continue exacte.",
        "La convergence de Newton près du bord réalisable, les autres "
        "ordres/quadratures et les limites de raffinement restent à qualifier ; "
        "le transport centré avec RK4 n'a pas ici de preuve générale "
        "de réalisabilité ou de stabilité entropique.",
    ),
    "M19": (
        "Le témoin est un véritable Vlasov–Poisson–BGK 1x1v, avec "
        "vitesses tronquées, flux nul au bord en vitesse et équilibre "
        "corrigé pour ses moments discrets ; le cas 3x3v et la limite "
        "de troncature ne sont pas qualifiés.",
        "La conservation des moments de collision ne prouve pas un "
        "théorème H, une énergie totale exactement conservée par "
        "l'avancement complet ni une limite BGK raide AP.",
    ),
    "M20": (
        "Le profil Fokker–Planck est inhomogène 1x1v avec deux espèces ; "
        "les mélanges généraux, l'espace des vitesses 3D et les limites "
        "de troncature/raffinement ne sont pas qualifiés.",
        "La projection discrète des défauts de quantité de mouvement et "
        "d'énergie assure les contraintes visées, mais ne transfère pas "
        "un théorème H ni une positivité générale au-delà des petits pas admis.",
    ),
    "M21": (
        "Le témoin électromagnétique slab f(z,v_parallel,mu) possède un "
        "dA_parallel/dt compatible avec le même opérateur en vitesse, "
        "mais remplace la géométrie perpendiculaire par un symbole "
        "k_perp fixé et une polarisation de Boussinesq.",
        "Le transport perpendiculaire complet, la courbure, le miroir, "
        "la géométrie tokamak 3D et la qualification de dispersion "
        "d'Alfvén du schéma de l'article restent hors de ce profil.",
    ),
    "M22": (
        "Le rayonnement gris M1–matière est écrit dans le repère du "
        "laboratoire avec échange explicite aux paramètres choisis, sans "
        "garde générale reliant le pas à l’opacité ; les "
        "corrections comobiles ou relativistes ne sont pas qualifiées.",
        "La limite de diffusion AP, les opacités raides et les "
        "conditions radiatives de Marshak restent à qualifier.",
    ),
    "M23": (
        "Le profil Hall-MHD évolue les équations complètes sur une onde "
        "plane ; son contrôle de div B ne qualifie pas une méthode CT "
        "pour des champs réellement multidimensionnels.",
        "L’amplitude totale et le contraste à résistivité nulle au même "
        "intervalle ne séparent pas quantitativement les amortissements "
        "numérique et ohmique. Les conditions magnétiques générales et "
        "le régime Hall raide restent hors du schéma explicite reçu.",
    ),
    "M24": (
        "Le système convectif de Baer–Nunziato emploie p_I=p_1, u_I=u_2 "
        "et un chemin conservé droit ; la relaxation raide des pressions "
        "et vitesses ainsi que la disparition d'une phase ne sont pas qualifiées.",
        "Les termes non conservatifs sont présents, mais leur oracle "
        "discret ne prouve ni une positivité générale ni la préservation "
        "exacte de tous les contacts matériels.",
    ),
    "M25": (
        "Le témoin GPR à 17 composantes comprend transport, chemin "
        "non conservatif et relaxations, mais reste une petite "
        "perturbation avec une vitesse de Rusanov fixée ; cette borne "
        "n'est pas certifiée pour des états généraux.",
        "Les limites raides visqueuse et conductrice ainsi que la "
        "préservation exacte de det A=rho/rho_0 ne sont pas qualifiées.",
    ),
    "M26": (
        "L'agrégation–diffusion est une PDE multicellule avec convolution "
        "périodique globale par un noyau cosinus lisse ; les noyaux "
        "singuliers et les configurations réellement 2D/3D ne sont pas qualifiés.",
        "La non-négativité finale est contrôlée pour le témoin fini. "
        "Ce contrôle, le bilan de masse et la baisse d'énergie ne "
        "prouvent pas une dissipation ou une positivité inconditionnelle "
        "de l'Euler explicite, ni une convergence par raffinement.",
    ),
    "M27": (
        "Le témoin Cahn–Hilliard possède le véritable opérateur spatial "
        "d'ordre quatre, avec mobilité constante et potentiel double puits ; "
        "la mobilité dégénérée et les évolutions réellement 2D/3D ne sont pas qualifiées.",
        "La diminution d'énergie aux petits pas choisis ne qualifie "
        "pas une stabilité énergétique inconditionnelle ni la "
        "convergence dans les régimes raides ou aux interfaces fines.",
    ),
    "M28": (
        "Le témoin ALE utilise des volumes mobiles 1D, des cas uniformes "
        "et non uniformes et un oracle Reynolds ; les faces mobiles "
        "multidimensionnelles et les changements de topologie ne sont pas qualifiés.",
        "Les autres quadratures mobiles et les changements de signe "
        "de la vitesse relative ne sont pas couverts par le profil "
        "amont à vitesse relative positive choisi.",
    ),
    "T01": (
        "Le témoin SSPRK2 résout Poisson à chaque étape d'une PDE "
        "centre-guide ; les tableaux RK d'ordre supérieur/adaptatifs "
        "et les frontières de stade non périodiques ne sont pas qualifiés.",
        "Les contrôles de fraîcheur et de résidu des champs ne "
        "constituent pas une preuve générale d'ordre pour tout couplage contraint.",
    ),
    "T02": (
        "Le profil IMEX-Euler couple une décroissance explicite à une "
        "diffusion globalement implicite périodique ; il ne qualifie "
        "pas les conditions d'ordre de familles RK additives ou partitionnées supérieures.",
        "L'ordre en régime raide, les limites AP et les couplages "
        "non linéaires généraux restent à qualifier.",
    ),
    "T03": (
        "Le Strang non autonome transporte correctement l'état et "
        "le temps étendu ; le lifting des frontières physiques "
        "inhomogènes d'une PDE scindée n'est pas qualifié.",
        "Le témoin fini ne prouve pas les conditions d'ordre ou "
        "la stabilité pour des opérateurs spatiaux non commutants généraux.",
    ),
    "T04": (
        "Le CG global et le Schur du témoin périodique à deux "
        "composantes ne qualifient pas un bloc éliminé lui-même "
        "non local ni les frontières non périodiques héritées du Schur.",
        "Les systèmes non symétriques ou indéfinis et les "
        "préconditionnements généraux restent hors de ce profil CG.",
    ),
    "T05": (
        "Le Newton imbriqué et la tangente implicite portent sur "
        "une fermeture strictement réalisable à quadrature fixée ; "
        "un certificat de faisabilité pour des quadratures générales "
        "et une convergence globale ne sont pas qualifiés.",
        "La différentiation à travers une sélection de racine, "
        "un Jacobien singulier ou un changement de branche reste hors du témoin.",
    ),
    "T06": (
        "Les deux relations hors diagonale distinguent effectivement "
        "état ancien et inconnue, mais seulement sur une petite "
        "branche positive explicitement choisie.",
        "Les relations à racines multiples, les garanties générales "
        "de domaine et leur résolution globalement couplée ne sont pas qualifiées.",
    ),
    "T07": (
        "Les historiques réels d'état, de pas et d'incrément soutiennent "
        "un BDF2 à pas variable ; leur transfert après remaillage AMR "
        "reste non qualifié.",
        "La provenance HistoryInterval et sa reprise doivent être "
        "jugées sur les reçus du binaire qui contient cette extension ; "
        "les anciens succès BDF2 ne les qualifient pas à eux seuls.",
        "Les méthodes multipas d'ordre supérieur et leur stabilité "
        "pour des suites de pas générales ne sont pas qualifiées par "
        "les trois pas du témoin.",
    ),
    "T08": (
        "La récursion à deux niveaux AMR avec sous-cyclage, reflux, "
        "synchronisation et rejet après synchronisation est couverte "
        "par ses propres tests ; les changements de hiérarchie "
        "avec transfert d'historiques restent hors de ce profil fixe.",
        "Les frontières non périodiques aux temps intermédiaires "
        "et les couplages non linéaires entre enfants ne sont pas "
        "qualifiés par l'advection périodique choisie.",
    ),
    "T09": (
        "Le témoin spatial Duhamel transporte deux vitesses avec "
        "reconstruction caractéristique et durée SSA dans le flux ; "
        "son équilibre BGK est figé sur l'intervalle.",
        "Un Maxwellien non linéaire reconstruit de manière "
        "auto-cohérente et les limites asymptotiques fluides/diffusives "
        "AP ne sont pas qualifiés par ce profil fini.",
    ),
    "T10": (
        "La relaxation de Heun partage gamma entre état et temps "
        "effectivement atteint, avec rejet/réessai ; le témoin ne "
        "qualifie pas un ledger d'échanges spatiaux sur l'intervalle gamma*h.",
        "La sélection globale d'une branche admissible et la "
        "relaxation de plusieurs invariants simultanés restent hors du témoin.",
    ),
    "T11": (
        "Le Patankar-Euler à deux espèces inclut inventaires nuls "
        "et circulation compensée ; les réseaux généraux et les "
        "variantes d'ordre supérieur ne sont pas qualifiés.",
        "La dépendance à la présentation en processus est testée "
        "sur ces décompositions, sans preuve d'équivalence ou "
        "d'admissibilité pour toute autre présentation.",
    ),
    "T12": (
        "L'arête finie colocalisée, le témoin scalaire sur faces "
        "cartésiennes et le nouveau témoin Euler à quatre composantes "
        "sont des profils distincts ; chacun exige sa propre réception "
        "native. Le succès de l'arête finie ne qualifie pas les faces spatiales.",
        "La correction à une passe teste la positivité scalaire ou "
        "la densité/pression Euler dans les profils choisis ; elle ne "
        "prouve pas une admissibilité générale, un algorithme MOOD "
        "itératif ni la compatibilité avec toutes les EOS.",
        "La réexécution des résolutions de champ dépendantes et "
        "le remplacement dans un ledger de faces AMR restent "
        "hors de ces témoins de remplacement versionné.",
    ),
}
