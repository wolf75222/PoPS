# Réception spatiale provisoire, 13 octobre 2026

Cinq cibles sont reçues dans chaque monde MPI : deux positifs Uniform (scalaire/cellules coupées, vecteur signé/permuté), le refus d'une source non finie active, le pôle source inactif ignoré, et le refus AMR non périodique au bind. Cela représente cinq résultats MPI1 et dix résultats MPI2, sans skip sur ces cibles.

**Le positif AMR n'est pas reçu.** Le profil périodique atteint deux niveaux et la géométrie cutcell, puis le garde `interaction history maturity differs across levels` refuse le premier pas. La revue source confirme une lecture de rings historiques LIVE entre callbacks des niveaux. Le ratio temporel reçu est 1/1, distinct du raffinement spatial 2×2. Une autorité historique stable reste requise.

Le package Python installé est `a9297394d5fbf6c191195bc69563822da3169bbe`; le Dim2 MPI est le binaire `f67ecd7e…`, SDK `e1b67851…`, ABI 13. Les corrections de fixture ultérieures sont identifiées par leurs SHA dans les gels et reçus. Ces résultats ne qualifient pas un futur binaire ou commit.

Le profil AMR positif sélectionne une topologie de maillage/halo périodique. Son C++ de programme, noyau `1+x[0]*y[1]-2*y[0]`, espace de sortie, géométrie EB, données initiales coarse et bindings initiaux sont identiques en processus frais, sans normalisation des identités. L'interaction reste une quadrature du domaine cartésien borné, sans image périodique implicite. Le profil non périodique original conserve NoFlux et ZeroFlux et doit échouer au bind. Aucune équivalence de données initiales fines avec ce profil impossible à binder n'est revendiquée.

Les équations source, instants, permutation, historique, checkpoints, replay exact et tolérances `rtol=atol=5e-12` sont conservés. La source seule est la même PDE sur les états capturés finis, puisque le flux authored reste `Mul(Const0,rho)` et les waves sont `Const0`.

[Reçu consolidé](receipt.json) contient les XML par rang, identités avant/après, captures réelles et leurs SHA. Les copies publiques texte sont assainies : `$AUDIT_ENV`, `$REPO` et `$EVIDENCE` remplacent les chemins locaux. Les textes originaux restent privés ; `origin_file_sha256` identifie leurs octets et `sanitized_public_sha256` identifie les copies publiques. Les hashes scientifiques embarqués gardent leur portée originale. Le lot MPI1 des deux positifs Uniform comprend aussi un échec AMR dans son XML complet : seules les deux testcase positives sont comptées. Deux startups -11 sans résultat pytest, les binds non supportés et le positif AMR bloqué sont explicitement exclus.

Cette preuve provisoire ne déclare ni CI verte, ni qualification scientifique M26, ni réception du positif AMR.


## Portabilité des preuves publiques

`$AUDIT_ENV` désigne la racine de l’environnement PoPS installé, `$REPO` la racine du checkout et `$EVIDENCE` la racine des preuves privées voisine du checkout. `$USER` désigne le composant utilisateur des répertoires temporaires pytest. Ces jetons sont des références de provenance à résoudre localement.

[PUBLIC_SANITIZATION.json](PUBLIC_SANITIZATION.json) distingue les SHA-256 originaux et ceux des copies publiques assainies. [SHA256.json](SHA256.json) vérifie les octets publics actuels. Les valeurs numériques scientifiques et les hashes originaux d’artefacts restent inchangés.

L’audit [NPZ_PATH_AUDIT.json](NPZ_PATH_AUDIT.json) couvre 56 archives, 2 707 membres NPY, dont 2 021 numériques et 686 textuels : aucun chemin personnel trouvé, aucune archive réécrite. Chaque archive est une copie exacte de son original privé, vérifiée par SHA. Toute archive comportant un chemin personnel serait conservée uniquement en privé et référencée par son SHA original.

Une collision de répertoire dans l’export public initial partageait le dossier `spatial-interaction` entre les deux positifs Uniform d’un groupe. Les répertoires sont maintenant uniques par cas et les fichiers manquants ont été restaurés depuis leurs originaux privés, sans modifier les octets scientifiques ni les réceptions enregistrées.
