# Contre-revue indépendante M15/M16 - c66f602

Aucun cœur PoPS modifié. Base examinée c66f602, appliquée comme dépendance b7fc57d dans le checkout isolé PoPS-principal-group. Le rapport auteur n'a pas servi d'oracle.

## Sources effectivement contrôlées

- `PoPS_Codex_handoff_0.4.0/context/CORPUS_ORIGINAL.md`, §2.2 et registre M15/M16 : M15 demande Fox–Laurent multiordre ; M16 demande la hiérarchie HyQMOM15 magnétique, normales obliques et admissibilité. Ce ne sont pas les deux campagnes complètes livrées par c66f602.
- Note originale `context/HyQMOM_Dupuy_2026_document_incomplet.pdf` p2 : S50 renseigné, S41 et S32 laissés vides ; pas une définition complète utilisable seule.
- Article primaire Bryngelson/Fox/Laurent, JCP566(2026)115242, page36, Eq.(B.1), téléchargé et inspecté visuellement : https://comp-physics.group/papers/bryngelson-JCP-26.pdf . Trois formules et permutation x/y concordent exactement avec la bibliothèque. L'article décrit aussi des corrections/projections distinctes ; elles ne sont pas imputées à la relation B.1 non modifiée.

## Mathématiques vérifiées indépendamment

`test_hyqmom_b1_independent_review.py` construit les moments du Gaussien covariance [[1,1/2],[1/2,1]] par récurrence de Wick en fractions exactes, puis différentie directement les formules B.1 et le changement de coordonnées au point considéré. Accord sur les 450 entrées Jx/Jy de l'exemple, donc les deux décalages d'indices Fx_pq=M_(p+1,q), Fy_pq=M_(p,q+1) sont effectivement contrôlés.

Une récurrence de Faddeev–LeVerrier rationnelle indépendante reproduit exactement
`lambda^15 -67/2 lambda^13 +5641/16 lambda^11 -1422 lambda^9 +9747/4 lambda^7 -1593 lambda^5 +4131/16 lambda^3` ; son résidu de Cayley–Hamilton est exactement nul. Les pivots LDL du Gram d'ordre4 sont tous positifs en arithmétique rationnelle. Les quatre racines obliques non réelles sont donc un témoin MATH de la relation B.1 non modifiée à un état strictement réalisable ; aucune impossibilité de toute méthode HyQMOM n'en découle.

Le Gaussien ne distingue pas B.1 de l'Eq.(40), leurs corrections et dérivées s'y annulant. Un contre-test supplémentaire utilise une mixture positive de Gaussiennes non symétrique à moyenne nulle. Il compare M32 et M23 à une conversion rationnelle directe B.1 et exige un écart avec la correction Eq.(40). Aucun cinquième moment de mixture n'est employé comme oracle de fermeture.

S50 axial est bien non linéaire, et la reconversion brute inclut les termes centrés 2/3/4/5 avec leurs coefficients binomiaux. Il s'agit exclusivement de cinq moments 1V ordre4, pas des récurrences multiordres Fox–Laurent. M16 construit les quinze flux de transport 2V ; aucune source électrique/magnétique n'y est appliquée.

## Finding d'admission corrigé

M15 ne déclarait aucune contrainte de domaine. Les moments `(-1,0,-1,0,-3)` donnaient un flux fini et le même spectre que la Gaussienne positive ; un contrôle de finitude ne certifie donc pas la réalisabilité. Test rouge reproduit avant correction.

L'exemple déclare maintenant la récupération identité et les conditions rho>0, detH2>0, detH3>=0 pour le Hankel 3x3. Cela rejette densité négative, variance nulle (la fermeture est indéfinie) et quatrième moment incompatible, tout en admettant la frontière réalisable à deux atomes de variance positive. Aucun plancher, clipping ou changement de S50 n'a été ajouté. Tests des expressions de domaine et de leur raccord au vrai brick de récupération émis.

M16 expose explicitement classification MATH. Son witness n'applique pas le projecteur optionnel attaché par la factory ; sa build_case reste source-only. L'admission native générale et la campagne magnétique n'ont pas été qualifiées ici et restent obligations actives.

## Validation

**7 tests source/math réussis**, incluant authoring→validate→resolve→émission des deux exemples. Aucun build lourd, aucune évolution native et aucune preuve MPI/GPU. Le nouveau commit contient seulement exemples/tests/rapport.
