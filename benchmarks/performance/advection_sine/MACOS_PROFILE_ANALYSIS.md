# Analyse macOS : prête pour les données, sans résultat inventé

## Réponse attendue

Après une acquisition qualifiée, l’analyse indiquera les quinze feuilles les
plus pondérées par `/usr/bin/sample`, leur composition par image, et confirmera
que cinq traces Time Profiler indépendantes avec table `time-profile` existent.
Aucun pourcentage, temps ou conclusion de performance n’est présent avant ces
preuves.

## Garde-fous d’interprétation

Les dix processus sont complets, avec compilation/bind/warmup hors acquisition,
mais leurs sorties servent uniquement à expliquer le coût local macOS. Elles ne
sont pas comparables aux mesures ROMEO et ne doivent pas alimenter strong/weak
scaling. Toute régression doit être rejouée avec la même campagne JSON, le même
interpréteur, une provenance READY identique et une nouvelle acquisition sans
collision de sortie.

## Tableau à remplir après collecte

| Élément | Preuve requise | Résultat |
|---|---|---|
| Point canonique | reçu de campagne JSON | à compléter |
| Source exacte | export Git-visible, manifeste et tree SHA | à compléter |
| Variante native | extension + variants.json + ABI/backend | à compléter |
| `/usr/bin/sample` | 5 rapports, feuilles repliées et hashes | à compléter |
| Time Profiler | 5 bundles récursivement hashés + TOC strict | à compléter |
| Top-15, images et icicle | summary collecté + trois PNG/SVG | à compléter |
