# Advection sinusoïdale périodique — index d'analyse

## État scientifique courant

Il n'existe pas encore de publication scientifique courante pour la matrice
versionnée de 37 cas. Une analyse n'est recevable que si elle est générée par
`plot_results.py --complete` depuis le `COMPLETE.json` correspondant : ce
consommateur vérifie les hashes des 37 paires NPZ/JSON, les obligations, les
reçus MPI et la convergence avant de créer ses figures et son propre
`ANALYSIS.md` immuable.

Les répertoires de tentative, les anciens rapports et les figures isolées ne
sont pas des preuves. En particulier, une tentative sans `COMPLETE.json` est
incomplète, même lorsqu'elle contient des sorties de cas; elle ne doit être ni
tracée, ni citée, ni utilisée pour conclure sur la précision, l'AMR ou MPI.

## Publication après une campagne complète

```sh
env -u PYTHONPATH python benchmarks/verification/advection/sine_wave/plot_results.py \
  --complete benchmarks/verification/advection/sine_wave/results/<campagne>/matrix-v1-prescribed-window/COMPLETE.json \
  --figures benchmarks/verification/advection/sine_wave/figures/<publication-nouvelle>
```

Le rapport généré documente, avec les médias associés, la convergence finale et
au probe, les visualisations 1D/2D/3D et GIF, la trajectoire du patch AMR
mobile, le coin MPI 3D à huit rangs, les comparaisons de blocs/subcycling, les
témoins demandés, la conservation et l'environnement d'exécution.

## Limites de lecture

Les visualisations sont des diagnostics utiles : elles ne remplacent pas les
normes, la conservation, les reçus d'ownership MPI ou les témoins AMR. Cette
vérification ne permet pas non plus de déduire un résultat CUDA, un profil ou
un scaling ROMEO ; ces campagnes sont publiées séparément sous
`benchmarks/performance/`.
