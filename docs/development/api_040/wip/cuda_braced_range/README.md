# Correctif CUDA en cours de réception

Ce dossier expose le patch candidat et la sonde exacte utilisée. Le patch est désormais
appliqué au runtime de cette PR, après réception indépendante de la sonde737830. Il remplace trois plages
bracées de amr_layout_transfer.cpp par des std::array nommés, sans modifier les
corps, leur ordre, les budgets, la sérialisation ou les gardes.

La sonde 737830 a terminé avec exit0 selon le premier retour du moniteur :
baseline exit1 aux trois sites connus, candidat exit0 avec objet ARM authentifié.
La réception complète des551sorties/modes et des fichiers NVCC est acquise. Cela ne constitue pas un build complet du Native CUDA ni un run PDE GPU.

probe.py/probe.sbatch/probe-argvs.json sont les scripts originaux de cette sonde.
Ils exigent les copies d'en-têtes, les pins de dépendances et le RootGO décrits
dans la préparation externe. Ils ne forment pas un raccourci autonome autorisant
un run à partir de ce dossier. Toutes les nouvelles sorties sont dans le scratch
personnel /gpfs/scratch/rmdraux ; les anciens emprunts projet sont lus seulement.
