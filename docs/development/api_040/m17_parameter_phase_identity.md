# M17 : identité canonique des paramètres après authentification

Le contre-exemple public non-auteur `test_sol61_atomic_parametric_fault_preparation.py::test_public_parameter_is_in_law_covector_and_resolved_graph` (gel Banach7865a008) conserve un RuntimeParam réel dans flux, matrices et covecteurs du nouveau port. Validate/resolve et brick réussissent ; la table du programme refusait ensuite le même paramètre : qualified_id authoring0 contre qualified_id du modèle résolu.

Le chemin résolu conserve correctement ses handles qualifiés. Le défaut était la comparaison des **phases d'identité** dans `program_emit_params.py` : les expressions ordinaires/table peuvent conserver leur handle authoring, tandis que les covecteurs capturés portent un handle résolu. Le lecteur et la table comparaient leurs qualified_id bruts.

Le correctif commun projette **les deux côtés** sur `handle._resolved().qualified_id`, après les contrôles existants d'owner exact et d'instance de bloc. Il ne crée aucun alias par nom, ne choisit aucun modèle représentatif, ne change aucun index/table/default et ne modifie aucun émetteur physique. Les instances étrangères et owners étrangers restent refusés avant projection. Aucune interface C++, ABI, IR, signature SDK ou équation n'est modifiée ; le paquet Python doit être réauthentifié par ROOT.

Trois nouveaux tests Source utilisent une autre composition publique (six moments rangés librement, advection commune et B=rhoI, RuntimeParam gain), vérifient la route réelle, l'égalité des identités authoring/résolue sans mutation et le refus d'un owner étranger homonyme. La cohorte avec les tests multi-model/graph/SymbolicPath ferme **17 PASS en13.97 s**, XML `/tmp/sol61-m17-param-final-source.xml`.

Le contre-exemple original Banach, inchangé, ferme **2 PASS en11.26 s** avec les modules PoPS du WT auteur explicitement préchargés, XML `/tmp/sol61-m17-param-banach-own-source.xml`. Une première commande pytest avec deux entrées pythonpath a chargé le vieux programme de Banach et conservé son RED ; ce résultat n'est pas compté comme test du correctif. Le rejeu GREEN imprime les chemins exacts `.../PoPS-sol61-m17-composition/python/pops/__init__.py` et `.../codegen/program_emit_params.py` avant pytest.

## Échec de non-régression séparé préservé

La cohorte élargie initiale donne26P/1F. Nœud exact : `tests/python/unit/codegen/test_param_core_contract.py::test_derived_contract_is_explicit_and_manifest_is_lossless`, ligne243 : expected ModuleManifest9, actual10. Ce test n'est ni modifié ni transformé en skip.

Pour borner la causalité, le même nœud est rejoué après chargement **en mémoire** du programme EmitParams exact du parent69fa712. Les fichiers du test et `_module_manifest.py` sont byte-identiques au parent, hashes respectifs `f33fda770c5a02aaead218c01bf9c63e8c02b87c3455ff161027399bca7ede3a` et `68e8b69893483fd1b067f3e658bc0e229cdc36a0d5b48a14dd13825f030d1e6e`. Le même échec expected9/actual10 est reproduit : **1 FAIL en0.19 s**, XML `/tmp/sol61-m17-parent-manifest-source.xml`. Cette preuve est une comparaison ciblée des fichiers et du code parent, pas un test d'un checkout complet indépendant. ROOT doit diagnostiquer le contrat de manifeste séparément ; ce lot ne change pas sa version.

Tous les résultats ici sont Source/host. Aucun paramètre énorme n'a été exécuté dans PoPS Native, aucune preuve rollback ni réception scientifique n'est déduite. ROOT et le contre-reviewer reçoivent le gel exact avant intégration.
