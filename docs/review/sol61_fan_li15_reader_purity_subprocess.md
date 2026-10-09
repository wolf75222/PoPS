# Eight-state reader purity test composability correction

ROOT's integrated Source cohort imported PoPS Source before the Wick math test.
The old assertion over the parent process sys.modules therefore failed although
the independent reader had not imported PoPS. The original red cohort is retained
by ROOT; no mathematical guard, Native result or receipt is changed.

The import-purity requirement now runs in a fresh Python subprocess from the
executed checkout, with PYTHONPATH removed. It imports the actual independent
reader and refuses both top-level and nested pops/_pops module names. The parent
cohort's sys.modules is never purged. An adversary creates an explicitly labelled
minimal temporary pops.py Source module, imports it in that child, and confirms
that the unchanged purity requirement refuses the process. This module is not a
runtime mock or Native proof.

Validation: reader plus frozen fixture-wire suite 19 PASS in 15.33s. A separate
Source launcher imported this checkout's real pops package before invoking the
Wick/purity subset: 3 PASS, 12 deselected, 0.37s. Thus the tests compose with a
Source cohort while independently maintaining the no-PoPS/no-Native reader rule.
No Main/Native/ENV/ROMEO or build freeze was modified.
