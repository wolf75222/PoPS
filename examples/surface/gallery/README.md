# The gallery of the user surface

These files are the cases and the library drafts of spec 0.3, section 2.6 (the user surface),
copied unchanged from `/mnt/project-files/dsl/cases/after/` as review R12c left them. They are
written in the spelling of spec 0.3 (`import pops`, `from pops.math import *`); until the
cut-over of section 21.7, the namespace `pops.surface` provides that spelling.

- `scripts/surface_lint.py` lints each file (SU-T1), and
  `tests/python/architecture/test_su_t1_surface_lint.py` pins the hash of each file.
- N15 lot 1 (ROM-98) elaborates level 1 of the surface. The L1 case of section 17.1 is
  elaborated at that level, with a fixed step, in `tests/python/unit/surface/test_su_t3_parity.py`.
  The other cases use constructs that later lots of N15 elaborate, and they do not run yet (SU-T4).
