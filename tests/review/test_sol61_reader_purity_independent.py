"""Non-author Source-only isolation adversaries; no Native execution."""
import os,sys
from pathlib import Path
import pops
from tests.review import test_sol61_fan_li15_eight_saved_reader as t
ROOT=Path(__file__).resolve().parents[2]

def test_real_source_import_in_parent_remains_and_child_is_pure(monkeypatch):
    assert Path(pops.__file__).resolve()==ROOT/"python/pops/__init__.py"
    retained=sys.modules["pops"]
    monkeypatch.setenv("PYTHONPATH","/SourceOnly-forbidden-child-precedence")
    before=os.environ["PYTHONPATH"]
    result=t._pure_reader_probe()
    assert result.returncode==0,result.stderr
    assert sys.modules["pops"] is retained and os.environ["PYTHONPATH"]==before
    t.test_wick_gram_flux_independent_of_generating_function()

def test_child_rejects_actual_checkout_pops_not_only_temporary_stub():
    prefix="import sys\nsys.path.insert(0,"+repr(str(ROOT/"python"))+")\nimport pops\nfrom pathlib import Path\nassert Path(pops.__file__).resolve()==Path("+repr(str(ROOT/"python/pops/__init__.py"))+")\n"
    result=t._pure_reader_probe(prefix)
    assert result.returncode!=0 and "reader imported forbidden PoPS/Native module" in result.stderr
