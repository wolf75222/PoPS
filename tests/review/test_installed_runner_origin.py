import importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
def load(name):
    spec=importlib.util.spec_from_file_location(name, ROOT/'docs/development/api_040'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
runner=load('run_installed_mpi_checks')
def test_symlink_origin_is_same_but_foreign_is_not(tmp_path):
    actual=tmp_path/'package.py';actual.write_text('same')
    alias=tmp_path/'alias.py';alias.symlink_to(actual)
    foreign=tmp_path/'foreign.py';foreign.write_text('same')
    before=dict(package_file=str(actual.resolve()),native_file='dso',native_sha256='hash',execution_environment={})
    row=dict(before,rank=0,package_file=str(alias.resolve()))
    assert runner.identity_mismatches(before,[row])==[]
    row['package_file']=str(foreign.resolve())
    mismatch=runner.identity_mismatches(before,[row])
    assert mismatch==[dict(rank=0,field='package_file',expected=str(actual.resolve()),observed=str(foreign.resolve()))]
def test_hash_and_environment_remain_strict():
    before=dict(package_file='p',native_file='n',native_sha256='h',execution_environment={'env':'a'})
    row=dict(before,rank=0,native_sha256='bad',execution_environment={'env':'b'})
    assert [m['field'] for m in runner.identity_mismatches(before,[row])]==['native_sha256','execution_environment']
def test_missing_xml_never_means_parity():
    assert not runner.rank_test_parity([{'rank':0,'status':'missing'},{'rank':1,'status':'missing'}])
    assert runner.rank_test_parity([{'nodes':[('a','b')]},{'nodes':[('a','b')]}])
def test_receipts_written_before_refusal_and_origin_writer_resolves():
    source=(ROOT/'docs/development/api_040/run_installed_mpi_checks.py').read_text()
    assert source.index('identity-check.json') < source.index('if mismatches:')
    writer=(ROOT/'docs/development/api_040/run_installed_checks.py').read_text()
    assert '"package_file": str(Path(pops.__file__).resolve())' in writer
