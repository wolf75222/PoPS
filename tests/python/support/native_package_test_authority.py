"""Distinct installed-prefix and authenticated CI build-artifact test origins."""
import hashlib
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]

def authenticate_ci_source_files(package, dimension, source_root=ROOT):
    source_root=Path(source_root).resolve();package=Path(package).resolve()
    expected=(source_root/'.pops-ci/python-packages'/f'dim{dimension}').resolve()
    if type(dimension) is not int or dimension not in (1,2,3) or package!=expected:
        raise RuntimeError('CI package is not the exact declared dimension artifact')
    source=source_root/'python/pops'
    rows={str(p.relative_to(source)):p for p in source.rglob('*.py')}
    observed={str(p.relative_to(package/'pops')):p for p in (package/'pops').rglob('*.py')}
    if not rows or set(rows)!=set(observed):
        raise RuntimeError('CI package Python source coverage differs')
    hashes={}
    for name,path in rows.items():
        expected_hash=hashlib.sha256(path.read_bytes()).hexdigest()
        if hashlib.sha256(observed[name].read_bytes()).hexdigest()!=expected_hash:
            raise RuntimeError('CI package Python source hash differs: '+name)
        hashes[name]=expected_hash
    return hashes

def authenticate_native_test_package(package_module,native,dimension):
    origin=Path(package_module.__file__).resolve()
    declared=os.environ.get('POPS_CI_NATIVE_PACKAGE')
    if declared is None:
        if not origin.is_relative_to(Path(sys.prefix).resolve()):
            raise RuntimeError('installed fixture package is outside the active prefix')
        return {'contract':'pops.native-test-origin@1','kind':'installed-prefix'}
    package=Path(declared).resolve()
    sources=authenticate_ci_source_files(package,dimension,ROOT)
    if origin!=(package/'pops/__init__.py').resolve():
        raise RuntimeError('imported Python package differs from declared CI artifact')
    from scripts.verify_installed_native import verify_installed_native
    extension=verify_installed_native(expect_dimension=dimension,expect_mpi=False,selector=lambda requested:native)
    if not extension.is_relative_to(package/'pops/_native'):
        raise RuntimeError('native DSO differs from declared CI artifact')
    from pops.codegen.abi import module_header_signature
    from pops.codegen.toolchain import pops_header_signature
    signature=module_header_signature()
    if not signature or signature!=pops_header_signature(ROOT/'include'):
        raise RuntimeError('CI native header signature differs from checkout')
    return {'contract':'pops.native-test-origin@1','kind':'ci-build-artifact','python_source_sha256':sources,'native_sha256':hashlib.sha256(extension.read_bytes()).hexdigest(),'headers':signature}
