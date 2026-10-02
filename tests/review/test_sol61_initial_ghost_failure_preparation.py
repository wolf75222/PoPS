"""Source package/resolve only; no native-module qualification."""
from pathlib import Path
import subprocess
import shutil
import sys
import pops
from pops import interfaces
from tests.python.support.initial_ghost_failure_component import package_data,load_component,InitialFailureBoundary
from tests.python.support.initial_field_ghost_native_case import build
ROOT=Path(__file__).resolve().parents[2]

def test_genuine_failure_package_and_field_dependency(tmp_path):
    assert Path(pops.__file__).resolve().is_relative_to(ROOT/'python')
    assert 'pops._pops' not in sys.modules
    component=load_component(tmp_path/'component')
    manifest,_=package_data();interfaces.GhostBoundary.require_manifest(manifest)
    assert interfaces.required_native_interface_tables(manifest.signature,interfaces.GhostBoundary)==((1,1,'PopsGhostBoundaryApiV1'),(11,1,'PopsAcceptedInitialGhostApiV1'))
    case,layout=build(boundary_composer=lambda base:InitialFailureBoundary(base,component))
    plan=pops.resolve(pops.validate(case),layout=layout,components=(component,));plan.verify()
    assert len(plan.field_plans)==1 and len(plan.component_inputs)==1
    assert 'pops._pops' not in sys.modules

def test_complete_generated_header_source_syntax(tmp_path):
    _,source=package_data();path=tmp_path/'initial.cpp';path.write_bytes(source)
    compiler=shutil.which('clang++') or shutil.which('c++');assert compiler
    mpi=Path('/Users/romaindespoulain/miniforge3/envs/pops-api040-ir17/include')
    assert (mpi/'mpi.h').is_file()
    subprocess.run([compiler,'-std=c++20','-fsyntax-only','-DPOPS_NATIVE_DIM=2','-Wall','-Wextra','-Werror','-I',str(ROOT/'include'),'-I',str(mpi),str(path)],check=True,capture_output=True)
