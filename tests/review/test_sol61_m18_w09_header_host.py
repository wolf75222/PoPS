"""Real common header probe, not installed PoPS execution."""
from pathlib import Path
import subprocess

def test_real_common_header_near_boundary_and_domain_failure(tmp_path):
    root=Path(__file__).resolve().parents[2]
    binary=tmp_path/'local-status'
    subprocess.run(['clang++','-std=c++20','-Wall','-Wextra','-Werror','-O2','-fno-fast-math','-ffp-contract=off','-I',str(root/'include'),str(root/'tests/review/sol61_m18_w09_local_status.cpp'),'-o',str(binary)],check=True,capture_output=True,text=True)
    output=subprocess.check_output([str(binary)],text=True)
    rows=[row.split() for row in output.splitlines()]
    assert len(rows)==4 and int(rows[0][1])==0 and int(rows[1][1])==0 and float(rows[1][3])<2e-11
    assert int(rows[2][1])!=0 and int(rows[3][1])!=0
