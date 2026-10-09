"""Source/host tests only; no claim of actual Native stage injection."""
from pathlib import Path
import subprocess
import shutil
import numpy as np
import pytest
from tests.python.integration.amr.test_public_accepted_halo_stage_failure import same_accepted_payload
from pops.runtime._checkpoint_manifest import MANIFEST_KEY,IDENTITY_KEY

ROOT=Path(__file__).resolve().parents[2]

def test_actual_standalone_request_header_host(tmp_path):
    compiler=shutil.which('clang++') or shutil.which('c++')
    if compiler is None:pytest.skip('host C++20 compiler unavailable')
    source=tmp_path/'request.cpp';binary=tmp_path/'request'
    source.write_text('''#include <pops/runtime/accepted_halo_test_failure.hpp>
#include <cassert>
int main(){
  pops::AcceptedHaloTestFailureRequest r{1,pops::AcceptedHaloTestFailurePhase::after_block_level_preparation,0,1,1};
  pops::validate_accepted_halo_test_failure_request(r,2,3,2);
  for(int n=0;n<5;++n){auto bad=r;
    if(n==0)bad.version=2;if(n==1)bad.phase=static_cast<pops::AcceptedHaloTestFailurePhase>(2);
    if(n==2)bad.block=2;if(n==3)bad.level=-1;if(n==4)bad.rank=2;
    bool refused=false;try{pops::validate_accepted_halo_test_failure_request(bad,2,3,2);}
    catch(const std::invalid_argument&){refused=true;}assert(refused);
  }
}''')
    subprocess.run([compiler,'-std=c++20','-Wall','-Wextra','-Werror','-I',str(ROOT/'include'),str(source),'-o',str(binary)],check=True,capture_output=True)
    subprocess.run([str(binary)],check=True,capture_output=True)

@pytest.mark.parametrize('member',('state','clock','history','diagnostics','field'))
def test_only_lifecycle_seals_can_change_in_comparator(member):
    before={key:np.array([1.]) for key in ('state','clock','history','diagnostics','field',MANIFEST_KEY,IDENTITY_KEY)}
    after={key:value.copy() for key,value in before.items()}
    after[MANIFEST_KEY][0]=2;after[IDENTITY_KEY][0]=2;same_accepted_payload(before,after)
    after[member][0]=np.nextafter(1.,2.)
    with pytest.raises(AssertionError,match=member):same_accepted_payload(before,after)

def test_failure_seam_is_inside_the_existing_try_vote_and_before_publication():
    source=(ROOT/'src/runtime/amr/amr_system.cpp').read_text()
    start=source.index('void AmrSystem<Dim>::prepare_accepted_halo_candidates(')
    end=source.index('void AmrSystem<Dim>::publish_prepared_amr_program_candidates(',start)
    body=source[start:end]
    injection=body.index('receipt.reached = receipt.consumed = receipt.before_publication = true;')
    fence=body.rfind('Kokkos::fence();',0,injection)
    prepare=body.rfind('prepare_generated_amr_block_level_state(',0,injection)
    vote=body.index('accepted halo test failure after block-level preparation fence failed collectively',injection)
    preparation_vote=body.index('accepted halo candidate preparation/copy/fence failed collectively',prepare)
    assert fence<preparation_vote<injection
    assert prepare<fence<injection<vote
    assert 'restore_accepted();' in body[vote:]
    assert 'publish_program_candidates' not in body
