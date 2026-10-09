"""Source ordering guards for the scheduler identity preparation; not a Native test."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HEADER=ROOT/'include/pops/runtime/multiblock/interface_flux_scheduler.hpp'
def test_identity_preparation_is_voted_before_consensus():
    text=HEADER.read_text()
    install=text[text.index('  void install('):text.index('  void install(',text.index('  void install(')+1)]
    assert 'const std::string collective_identity = prepare_collective_plan_identity_(' in install
    assert install.index('prepare_collective_plan_identity_(')<install.index('const ExactOrderedBytePair exact_contract_pair')<install.index('all_ranks_agree_exact_ordered_byte_pairs(')
    assert 'std::string communicator_identity;' in install
    assert install.index('try {')<install.index('communicator_identity.assign(')<install.index('catch (...)')
def test_serializer_and_vote_are_separate_from_materialization():
    text=HEADER.read_text()
    helper=text[text.index('  static std::string prepare_collective_plan_identity_('):text.index('  static std::string collective_plan_identity_(')]
    assert helper.index('try {')<helper.index('identity = collective_plan_identity_(')<helper.index('catch (...)')<helper.index('finish_collective_preflight_(')<helper.index('return identity;')
    assert 'communicator, preparation_failure, "route identity preparation"' in helper
    assert 'std::current_exception()' in helper
    assert 'all_ranks_agree_exact_ordered_byte_pairs' not in helper
