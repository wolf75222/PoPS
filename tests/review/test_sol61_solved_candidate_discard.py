"""Actual SolveOutcome/SolveReport host bodies, debug and NDEBUG; no native runtime."""
from pathlib import Path
import signal
import re
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("release", (False, True))
def test_actual_outcome_discard_and_fail_stop_in_both_build_modes(tmp_path, release):
    report = (ROOT / "include/pops/numerics/elliptic/linear/solve_report.hpp").read_text()
    outcome = (ROOT / "include/pops/numerics/elliptic/linear/solve_outcome.hpp").read_text()
    # The bodies remain unchanged; only PoPS scalar/communicator declarations are
    # replaced with host seams. These serial probes do not qualify native MPI.
    source = """
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
namespace pops {
using Real=double;
struct ExecutionLane {};
long all_reduce_max(long x) { return x; }
long all_reduce_min(long x) { return x; }
long all_reduce_max(long x, const ExecutionLane&) { return x; }
long all_reduce_min(long x, const ExecutionLane&) { return x; }
}
""" + report[report.index("namespace pops {"):] + outcome[outcome.index("namespace pops {"):]
    source += r'''
struct Hooks {
  int accepts=0, rejects=0, releases=0, failures=0, validations=0;
  bool refuse=true;
  pops::SolveOutcome::PublicationHooks hooks() {
    return {this, [](void* p) noexcept { ++static_cast<Hooks*>(p)->accepts; },
      [](void* p) { ++static_cast<Hooks*>(p)->rejects; },
      [](void* p) noexcept { ++static_cast<Hooks*>(p)->releases; }, {},
      [](void* p) { auto& h=*static_cast<Hooks*>(p); ++h.validations;
        if(h.refuse) throw std::logic_error("refused"); },
      [](void* p,pops::SolveConsumption) { ++static_cast<Hooks*>(p)->failures; }};
  }
};
void check(bool condition) { if(!condition) throw std::runtime_error("probe check"); }
template<class Body> void refused(Body body) {
  bool threw=false; try { body(); } catch(const std::logic_error&) { threw=true; } check(threw);
}
int main(int argc,char**) {
  using namespace pops;
  static_assert(SolveOutcome::publication_contract_version==2);
  static_assert(int(SolveConsumption::kAccept)==0 && int(SolveConsumption::kRejectAttempt)==1
                && int(SolveConsumption::kFailRun)==2 && int(SolveConsumption::kDiscardCandidate)==3);
  SolveReport report; report.mark_solved("exact numerical result");
  report.iters=3; report.residual_norm=.125; report.reference_residual_norm=2;
  if(argc>1) { Hooks h; auto outcome=SolveOutcome::serial(report,h.hooks());
    refused([&]{(void)outcome.consume(SolveConsumption::kAccept);});
    return 0; } // This pending Outcome must abort, with and without NDEBUG.
  {
    Hooks h; auto outcome=SolveOutcome::serial(report,h.hooks());
    refused([&]{(void)outcome.consume(SolveConsumption::kAccept);});
    check(h.releases==0);
    outcome.discard_candidate();
    check(h.accepts==0 && h.rejects==0 && h.failures==0 && h.releases==1 && h.validations==1);
    check(outcome.report().status==report.status && outcome.report().action==report.action
          && outcome.report().reason==report.reason && outcome.report().iters==report.iters
          && outcome.report().residual_norm==report.residual_norm);
    refused([&]{outcome.discard_candidate();});
    refused([&]{(void)outcome.consume(SolveConsumption::kAccept);}); check(h.releases==1);
  }
  {
    Hooks h; auto outcome=SolveOutcome::serial(report,h.hooks());
    refused([&]{(void)outcome.consume(SolveConsumption::kAccept);});
    h.refuse=false; (void)outcome.consume(SolveConsumption::kAccept);
    check(h.accepts==1 && h.releases==1 && h.validations==2);
  }
  {
    report.mark_failed(SolveStatus::kBreakdown,SolveAction::kFailRun,"failed original");
    Hooks h; auto outcome=SolveOutcome::serial(report,h.hooks());
    refused([&]{outcome.discard_candidate();}); check(h.releases==0);
    (void)outcome.consume(SolveConsumption::kFailRun);
    check(h.failures==1 && h.accepts==0 && h.releases==1);
  }
}
'''
    cpp, exe = tmp_path / "actual_outcome.cpp", tmp_path / "actual_outcome"
    cpp.write_text(source)
    command = ["/usr/bin/clang++", "-std=c++20", "-O0", str(cpp), "-o", str(exe)]
    if release:
        command.insert(1, "-DNDEBUG")
    subprocess.run(command, check=True, timeout=30, capture_output=True)
    subprocess.run([str(exe)], check=True, timeout=5, capture_output=True)
    pending = subprocess.run([str(exe), "leave-pending"], timeout=5, capture_output=True)
    assert pending.returncode == -signal.SIGABRT
    assert b"SolveOutcome destroyed before explicit consumption" in pending.stderr


def test_native_attempt_fragment_is_really_discovered_by_cmake(tmp_path):
    cmake = (ROOT / "tests/CMakeLists.txt").read_text()
    declaration = cmake.split("pops_add_gtest_suite(NAME test_program_context_contract", 1)[1].split(")", 1)[0]
    assert "DISCOVERY_SOURCES" in declaration
    assert "solve_outcome_attempt_review.inc" in declaration
    cpp = ROOT / "tests/cpp/unit/runtime/test_program_context_contract.cpp"
    fragment = cpp.with_name("solve_outcome_attempt_review.inc")
    expected = {suite+"."+name for suite, name in re.findall(
        r"TEST\((ProgramContextContract),\s*(\w+)\)", cpp.read_text()+fragment.read_text())}
    historical_fragment = subprocess.check_output(
        ["git", "show", "9fb7e8fb64c3e3de289d9c139ba310e9e47e43a6:" + str(fragment.relative_to(ROOT))],
        cwd=ROOT, text=True)
    historical = {suite+"."+name for suite, name in re.findall(
        r"TEST\((ProgramContextContract),\s*(\w+)\)", historical_fragment)}
    assert len(historical) == 3
    assert historical <= expected
    assert "ProgramContextContract.DiscardSolvedNativeCandidateClosesPendingAttempt" in expected
    (tmp_path / "CMakeLists.txt").write_text('''cmake_minimum_required(VERSION 3.20)
project(InventoryOnly LANGUAGES NONE)
enable_testing()
include(GoogleTest)
add_executable(inventory IMPORTED)
set_target_properties(inventory PROPERTIES IMPORTED_LOCATION "/not-executed/inventory")
gtest_add_tests(TARGET inventory SOURCES "%s" "%s" TEST_LIST cases)
file(WRITE "${CMAKE_BINARY_DIR}/cases.txt" "${cases}")
''' % (cpp, fragment))
    subprocess.run(["cmake", "-S", str(tmp_path), "-B", str(tmp_path / "inventory")],
                   check=True, timeout=30, capture_output=True)
    found = set((tmp_path / "inventory/cases.txt").read_text().split(";"))
    assert found == expected
