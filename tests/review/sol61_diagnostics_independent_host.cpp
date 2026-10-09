#include <pops/runtime/program/program_diagnostics_checkpoint.hpp>
#include <atomic>
#include <exception>
#include <iostream>
#include <memory>

// Real codec and source-extracted restore bodies; lane/storage/fence are host adapters.
namespace Kokkos {
inline bool fail_fence = false;
inline void fence() { if (fail_fence) throw std::runtime_error("host fence failure"); }
}
struct Lane { int rank() const { return 1; } int size() const { return 2; } };
inline bool peer_failure = false;
inline int votes = 0;
inline void collectively_rethrow_exception(std::exception_ptr e, const Lane&, const char*) {
  ++votes;
  if (peer_failure) throw std::runtime_error("host peer preparation failure");
  if (e) std::rethrow_exception(e);
}
namespace pops {
using Table = std::map<std::string, Real>;
struct Program { Table diagnostics_; };
struct Authority { std::atomic<int> pending{0}; };
template<int Dim> struct System {
  struct Impl { Program program_; bool external_restart_transaction_=true;
    bool external_step_transaction_committed_=false; };
  std::unique_ptr<Impl> p_=std::make_unique<Impl>();
  std::unique_ptr<Authority> solve_outcome_authority_=std::make_unique<Authority>();
  Lane lane;
  const Lane& prepared_boundary_execution_lane() const { return lane; }
  void restore_checkpoint_program_diagnostics(std::span<const std::uint8_t> (*)(const void*), const void*);
};
template<int Dim> struct AmrSystem {
  struct Impl { Program program; bool restart_transaction=true;
    bool restart_transaction_committed=false; Lane lane;
    const Lane& require_package_assembly_lane() const { return lane; } };
  std::unique_ptr<Impl> p_=std::make_unique<Impl>();
  int depth=0;
  int step_transaction_depth() const { return depth; }
  void restore_checkpoint_program_diagnostics(std::span<const std::uint8_t> (*)(const void*), const void*);
};
// SOURCE_EXTRACTED_BODIES
}
using namespace pops;
using namespace pops::runtime::program;
int checks=0;
void check(bool good) { ++checks; if (!good) throw std::runtime_error("independent check failed"); }
template<class F> void refuses(F f) { bool bad=false; try { f(); } catch (...) { bad=true; } check(bad); }
std::vector<std::uint8_t> wire(const Table& t) { return checkpoint_program_diagnostics(t,1,2); }
bool exact(const Table& a,const Table& b) { return wire(a)==wire(b); }
std::span<const std::uint8_t> produce(const void* c) {
  return *static_cast<const std::vector<std::uint8_t>*>(c);
}
std::span<const std::uint8_t> bad_producer(const void*) { throw std::bad_alloc(); }
void word(std::vector<std::uint8_t>& b, std::size_t at, std::uint64_t n) {
  for (int i=0;i<8;++i) b[at+i]=std::uint8_t(n>>(8*i));
}
template<class Owner> void atomic_restore(Owner& s,Table& live) {
  const auto original=live;
  auto replacement=wire(Table{{"after",Real(9)}});
  peer_failure=true;
  refuses([&]{s.restore_checkpoint_program_diagnostics(produce,&replacement);});
  check(exact(live,original)); peer_failure=false;
  Kokkos::fail_fence=true;
  refuses([&]{s.restore_checkpoint_program_diagnostics(produce,&replacement);});
  check(exact(live,original)); Kokkos::fail_fence=false;
  refuses([&]{s.restore_checkpoint_program_diagnostics(bad_producer,nullptr);});
  check(exact(live,original));
  auto malformed=replacement; malformed.push_back(0);
  refuses([&]{s.restore_checkpoint_program_diagnostics(produce,&malformed);});
  check(exact(live,original));
  s.restore_checkpoint_program_diagnostics(produce,&replacement);
  check(wire(live)==replacement);
  // Model the enclosing accepted snapshot rollback separately from the restore method.
  live=original; check(exact(live,original));
  std::vector<std::uint8_t> legacy;
  s.restore_checkpoint_program_diagnostics(produce,&legacy); check(live.empty());
}
int main() {
  const RealBits nan=sizeof(RealBits)==8 ? RealBits(0x7ff8000000000142ULL) : RealBits(0x7fc00142);
  const RealBits sign=RealBits(1)<<(8*sizeof(RealBits)-1);
  const Table table{{"",std::bit_cast<Real>(sign)},
    {std::string("\0opaque",7),std::bit_cast<Real>(nan)},
    {std::string(1,char(0xff)),Real(0)}};
  const auto encoded=wire(table);
  check(encoded.size()==40+3*16+8);
  check(wire(read_program_diagnostics_checkpoint(encoded,1,2))==encoded);
  for(std::size_t n=0;n<encoded.size();++n)
    refuses([&]{read_program_diagnostics_checkpoint(std::span(encoded).first(n),1,2);});
  for(auto offset:{8,16,24,32,40}) {
    auto bad=encoded; word(bad,offset,~std::uint64_t(0));
    refuses([&]{read_program_diagnostics_checkpoint(bad,1,2);});
  }
  auto trailing=encoded; trailing.push_back(0);
  refuses([&]{read_program_diagnostics_checkpoint(trailing,1,2);});
  auto duplicate=wire(Table{{"a",1},{"b",2}}); duplicate[65]='a';
  refuses([&]{read_program_diagnostics_checkpoint(duplicate,1,2);});
  refuses([&]{wire(Table{{"pops.balance-term.foreign",1}});});
  check(wire(Table{{"xpops.balance-term",1}}).size()>40);
  auto empty=wire({}); check(empty.size()==40);
  check(read_program_diagnostics_checkpoint(empty,1,2).empty());
  System<1> u; u.p_->program_.diagnostics_=table;
  atomic_restore(u,u.p_->program_.diagnostics_);
  AmrSystem<1> a; a.p_->program.diagnostics_=table;
  atomic_restore(a,a.p_->program.diagnostics_);
  check(votes==12);
  std::cout<<"independent assertions="<<checks<<" real_bits="<<sizeof(RealBits)*8<<"\n";
}
