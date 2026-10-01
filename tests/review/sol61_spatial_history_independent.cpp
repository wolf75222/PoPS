#include <pops/runtime/program/spatial_interaction_history_source.hpp>
#include <cassert>
#include <iostream>
using namespace pops;
using namespace pops::runtime::program;
int main() {
  HistoryManager<1> h;
  mesh::BoxArray<1> boxes({Box<1>(Index<1>{0}, Index<1>{1})});
  mesh::RankSpace<1> ranks(Index<1>{0}, Extent<1>{1});
  auto dist = mesh::Distribution<1>::replicated(boxes, ranks);
  MultiFab<1> seed(boxes, dist, Index<1>{0}, 1, Extent<1>{}); seed.set_val(Real(2));
  h.histories["T"] = {seed, seed, seed}; h.depth["T"] = 3; h.owner["T"] = 7;
  h.state_identity["T"] = "T-state"; h.space_identity["T"] = "T-space";
  h.clock_identity["T"] = "macro"; h.interpolation_identity["T"] = "none";
  h.initialized["T"] = false; h.fill_count["T"] = 0; h.store_pending["T"] = false;
  h.slot_dt["T"] = {Real(0),Real(0),Real(0)};
  h.slot_sample["T"] = {HistorySampleIdentity::zero_start(),HistorySampleIdentity::zero_start(),HistorySampleIdentity::zero_start()};
  auto before = h.slot_sample.at("T");
  int checks = 0;
  auto select = [&] { ExactContractBuilder exact; return interaction_history_selected(h,"T",2,7,"T-state","T-space","macro","none",exact); };
  assert(select()); ++checks;
  assert(h.slot_sample.at("T") == before && h.fill_count.at("T") == 0 && !h.initialized.at("T")); ++checks;
  auto sample = h.prepare_sample_store("T",0.,.125);
  h.slot_sample["T"] = sample; h.slot_dt["T"] = {Real(.125),Real(.125),Real(.125)};
  h.initialized["T"] = true; h.store_pending["T"] = true;
  assert(select()); ++checks;
  interaction_history_cold_equal(h.histories["T"][2],seed); ++checks;
  MultiFab<1> Q(boxes,dist,Index<1>{0},1,Extent<1>{}); Q.set_val(Real(6));
  bool bad = false;
  try { interaction_history_cold_equal(h.histories["T"][2],Q); } catch(const std::invalid_argument&) {bad=true;}
  assert(bad); ++checks;
  h.fill_count["T"] = 3; h.store_pending["T"] = false;
  assert(!select()); ++checks;
  auto retained = h.slot_sample["T"][2];
  h.slot_sample["T"] = h.prepare_sample_store("T",.125,.0625);
  h.slot_dt["T"][0] = Real(.0625); h.store_pending["T"] = true;
  assert(!select() && h.slot_sample["T"][2] == retained); ++checks;
  h.owner["T"] = 8; bad=false;
  try { select(); } catch(const std::invalid_argument&) {bad=true;}
  assert(bad); ++checks; h.owner["T"] = 7;
  h.slot_dt["T"][2] = Real(.25); bad=false;
  try { select(); } catch(const std::invalid_argument&) {bad=true;}
  assert(bad); ++checks; h.slot_dt["T"][2] = Real(.125);
  h.slot_sample["T"][2].kind = HistorySampleKind::UnknownLegacy; bad=false;
  try { select(); } catch(const std::invalid_argument&) {bad=true;}
  assert(bad); ++checks;
  std::cout << "independent-history checks=" << checks << " Realbits=" << sizeof(Real)*8 << '\n';
}
