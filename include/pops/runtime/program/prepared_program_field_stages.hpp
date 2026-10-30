#pragma once

#include <pops/core/foundation/kokkos_env.hpp>
#include <pops/parallel/collective_exception.hpp>
#include <pops/runtime/program/prepared_resource_lifetime.hpp>
#include <pops/runtime/system/prepared_field_rhs_inputs.hpp>

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pops::runtime::program {
/// Retains exact per-level Field inputs until one authenticated hierarchy publication.
/// Contexts provide owner/slot admission; this authority owns ordering and retained proofs.
template <int Dim>
class PreparedProgramFieldStages {
 public:
  using field_type = MultiFab<Dim>;
  using request_type = runtime::system::FieldSolveRequest<Dim>;
  struct Stamp {
    runtime::multiblock::BoundaryEvaluationPoint point;
    std::uint64_t epoch = 0, generation = 0, sequence = 0, attempt = 0;
    friend bool operator==(const Stamp&, const Stamp&) = default;
  };
  struct Sources {
    Stamp stamp;
    std::string field;
    std::vector<typename request_type::Source> sources;
  };
  struct WarmStart {
    Stamp stamp;
    std::string slot, identity;
    int owner = -1;
    std::int64_t source_value = -1;
    int source_subslot = -1;
    const field_type* source = nullptr;
    std::shared_ptr<const field_type> image;
    PreparedResourceAttempt resource;
  };
  struct WarmStartPublication {
    const WarmStart* root = nullptr;
    std::vector<const field_type*> images;
  };

  template <class Overrides, class StageSource, class AcceptedSource>
  static Sources prepare_sources(Stamp stamp, std::int64_t field_node, std::string field,
                                 const ProgramValuePlan& plan, const Overrides& overrides,
                                 StageSource&& stage_source, AcceptedSource&& accepted_source) {
    Sources result{std::move(stamp), std::move(field), {}};
    std::set<int> seen;
    for (const auto& item : overrides) {
      if (!item.state || item.source_ssa < 0 || !seen.insert(item.program_block).second)
        throw std::invalid_argument("AMR SSA Field stage requires unique successful SSA sources");
      result.sources.push_back(stage_source(item));
    }
    for (const auto& edge : plan.fields) {
      if (edge.field_node != field_node || seen.contains(edge.program_block))
        continue;
      if (!edge.accepted_only)
        throw std::invalid_argument("AMR Field stage lacks its declared successful SSA source");
      result.sources.push_back(accepted_source(edge.program_block));
    }
    if (result.sources.empty())
      throw std::invalid_argument("AMR Field stage has no installed source authority");
    return result;
  }

  void stage_sources(std::int64_t field_node, Sources stage) {
    auto& levels = sources_[field_node];
    if (levels.size() != static_cast<std::size_t>(stage.stamp.point.level))
      throw std::logic_error("AMR Field stage is duplicate, missing, or out of hierarchy order");
    levels.push_back(std::move(stage));
  }

  template <class PrepareSingle, class CreateRequest>
  request_type prepare_request(const Stamp& stamp, std::int64_t field_node,
                               const std::string& field, int level_count, bool root_callback,
                               PrepareSingle&& prepare_single,
                               CreateRequest&& create_request) const {
    const auto found = sources_.find(field_node);
    std::vector<Sources> single;
    const std::vector<Sources>* levels = nullptr;
    if (found != sources_.end()) {
      if (stamp.point.level != 0 || !root_callback)
        throw std::logic_error("composite Field solve requires the unique root hierarchy callback");
      levels = &found->second;
    } else {
      if (level_count != 1)
        throw std::invalid_argument(
            "composite Field solve requires every exact level stage; accepted fallback is "
            "forbidden");
      single.push_back(prepare_single());
      levels = &single;
    }
    if (levels->size() != static_cast<std::size_t>(level_count))
      throw std::invalid_argument("composite Field solve lacks a live hierarchy source level");
    std::vector<typename request_type::Source> sources;
    for (std::size_t level = 0; level < levels->size(); ++level) {
      const auto& staged = levels->at(level);
      auto expected = stamp;
      expected.point.level = static_cast<int>(level);
      if (staged.stamp != expected || staged.field != field)
        throw std::invalid_argument(
            "composite Field stage lost clock, attempt, topology, or materialization authority");
      for (const auto& source : staged.sources) {
        source.validate();
        sources.push_back(source);
      }
    }
    return create_request(field, stamp.point, std::move(sources));
  }

  void stage_warm_start(Stamp stamp, const std::string& slot, const std::string& identity,
                        int owner, std::int64_t source_value, int source_subslot,
                        const field_type& potential, PreparedResourceAttempt resource) {
    if (slot.empty() || identity.empty() || potential.ncomp() != 1)
      throw std::invalid_argument("AMR field warm start requires its exact scalar hierarchy stage");
    auto& levels = warm_starts_[stamp.point.stage];
    if (levels.size() != static_cast<std::size_t>(stamp.point.level))
      throw std::invalid_argument("AMR warm-start stage is duplicate, missing, or out of order");
    device_fence();
    levels.push_back({std::move(stamp), slot, identity, owner, source_value, source_subslot,
                      &potential, std::make_shared<const field_type>(potential),
                      std::move(resource)});
  }

  template <class AuthenticateSource>
  WarmStartPublication prepare_warm_start(const Stamp& stamp, int evaluation_id, int level_count,
                                          bool root_callback,
                                          const PreparedResourceAttempt& resource,
                                          AuthenticateSource&& authenticate_source) const {
    const auto found = warm_starts_.find(evaluation_id);
    if (stamp.point.level != 0 || !root_callback || found == warm_starts_.end() ||
        found->second.size() != static_cast<std::size_t>(level_count))
      throw std::invalid_argument(
          "AMR warm start requires one complete root hierarchy publication");
    WarmStartPublication publication{&found->second.front(), {}};
    if (publication.root->stamp.point != stamp.point)
      throw std::invalid_argument("AMR warm start lost its current root point");
    for (std::size_t level = 0; level < found->second.size(); ++level) {
      const auto& staged = found->second[level];
      auto expected = stamp;
      expected.point.level = static_cast<int>(level);
      if (staged.stamp != expected || staged.slot != publication.root->slot ||
          staged.identity != publication.root->identity ||
          staged.owner != publication.root->owner || !staged.resource.visible() ||
          !staged.resource.same_attempt(resource) ||
          !authenticate_source(static_cast<int>(level), staged) || !staged.image)
        throw std::invalid_argument(
            "AMR warm start lost scalar, clock, attempt, or hierarchy authority");
      publication.images.push_back(staged.image.get());
    }
    return publication;
  }

  template <class PrepareRequest, class Solve>
  auto solve_sources(std::int64_t field_node, const ExecutionLane& lane,
                     PrepareRequest&& prepare_request, Solve&& solve) {
    std::optional<request_type> request;
    std::exception_ptr error;
    try {
      request.emplace(prepare_request());
    } catch (...) {
      error = std::current_exception();
    }
    collectively_rethrow_exception(error, lane, "AMR SSA Field request failed collectively");
    // The request keeps source proofs alive after staged storage is released.
    erase_sources(field_node);
    return solve(*request);
  }

  template <class PreparePublication, class Publish>
  void publish_warm_start(int evaluation_id, const ExecutionLane& lane,
                          PreparePublication&& prepare_publication, Publish&& publish) {
    WarmStartPublication publication;
    std::exception_ptr error;
    try {
      publication = prepare_publication();
    } catch (...) {
      error = std::current_exception();
    }
    collectively_rethrow_exception(error, lane,
                                   "AMR warm-start hierarchy admission failed collectively");
    publish(publication);
    erase_warm_start(evaluation_id);
  }

  void erase_sources(std::int64_t field_node) { sources_.erase(field_node); }
  void erase_warm_start(int evaluation_id) { warm_starts_.erase(evaluation_id); }
  void clear() noexcept {
    sources_.clear();
    warm_starts_.clear();
  }
  [[nodiscard]] bool empty() const noexcept { return sources_.empty() && warm_starts_.empty(); }

 private:
  std::map<std::int64_t, std::vector<Sources>> sources_;
  std::map<std::int64_t, std::vector<WarmStart>> warm_starts_;
};
}  // namespace pops::runtime::program
