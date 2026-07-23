#pragma once

#include <pops/mesh/boundary/fill_boundary.hpp>
#include <pops/numerics/time/amr/prepared_coarse_fine_operator.hpp>

#include <algorithm>
#include <functional>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace pops {
class AmrRuntime;
class MultiFab;
}  // namespace pops

namespace pops::runtime::amr {

enum class TransferCentering { Cell, FaceX, FaceY, Node };

struct TransferRouteDescriptor {
  std::string space;
  std::string centering;
  std::string representation;
  std::string storage;
  std::string operation;
  int order;
  std::vector<int> ghost_depth;
  int dimension;
  int refinement_ratio;
};

struct IndexTransform {
  std::vector<int> coarse_origin;
  std::vector<int> fine_origin;
  std::vector<int> refinement_ratio;
};

struct SpatialTransferContext {
  int coarse_level = 0;
  int fine_level = 0;
  int components = 0;
  IndexTransform index;
  Box2D logical_coarse_domain{};
  Box2D logical_fine_domain{};
  bool replicated_parent = false;
  Periodicity periodicity{};
};

struct TransferTimePoint {
  int step = 0;
  double physical_time = 0.0;
};

struct TemporalTransferContext {
  TransferTimePoint old_point;
  TransferTimePoint new_point;
  TransferTimePoint target_point;
  std::int64_t exact_alpha_numerator = 0;
  std::int64_t exact_alpha_denominator = 0;

  double alpha() const {
    if (old_point.step >= new_point.step || !(old_point.physical_time < new_point.physical_time) ||
        target_point.step < old_point.step || target_point.step > new_point.step ||
        target_point.physical_time < old_point.physical_time ||
        target_point.physical_time > new_point.physical_time)
      throw std::runtime_error("invalid native AMR temporal interpolation window");
    if (exact_alpha_denominator != 0) {
      if (exact_alpha_denominator < 0 || exact_alpha_numerator < 0 ||
          exact_alpha_numerator > exact_alpha_denominator)
        throw std::runtime_error("invalid exact native AMR temporal interpolation coordinate");
      return static_cast<double>(exact_alpha_numerator) /
             static_cast<double>(exact_alpha_denominator);
    }
    return (target_point.physical_time - old_point.physical_time) /
           (new_point.physical_time - old_point.physical_time);
  }
};

struct MaterializationContext {
  std::string subject;
  std::string target;
  std::string operation;
  int level = 0;
};

/// Capabilities authenticated by the route descriptor that prepared a kernel.
///
/// Keep these facts attached to the executable authority: runtime consumers must not infer an
/// interpolation order from a kernel id, a stencil radius, or the current builtin catalogue.
struct PreparedTransferCapabilities {
  int order = 0;
  std::vector<int> ghost_depth;
};

struct PreparedTransferKernel {
  PreparedTransferCapabilities capabilities;
  /// Prepared spatial identity used by allocation-free FillPatch.  A coarse/fine callable without
  /// this value is not eligible for the native subcycling route.
  std::shared_ptr<const PreparedCoarseFineOperator> prepared_coarse_fine;
  std::function<void(const MultiFab&, MultiFab&, const SpatialTransferContext&)> spatial;
  std::function<void(const MultiFab&, MultiFab&, const SpatialTransferContext&)> coarse_fine;
  std::function<void(const MultiFab&, const MultiFab&, MultiFab&, MultiFab&,
                     const SpatialTransferContext&)>
      face_vector;
  std::function<void(const MultiFab&, const MultiFab&, MultiFab&, const TemporalTransferContext&)>
      temporal;
  std::function<std::int64_t(AmrRuntime&, const MaterializationContext&)> materialize;
};

/// Small extension protocol for one native transfer kernel.  Builtins and external backends register
/// the same authenticated manifest; the route registry never branches on concrete kernel IDs.
struct TransferKernelManifest {
  std::string qualified_id;
  std::function<bool(const TransferRouteDescriptor&)> accepts;
  std::function<PreparedTransferKernel(const TransferRouteDescriptor&)> prepare;
};

class TransferKernelRegistry {
 public:
  void add(TransferKernelManifest manifest) {
    if (manifest.qualified_id.empty() || !manifest.accepts || !manifest.prepare ||
        manifests_.count(manifest.qualified_id) != 0)
      throw std::runtime_error("invalid or duplicate native AMR transfer kernel manifest");
    manifests_.emplace(manifest.qualified_id, std::move(manifest));
  }

  void add_exact(std::string qualified_id, TransferRouteDescriptor descriptor,
                 std::function<PreparedTransferKernel(const TransferRouteDescriptor&)> prepare) {
    const auto exact_descriptor = descriptor;
    add(TransferKernelManifest{
        qualified_id,
        [exact_descriptor](const TransferRouteDescriptor& row) {
          return row.space == exact_descriptor.space &&
                 row.centering == exact_descriptor.centering &&
                 row.representation == exact_descriptor.representation &&
                 row.storage == exact_descriptor.storage &&
                 row.operation == exact_descriptor.operation &&
                 row.order == exact_descriptor.order &&
                 row.ghost_depth == exact_descriptor.ghost_depth &&
                 row.dimension == exact_descriptor.dimension &&
                 row.refinement_ratio == exact_descriptor.refinement_ratio;
        },
        std::move(prepare)});
    catalogue_.push_back({std::move(qualified_id), std::move(descriptor)});
  }

  PreparedTransferKernel prepare(const std::string& qualified_id,
                                 const TransferRouteDescriptor& descriptor) const {
    const auto found = manifests_.find(qualified_id);
    if (found == manifests_.end())
      throw std::runtime_error("unregistered native AMR transfer kernel '" + qualified_id + "'");
    if (!found->second.accepts(descriptor))
      throw std::runtime_error(
          "native AMR transfer route is incompatible with its authenticated kernel manifest");
    PreparedTransferKernel prepared = found->second.prepare(descriptor);
    if (static_cast<bool>(prepared.coarse_fine) !=
        static_cast<bool>(prepared.prepared_coarse_fine))
      throw std::runtime_error(
          "native AMR coarse/fine provider must pair its callable with one prepared spatial "
          "identity");
    if (descriptor.operation == "coarse_fine_fill" && !prepared.coarse_fine &&
        !prepared.materialize)
      throw std::runtime_error(
          "native AMR coarse/fine provider omitted its executable authority");
    // The manifest acceptance predicate authenticates the descriptor.  Publish exactly those
    // capabilities with the callable so downstream code never needs a route-name switch.
    prepared.capabilities = PreparedTransferCapabilities{descriptor.order, descriptor.ghost_depth};
    return prepared;
  }

  PreparedTransferKernel prepare_minimum(const TransferRouteDescriptor& requirement) const {
    if (requirement.dimension < 1 || requirement.order < 1 || requirement.refinement_ratio < 2 ||
        (requirement.ghost_depth.size() != 1 &&
         requirement.ghost_depth.size() != static_cast<std::size_t>(requirement.dimension)) ||
        std::any_of(requirement.ghost_depth.begin(), requirement.ghost_depth.end(),
                    [](int depth) { return depth < 0; }))
      throw std::runtime_error("invalid native AMR transfer capability requirement");
    struct Match {
      const std::string* qualified_id;
      const TransferRouteDescriptor* descriptor;
      int order_surplus;
      std::vector<int> ghost_surplus;
    };
    std::vector<Match> matches;
    const auto expanded = [&](const std::vector<int>& ghost) {
      return ghost.size() == 1 ? std::vector<int>(static_cast<std::size_t>(requirement.dimension),
                                                  ghost.front())
                               : ghost;
    };
    const std::vector<int> needed = expanded(requirement.ghost_depth);
    for (const auto& [qualified_id, descriptor] : catalogue_) {
      if (descriptor.space != requirement.space || descriptor.centering != requirement.centering ||
          descriptor.representation != requirement.representation ||
          descriptor.storage != requirement.storage ||
          descriptor.operation != requirement.operation ||
          descriptor.dimension != requirement.dimension ||
          descriptor.refinement_ratio != requirement.refinement_ratio ||
          descriptor.order < requirement.order)
        continue;
      const std::vector<int> available = expanded(descriptor.ghost_depth);
      if (available.size() != needed.size())
        continue;
      std::vector<int> ghost_surplus;
      ghost_surplus.reserve(available.size());
      bool supports = true;
      for (std::size_t axis = 0; axis < available.size(); ++axis) {
        if (available[axis] < needed[axis])
          supports = false;
        ghost_surplus.push_back(available[axis] - needed[axis]);
      }
      if (supports)
        matches.push_back(Match{&qualified_id, &descriptor,
                                descriptor.order - requirement.order,
                                std::move(ghost_surplus)});
    }
    if (matches.empty())
      throw std::runtime_error(
          "native AMR transfer registry has no capability route satisfying the requirement");
    const auto dominates = [](const Match& left, const Match& right) {
      if (left.order_surplus > right.order_surplus ||
          left.ghost_surplus.size() != right.ghost_surplus.size())
        return false;
      bool strictly_better = left.order_surplus < right.order_surplus;
      for (std::size_t axis = 0; axis < left.ghost_surplus.size(); ++axis) {
        if (left.ghost_surplus[axis] > right.ghost_surplus[axis])
          return false;
        strictly_better = strictly_better ||
                          left.ghost_surplus[axis] < right.ghost_surplus[axis];
      }
      return strictly_better;
    };
    std::vector<const Match*> frontier;
    for (const Match& candidate : matches) {
      const bool dominated = std::any_of(
          matches.begin(), matches.end(), [&](const Match& other) {
            return &candidate != &other && dominates(other, candidate);
          });
      if (!dominated)
        frontier.push_back(&candidate);
    }
    if (frontier.size() != 1)
      throw std::runtime_error(
          "native AMR transfer registry has ambiguous non-dominated capability routes");
    return prepare(*frontier.front()->qualified_id, *frontier.front()->descriptor);
  }

 private:
  std::map<std::string, TransferKernelManifest> manifests_;
  std::vector<std::pair<std::string, TransferRouteDescriptor>> catalogue_;
};

struct TransferRoute {
  std::string provider_identity;
  std::string kernel_identity;
  TransferRouteDescriptor descriptor;
  PreparedTransferKernel executable;

  auto exact_key() const {
    return std::tuple{provider_identity,          kernel_identity,
                      descriptor.space,           descriptor.centering,
                      descriptor.representation,  descriptor.storage,
                      descriptor.operation,       descriptor.order,
                      descriptor.ghost_depth,     descriptor.dimension,
                      descriptor.refinement_ratio};
  }
};

class TransferRouteRegistry {
 public:
  explicit TransferRouteRegistry(TransferKernelRegistry kernels) : kernels_(std::move(kernels)) {}

  void add(std::string identity, TransferRoute route) {
    if (identity.empty() || route.provider_identity.empty())
      throw std::runtime_error("native AMR transfer route identity/provider must be non-empty");
    route.executable = kernels_.prepare(route.kernel_identity, route.descriptor);
    if (routes_.count(identity) != 0)
      throw std::runtime_error("duplicate native AMR transfer route identity");
    routes_.emplace(std::move(identity), std::move(route));
  }

  const TransferRoute& at(const std::string& identity) const {
    const auto found = routes_.find(identity);
    if (found == routes_.end())
      throw std::runtime_error("native AMR transfer route identity is not registered");
    return found->second;
  }

  std::size_t size() const { return routes_.size(); }

  PreparedTransferKernel prepare_minimum(
      const TransferRouteDescriptor& requirement) const {
    return kernels_.prepare_minimum(requirement);
  }

 private:
  TransferKernelRegistry kernels_;
  std::map<std::string, TransferRoute> routes_;
};

}  // namespace pops::runtime::amr
