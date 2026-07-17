#pragma once

#include <functional>
#include <cstdint>
#include <map>
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
  bool replicated_parent = false;
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

struct PreparedTransferKernel {
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

  PreparedTransferKernel prepare(const std::string& qualified_id,
                                 const TransferRouteDescriptor& descriptor) const {
    const auto found = manifests_.find(qualified_id);
    if (found == manifests_.end())
      throw std::runtime_error("unregistered native AMR transfer kernel '" + qualified_id + "'");
    if (!found->second.accepts(descriptor))
      throw std::runtime_error(
          "native AMR transfer route is incompatible with its authenticated kernel manifest");
    return found->second.prepare(descriptor);
  }

 private:
  std::map<std::string, TransferKernelManifest> manifests_;
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

 private:
  TransferKernelRegistry kernels_;
  std::map<std::string, TransferRoute> routes_;
};

}  // namespace pops::runtime::amr
