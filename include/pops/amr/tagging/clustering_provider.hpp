#pragma once

#include <pops/amr/tagging/cluster.hpp>
#include <pops/amr/tagging/tag_box.hpp>
#include <pops/mesh/index/box2d.hpp>

#include <vector>

namespace pops::amr {

/// Core clustering extension seam.  It depends only on AMR geometry; dynamic loading and runtime
/// preparation remain in the adapter layer.
class ClusteringProvider {
 public:
  virtual ~ClusteringProvider() = default;
  virtual std::vector<Box2D> cluster(const TagBox& tags) const = 0;
};

class BergerRigoutsosProvider final : public ClusteringProvider {
 public:
  explicit BergerRigoutsosProvider(ClusterParams params) : params_(params) {}
  std::vector<Box2D> cluster(const TagBox& tags) const override {
    return berger_rigoutsos(tags, params_);
  }

 private:
  ClusterParams params_;
};

}  // namespace pops::amr
