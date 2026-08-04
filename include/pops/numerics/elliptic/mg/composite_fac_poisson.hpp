/// @file
/// @brief Capability-qualified boundary for the historical two-dimensional composite FAC backend.

#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/core/identity/prepared_provider.hpp>
#include <pops/numerics/elliptic/linear/solve_report.hpp>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace pops::numerics::elliptic {

/// Geometry families understood by an elliptic provider capability query.
enum class CompositeEllipticGeometry : std::uint8_t {
  Cartesian = 0,
  Polar = 1,
  EmbeddedBoundary = 2,
};

/// Dynamic request presented only at the provider-selection boundary.
///
/// Python resolves the dimension before native allocation.  The integer here authenticates that
/// the selected extension and the compiled plan agree; it is not an execution-time dispatch tag.
struct CompositeFacCapabilityRequest {
  int dimension = 0;
  CompositeEllipticGeometry geometry = CompositeEllipticGeometry::Cartesian;
  bool cell_centered = true;
  bool nested_hierarchy = true;
};

/// Opaque invocation owned by a separately built, explicitly two-dimensional backend.
///
/// The generic PoPS headers deliberately do not embed rank-specialized field storage here: doing so
/// would install a second rank authority inside the 1D and 3D variants.  A selected extension prepares
/// its workspace
/// from the exact spatial and operator contracts, then exposes that workspace through this narrow
/// trusted boundary.
struct CompositeFacPoisson2DInvocation {
  std::string_view spatial_contract{};
  std::string_view operator_contract{};
  void* prepared_workspace = nullptr;
  int maximum_iterations = 0;
  Real relative_tolerance = Real(0);
};

using PreparedCompositeFacPoisson2DKernel =
    PreparedProvider<SolveReport(const CompositeFacPoisson2DInvocation&)>;

/// Explicit capability wrapper for the legacy FAC algorithm.
///
/// This type intentionally has no `Dim` template parameter and cannot masquerade as an ND solver.
/// A future real ND composite elliptic algorithm belongs in a separate ranked provider.  Until
/// then, non-Cartesian or non-2D plans fail during capability resolution rather than falling back to
/// a partially used coordinate or a hidden 2D allocation.
class CompositeFacPoisson2DProvider {
 public:
  static constexpr int supported_dimension = 2;

  explicit CompositeFacPoisson2DProvider(PreparedCompositeFacPoisson2DKernel kernel)
      : kernel_(std::move(kernel)) {
    if (!kernel_)
      throw std::invalid_argument("composite FAC 2D provider requires a prepared kernel");
    ExactContractBuilder contract;
    contract.text("pops.composite-fac-poisson.specialized-provider")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{supported_dimension})
        .scalar(CompositeEllipticGeometry::Cartesian)
        .bytes(kernel_.collective_contract());
    collective_contract_ = std::move(contract).release();
  }

  [[nodiscard]] explicit operator bool() const noexcept { return static_cast<bool>(kernel_); }

  [[nodiscard]] static constexpr PreparedProviderSupport support(
      const CompositeFacCapabilityRequest& request) noexcept {
    if (request.dimension != supported_dimension)
      return PreparedProviderSupport::reject(1, "composite FAC backend is specialized for 2D");
    if (request.geometry != CompositeEllipticGeometry::Cartesian)
      return PreparedProviderSupport::reject(2,
                                             "composite FAC backend requires Cartesian geometry");
    if (!request.cell_centered)
      return PreparedProviderSupport::reject(3, "composite FAC backend requires cell centering");
    if (!request.nested_hierarchy)
      return PreparedProviderSupport::reject(4,
                                             "composite FAC backend requires a nested hierarchy");
    return PreparedProviderSupport::accept();
  }

  [[nodiscard]] std::string_view collective_contract() const noexcept {
    return collective_contract_;
  }

  SolveReport solve(const CompositeFacCapabilityRequest& capability,
                    const CompositeFacPoisson2DInvocation& invocation) const {
    const PreparedProviderSupport decision = support(capability);
    if (!decision.accepted())
      throw std::invalid_argument(std::string(decision.reason));
    if (invocation.spatial_contract.empty() || invocation.operator_contract.empty() ||
        invocation.prepared_workspace == nullptr)
      throw std::invalid_argument("composite FAC 2D invocation is not fully prepared");
    if (invocation.maximum_iterations < 0 || !std::isfinite(invocation.relative_tolerance) ||
        invocation.relative_tolerance < Real(0))
      throw std::invalid_argument("composite FAC 2D solve controls are invalid");

    SolveReport report = kernel_(invocation);
    if (!solve_report_is_publishable(report, invocation.maximum_iterations))
      throw std::runtime_error("composite FAC 2D provider returned a malformed solve report");
    return report;
  }

 private:
  PreparedCompositeFacPoisson2DKernel kernel_{};
  std::string collective_contract_{};
};

}  // namespace pops::numerics::elliptic
