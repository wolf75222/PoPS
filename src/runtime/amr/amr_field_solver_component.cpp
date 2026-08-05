/// @file
/// @brief Compile-time-ranked external field-solver component boundary.

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/runtime/amr/exact_field_solver_provider.hpp>
#include <pops/runtime/system/prepared_field_solver_component.hpp>

namespace pops::runtime::amr {
namespace {

// External FieldTopology@2 + FieldSolver@2 components are materialized directly by the ranked
// System route.  The former AmrSystem adapter copied exact-ranked fields through Fab2D/Box2D and
// reintroduced a second provider hierarchy.  Keeping this translation unit as a compile-time
// boundary makes the retirement explicit while preserving the library's source partition.
static_assert(runtime::field::PreparedFieldSolverComponent<kNativeDimension>::dimension ==
              kNativeDimension);
static_assert(ExactAmrFieldSolverBuildRequest<kNativeDimension>::hierarchy_type::dimension ==
              kNativeDimension);

}  // namespace
}  // namespace pops::runtime::amr
