#pragma once

namespace pops {

/// Exact dimension carried by the current Box2D/Fab2D runtime. Dimension-generic local providers
/// advertise their own compile-time dimension and do not change this runtime fact.
inline constexpr int kNativeDimension = 2;

}  // namespace pops
