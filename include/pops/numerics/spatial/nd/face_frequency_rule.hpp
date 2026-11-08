/// Rules by which the recorded face speeds of a cell bound an explicit update (ROM-107).
#pragma once

namespace pops::nd {

/// s_a = max(s_lower, s_upper) on the two faces of a cell along axis a.
/// incident_sum: sum_a s_a / h_a, the convex budget of the incident faces.
/// largest_axis: max_a s_a / min_a h_a, which a step bound c min_a h_a / max_a s_a (2.6.9) admits.
enum class FaceFrequencyRule : int { incident_sum = 0, largest_axis = 1 };

}  // namespace pops::nd
