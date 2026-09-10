#pragma once

#include <pops/numerics/elliptic/amr/composite_fac_poisson.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace_workspace.hpp>
#include <pops/runtime/amr/hierarchy_tensor_solver_provider.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <vector>

namespace pops::elliptic::nd {

namespace general_composite_detail {
inline constexpr std::string_view provider_id = "pops.hierarchy.general-field.gmres@1";
inline constexpr std::string_view options_id = "pops.hierarchy.general-field@1";
inline constexpr std::string_view coefficient_slot = "pops.general-field.coefficients";
inline constexpr std::string_view rhs_slot = "pops.general-field.rhs";
inline constexpr std::string_view solution_slot = "pops.general-field.solution";

inline std::size_t product(std::size_t a, std::size_t b) {
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
    throw std::length_error("composite field preparation extent overflows size_t");
  return a * b;
}
inline int integer(const PreparedProviderOptions& options, const std::string& key,
                   int fallback = -1) {
  const auto found = options.values.find(key);
  if (found == options.values.end()) {
    if (fallback >= 0)
      return fallback;
    throw std::invalid_argument("missing composite field integer option " + key);
  }
  std::uint64_t value;
  if (const auto* v = std::get_if<std::uint64_t>(&found->second))
    value = *v;
  else if (const auto* v = std::get_if<std::int64_t>(&found->second); v && *v >= 0)
    value = static_cast<std::uint64_t>(*v);
  else
    throw std::invalid_argument("invalid composite field integer option " + key);
  if (value > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
    throw std::invalid_argument("composite field option exceeds native int");
  return static_cast<int>(value);
}
inline Real scalar(const PreparedProviderOptions& options, const std::string& key) {
  const auto found = options.values.find(key);
  if (found == options.values.end())
    throw std::invalid_argument("missing composite field scalar option " + key);
  const auto* value = std::get_if<double>(&found->second);
  if (value == nullptr || !std::isfinite(*value))
    throw std::invalid_argument("non-finite or untyped composite field scalar option " + key);
  return static_cast<Real>(*value);
}

struct Options {
  int components = 0, coefficients = 0, restart = 30;
  std::vector<Real> reaction;
  std::vector<std::vector<Real>> modes;
  std::vector<Real> gauge;
};

template <int Dim>
Options decode(const PreparedProviderOptions& options, int components) {
  if (options.schema_identity != options_id || components < 1)
    throw std::invalid_argument("invalid composite field options schema or arity");
  const auto square = product(static_cast<std::size_t>(components), components);
  if (square > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::length_error("composite field matrix exceeds native component indexing");
  Options result;
  result.components = components;
  result.coefficients = integer(options, "coefficient_components");
  result.restart = integer(options, "restart", 30);
  if (result.restart < 1 || result.restart == std::numeric_limits<int>::max() ||
      (result.coefficients != components && result.coefficients != static_cast<int>(square)))
    throw std::invalid_argument("invalid composite field coefficient width or GMRES restart");
  const int count = integer(options, "modes.count", 0);
  if (count > components)
    throw std::invalid_argument("composite field has more constant modes than components");
  std::set<std::string> known{"coefficient_components", "restart", "modes.count"};
  result.reaction.resize(square);
  for (int i = 0; i < components; ++i)
    for (int j = 0; j < components; ++j) {
      const auto key = "reaction." + std::to_string(i) + "." + std::to_string(j);
      known.insert(key);
      result.reaction[i * components + j] = scalar(options, key);
    }
  result.modes.resize(count, std::vector<Real>(components));
  result.gauge.resize(count);
  for (int k = 0; k < count; ++k) {
    const auto gauge_key = "gauge." + std::to_string(k);
    known.insert(gauge_key);
    result.gauge[k] = scalar(options, gauge_key);
    for (int i = 0; i < components; ++i) {
      const auto key = "modes." + std::to_string(k) + "." + std::to_string(i);
      known.insert(key);
      result.modes[k][i] = scalar(options, key);
    }
  }
  for (int axis = 0; axis < Dim; ++axis)
    for (int side = 0; side < 2; ++side) {
      const auto key = "physical." + std::to_string(axis) + "." + std::to_string(side);
      known.insert(key);
      if (const auto found = options.values.find(key); found != options.values.end()) {
        const auto* value = std::get_if<std::string>(&found->second);
        if (!value || (*value != "periodic" && *value != "neumann"))
          throw std::invalid_argument("unsupported composite field physical boundary");
      }
    }
  for (const auto& [key, value] : options.values)
    if (!known.contains(key))
      throw std::invalid_argument("unknown composite field option " + key);

  // GMRES also accepts nonsymmetric or indefinite reaction without declared modes.
  // With modes, authenticate the complete symmetric PSD kernel contract below.
  if (count == 0)
    return result;
  // Symmetric PSD reaction and its complete constant kernel are authored data.
  // No shared-(1,...,1) physical coupling is inferred here.
  Real scale = Real(0);
  for (Real value : result.reaction)
    scale = std::max(scale, std::abs(value));
  const Real tolerance = Real(128) * std::numeric_limits<Real>::epsilon() *
                         (scale > Real(0) ? scale : Real(1)) * Real(components);
  auto reduced = result.reaction;
  int rank = 0;
  for (int i = 0; i < components; ++i)
    for (int j = 0; j < components; ++j)
      if (result.reaction[i * components + j] != result.reaction[j * components + i])
        throw std::invalid_argument("composite field reaction must be symmetric");
  for (int k = 0; k < components; ++k) {
    const Real diagonal = reduced[k * components + k];
    if (diagonal < -tolerance || !std::isfinite(diagonal))
      throw std::invalid_argument("composite field reaction must be positive semidefinite");
    if (diagonal <= tolerance) {
      for (int j = k + 1; j < components; ++j)
        if (std::abs(reduced[k * components + j]) > tolerance)
          throw std::invalid_argument("composite field reaction has an indefinite zero pivot");
      continue;
    }
    ++rank;
    for (int i = k + 1; i < components; ++i)
      for (int j = k + 1; j < components; ++j)
        reduced[i * components + j] -=
            reduced[i * components + k] * (reduced[k * components + j] / diagonal);
  }
  if (rank + count != components)
    throw std::invalid_argument("declared composite field modes do not cover the reaction kernel");
  for (const auto& mode : result.modes) {
    Real magnitude = Real(0);
    for (Real value : mode)
      magnitude = std::max(magnitude, std::abs(value));
    if (!(magnitude > Real(0)))
      throw std::invalid_argument("composite field constant mode is zero");
    for (int i = 0; i < components; ++i) {
      Real image = Real(0);
      for (int j = 0; j < components; ++j)
        image += result.reaction[i * components + j] * mode[j];
      if (std::abs(image) > tolerance * magnitude)
        throw std::invalid_argument("declared constant mode is not in the reaction kernel");
    }
  }
  return result;
}

template <int Dim>
amr::CompositeFacBuildRequest<Dim> fac_request(
    const runtime::program::HierarchyTensorSolverBuildRequest<Dim>& request) {
  amr::CompositeFacBuildRequest<Dim> result;
  result.ratios = request.ratios;
  std::size_t boxes = 1, cells = 1;
  for (const auto& level : request.levels) {
    boxes = std::max(boxes, level.layout.size());
    const auto count = level.geometry.domain().grow(4).numPts();
    if (count < 1)
      throw std::invalid_argument("invalid composite field domain volume");
    cells = std::max(cells, static_cast<std::size_t>(count));
    const auto pairs = product(level.layout.size(), level.layout.size());
    result.levels.push_back({level.geometry,
                             level.layout,
                             level.distribution,
                             level.local_rank,
                             level.boundary,
                             Extent<Dim>{},
                             amr::fac_detail::unit_ghosts<Dim>(),
                             {level.layout.size(), pairs}});
  }
  std::size_t images = 1;
  for (int axis = 0; axis < Dim; ++axis)
    images = product(images, 3);
  const auto pairs = product(boxes, boxes);
  // Conservative checked envelopes for canonical patch/image jobs. Budgets do not allocate;
  // FAC allocates only the jobs/regions present in the exact authenticated hierarchy.
  const auto jobs = product(product(pairs, images), 32 * Dim);
  const auto peers =
      product(product(boxes, 2), request.levels.front().distribution.rank_space().size());
  const auto elements =
      product(product(jobs, cells), request.levels.front().distribution.rank_space().size());
  auto& budget = result.budget;
  budget.levels = request.levels.size();
  budget.connections = request.ratios.size();
  budget.parent_child_patch_pairs = pairs;
  budget.interpolation_regions = jobs;
  budget.local_scratch_cells = product(product(boxes, cells), 8);
  budget.same_level_halo = {{boxes, pairs}, jobs,     jobs,     images,
                            peers,          elements, elements, elements};
  budget.parent_gather = {jobs, peers, elements, elements, elements};
  budget.fine_restriction = {jobs, peers, elements, elements, elements};
  return result;
}
}  // namespace general_composite_detail

/// One N-component composite solve, with the field-owned provider protocol as its only
/// publication authority. Each diffusion entry reuses FAC's prepared conservative operator;
/// entries never invoke a scalar solve. All Krylov storage is materialized at preparation.
template <int Dim>
class PreparedCompositeGeneralField final
    : public runtime::program::PreparedHierarchyTensorSolver<Dim> {
 public:
  using field_type = MultiFab<Dim>;
  using request_type = runtime::program::HierarchyTensorSolverBuildRequest<Dim>;
  using hierarchy_type = std::vector<field_type>;

  PreparedCompositeGeneralField(request_type request, std::string contract,
                                const ExecutionLane& lane)
      : request_(std::move(request)),
        options_(general_composite_detail::decode<Dim>(request_.options, request_.components)),
        contract_(std::move(contract)),
        lane_(&lane) {
    const int n = options_.components;
    std::optional<amr::CompositeFacBuildRequest<Dim>> scalar_request;
    // Allocate locally before entering any nested collective preparation.
    long failure = 0;
    try {
      scalar_request.emplace(general_composite_detail::fac_request(request_));
      for (const auto& level : request_.levels) {
        const auto ghosts = amr::fac_detail::unit_ghosts<Dim>();
        coefficients_.emplace_back(level.layout, level.distribution, level.local_rank,
                                   options_.coefficients, ghosts);
        validation_.emplace_back(level.layout, level.distribution, level.local_rank,
                                 options_.coefficients, Extent<Dim>{});
        rhs_.emplace_back(level.layout, level.distribution, level.local_rank, n, Extent<Dim>{});
        initial_.emplace_back(level.layout, level.distribution, level.local_rank, n, ghosts);
        solution_.emplace_back(level.layout, level.distribution, level.local_rank, n, ghosts);
        initial_.back().set_val(Real(0));
        solution_.back().set_val(Real(0));
        rhs_.back().set_val(Real(0));
        coefficients_.back().set_val(Real(0));
        Real measure = Real(1);
        for (int axis = 0; axis < Dim; ++axis)
          measure *= level.geometry.spacing(axis);
        measures_.push_back(measure);
      }
      residual_ = make_vector_();
      image_ = make_vector_();
      work_ = make_vector_();
      for (int i = 0; i <= options_.restart; ++i)
        basis_.push_back(make_vector_());
      const auto restart = static_cast<std::size_t>(options_.restart);
      hessenberg_.resize(general_composite_detail::product(restart, restart + 1));
      cosine_.resize(restart);
      sine_.resize(restart);
      y_.resize(restart);
      rotated_.resize(restart + 1);
      entries_.reserve(options_.coefficients);
      rhs_views_.reserve(rhs_.size());
      solution_views_.reserve(solution_.size());
    } catch (...) {
      failure = 1;
    }
    if (all_reduce_max(failure, lane) != 0)
      throw std::runtime_error("composite field tuple allocation failed collectively");
    for (int entry = 0; entry < options_.coefficients; ++entry)
      entries_.push_back(std::make_unique<amr::CompositeFacPoisson<Dim>>(
          *scalar_request, CompositeFacOptions{}, Real(0), &lane, true));
    prepare_nullspace_();
  }

  std::string_view provider_identity() const noexcept override {
    return general_composite_detail::provider_id;
  }
  std::uint64_t provider_version() const noexcept override { return 1; }
  std::string_view exact_prepared_contract() const noexcept override { return contract_; }
  runtime::program::HierarchyTensorSolverExecutionPath execution_path() const noexcept override {
    return runtime::program::HierarchyTensorSolverExecutionPath::DirectProvider;
  }
  int level_count() const noexcept override { return static_cast<int>(solution_.size()); }
  field_type& assembly_target(std::string_view slot, int level) override {
    if (slot == general_composite_detail::coefficient_slot)
      return coefficients_.at(level);
    if (slot == general_composite_detail::rhs_slot)
      return rhs_.at(level);
    throw std::invalid_argument("unknown composite general field assembly slot");
  }
  field_type& solution(int level) override { return solution_.at(level); }
  void stage_initial_guess(int level, const field_type* guess) override {
    auto& destination = initial_.at(level);
    if (guess == nullptr)
      destination.set_val(Real(0));
    else
      copy_field_(*guess, destination);
  }
  const field_type& active_cell_mask(int level) const override { return active_cells(level); }
  const field_type& active_cells(int level) const {
    return entries_.front()->linear_active_level(level);
  }

  /// Prepared operator witness useful for MMS and conservation checks. This does not solve
  /// or publish. Coefficients must be frozen through prepare_coefficients first.
  void prepare_coefficients() {
    if (!validate_coefficients_())
      throw std::invalid_argument("composite diffusion matrix is not finite symmetric SPD");
    for (int entry = 0; entry < options_.coefficients; ++entry) {
      for (int level = 0; level < level_count(); ++level)
        copy_component_(coefficients_[level], entry,
                        entries_[entry]->linear_coefficient_level(level), 0);
      entries_[entry]->prepare_linear_coefficients();
    }
  }
  void apply(const hierarchy_type& input, hierarchy_type& output) {
    authenticate_(input);
    authenticate_(output);
    for (auto& field : output)
      field.set_val(Real(0));
    const int n = options_.components;
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) {
        if (options_.coefficients == n && i != j)
          continue;
        const int entry = options_.coefficients == n ? i : i * n + j;
        auto& scalar = *entries_[entry];
        for (int level = 0; level < level_count(); ++level)
          copy_component_(input[level], j, scalar.phi_level(level), 0);
        scalar.apply_linear_composite(options_.coefficients != n);
        for (int level = 0; level < level_count(); ++level)
          add_component_(scalar.linear_image_level(level), 0, output[level], i, Real(1));
      }
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j)
        if (const Real reaction = options_.reaction[i * n + j]; reaction != Real(0))
          for (int level = 0; level < level_count(); ++level)
            add_component_(input[level], j, output[level], i, reaction);
    mask_(output);
  }
  Real composite_dot(const hierarchy_type& left, const hierarchy_type& right) const {
    authenticate_(left);
    authenticate_(right);
    Real local_sum = Real(0);
    const int n = options_.components;
    for (int level = 0; level < level_count(); ++level) {
      const auto& a = left[level];
      if (a.distribution().replicated() && lane_->rank() != 0)
        continue;
      for (std::size_t patch = 0; patch < a.local_size(); ++patch) {
        const auto av = a.fab(patch).view(), bv = right[level].fab(patch).view();
        const auto mask = active_cells(level).fab(patch).view();
        const Real measure = measures_[level];
        local_sum += for_each_cell_reduce_sum(a.box(patch), [=] POPS_HD(const Index<Dim>& cell) {
          if (mask(cell, 0) < Real(0.5))
            return Real(0);
          Real sum = Real(0);
          for (int component = 0; component < n; ++component)
            sum += av(cell, component) * bv(cell, component);
          return measure * sum;
        });
      }
    }
    return all_reduce_sum(local_sum, *lane_);
  }

 protected:
  SolveReport solve(const runtime::program::HierarchyTensorSolveControls& controls,
                    const ExecutionLane& lane) override {
    if (&lane != lane_)
      throw std::logic_error("composite field execution lane changed");
    SolveReport report;
    try {
      prepare_coefficients();
      copy_(initial_, solution_);
      if (nullspace_) {
        nullspace_->require_compatible(rhs_views_);
        nullspace_->apply_gauge(solution_views_);
      }
    } catch (const FieldNullspaceIncompatibleRhs& error) {
      report.mark_failed(SolveStatus::kIncompatibleRhs, SolveAction::kFailRun, error.what());
      return report;
    } catch (const FieldNullspaceInvalidEvaluation& error) {
      report.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kFailRun, error.what());
      return report;
    } catch (const std::invalid_argument& error) {
      report.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kFailRun, error.what());
      return report;
    }
    compute_residual_();
    report.evaluations = 1;
    const Real reference = norm_(residual_);
    report.reference_residual_norm = reference;
    const Real stop =
        std::max(controls.absolute_tolerance, controls.relative_tolerance * reference);
    update_report_(report, reference);
    if (!std::isfinite(reference))
      return failed_(report, SolveStatus::kInvalidEvaluation);
    if (reference <= stop)
      return accepted_(report);
    while (report.iters < controls.maximum_iterations) {
      const Real beta = report.residual_norm;
      copy_(residual_, basis_[0]);
      scale_(basis_[0], Real(1) / beta);
      std::fill(hessenberg_.begin(), hessenberg_.end(), Real(0));
      std::fill(rotated_.begin(), rotated_.end(), Real(0));
      rotated_[0] = beta;
      int used = 0;
      const int cycle = std::min(options_.restart, controls.maximum_iterations - report.iters);
      for (int column = 0; column < cycle; ++column) {
        apply(basis_[column], work_);
        ++report.evaluations;
        // Twice-modified Gram-Schmidt limits loss of orthogonality near the constant kernel.
        for (int pass = 0; pass < 2; ++pass)
          for (int row = 0; row <= column; ++row) {
            const Real projection = composite_dot(work_, basis_[row]);
            h_(row, column) += projection;
            axpy_(work_, -projection, basis_[row]);
          }
        const Real next = norm_(work_);
        if (!std::isfinite(next))
          return failed_(report, SolveStatus::kInvalidEvaluation);
        h_(column + 1, column) = next;
        if (next > Real(0)) {
          copy_(work_, basis_[column + 1]);
          scale_(basis_[column + 1], Real(1) / next);
        }
        for (int rotation = 0; rotation < column; ++rotation) {
          const Real a = h_(rotation, column), b = h_(rotation + 1, column);
          h_(rotation, column) = cosine_[rotation] * a + sine_[rotation] * b;
          h_(rotation + 1, column) = -sine_[rotation] * a + cosine_[rotation] * b;
        }
        const Real diagonal = h_(column, column), lower = h_(column + 1, column);
        const Real magnitude = std::hypot(diagonal, lower);
        if (!(magnitude > Real(0)) || !std::isfinite(magnitude))
          break;
        cosine_[column] = diagonal / magnitude;
        sine_[column] = lower / magnitude;
        h_(column, column) = magnitude;
        h_(column + 1, column) = Real(0);
        const Real old = rotated_[column];
        rotated_[column] = cosine_[column] * old;
        rotated_[column + 1] = -sine_[column] * old;
        used = column + 1;
        ++report.iters;
        if (std::abs(rotated_[column + 1]) <= stop || next == Real(0))
          break;
      }
      if (used == 0)
        return failed_(report, SolveStatus::kBreakdown);
      for (int row = used; row-- > 0;) {
        Real value = rotated_[row];
        for (int column = row + 1; column < used; ++column)
          value -= h_(row, column) * y_[column];
        y_[row] = value / h_(row, row);
        if (!std::isfinite(y_[row]))
          return failed_(report, SolveStatus::kBreakdown);
      }
      for (int row = 0; row < used; ++row)
        axpy_(solution_, y_[row], basis_[row]);
      if (nullspace_)
        nullspace_->apply_gauge(solution_views_);
      // Only the full, refluxed true residual can accept a tuple; the Hessenberg
      // estimate is exclusively a restart trigger.
      compute_residual_();
      ++report.evaluations;
      update_report_(report, norm_(residual_));
      if (!std::isfinite(report.residual_norm))
        return failed_(report, SolveStatus::kInvalidEvaluation);
      if (report.residual_norm <= stop)
        return accepted_(report);
    }
    return failed_(report, SolveStatus::kIterationLimit);
  }

 private:
  hierarchy_type make_vector_() const {
    hierarchy_type result;
    result.reserve(request_.levels.size());
    for (const auto& level : request_.levels)
      result.emplace_back(level.layout, level.distribution, level.local_rank, options_.components,
                          Extent<Dim>{});
    return result;
  }
  void authenticate_(const hierarchy_type& value) const {
    if (value.size() != solution_.size())
      throw std::invalid_argument("composite tuple depth mismatch");
    for (std::size_t level = 0; level < value.size(); ++level)
      if (value[level].ncomp() != options_.components ||
          value[level].layout() != solution_[level].layout() ||
          value[level].distribution() != solution_[level].distribution() ||
          value[level].local_rank() != solution_[level].local_rank())
        throw std::invalid_argument("composite tuple exact level layout mismatch");
  }
  static void copy_component_(const field_type& source, int from, field_type& target, int to) {
    if (source.layout() != target.layout() || source.distribution() != target.distribution() ||
        source.local_rank() != target.local_rank() || from < 0 || from >= source.ncomp() ||
        to < 0 || to >= target.ncomp())
      throw std::invalid_argument("composite field component copy layout mismatch");
    for (std::size_t local = 0; local < target.local_size(); ++local) {
      const auto input = source.fab(local).view();
      const auto output = target.fab(local).view();
      for_each_cell(target.box(local),
                    [=] POPS_HD(const Index<Dim>& cell) { output(cell, to) = input(cell, from); });
    }
  }
  static void add_component_(const field_type& source, int from, field_type& target, int to,
                             Real a) {
    for (std::size_t local = 0; local < target.local_size(); ++local) {
      const auto input = source.fab(local).view();
      const auto output = target.fab(local).view();
      for_each_cell(target.box(local), [=] POPS_HD(const Index<Dim>& cell) {
        output(cell, to) += a * input(cell, from);
      });
    }
  }
  static void copy_field_(const field_type& source, field_type& destination) {
    if (source.ncomp() != destination.ncomp())
      throw std::invalid_argument("composite field arity mismatch");
    for (int component = 0; component < source.ncomp(); ++component)
      copy_component_(source, component, destination, component);
  }
  static void copy_(const hierarchy_type& source, hierarchy_type& destination) {
    for (std::size_t level = 0; level < source.size(); ++level)
      copy_field_(source[level], destination[level]);
  }
  static void axpy_(hierarchy_type& target, Real a, const hierarchy_type& source) {
    for (std::size_t level = 0; level < target.size(); ++level)
      for (int component = 0; component < target[level].ncomp(); ++component)
        add_component_(source[level], component, target[level], component, a);
  }
  static void scale_(hierarchy_type& target, Real a) {
    for (auto& field : target)
      scale(field, a);
  }
  void mask_(hierarchy_type& fields) const {
    const int n = options_.components;
    for (int level = 0; level < level_count(); ++level)
      for (std::size_t patch = 0; patch < fields[level].local_size(); ++patch) {
        const auto value = fields[level].fab(patch).view();
        const auto active = active_cells(level).fab(patch).view();
        for_each_cell(fields[level].box(patch), [=] POPS_HD(const Index<Dim>& cell) {
          if (active(cell, 0) < Real(0.5))
            for (int component = 0; component < n; ++component)
              value(cell, component) = Real(0);
        });
      }
  }
  Real norm_(const hierarchy_type& value) const {
    const Real squared = composite_dot(value, value);
    return squared >= Real(0) ? std::sqrt(squared) : std::numeric_limits<Real>::quiet_NaN();
  }
  void compute_residual_() {
    apply(solution_, image_);
    copy_(rhs_, residual_);
    axpy_(residual_, Real(-1), image_);
    mask_(residual_);
  }
  Real& h_(int row, int column) {
    return hessenberg_[static_cast<std::size_t>(column) * (options_.restart + 1) + row];
  }
  static void update_report_(SolveReport& report, Real norm) {
    report.residual_norm = norm;
    report.rel_residual =
        report.reference_residual_norm > Real(0) ? norm / report.reference_residual_norm : norm;
  }
  static SolveReport failed_(SolveReport report, SolveStatus status) {
    report.mark_failed(status, SolveAction::kFailRun, "composite_general_field_gmres_failed");
    return report;
  }
  SolveReport accepted_(SolveReport report) {
    // Refresh the covered values and physical ghosts once for the accepted full tuple.
    auto& scalar = *entries_.front();
    for (int component = 0; component < options_.components; ++component) {
      for (int level = 0; level < level_count(); ++level)
        copy_component_(solution_[level], component, scalar.phi_level(level), 0);
      scalar.synchronize_linear_solution();
      for (int level = 0; level < level_count(); ++level)
        for (std::size_t patch = 0; patch < solution_[level].local_size(); ++patch) {
          const auto input = std::as_const(scalar.phi_level(level)).fab(patch).view();
          const auto output = solution_[level].fab(patch).view();
          for_each_cell(
              solution_[level].fab(patch).grown_box(),
              [=] POPS_HD(const Index<Dim>& cell) { output(cell, component) = input(cell, 0); });
        }
    }
    report.mark_solved("composite_general_field_true_residual");
    return report;
  }

  bool validate_coefficients_() {
    const int n = options_.components, width = options_.coefficients;
    Real invalid = Real(0);
    for (int level = 0; level < level_count(); ++level)
      for (std::size_t patch = 0; patch < coefficients_[level].local_size(); ++patch) {
        const auto input = std::as_const(coefficients_[level]).fab(patch).view();
        const auto scratch = validation_[level].fab(patch).view();
        invalid += for_each_cell_reduce_sum(
            coefficients_[level].box(patch), [=] POPS_HD(const Index<Dim>& cell) {
              for (int slot = 0; slot < width; ++slot) {
                const Real value = input(cell, slot);
                if (!std::isfinite(value))
                  return Real(1);
                scratch(cell, slot) = value;
              }
              if (width == n) {
                for (int i = 0; i < n; ++i)
                  if (!(input(cell, i) > Real(0)))
                    return Real(1);
                return Real(0);
              }
              for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                  if (input(cell, i * n + j) != input(cell, j * n + i))
                    return Real(1);
              for (int k = 0; k < n; ++k) {
                const Real pivot = scratch(cell, k * n + k);
                if (!(pivot > Real(0)) || !std::isfinite(pivot))
                  return Real(1);
                for (int i = k + 1; i < n; ++i)
                  for (int j = k + 1; j < n; ++j)
                    scratch(cell, i * n + j) -=
                        scratch(cell, i * n + k) * (scratch(cell, k * n + j) / pivot);
              }
              return Real(0);
            });
      }
    return all_reduce_max(invalid, *lane_) == Real(0);
  }
  void prepare_nullspace_() {
    FieldNullspacePlan<Dim> plan;
    plan.identity = request_.plan_identity + ":constant-modes";
    plan.layout_identity = contract_;
    std::vector<PreparedVectorDistribution<Dim>> distributions;
    long failure = 0;
    try {
      for (int level = 0; level < level_count(); ++level) {
        rhs_views_.push_back(&rhs_[level]);
        solution_views_.push_back(&solution_[level]);
        distributions.push_back(rhs_[level].distribution().replicated()
                                    ? PreparedVectorDistribution<Dim>::replicated()
                                    : PreparedVectorDistribution<Dim>::distributed());
      }
      for (std::size_t k = 0; k < options_.modes.size(); ++k) {
        FieldNullspaceBasis<Dim> basis;
        basis.identity = plan.identity + ":" + std::to_string(k);
        basis.provenance = request_.operator_contract_identity;
        basis.recipe_identity = "authored-constant-component-vector@1";
        basis.component_count = options_.components;
        basis.cell_measure = measures_;
        for (int level = 0; level < level_count(); ++level) {
          const auto& layout = request_.levels[level];
          auto mask =
              std::make_shared<field_type>(layout.layout, layout.distribution, layout.local_rank,
                                           options_.components, Extent<Dim>{});
          auto coverage = std::make_shared<field_type>(layout.layout, layout.distribution,
                                                       layout.local_rank, 1, Extent<Dim>{});
          copy_component_(active_cells(level), 0, *coverage, 0);
          for (int component = 0; component < options_.components; ++component) {
            const Real value = options_.modes[k][component];
            for (std::size_t patch = 0; patch < mask->local_size(); ++patch) {
              const auto output = mask->fab(patch).view();
              for_each_cell(mask->box(patch), [=] POPS_HD(const Index<Dim>& cell) {
                output(cell, component) = value;
              });
            }
          }
          basis.masks.push_back(std::move(mask));
          basis.coverage.push_back(std::move(coverage));
        }
        plan.gauges.push_back({basis.identity, options_.gauge[k]});
        plan.bases.push_back(std::move(basis));
      }
    } catch (...) {
      failure = 1;
    }
    if (all_reduce_max(failure, *lane_) != 0)
      throw std::runtime_error("composite constant-mode preparation failed collectively");
    if (!plan.empty())
      nullspace_ = std::make_unique<FieldNullspaceWorkspace<Dim>>(std::move(plan), rhs_views_,
                                                                  std::move(distributions), *lane_);
  }

  request_type request_;
  general_composite_detail::Options options_;
  std::string contract_;
  const ExecutionLane* lane_;
  hierarchy_type coefficients_, validation_, rhs_, initial_, solution_, residual_, image_, work_;
  std::vector<hierarchy_type> basis_;
  std::vector<Real> measures_, hessenberg_, cosine_, sine_, rotated_, y_;
  std::vector<std::unique_ptr<amr::CompositeFacPoisson<Dim>>> entries_;
  std::vector<const field_type*> rhs_views_;
  std::vector<field_type*> solution_views_;
  std::unique_ptr<FieldNullspaceWorkspace<Dim>> nullspace_;
};

template <int Dim>
class CompositeGeneralFieldProvider final
    : public runtime::program::HierarchyTensorSolverProvider<Dim> {
 public:
  using request_type = runtime::program::HierarchyTensorSolverBuildRequest<Dim>;
  using solver_type = runtime::program::PreparedHierarchyTensorSolver<Dim>;
  std::string_view identity() const noexcept override {
    return general_composite_detail::provider_id;
  }
  std::uint64_t interface_version() const noexcept override { return 1; }
  std::string_view collective_contract() const noexcept override {
    return "pops.general-field.composite-conservative-gmres.runtime-components@1";
  }
  std::vector<std::string> capability_contracts() const override {
    return {"pops.general-field.composite.coverage-metric@1",
            "pops.general-field.composite.conservative-matrix-flux@1",
            "pops.general-field.composite.authored-constant-modes@1",
            "pops.general-field.composite.prepared-lane@1"};
  }
  PreparedProviderOptions default_options() const override {
    return {std::string(general_composite_detail::options_id), {{"restart", std::uint64_t{30}}}};
  }
  PreparedProviderSupport accepts_options(
      const PreparedProviderOptions& options) const noexcept override {
    return options.schema_identity == general_composite_detail::options_id
               ? PreparedProviderSupport::accept()
               : PreparedProviderSupport::reject(1, "invalid general field schema");
  }
  PreparedProviderSupport supports(const request_type& request) const noexcept override {
    try {
      runtime::program::hierarchy_tensor_detail::validate_request(request);
      if (request.assembly_field_slots !=
              std::vector<std::string>{std::string(general_composite_detail::coefficient_slot),
                                       std::string(general_composite_detail::rhs_slot)} ||
          request.solution_field_slot != general_composite_detail::solution_slot)
        return PreparedProviderSupport::reject(2, "invalid general field assembly envelope");
      (void)general_composite_detail::decode<Dim>(request.options, request.components);
      for (const auto& level : request.levels)
        for (int axis = 0; axis < Dim; ++axis)
          for (int side = 0; side < 2; ++side) {
            const Face<Dim> face{axis, side == 0 ? BoundarySide::lower : BoundarySide::upper};
            const bool periodic = level.boundary.topology().is_periodic(face);
            const auto& law = level.boundary.at(face);
            if (!periodic && (law.kind != PhysicalBoundaryKind::neumann || law.value != Real(0)))
              return PreparedProviderSupport::reject(
                  3, "general field requires periodic or zero-Neumann laws");
            const auto key = "physical." + std::to_string(axis) + "." + std::to_string(side);
            if (const auto found = request.options.values.find(key);
                found != request.options.values.end())
              if (std::get<std::string>(found->second) != (periodic ? "periodic" : "neumann"))
                return PreparedProviderSupport::reject(
                    4, "general field physical law conflicts with request");
          }
      return PreparedProviderSupport::accept();
    } catch (...) {
      return PreparedProviderSupport::reject(5,
                                             "invalid general field hierarchy or matrix contract");
    }
  }
  PreparedProviderSupport accepts_execution(
      const request_type& request,
      runtime::program::HierarchyTensorSolverExecutionPath execution) const noexcept override {
    if (!supports(request).accepted())
      return supports(request);
    return execution == runtime::program::HierarchyTensorSolverExecutionPath::DirectProvider
               ? PreparedProviderSupport::accept()
               : PreparedProviderSupport::reject(6, "general field requires direct tuple solve");
  }
  std::string expected_prepared_contract(const request_type& request) const override {
    if (!supports(request).accepted())
      throw std::invalid_argument("general field provider rejected request");
    ExactContractBuilder result;
    result.text(identity())
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .bytes(runtime::program::hierarchy_tensor_detail::request_contract(request));
    return std::move(result).release();
  }
  std::unique_ptr<solver_type> prepare(const request_type& request,
                                       const ExecutionLane& lane) const override {
    return std::make_unique<PreparedCompositeGeneralField<Dim>>(
        request, expected_prepared_contract(request), lane);
  }
};

template <int Dim>
std::shared_ptr<const runtime::program::HierarchyTensorSolverProvider<Dim>>
make_composite_general_field_provider() {
  return std::make_shared<CompositeGeneralFieldProvider<Dim>>();
}

}  // namespace pops::elliptic::nd
