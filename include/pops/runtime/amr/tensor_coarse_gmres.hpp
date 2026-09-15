#pragma once

/// @file
/// @brief Prepared nonsingular GMRES correction for the complete tensor FAC base grid.

#include <pops/mesh/boundary/fill_boundary.hpp>
#include <pops/mesh/boundary/halo_exchange.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/numerics/elliptic/linear/generic_krylov.hpp>
#include <pops/numerics/elliptic/nd/cartesian_tensor_operator.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>

namespace pops::runtime::program::tensor_fac {

/// The Krylov basis and coefficient/diagonal layouts are constructed eagerly. Operator sessions
/// first prepare from real staged coefficients, before the first recurrence, and subsequently
/// reuse their private ghost carrier and communication buffers. No matrix application allocates.
/// This class has no composite mask: every owned valid cell of the FAC base grid is an unknown.
template <int Dim>
class TensorCoarseGmres {
 public:
  using field_type = MultiFab<Dim>;
  using coefficient_fields = std::array<const field_type*, static_cast<std::size_t>(Dim * Dim)>;
  using stencil_type = elliptic::nd::CartesianTensorOperator<
      Dim, elliptic::nd::CartesianTensorDivergenceSign::negative_divergence,
      elliptic::nd::SplitCartesianTensorCoefficients<Dim>>;

 private:
  struct Data {
    Geometry<Dim> geometry;
    const HaloSchedule<Dim>* halo;
    const PreparedPhysicalBoundary<Dim>* boundary;
    elliptic::nd::CartesianTensorStencilOptions options;
    std::string contract;
    std::array<field_type, static_cast<std::size_t>(Dim * Dim)> coefficients;
    field_type inverse_diagonal;
    OperatorEvaluationSnapshot snapshot{};
    PreparedVectorDistribution<Dim> distribution;

    Data(const field_type& prototype, const Geometry<Dim>& geometry_value,
         const HaloSchedule<Dim>& halo_value, const PreparedPhysicalBoundary<Dim>& boundary_value,
         elliptic::nd::CartesianTensorStencilOptions stencil_options, std::string_view owner)
        : geometry(geometry_value), halo(&halo_value), boundary(&boundary_value),
          options(stencil_options), contract(owner),
          inverse_diagonal(prototype.layout(), prototype.distribution(), prototype.local_rank(),
                           1, Extent<Dim>{}),
          distribution(prototype.distribution().replicated()
                           ? PreparedVectorDistribution<Dim>::replicated()
                           : PreparedVectorDistribution<Dim>::distributed()) {
      ExactContractBuilder parameters;
      parameters.text(contract).text(distribution.layout_contract(prototype))
          .scalar(options.zero_flux_faces).scalar(options.dirichlet_faces)
          .scalar(options.arithmetic_diagonal);
      for (int axis = 0; axis < Dim; ++axis) {
        parameters.scalar(geometry.domain().lo[axis]).scalar(geometry.domain().hi[axis])
            .scalar(geometry.lower()[axis]).scalar(geometry.upper()[axis])
            .scalar(boundary->conditions().spacing()[axis]);
        for (BoundarySide side : {BoundarySide::lower, BoundarySide::upper}) {
          const Face<Dim> face{axis, side};
          const auto& value = boundary->conditions().at(face);
          parameters.scalar(boundary->conditions().topology().is_periodic(face))
              .scalar(value.kind).scalar(value.value).scalar(value.alpha).scalar(value.beta);
        }
      }
      contract = std::move(parameters).release();
      for (auto& coefficient : coefficients)
        coefficient = field_type(prototype.layout(), prototype.distribution(),
                                 prototype.local_rank(), 1, prototype.ghosts());
      snapshot.authority = ::pops::detail::fingerprint_seed();
      ::pops::detail::fingerprint_mix(snapshot.authority, contract);
      ::pops::detail::fingerprint_mix(snapshot.authority, "tensor-coarse-gmres-operator-v1");
      snapshot.topology = ::pops::detail::layout_fingerprint(prototype, distribution);
      snapshot.topology_revision = 1;
      snapshot.resources = snapshot.authority;
      ::pops::detail::fingerprint_mix(snapshot.resources, "private-coefficient-and-diagonal-bank");
      // revision remains zero until the first real coefficient image has been copied.
    }

    stencil_type stencil(std::size_t local, const field_type& input) const {
      std::array<FieldView<const Real, Dim>, static_cast<std::size_t>(Dim * Dim)> views{};
      for (std::size_t slot = 0; slot < views.size(); ++slot)
        views[slot] = coefficients[slot].fab(local).view();
      return elliptic::nd::make_cartesian_tensor_operator<
          elliptic::nd::CartesianTensorDivergenceSign::negative_divergence>(
          input.fab(local).view(), elliptic::nd::split_cartesian_tensor_coefficients<Dim>(views),
          geometry, options);
    }
  };

  struct CopyKernel {
    FieldView<Real, Dim> out;
    FieldView<const Real, Dim> in;
    POPS_HD void operator()(const Index<Dim>& index) const { out(index, 0) = in(index, 0); }
  };

  struct ApplyKernel {
    FieldView<Real, Dim> out;
    stencil_type stencil;
    POPS_HD void operator()(const Index<Dim>& index) const {
      out(index, 0) = stencil.image(index);
    }
  };

  struct DiagonalKernel {
    FieldView<Real, Dim> out;
    stencil_type stencil;
    POPS_HD Real operator()(const Index<Dim>& index) const {
      const Real diagonal = stencil.diagonal(index);
      const Real inverse = Real(1) / diagonal;
      constexpr Real infinity = std::numeric_limits<Real>::infinity();
      const bool valid = diagonal > Real(0) && diagonal < infinity && inverse > Real(0) &&
                         inverse < infinity;
      out(index, 0) = valid ? inverse : std::numeric_limits<Real>::quiet_NaN();
      return valid ? Real(0) : Real(1);
    }
  };

  struct PreconditionKernel {
    FieldView<Real, Dim> out;
    FieldView<const Real, Dim> in;
    FieldView<const Real, Dim> inverse_diagonal;
    POPS_HD void operator()(const Index<Dim>& index) const {
      out(index, 0) = inverse_diagonal(index, 0) * in(index, 0);
    }
  };

  struct ApplySession {
    struct Storage {
      std::shared_ptr<const Data> data;
      const ExecutionLane* lane;
      field_type ghosts;
      std::optional<HaloExchange<Dim>> exchange;
      bool prepared = false;

      Storage(std::shared_ptr<const Data> source, const ExecutionLane& execution_lane)
          : data(std::move(source)), lane(&execution_lane),
            ghosts(data->coefficients[0].layout(), data->coefficients[0].distribution(),
                   data->coefficients[0].local_rank(), 1, data->coefficients[0].ghosts()) {}
    };
    std::unique_ptr<Storage> storage;

    ApplySession(std::shared_ptr<const Data> data, const ExecutionLane& lane)
        : storage(std::make_unique<Storage>(std::move(data), lane)) {}

    void prepare() {
      auto& state = *storage;
      if (!state.data->snapshot.valid())
        throw std::logic_error("tensor coarse GMRES has no real coefficient snapshot");
      if (!state.prepared) {
        const bool remote = all_reduce_max(state.data->halo->has_remote_jobs() ? 1L : 0L,
                                           *state.lane) != 0;
        if (remote) {
          HaloExchangeContext context{};
          context.context_generation = 1;
          context.schedule_generation = 1;
          state.exchange.emplace(*state.data->halo, *state.lane, context);
        }
        state.prepared = true;
      }
    }

    PreparedApplyResult apply(field_type& out, const field_type& in) noexcept {
      auto& state = *storage;
      long copy_failed = 0;
      try {
        if (!state.prepared || !state.data->snapshot.valid())
          throw std::logic_error("tensor coarse GMRES operator is unprepared");
        for (std::size_t local = 0; local < in.local_size(); ++local)
          for_each_cell(in.box(local), CopyKernel{state.ghosts.fab(local).view(),
                                                 in.fab(local).view()});
        Kokkos::fence();
      } catch (...) {
        copy_failed = 1;
      }
      try {
        // Agree before entering the halo trace if any rank's local copy/launch failed.
        if (all_reduce_max(copy_failed, *state.lane) != 0)
          return PreparedApplyResult::unknown_failure();
        if (state.exchange)
          state.exchange->execute(state.ghosts, *state.lane);
        else
          fill_boundary(state.ghosts, *state.data->halo);
        fill_physical_boundary(state.ghosts, *state.data->boundary);
        for (std::size_t local = 0; local < out.local_size(); ++local)
          for_each_cell(out.box(local),
                        ApplyKernel{out.fab(local).view(),
                                    state.data->stencil(local, state.ghosts)});
        Kokkos::fence();
        return PreparedApplyResult::success();
      } catch (...) {
        return PreparedApplyResult::unknown_failure();
      }
    }

    std::size_t allocation_count() const noexcept { return 1; }
  };

  struct ApplySource {
    std::shared_ptr<const Data> data;
    static constexpr PreparedProviderIdentity provider_identity() noexcept {
      return {"pops.tensor-fac.coarse-exact-tensor", 1};
    }
    void serialize_exact_parameters(ExactContractBuilder& contract) const {
      contract.text(data->contract);
    }
    ApplySession make_session(const ExecutionLane& lane) const { return {data, lane}; }
  };

  struct PreconditionSession {
    std::shared_ptr<const Data> data;
    void prepare() {
      if (!data->snapshot.valid())
        throw std::logic_error("tensor coarse GMRES diagonal is unprepared");
    }
    PreparedApplyResult apply(field_type& out, const field_type& in) noexcept {
      try {
        for (std::size_t local = 0; local < out.local_size(); ++local)
          for_each_cell(out.box(local),
                        PreconditionKernel{out.fab(local).view(), in.fab(local).view(),
                                           data->inverse_diagonal.fab(local).view()});
        Kokkos::fence();
        return PreparedApplyResult::success();
      } catch (...) {
        return PreparedApplyResult::unknown_failure();
      }
    }
    std::size_t allocation_count() const noexcept { return 0; }
  };

  struct PreconditionSource {
    std::shared_ptr<const Data> data;
    static constexpr PreparedProviderIdentity provider_identity() noexcept {
      return {"pops.tensor-fac.coarse-fixed-diagonal", 1};
    }
    void serialize_exact_parameters(ExactContractBuilder& contract) const {
      contract.text(data->contract);
    }
    PreconditionSession make_session(const ExecutionLane&) const { return {data}; }
  };

 public:
  TensorCoarseGmres(const field_type& prototype, const Geometry<Dim>& geometry,
                   const HaloSchedule<Dim>& halo,
                   const PreparedPhysicalBoundary<Dim>& homogeneous_boundary,
                   elliptic::nd::CartesianTensorStencilOptions options,
                   const ExecutionLane& parent, std::string_view exact_owner, int restart)
      : parent_(&parent), restart_(restart) {
    long failed = 0;
    try {
      if (restart < 1 || restart >= KrylovWorkspace<Dim>::max_batched_basis_extent())
        throw std::invalid_argument("tensor coarse GMRES restart exceeds its prepared basis limit");
      data_ = std::make_shared<Data>(prototype, geometry, halo, homogeneous_boundary, options,
                                     exact_owner);
    } catch (...) {
      failed = 1;
    }
    if (all_reduce_max(failed, parent) != 0)
      throw std::invalid_argument("tensor coarse GMRES local preparation failed collectively");
#ifdef POPS_HAS_MPI
    const auto parent_communicator = ExecutionCommunicator::borrowed(
        "pops.tensor-fac.coarse-gmres.parent", parent.native_handle());
#else
    const auto parent_communicator = ExecutionCommunicator::world();
#endif
    const KrylovFootprint<Dim> footprint{1, prototype.ghosts(), true};
    problem_ = PreparedAffineLinearProblem<Dim>::make_shared_collectively(
        parent_communicator, "pops.tensor-fac.coarse-gmres.problem", [&] {
          return typename PreparedAffineLinearProblem<Dim>::ConstructionInputs{
              std::cref(prototype), PreparedAffineOperatorProvider<Dim>(ApplySource{data_}),
              PreparedLinearPreconditioner<Dim>(
                  prototype, PreparedLinearPreconditionerProvider<Dim>(PreconditionSource{data_}),
                  data_->distribution),
              LinearOperatorProperties::general(), footprint,
              PreparedNullspacePolicy<Dim>::nonsingular(),
              [data = data_] { return data->snapshot; }, {}, data_->distribution, {}};
        });
    workspace_ = KrylovWorkspace<Dim>::make_shared_collectively(
        parent_communicator, "pops.tensor-fac.coarse-gmres.workspace", [&] {
          return typename KrylovWorkspace<Dim>::ConstructionInputs{
              data_->contract, std::cref(prototype), gmres_krylov_method<Dim>(restart_), footprint,
              data_->distribution, {}};
        });
  }

  TensorCoarseGmres(const TensorCoarseGmres&) = delete;
  TensorCoarseGmres& operator=(const TensorCoarseGmres&) = delete;

  /// Copy the full coefficient image once per FAC solve. No alias into staged user storage is
  /// visible to an active GMRES recurrence, and all applications use the same fixed diagonal.
  void prepare_coefficients(const coefficient_fields& source) {
    // A failed refresh must never leave a usable solver pointing at a partially replaced bank.
    prepared_ = false;
    long failed = 0;
    try {
      if (data_->snapshot.revision == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("tensor coarse GMRES coefficient revision overflow");
      for (std::size_t slot = 0; slot < source.size(); ++slot) {
        auto& destination = data_->coefficients[slot];
        if (source[slot] == nullptr || source[slot]->layout() != destination.layout() ||
            source[slot]->distribution() != destination.distribution() ||
            source[slot]->local_rank() != destination.local_rank() ||
            source[slot]->ghosts() != destination.ghosts() || source[slot]->ncomp() != 1)
          throw std::invalid_argument("tensor coarse GMRES coefficient image changed layout");
        for (std::size_t local = 0; local < destination.local_size(); ++local)
          Kokkos::deep_copy(destination.fab(local).storage(), source[slot]->fab(local).storage());
      }
      Kokkos::fence();
      for (std::size_t local = 0; local < data_->inverse_diagonal.local_size(); ++local)
        if (for_each_cell_reduce_max(
                data_->inverse_diagonal.box(local),
                DiagonalKernel{data_->inverse_diagonal.fab(local).view(),
                               data_->stencil(local, data_->coefficients[0])}) != Real(0))
          failed = 1;
      Kokkos::fence();
    } catch (...) {
      failed = 1;
    }
    if (all_reduce_max(failed, *parent_) != 0)
      throw std::invalid_argument("tensor coarse GMRES requires an exact finite positive diagonal");
    ++data_->snapshot.revision;
    problem_->prepare(data_->snapshot);
    workspace_->bind(*problem_);
    prepared_ = true;
  }

  /// The physical convergence norm is the original FAC infinity norm; Arnoldi retains its
  /// Euclidean inner product. The caller independently rechecks the original coefficient image.
  SolveReport solve(field_type& correction, const field_type& rhs, Real tau, int maximum) {
    if (!prepared_ || !std::isfinite(static_cast<double>(tau)) || tau <= Real(0) || maximum < 1)
      throw std::invalid_argument("tensor coarse GMRES solve has no valid prepared tolerance");
    correction.set_val(Real(0));
    KrylovControls<Dim> controls{gmres_krylov_method<Dim>(restart_), Real(0), tau, maximum};
    controls.physical_norm = KrylovPhysicalNorm::component_linf;
    return ::pops::detail::solve_prepared_affine_in_place(
        *problem_, *workspace_, correction, rhs, controls);
  }

 private:
  const ExecutionLane* parent_;
  int restart_;
  std::shared_ptr<Data> data_;
  std::shared_ptr<PreparedAffineLinearProblem<Dim>> problem_;
  std::shared_ptr<KrylovWorkspace<Dim>> workspace_;
  bool prepared_ = false;
};

}  // namespace pops::runtime::program::tensor_fac
