// Included inside ProgramContext's public section. No second runtime/stepper.
  const MovingIntervalGeometry<Dim>& moving_interval_geometry(const std::string& identity) const {
    return runtime_state().moving_interval_geometry_.at(identity);
  }

  void initialize_moving_interval_geometry(const std::string& identity, int program_block) const {
    const auto& lane = prepared_execution_lane();
    std::optional<MovingIntervalGeometry<Dim>> candidate;
    std::exception_ptr error;
    try {
      if constexpr (Dim != 1) {
        throw std::logic_error("moving interval provider requires 1D face geometry; "
                               "higher-dimensional swept surfaces are not prepared");
      } else {
        if (identity.empty() || runtime_state().moving_interval_geometry_.contains(identity) ||
            system_->step_transaction_depth() != 0)
          throw std::invalid_argument("moving geometry initialization requires a fresh identity "
                                      "at an accepted boundary outside a transaction");
        const auto& prototype = state(program_block);
        candidate.emplace();
        candidate->runtime_block = sys_block(program_block);
        candidate->measures = scalar_field_like_(prototype, 1, 0);
        const auto geom = geometry();
        for (std::size_t patch = 0; patch < prototype.local_size(); ++patch) {
          candidate->coordinates.emplace_back(prototype.box(patch), 1);
          candidate->swept_volumes.emplace_back(prototype.box(patch), 1);
          candidate->swept_volumes.back().set_val(Real(0));
          const auto coordinate = candidate->coordinates.back().template field<0>().view();
          for_each_cell(nd::face_box(prototype.box(patch), 0),
                        [=] POPS_HD(const Index<Dim>& face) {
            coordinate(face) = geom.face_coordinate(0, face[0]);
          });
          const auto measure = candidate->measures.fab(patch).view();
          for_each_cell(prototype.box(patch), [=] POPS_HD(const Index<Dim>& cell) {
            Index<Dim> upper = cell;
            ++upper[0];
            measure(cell) = coordinate(upper) - coordinate(cell);
          });
        }
        Kokkos::fence();
      }
    } catch (...) { error = std::current_exception(); }
    collectively_rethrow_exception(error, lane, "moving geometry initialization failed collectively");
    ExactContractBuilder declaration;
    declaration.text("pops.moving-interval.v1").text(identity).scalar(std::int32_t(program_block));
    if (!all_ranks_agree_exact_ordered_byte_pairs({{"moving-geometry", std::move(declaration).release()}}, lane))
      throw std::invalid_argument("moving geometry declaration differs across ranks");
    bool inserted = false;
    try {
      inserted = runtime_state().moving_interval_geometry_.emplace(identity, std::move(*candidate)).second;
    } catch (...) { error = std::current_exception(); }
    if (all_reduce_max(error ? 1L : 0L, lane) != 0) {
      if (inserted) runtime_state().moving_interval_geometry_.erase(identity);
      collectively_rethrow_exception(error, lane, "moving geometry registration failed collectively");
    }
  }

  /// Update the real native state using supplied shared-face reconstructions and
  /// INTEGRATED physical flux/source. The same supplied swept volumes enter GCL
  /// and amount exchange. Accepted geometry/state/ledger remain one transaction.
  void advance_moving_intervals(
      const std::string& identity, int program_block,
      const std::vector<nd::FaceField<Dim>>& proposed_coordinates,
      const std::vector<nd::FaceField<Dim>>& swept_volumes,
      const std::vector<nd::FaceField<Dim>>& integrated_physical_flux,
      const std::vector<nd::FaceField<Dim>>& face_density,
      const field_type& integrated_source, Real geometry_tolerance) const {
    const auto& lane = prepared_execution_lane();
    std::optional<MovingIntervalGeometry<Dim>> candidate;
    std::optional<field_type> updated;
    std::string interval;
    std::vector<std::vector<Real>> local_endpoints;
    std::exception_ptr error;
    try {
      if constexpr (Dim != 1) {
        throw std::logic_error("moving interval provider has no higher-dimensional swept surfaces");
      } else {
        if (system_->step_transaction_depth() == 0 || !std::isfinite(geometry_tolerance) ||
            geometry_tolerance < Real(0))
          throw std::invalid_argument("moving interval update requires an enclosing transaction "
                                      "and a finite nonnegative geometry tolerance");
        const auto& accepted = moving_interval_geometry(identity);
        const auto& input = state(program_block);
        if (accepted.runtime_block != sys_block(program_block))
          throw std::invalid_argument("moving geometry belongs to a different physical state");
        mf_arith_detail::require_same_layout(input, integrated_source, "moving source");
        mf_arith_detail::require_same_layout(input, accepted.measures, "moving measure", false);
        for (const auto* values : {&proposed_coordinates, &swept_volumes,
                                   &integrated_physical_flux, &face_density})
          if (values->size() != input.local_size())
            throw std::invalid_argument("moving interval faces do not match local patch ownership");
        for (std::size_t patch = 0; patch < input.local_size(); ++patch) {
          if (proposed_coordinates[patch].cell_box() != input.box(patch) ||
              swept_volumes[patch].cell_box() != input.box(patch) ||
              integrated_physical_flux[patch].cell_box() != input.box(patch) ||
              face_density[patch].cell_box() != input.box(patch) ||
              proposed_coordinates[patch].ncomp() != 1 || swept_volumes[patch].ncomp() != 1 ||
              integrated_physical_flux[patch].ncomp() != input.ncomp() ||
              face_density[patch].ncomp() != input.ncomp())
            throw std::invalid_argument("moving interval face boxes/component roles disagree");
        }
        ExchangeRecord point{"moving-interval", "interval", identity, "endpoint-swept@1", 1, 1, 0, 1, 1};
        point.qualify_runtime_point(boundary_evaluation_point(0));
        interval = point.evaluation_context;
        if (accepted.last_interval == interval ||
            accepted.generation == std::numeric_limits<std::uint64_t>::max())
          throw std::invalid_argument("moving interval occurrence repeated or generation exhausted");
        candidate = accepted;
        candidate->coordinates = proposed_coordinates;
        candidate->swept_volumes = swept_volumes;
        candidate->last_interval = interval;
        ++candidate->generation;
        updated = input;
        local_endpoints.resize(input.local_size());
        // Only patch endpoints cross ranks. Interior faces already have a single
        // owning FaceField storage. No global dense grid/state gather is used.
        for (std::size_t patch = 0; patch < input.local_size(); ++patch) {
          const auto old = accepted.coordinates[patch].template field<0>().create_host_mirror();
          const auto pos = proposed_coordinates[patch].template field<0>().create_host_mirror();
          const auto sw = swept_volumes[patch].template field<0>().create_host_mirror();
          const auto flux = integrated_physical_flux[patch].template field<0>().create_host_mirror();
          const auto density = face_density[patch].template field<0>().create_host_mirror();
          accepted.coordinates[patch].template field<0>().copy_to_host(old);
          proposed_coordinates[patch].template field<0>().copy_to_host(pos);
          swept_volumes[patch].template field<0>().copy_to_host(sw);
          integrated_physical_flux[patch].template field<0>().copy_to_host(flux);
          face_density[patch].template field<0>().copy_to_host(density);
          const std::size_t count = static_cast<std::size_t>(input.box(patch).length(0)) + 1;
          for (std::size_t endpoint : {std::size_t{0}, count - 1}) {
            auto& packet = local_endpoints[patch];
            packet.insert(packet.end(), {old(endpoint), pos(endpoint), sw(endpoint)});
            for (int component = 0; component < input.ncomp(); ++component) {
              const auto offset = endpoint + count * static_cast<std::size_t>(component);
              packet.push_back(flux(offset));
              packet.push_back(density(offset));
            }
          }
        }
      }
    } catch (...) { error = std::current_exception(); }
    collectively_rethrow_exception(error, lane, "moving interval preparation failed collectively");

    ExactContractBuilder publication;
    try {
    const auto& input = state(program_block);
    publication.text("pops.moving-interval-publication.v2").text(identity).text(interval)
        .scalar(std::int32_t(program_block)).scalar(geometry_tolerance)
        .scalar(std::int32_t(input.ncomp()))
        .scalar(static_cast<std::uint64_t>(input.layout().size()))
        .scalar(candidate->generation);
    publication.text(lane.identity()).scalar(std::int32_t(lane.size()))
        .scalar(std::int32_t(input.distribution().replicated()));
    for (int axis = 0; axis < Dim; ++axis)
      publication.scalar(std::int32_t(input.rank_space().origin()[axis]))
          .scalar(std::int32_t(input.rank_space().extent()[axis]))
          .scalar(std::int32_t(geometry().domain().lo[axis]))
          .scalar(std::int32_t(geometry().domain().hi[axis]))
          .scalar(std::int32_t(scalar_boundary_topology_().is_periodic(
              Face<Dim>{axis, BoundarySide::lower})));
    for (std::size_t global = 0; global < input.layout().size(); ++global) {
      for (int axis = 0; axis < Dim; ++axis)
        publication.scalar(std::int32_t(input.layout()[global].lo[axis]))
            .scalar(std::int32_t(input.layout()[global].hi[axis]));
      const auto owner = input.distribution().replicated() ? std::size_t{0} :
          input.rank_space().linear_rank(input.distribution().owner(global));
      if (owner >= static_cast<std::size_t>(lane.size()))
        throw std::invalid_argument("moving interval owner is outside the prepared execution lane");
      publication.scalar(static_cast<std::uint64_t>(owner));
    }
    } catch (...) { error = std::current_exception(); }
    collectively_rethrow_exception(error, lane, "moving publication contract failed collectively");
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{"moving-publication", std::move(publication).release()}}, lane))
      throw std::invalid_argument("moving interval point/state/tolerance/generation differs across ranks");

    if constexpr (Dim == 1) {
      const auto& input = state(program_block);
      const auto& accepted = moving_interval_geometry(identity);
      const int components = input.ncomp();
      const auto* active_mask = pointwise_active_mask(program_block, input);
      if (all_reduce_max(active_mask ? 1L : 0L, lane) != 0)
        throw std::invalid_argument("moving interval provider requires full physical cells; "
                                    "embedded-boundary moving measures are not prepared");
      std::vector<std::size_t> ordered;
      std::vector<std::vector<Real>> endpoints;
      try {
        ordered.resize(input.layout().size());
        for (std::size_t global = 0; global < ordered.size(); ++global) ordered[global] = global;
        std::sort(ordered.begin(), ordered.end(), [&](auto left, auto right) {
          return input.layout()[left].lo[0] < input.layout()[right].lo[0];
        });
        endpoints.assign(ordered.size(), std::vector<Real>(2 * (3 + 2 * components)));
        int expected = geometry().domain().lo[0];
        for (auto global : ordered) {
          const auto& box = input.layout()[global];
          if (box.lo[0] != expected)
            throw std::invalid_argument("moving interval topology is not an exact full-domain partition");
          expected = box.hi[0] + 1;
        }
        if (expected != geometry().domain().hi[0] + 1)
          throw std::invalid_argument("moving interval partition omits physical cells");
      } catch (...) { error = std::current_exception(); }
      collectively_rethrow_exception(error, lane, "moving shared-face topology failed collectively");
      bool shared_mismatch = false;
      for (std::size_t row = 0; row < ordered.size(); ++row) {
        const auto global = ordered[row];
        const auto local = input.local_index_of(global);
        const int owner = input.distribution().replicated() ? 0 : static_cast<int>(
            input.rank_space().linear_rank(input.distribution().owner(global)));
        if (local != field_type::not_local) endpoints[row] = local_endpoints[local];
        broadcast_bytes_inplace(reinterpret_cast<char*>(endpoints[row].data()),
                                endpoints[row].size() * sizeof(Real), lane, owner);
        if (local != field_type::not_local && endpoints[row] != local_endpoints[local])
          shared_mismatch = true;
        const std::size_t width = endpoints[row].size() / 2;
        if (row != 0 && !std::equal(endpoints[row].begin(), endpoints[row].begin() + width,
                                    endpoints[row - 1].begin() + width))
          shared_mismatch = true;
      }
      const std::size_t width = endpoints.front().size() / 2;
      // This provider moves mesh faces inside a fixed physical interval. A moving
      // physical boundary needs its own law/continuation provider.
      if (endpoints.front()[0] != endpoints.front()[1] ||
          endpoints.back()[width] != endpoints.back()[width + 1]) shared_mismatch = true;
      if (scalar_boundary_topology_().is_periodic(Face<Dim>{0, BoundarySide::lower})) {
        for (std::size_t value = 3; value < width; ++value)
          if (endpoints.front()[value] != endpoints.back()[width + value])
            shared_mismatch = true;
      }
      if (all_reduce_max(shared_mismatch ? 1L : 0L, lane) != 0)
        throw std::invalid_argument("moving shared face or fixed-domain boundary is inconsistent");

      Real invalid = 0;
      try {
      for (std::size_t patch = 0; patch < input.local_size(); ++patch) {
        const auto old = accepted.coordinates[patch].template field<0>().view();
        const auto pos = proposed_coordinates[patch].template field<0>().view();
        const auto sw = swept_volumes[patch].template field<0>().view();
        const auto physical = integrated_physical_flux[patch].template field<0>().view();
        const auto density = face_density[patch].template field<0>().view();
        const auto initial = input.fab(patch).view();
        const auto source = integrated_source.fab(patch).view();
        const auto previous_measure = accepted.measures.fab(patch).view();
        const auto new_measure = candidate->measures.fab(patch).view();
        const auto output = updated->fab(patch).view();
        const Real tol = geometry_tolerance;
        const Real status = for_each_cell_reduce_max(input.box(patch),
            [=] POPS_HD(const Index<Dim>& cell) {
          const auto finite = [] POPS_HD(Real v) { return v == v && v <= std::numeric_limits<Real>::max() &&
                                                 v >= -std::numeric_limits<Real>::max(); };
          Index<Dim> right = cell; ++right[0];
          const Real v0 = old(right) - old(cell), v1 = pos(right) - pos(cell);
          const Real s0 = sw(cell), s1 = sw(right);
          if (!finite(v0) || !finite(v1) || !finite(s0) || !finite(s1) || v0 <= 0 || v1 <= 0 ||
              !finite(old(cell)) || !finite(pos(cell)) || !finite(old(right)) || !finite(pos(right)) ||
              !finite(previous_measure(cell)) ||
              Kokkos::abs(previous_measure(cell) - v0) > tol ||
              Kokkos::abs((pos(cell) - old(cell)) - s0) > tol ||
              Kokkos::abs((pos(right) - old(right)) - s1) > tol ||
              Kokkos::abs((v1 - v0) - (s1 - s0)) > tol) return Real(1);
          new_measure(cell) = v1;
          for (int component = 0; component < components; ++component) {
            if (!finite(initial(cell, component)) || !finite(source(cell, component)) ||
                !finite(density(cell, component)) || !finite(density(right, component)) ||
                !finite(physical(cell, component)) || !finite(physical(right, component))) return Real(1);
            const Real amount = SweptInterval::integrated_amount_update(
                initial(cell, component) * v0, density(cell, component), density(right, component),
                physical(cell, component), physical(right, component), source(cell, component), s0, s1);
            const Real result = amount / v1;
            if (!finite(result)) return Real(1);
            output(cell, component) = result;
          }
          return Real(0);
        });
        invalid = std::max(invalid, status);
      }
      Kokkos::fence();
      } catch (...) { error = std::current_exception(); }
      collectively_rethrow_exception(error, lane, "moving numeric preparation failed collectively");
      if (all_reduce_max(static_cast<double>(invalid), lane) != 0)
        throw StepAttemptRejected(SolveStatus::kInvalidEvaluation, "moving-GCL",
                                   "nonfinite/inverted cell or endpoint/sweep/GCL mismatch");
      // Preserve the physical block's prepared recovery/admissibility contract.
      // A finite density alone does not establish a valid physical state.
      const auto* publication_mask = system_->validate_program_state_publication_candidate_(
          sys_block(program_block), *updated, lane);
      if (all_reduce_max(publication_mask ? 1L : 0L, lane) != 0)
        throw std::invalid_argument("moving interval publication cannot use a static EB mask");

      stage_exchange_batch([&](auto&& stage) {
        for (std::size_t patch = 0; patch < input.local_size(); ++patch) {
          const auto sw = swept_volumes[patch].template field<0>().create_host_mirror();
          const auto flux = integrated_physical_flux[patch].template field<0>().create_host_mirror();
          const auto density = face_density[patch].template field<0>().create_host_mirror();
          const auto source = integrated_source.fab(patch).create_host_mirror();
          swept_volumes[patch].template field<0>().copy_to_host(sw);
          integrated_physical_flux[patch].template field<0>().copy_to_host(flux);
          face_density[patch].template field<0>().copy_to_host(density);
          integrated_source.fab(patch).copy_to_host(source);
          const auto source_view = integrated_source.fab(patch).view();
          const auto box = input.box(patch);
          const std::size_t count = static_cast<std::size_t>(box.length(0)) + 1;
          for (int cell = box.lo[0]; cell <= box.hi[0]; ++cell) {
            for (int component = 0; component < components; ++component) {
              const auto offset = static_cast<std::size_t>(cell - source_view.origin[0]) *
                                      source_view.strides[0] +
                                  static_cast<std::size_t>(component) * source_view.component_stride;
              stage(ExchangeRecord{"source:" + identity + "/component:" + std::to_string(component),
                                   "cell:" + std::to_string(cell), identity, "endpoint-swept@1", 1,
                                   1., static_cast<double>(source(offset)), 1., 1});
            }
            for (int side = 0; side < 2; ++side) {
              const auto face = static_cast<std::size_t>(cell - box.lo[0] + side);
              const auto occurrence = "cell:" + std::to_string(cell) + "/side:" + std::to_string(side);
              const int outward = side == 0 ? -1 : 1;
              stage(ExchangeRecord{"geometry:" + identity, occurrence, identity, "endpoint-swept@1",
                                   outward, 1., static_cast<double>(sw(face)), 1., 1});
              for (int component = 0; component < components; ++component) {
                const auto offset = face + count * static_cast<std::size_t>(component);
                const Real relative = flux(offset) - density(offset) * sw(face);
                stage(ExchangeRecord{"amount:" + identity + "/component:" + std::to_string(component),
                                     occurrence, identity, "endpoint-swept@1", outward, 1.,
                                     static_cast<double>(relative), 1., 1});
              }
            }
          }
        }
      });
      // All allocations, numeric checks and ledger append completed collectively.
      // Both remaining swaps are nonthrowing; no published half-geometry exists.
      static_assert(std::is_nothrow_swappable_v<MovingIntervalGeometry<Dim>>);
      static_assert(std::is_nothrow_swappable_v<field_type>);
      std::swap(runtime_state().moving_interval_geometry_.at(identity), *candidate);
      std::swap(state(program_block), *updated);
    }
  }
