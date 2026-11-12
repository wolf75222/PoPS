// Included in AmrProgramContext after the regrid frame. N10 lot 3b (DM13, DM6, DM14).
// regrid_read: each cell of the grown target box reads the source cell that is its parent
// under Parent(r), r the ratio from the source level to the active child level, with the
// identification of a periodic domain. It is always a copy into new storage of H'. The one
// relayout is the scratch on the coarsening of each grown target box. No prolongation law
// lives here (DM14): the map is an injection.
public:
std::shared_ptr<field_type> regrid_read(const field_type& source, std::int64_t source_level) const {
  using Job = ::pops::mesh::parallel::RegionTransferJob<Dim>;
  struct Prepared {
    std::shared_ptr<field_type> field, scratch;
    std::vector<Job> jobs;
    Index<Dim> ratio{};
    std::size_t elements = 0;
    std::uint64_t bytes = 0;
  };
  auto prepared = continuation_local_phase(
      [&] {
        if (continuation_parent_phase_)
          throw std::invalid_argument("regrid_read is a law of a regrid child, not of the parent");
        const auto qualification = regrid_field_qualification_(source);
        if (!qualification || qualification->side != FieldTopology::Side::Source)
          throw std::invalid_argument("regrid_read requires a Field that regrid_source issued");
        if (qualification->level != source_level)
          throw std::invalid_argument("regrid_read source level differs from its Field producer");
        const std::int64_t target_level = active_level_;
        // DM13b: a finer source would need a restriction law, which phase 1 does not have (G11).
        if (source_level < 0 || source_level > target_level)
          throw std::invalid_argument(
              "regrid_read reads a source level finer than its target: restriction is left to G11");
        const bool provider = !qualification->provider.empty();
        const auto source_view =
            provider ? continuation_source_topology(qualification->provider, source_level, source)
                     : continuation_source_topology(qualification->block, source_level, source);
        require_regrid_field_topology(source, source_view);
        const auto target =
            provider ? continuation_target_topology(qualification->provider, target_level, source)
                     : continuation_target_topology(qualification->block, target_level, source);
        if (source.distribution().rank_space() != target.distribution().rank_space())
          throw std::invalid_argument("regrid_read source and target have different rank spaces");
        const auto& source_domain = source_view.geometry().domain();
        const auto& target_domain = target.geometry().domain();
        Prepared result;
        Extent<Dim> extent{};
        for (int axis = 0; axis < Dim; ++axis) {
          const auto coarse = source_domain.length(axis), fine = target_domain.length(axis);
          if (coarse <= 0 || fine < coarse || fine % coarse != 0 ||
              fine / coarse > std::numeric_limits<int>::max())
            throw std::invalid_argument("regrid_read levels have no integer refinement ratio");
          result.ratio[axis] = static_cast<int>(fine / coarse);
          extent[axis] = fine / coarse;
        }
        if (refine(source_domain, extent) != target_domain)
          throw std::invalid_argument("regrid_read target domain is not the refined source domain");
        const BoundaryTopology<Dim> boundary = facade_->prepared_amr_boundary_topology();
        std::array<bool, Dim> periodic{};
        for (int axis = 0; axis < Dim; ++axis)
          periodic[static_cast<std::size_t>(axis)] =
              boundary.is_periodic(Face<Dim>{axis, BoundarySide::lower}) &&
              boundary.is_periodic(Face<Dim>{axis, BoundarySide::upper});

        auto field = std::make_shared<field_type>(target.layout(), target.distribution(),
                                                  target.local_rank(), target.components(),
                                                  target.ghosts());
        field->set_val(Real{0});
        target.require_field(*field);
        // Retained until the frame is revoked, so the registered pointer is never recycled.
        require_continuation_regrid_().allocated_fields.push_back(field);
        continuation_bind_regrid_import_(*field, target, source);
        result.field = field;

        // Scratch: the coarsening of each grown target box, owned like its target patch.
        const auto& layout = target.layout();
        const auto& distribution = target.distribution();
        const auto& space = distribution.rank_space();
        std::vector<Box<Dim>> scratch_boxes;
        scratch_boxes.reserve(layout.size());
        for (std::size_t patch = 0; patch < layout.size(); ++patch) {
          Box<Dim> grown = layout[patch];
          for (int axis = 0; axis < Dim; ++axis)
            grown = grown.grow(axis, static_cast<int>(target.ghosts()[axis]));
          scratch_boxes.push_back(coarsen(grown, extent));
        }
        typename field_type::layout_type scratch_layout(scratch_boxes);
        auto scratch_distribution =
            distribution.replicated()
                ? field_type::distribution_type::replicated(scratch_layout, space)
                : field_type::distribution_type::partitioned(scratch_layout, space,
                                                             distribution.owners());
        result.scratch = std::make_shared<field_type>(scratch_layout, scratch_distribution,
                                                      target.local_rank(), source.ncomp(),
                                                      Extent<Dim>{});
        result.scratch->set_val(Real{0});

        // Images of a source box that meet a box, under the periodic identification.
        const auto images = [&](const Box<Dim>& region, const Box<Dim>& box, auto&& visit) {
          std::array<std::int64_t, Dim> low{}, high{}, length{};
          for (int axis = 0; axis < Dim; ++axis) {
            const auto a = static_cast<std::size_t>(axis);
            length[a] = source_domain.length(axis);
            if (!periodic[a]) {
              low[a] = high[a] = 0;
              continue;
            }
            const auto floor_div = [](std::int64_t n, std::int64_t d) {
              return n / d - ((n % d != 0) && ((n < 0) != (d < 0)) ? 1 : 0);
            };
            // Shifts k with box + k L meeting region: k in [ceil((lo_r - hi_b)/L), floor((hi_r - lo_b)/L)].
            low[a] = -floor_div(static_cast<std::int64_t>(box.hi[axis]) - region.lo[axis], length[a]);
            high[a] = floor_div(static_cast<std::int64_t>(region.hi[axis]) - box.lo[axis], length[a]);
          }
          std::array<std::int64_t, Dim> k = low;
          while (true) {
            Index<Dim> offset{};
            for (int axis = 0; axis < Dim; ++axis)
              offset[axis] = static_cast<int>(k[static_cast<std::size_t>(axis)] *
                                              length[static_cast<std::size_t>(axis)]);
            const auto shifted = box.shift(offset);
            const auto meet = region.intersect(shifted);
            if (!meet.empty())
              visit(meet, offset);
            int axis = 0;
            for (; axis < Dim; ++axis) {
              const auto a = static_cast<std::size_t>(axis);
              if (++k[a] <= high[a])
                break;
              k[a] = low[a];
            }
            if (axis == Dim)
              break;
          }
        };

        // Coverage (DM13): every valid target cell has a valid source parent. Global metadata
        // only, so every rank reaches the same verdict.
        const auto& source_layout = source.layout();
        for (std::size_t patch = 0; patch < layout.size(); ++patch) {
          const auto parents = coarsen(layout[patch], extent);
          std::int64_t covered = 0;
          for (std::size_t box = 0; box < source_layout.size(); ++box)
            images(parents, source_layout[box],
                   [&](const Box<Dim>& meet, const Index<Dim>&) { covered += meet.numPts(); });
          if (covered != parents.numPts())
            throw std::invalid_argument(
                "regrid_read: a valid target cell has no valid source parent");
        }

        // Jobs: scratch patch ∩ periodic image of a source patch, in global order.
        const auto ncomp = static_cast<std::size_t>(source.ncomp());
        for (std::size_t patch = 0; patch < scratch_layout.size(); ++patch) {
          for (std::size_t box = 0; box < source_layout.size(); ++box)
            images(scratch_layout[patch], source_layout[box],
                   [&](const Box<Dim>& meet, const Index<Dim>& offset) {
                     Index<Dim> back{};
                     for (int axis = 0; axis < Dim; ++axis)
                       back[axis] = -offset[axis];
                     const auto add = [&](const Index<Dim>& destination) {
                       const auto origin = source.distribution().replicated()
                                               ? destination
                                               : source.distribution().owner(box);
                       result.jobs.push_back(
                           Job{box, patch, origin, destination, meet.shift(back), meet, 0, 0});
                       result.elements += static_cast<std::size_t>(meet.numPts()) * ncomp;
                     };
                     if (distribution.replicated())
                       for (std::size_t rank = 0; rank < space.size(); ++rank)
                         add(space.coordinate(rank));
                     else
                       add(distribution.owner(patch));
                   });
          result.bytes += static_cast<std::uint64_t>(scratch_layout[patch].numPts()) * ncomp *
                          sizeof(Real);
        }
        return result;
      },
      "regrid_read preparation");

  const auto& lane = prepared_execution_lane();
  const auto jobs = prepared.jobs.size();
  const auto& space = prepared.scratch->distribution().rank_space();
  ::pops::mesh::parallel::RegionTransport<Dim> transport(
      ::pops::mesh::parallel::RegionTransferPlan<Dim>{
          space, prepared.scratch->local_rank(), prepared.scratch->ncomp(),
          std::move(prepared.jobs),
          {jobs, space.size(), prepared.elements, prepared.elements, prepared.elements}});
  transport.prepare_collectively(lane);
  transport.execute(
      [&](const auto& job) { return source.fab_global(job.source_patch).view(); },
      [&](const auto& job) { return prepared.scratch->fab_global(job.destination_patch).view(); });

  continuation_local_phase(
      [&] {
        auto& field = *prepared.field;
        const auto& scratch = std::as_const(*prepared.scratch);
        for (std::size_t local = 0; local < field.local_size(); ++local) {
          const auto global = field.global_index(local);
          pops::for_each_cell(
              field.fab(local).grown_box(),
              detail::RegridParentKernel<Dim>{field.fab(local).view(),
                                              scratch.fab_global(global).view(), prepared.ratio,
                                              field.ncomp()});
        }
        relayout_bytes().fetch_add(prepared.bytes, std::memory_order_relaxed);
      },
      "regrid_read parent map");
  return prepared.field;
}

// Σ field(c, k)·volume·κ over the active, uncovered cells of the field's target level, as
// CompositeIntegralKernel measures them, then one collective sum (DM9).
Real regrid_integral(const field_type& field, int component) const {
  const auto& lane = prepared_execution_lane();
  std::exception_ptr error;
  FiniteCompensatedSum accumulated;
  try {
    const auto qualification = regrid_field_qualification_(field);
    if (!qualification || qualification->side != FieldTopology::Side::Target)
      throw std::invalid_argument("regrid_integral requires a Field read onto the target level");
    if (component < 0 || component >= field.ncomp())
      throw std::invalid_argument("regrid_integral component is outside its Field");
    const auto view = qualification->provider.empty()
                          ? continuation_target_topology(qualification->block, qualification->level)
                          : continuation_provider_topology_(qualification->provider,
                                                            qualification->level,
                                                            FieldTopology::Side::Target);
    if (field.layout() != view.layout() || field.distribution() != view.distribution() ||
        field.local_rank() != view.local_rank())
      throw std::invalid_argument("regrid_integral Field differs from its target level layout");
    const auto* active = view.active_mask();
    const auto* fraction = view.volume_fraction();
    const auto& coverage = view.coverage_mask();
    mf_arith_detail::validate_measure(field, {active, nullptr}, "regrid_integral");
    mf_arith_detail::validate_measure(field, {fraction, nullptr},
                                      "regrid_integral volume fraction");
    mf_arith_detail::validate_measure(field, {&coverage, nullptr}, "regrid_integral");
    double volume = 1.;
    for (int axis = 0; axis < Dim; ++axis)
      volume *= view.geometry().spacing(axis);
    if (!(volume > 0.) || !std::isfinite(volume))
      throw std::invalid_argument("regrid_integral requires finite positive cell volumes");
    for (std::size_t local = 0; local < field.local_size(); ++local) {
      const auto sum = pops::for_each_cell_reduce_finite_sum(
          field.box(local),
          detail::CompositeIntegralKernel<Dim, detail::RegridComponentIntegrand<Dim>>{
              detail::RegridComponentIntegrand<Dim>{field.fab(local).view(), component},
              active ? active->fab(local).view() : FieldView<const Real, Dim>{},
              coverage.fab(local).view(),
              fraction ? fraction->fab(local).view() : FieldView<const Real, Dim>{},
              Real(volume), active != nullptr, fraction != nullptr});
      if (!sum.finite())
        throw std::overflow_error("regrid_integral has nonfinite active values");
      if (!field.distribution().replicated() || lane.rank() == 0) {
        accumulated.add(sum.high);
        accumulated.add(sum.low);
      }
    }
    if (!accumulated.finite())
      throw std::overflow_error("regrid_integral has nonfinite accumulation");
  } catch (...) {
    error = std::current_exception();
  }
  const CollectiveCompensatedSum reduced = pops::collective_finite_compensated_sum_or_failure(
      accumulated, static_cast<bool>(error), lane.communicator());
  return reduced.or_vote(
      [&] { converge_owner_reduction_(error, lane, "regrid_integral of a read Field"); });
}
